import sys
import os
import torch
import numpy as np
from eggnet.utils.cluster import cluster
import matplotlib as mpl
import matplotlib.pyplot as plt
import atlasify

INFER_PYG_FILE = '/global/cfs/projectdirs/m3443/usr/lynkallo/EggNet/experiment/parameter_prediction_confidence/confidence_staging_dir/valset/event000001150.pyg'
USE_TARGET_PARTICLES = True

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert os.path.isfile(INFER_PYG_FILE), f"ERROR: {INFER_PYG_FILE} is not a regular file!"
    infer_data = torch.load(INFER_PYG_FILE)
    infer_data = infer_data.to(device)
    print(f'loaded {infer_data.hit_particle_id.size()} hits and {infer_data.track_particle_id.size()} tracks')
    eps = 0.1
    event = infer_data
    cluster(event, eps, 3, time_yes=False)
    event.hit_target_mask = (event.hit_particle_id != 0)
    assert torch.sum(event.hit_target_mask.int()).item() > 0
    uni_labels, inv_idx, count = torch.unique(
        event.hit_label, return_counts=True, return_inverse=True
    )
    event.hit_track_length = count[inv_idx]
    hit_track_info = torch.stack(
        [
            event.hit_label,
            event.hit_particle_id,
            event.hit_track_length,
            event.hit_particle_pt,
        ],
        dim=0,
    )
    if not USE_TARGET_PARTICLES:
        
        uni_track_info, inv_idx1, n_matched_hits = torch.unique(
            hit_track_info, dim=1, return_counts=True, return_inverse=True
        )
        match_mask = (uni_track_info[0] >= 0) & (n_matched_hits / uni_track_info[2] > 0.5)
        matched_track_particle_id = uni_track_info[1][
            match_mask
        ] 
        matched_track_label_id = uni_track_info[0][
            match_mask
        ]
        event.matched_track_label_id = matched_track_label_id
        event.matched_track_particle_id = matched_track_particle_id
        event.hit_match_mask = match_mask[inv_idx1]
        event.hit_matched_track_particle_id = torch.where(event.hit_match_mask, hit_track_info[1], 0)
    
    if USE_TARGET_PARTICLES:
        hit_target_track_info = hit_track_info[:, event.hit_target_mask]
        uni_target_track_info, inv_idx1, n_matched_target_hits = torch.unique(
            hit_target_track_info, dim=1, return_counts=True, return_inverse=True
        )
        match_mask:torch.Tensor = (uni_target_track_info[0] >= 0) & (n_matched_target_hits / uni_target_track_info[2] > 0.5)
        # matched_target_tracks = uni_target_track_info[
        #     0
        # ] [match_mask] # for broadcasting purposes
        # matched_target_particles = torch.unique(matched_target_tracks, dim=1)
        matched_track_particle_id = uni_target_track_info[1][
            match_mask
        ] 
        matched_track_label_id = uni_target_track_info[0][
            match_mask
        ]
        
        event.matched_target_track_label_id = matched_track_label_id
        event.matched_track_particle_id = matched_track_particle_id
        event.hit_match_mask = match_mask[inv_idx1]
        event.hit_matched_track_particle_id = torch.where(event.hit_match_mask, hit_track_info[1], 0)
    
    # remap matched_target particles to hit
    # print(f"{hit_track_info.shape=}")
    # print(f"{uni_track_info.shape=}")
    # print(f"{inv_idx1.shape=}") 
    # print(f"{inv_idx1.max()=}") 
    # event.hit_label_particle_id = matched_track_particle_id[inv_idx1]
    
    # print(f"{matched_track_particle_id.shape=}")
    # print(f"{matched_target_particles.shape=}")
    # print(f"{matched_target_tracks.shape=}")
    # Plot everything
    plot(event, title_suffix='ree')
    # Plot only particles with a high degree of confidence 
        
    if "hit_parameter_confidence" in event and 'matched_track_particle_id' in event and 'hit_parameters' in event:
        mean_confidence = torch.mean(event.hit_parameter_confidence).item()
        print(f"DEBUG: Mean hit parameter confidence: {mean_confidence:.4f}")
        std_confidence = torch.std(event.hit_parameter_confidence).item()
        print(f"DEBUG: Std Dev of hit parameter confidence: {std_confidence:.4f}")
        z_score = -1.0
        threshold = mean_confidence + z_score * std_confidence
        print(f"DEBUG: Plotting hits with threshold {threshold:.4f} (z={z_score})")
        hit_particle_ids = event.hit_matched_track_particle_id
        valid_hit_mask = hit_particle_ids != 0
        if torch.any(valid_hit_mask):
            valid_particle_ids = hit_particle_ids[valid_hit_mask]
            valid_confidence = event.hit_parameter_confidence[valid_hit_mask]
            unique_pids, inv_idx = torch.unique(valid_particle_ids, return_inverse=True)
            hit_pass = valid_confidence < threshold
            counts = torch.bincount(inv_idx, minlength=unique_pids.numel())
            pass_counts = torch.bincount(
                inv_idx, weights=hit_pass.to(torch.int64), minlength=unique_pids.numel()
            )
            all_pass = pass_counts == counts
            high_confidence_particle_ids = unique_pids[all_pass]
            high_confidence_hit_mask = torch.zeros_like(valid_hit_mask, dtype=torch.bool)
            high_confidence_hit_mask[valid_hit_mask] = all_pass[inv_idx]
            matched_pid_mask = torch.isin(high_confidence_particle_ids, event.matched_track_particle_id)
            high_confidence_particle_ids = high_confidence_particle_ids[matched_pid_mask]
            high_confidence_hit_mask &= torch.isin(hit_particle_ids, high_confidence_particle_ids)
        else:
            high_confidence_hit_mask = torch.zeros_like(valid_hit_mask, dtype=torch.bool)
            high_confidence_particle_ids = event.matched_track_particle_id.new_empty((0,))

        high_confidence_data = {}
        for key in ['hit_particle_id', 'hit_matched_track_particle_id', 'hit_x', 'hit_y', 'hit_z', 'hit_charge_pt_ratio', 'hit_parameter_confidence']:
            high_confidence_data[key] = event[key][high_confidence_hit_mask]
        high_confidence_data['hit_parameters'] = event['hit_parameters'][high_confidence_hit_mask]
        high_confidence_data['matched_track_particle_id'] = high_confidence_particle_ids
        plot(high_confidence_data, title_suffix='high_confidence')
        plot(high_confidence_data, color_based_on_pq_ratio=True, title_suffix='high_confidence_colored')
    else:
        print("WARNING: Could not plot high confidence tracks - missing data keys")
        # find missing datakey
        for key in ['matched_track_particle_id', 'hit_particle_id', 'hit_matched_track_particle_id', 'hit_x', 'hit_y', 'hit_z', 'hit_charge_pt_ratio']:
            if key not in event:
                print(f"DEBUG: Missing data key: {key}")
        
        for key in ['hit_parameter_confidence', 'hit_parameters']:
            if key not in event:
                print(f"DEBUG: Missing data key: {key}")



def plot(data, 
         num_tracks_to_plot=60,
         color_based_on_pq_ratio=False, 
         plot_ground_truth=False, 
         title_suffix=None):
    '''Where data is a dict of tensors containing at least the keys required below'''
    fig = plt.figure()
    all_particle_ids = data['matched_track_particle_id'].cpu().numpy()
    np.random.seed(42)
    ridxs = np.random.randint(0, all_particle_ids.size, num_tracks_to_plot)
    selected_particle_ids = np.sort(all_particle_ids[ridxs])
    # if 'hit_parameter_confidence' in data:
    #     for id in selected_particle_ids:
    #         print("DEBUG: Selected particle id:", id)
    #         print(f"DEUBUG: Selected particle confidence max: {torch.max(data['hit_parameter_confidence'][data['hit_matched_track_particle_id'] == id]).item():.4f}")
    plt.gca().set_aspect("equal", adjustable="box")

    if selected_particle_ids[0].item() == 0: 
        print("DEBUG: zero id selected.")
        selected_particle_ids = selected_particle_ids[1:]
    if plot_ground_truth:
        hit_particle_ids = data['hit_particle_id'].cpu().numpy()
    else:
        hit_particle_ids = data['hit_matched_track_particle_id'].cpu().numpy()
    hit_bitmasks = [hit_particle_ids == id for id in selected_particle_ids]
    if color_based_on_pq_ratio:
        if 'hit_parameters' not in data:
            print("WARNING: Missing data key 'hit_parameters'; skipping plot")
            return
        else:
            charge_momentum_ratio = data['hit_parameters']
            if torch.is_tensor(charge_momentum_ratio):
                charge_momentum_ratio = charge_momentum_ratio.detach().cpu().numpy()
            else:
                charge_momentum_ratio = np.asarray(charge_momentum_ratio)
        if charge_momentum_ratio.size:
            c_min = charge_momentum_ratio.min()
            c_max = charge_momentum_ratio.max()
        else:
            c_min = 0.0
            c_max = 1.0
        if c_max <= c_min:
            c_max = c_min + 1e-6
        norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max, clip=True)
        cmap = mpl.colormaps.get_cmap("seismic")
    i = 0
    for pid, hit_bitmask in zip(selected_particle_ids, hit_bitmasks):
        # print(f"DEBUG: Plotting partcle id {pid}")
        # assert np.sum(hit_bitmask.astype(int)) != 0, np.sum(hit_bitmask.astype(int))
        xs = data['hit_x'].cpu().numpy()[hit_bitmask]
        ys = data['hit_y'].cpu().numpy()[hit_bitmask]
        zs = data['hit_z'].cpu().numpy()[hit_bitmask]
        # assert all([len(d) != 0 for d in [xs, ys, zs]])
        if np.sum(hit_bitmask.astype(int)) == 0: continue
        # Sort based on distance from the vertex
        square_distance:np.ndarray = xs ** 2 + ys ** 2 + zs ** 2
        sorted_idxs = square_distance.argsort()
        if color_based_on_pq_ratio:
            ratio_vals = charge_momentum_ratio[hit_bitmask]
            colors = cmap(norm(ratio_vals))
            xs_sorted = xs[sorted_idxs]
            ys_sorted = ys[sorted_idxs]
            if xs_sorted.size > 1:
                line_colors = (colors[sorted_idxs][:-1] + colors[sorted_idxs][1:]) * 0.5
                points = np.column_stack([xs_sorted, ys_sorted])
                segments = np.stack([points[:-1], points[1:]], axis=1)
                line = mpl.collections.LineCollection(
                    segments, colors=line_colors, linewidths=1.5
                )
                plt.gca().add_collection(line)
            else:
                plt.plot(xs_sorted, ys_sorted, color=colors[sorted_idxs][0], linewidth=1.5)
            plt.scatter(xs_sorted, ys_sorted, c=colors[sorted_idxs], s=16)
        else:
            plt.plot(xs[sorted_idxs], ys[sorted_idxs], linewidth=1.5)
            plt.scatter(xs, ys, s=16)
        i += 1
    atlasify.atlasify(
        atlas="Internal",
        # enlarge=False,
        outside=True,
        font_size=10,
        sub_font_size=8,
        axes=plt.gca(),
        label_font_size=10,
        subtext=(
            r"$\sqrt{s}=14$TeV, $t \bar{t}$, $\langle \mu \rangle = 200$, primaries"
            r" $t \bar{t}$ and soft interactions) " + "\n"
        ),
        offset=8
    )
    # plt.xlabel(r'$x$ position') 
    # plt.ylabel(r'$y$ position')
    title_suffix:str
    if title_suffix and 'high_confidence' in title_suffix:
        plt.title(r'High Confidence ($> 1 \sigma$) Track Visualization')
    else:
        plt.title(r'Track Visualization')
    if color_based_on_pq_ratio:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=plt.gca(), label=r'$q/p$ prediction')
    ax = plt.gca()
    # ax.tick_params(labelbottom=True, labelleft=True, direction="in", pad=-8)
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_alpha(0.6)
    #     label.set_clip_on(True)
    plt.tight_layout()
    fp = os.path.join(f"tracks{'_' + title_suffix if title_suffix is not None else ''}.png")
    plt.savefig(fp)
    print("INFO: Saved figure to " + fp)


def export_path_file(path, data) -> bool:
    '''Export the data to a file at the given path'''
    ...

if __name__ == '__main__': main()
