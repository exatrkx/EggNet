import sys
import os
import torch
import numpy as np
from eggnet.utils.cluster import cluster
import matplotlib as mpl
import matplotlib.pyplot as plt

INFER_PYG_FILE = '/global/cfs/projectdirs/m3443/usr/lynkallo/EggNet/experiment/test/testset/event000010000.pyg'
USE_TARGET_PARTICLES = True

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    plot(event)

def plot(data, plot_ground_truth=False):
    '''Where data is a dict of tensors containing at least the keys required below'''
    all_particle_ids = data['matched_track_particle_id'].cpu().numpy()
    np.random.seed(0)
    ridxs = np.random.randint(0, all_particle_ids.size, 30)
    selected_particle_ids = np.sort(all_particle_ids[ridxs])
    print(f"{selected_particle_ids = }")
    if selected_particle_ids[0].item() == 0: 
        print("DEBUG: zero id selected.")
        selected_particle_ids = selected_particle_ids[1:]
    if plot_ground_truth:
        hit_particle_ids = data['hit_particle_id'].cpu().numpy()
    else:
        hit_particle_ids = data['hit_matched_track_particle_id'].cpu().numpy()
    hit_bitmasks = [hit_particle_ids == id for id in selected_particle_ids]
    i = 0
    for pid, hit_bitmask in zip(selected_particle_ids, hit_bitmasks):
        print(f"DEBUG: Plotting partcle id {pid}")
        # assert np.sum(hit_bitmask.astype(int)) != 0, np.sum(hit_bitmask.astype(int))
        xs = data['hit_x'].cpu().numpy()[hit_bitmask]
        ys = data['hit_y'].cpu().numpy()[hit_bitmask]
        zs = data['hit_z'].cpu().numpy()[hit_bitmask]
        # assert all([len(d) != 0 for d in [xs, ys, zs]])
        if np.sum(hit_bitmask.astype(int)) == 0: continue
        # Sort based on distance from the vertex
        square_distance:np.ndarray = xs ** 2 + ys ** 2 + zs ** 2
        sorted_idxs = square_distance.argsort()
        plt.plot(xs[sorted_idxs], ys[sorted_idxs])
        plt.scatter(xs, ys, color='red')
        i += 1
            
    fp = os.path.join(f'tracks{("_target" if USE_TARGET_PARTICLES else "")}.png')
    plt.savefig(fp)
    print("INFO: Saved figure to " + fp)

def export_path_file(path, data) -> bool:
    '''Export the data to a file at the given path'''
    ...

if __name__ == '__main__': main()

