import pandas as pd
import numpy as np
import torch
import cuml

from eggnet.utils.mapping import get_node_target_mask
from eggnet.utils.timing import time_function


@time_function
def cluster(event, eps, min_samples):

    clusterer = cuml.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    # clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=3, allow_single_cluster=True, cluster_selection_epsilon=0)
    hit_label = clusterer.fit_predict(event.hit_embedding)
    event.hit_label = torch.as_tensor(hit_label, device=event.hit_embedding.device)


def cluster_and_match(event, eps, eval_config, time_yes=False):

    if eval_config.get("pT_unit", "MeV") == "MeV":
        pt_min, pt_max = 1000, 50000
    else:
        pt_min, pt_max = 1, 50
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

    if eval_config.get("plot_eta", True):
        eta_bins = np.linspace(-4, 4)

    event.hit_target_mask = get_node_target_mask(
        event, eval_config.get("target_tracks", None)
    )
    stack_data = torch.stack([event.hit_particle_id, event.hit_particle_pt])
    if eval_config.get("plot_eta", True):
        stack_data = torch.concat([stack_data, event.hit_particle_eta.unsqueeze(0)])
    particles = torch.unique(stack_data[:, event.hit_target_mask], dim=1)

    particles_pt_hist = np.histogram(particles[1].cpu().numpy(), bins=pt_bins)[0]
    if eval_config.get("plot_eta", True):
        particles_eta_hist = np.histogram(particles[2].cpu().numpy(), bins=eta_bins)[0]
    else:
        particles_eta_hist = None

    eps_data = pd.DataFrame({
        "eps": [0],
        "n_particles": 0,
        "n_matched_particles": 0,
        "n_matched_tracks": 0,
        "n_matched_target_particles": 0,
        "n_matched_target_tracks": 0,
        "n_tracks": 0,
    })

    cluster(event, eps, 3, time_yes=time_yes)
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
        ]
        + ([event.hit_particle_eta] if eval_config.get("plot_eta", True) else []),
        dim=0,
    )
    uni_track_info, inv_idx, n_matched_hits = torch.unique(
        hit_track_info, dim=1, return_counts=True, return_inverse=True
    )
    matched_track_particle_id = uni_track_info[1][
        (uni_track_info[0] >= 0) & (n_matched_hits / uni_track_info[2] > 0.5)
    ]
    hit_target_track_info = hit_track_info[:, event.hit_target_mask]
    uni_target_track_info, inv_idx, n_matched_target_hits = torch.unique(
        hit_target_track_info, dim=1, return_counts=True, return_inverse=True
    )
    matched_target_tracks = uni_target_track_info[
        [1, 3] + ([4] if eval_config.get("plot_eta", True) else [])
    ][
        :,
        (uni_target_track_info[0] >= 0)
        & (n_matched_target_hits / uni_target_track_info[2] > 0.5),
    ]
    matched_target_particles = torch.unique(matched_target_tracks, dim=1)

    eps_data.n_particles += len(particles[0])
    eps_data.n_matched_particles += len(torch.unique(matched_track_particle_id))
    eps_data.n_matched_tracks += len(matched_track_particle_id)
    eps_data.n_matched_target_particles += len(matched_target_particles[0])
    eps_data.n_matched_target_tracks += len(matched_target_tracks[0])
    eps_data.n_tracks += len(uni_labels[uni_labels >= 0])

    matched_target_particles_pt_hist = np.histogram(
        matched_target_particles[1].cpu().numpy(), bins=pt_bins
    )[0]
    if eval_config.get("plot_eta", True):
        matched_target_particles_eta_hist = np.histogram(
            matched_target_particles[2].cpu().numpy(), bins=eta_bins
        )[0]
    else:
        matched_target_particles_eta_hist = None

    return eps_data, particles_pt_hist, matched_target_particles_pt_hist, particles_eta_hist, matched_target_particles_eta_hist
