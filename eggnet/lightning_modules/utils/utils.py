import torch

from eggnet.utils.mapping import get_node_target_mask, get_target, get_weight, get_number_of_true_edges
from eggnet.utils.cluster import cluster


def get_optimizers(parameters, hparams):
    """Get the optimizer and scheduler."""
    weight_decay = hparams.get("lr_weight_decay", 0.01)
    optimizer = [
        torch.optim.AdamW(
            parameters,
            lr=(hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=weight_decay,
        )
    ]

    if (
        "scheduler" not in hparams
        or hparams["scheduler"] is None
        or hparams["scheduler"] == "StepLR"
    ):
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=hparams["patience"],
                    gamma=hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    elif hparams["scheduler"] == "ReduceLROnPlateau":
        metric_mode = hparams.get("metric_mode", "min")
        metric_to_monitor = hparams.get("metric_to_monitor", "val_loss")
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode=metric_mode,
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": metric_to_monitor,
            }
        ]
    elif hparams["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer[0],
                    T_0=hparams["patience"],
                    T_mult=2,
                    eta_min=1e-8,
                    last_epoch=-1,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    else:
        raise ValueError(f"Unknown scheduler: {hparams['scheduler']}")

    return optimizer, scheduler


def cluster_eval(batch, hparams):

    if (
        "true_default" not in hparams.get("weighting")
        or hparams.get("weighting")["true_default"] != 0
    ):
        signal_mask = batch.hit_particle_id != 0
    else:
        signal_mask = torch.zeros_like(batch.hit_particle_nhits, dtype=torch.bool)

    if (
        "conditional_weighting" in hparams.get("weighting")
        and hparams.get("weighting")["conditional_weighting"]
    ):
        for weight_spec in hparams.get("weighting")["conditional_weighting"]:
            graph_mask = get_node_target_mask(batch, target_tracks=weight_spec["conditions"])

            if weight_spec["weight"] == 0:
                signal_mask &= ~graph_mask
            else:
                signal_mask |= graph_mask

    particles = torch.unique(batch.hit_particle_id[batch.hit_particle_id != 0])
    target_particles = torch.unique(batch.hit_particle_id[signal_mask])

    cluster(batch, 0.1, 3, node_filter=True if hparams.get("node_hard_filter") else False)
    uni_labels, inv_idx, count = torch.unique(
        batch.hit_label, return_counts=True, return_inverse=True
    )
    batch.hit_track_length = count[inv_idx]
    hit_track_info = torch.stack(
        [
            batch.hit_label,
            batch.hit_particle_id,
            batch.hit_track_length,
        ], dim=0,
    )
    uni_track_info, inv_idx, n_matched_hits = torch.unique(
        hit_track_info, dim=1, return_counts=True, return_inverse=True
    )
    matched_track_particle_id = uni_track_info[1][
        (uni_track_info[0] >= 0) & (n_matched_hits / uni_track_info[2] > 0.5)
    ]

    hit_target_track_info = hit_track_info[:, signal_mask]
    uni_target_track_info, inv_idx, n_matched_target_hits = torch.unique(
        hit_target_track_info, dim=1, return_counts=True, return_inverse=True
    )
    matched_target_tracks = uni_target_track_info[1][
        (uni_target_track_info[0] >= 0)
        & (n_matched_target_hits / uni_target_track_info[2] > 0.5),
    ]
    matched_target_particles = torch.unique(matched_target_tracks)

    n_particles = len(particles)
    n_matched_particles = len(torch.unique(matched_track_particle_id))
    n_matched_tracks = len(matched_track_particle_id)
    n_target_particles = len(target_particles)
    n_matched_target_particles = len(matched_target_particles)
    n_matched_target_tracks = len(matched_target_tracks)
    n_tracks = len(uni_labels[uni_labels >= 0])

    eff = n_matched_particles / n_particles if n_particles != 0 else 0
    signal_eff = n_matched_target_particles / n_target_particles if n_target_particles != 0 else 0
    dup = (
        n_matched_target_tracks - n_matched_target_particles
    ) / n_matched_target_particles if n_matched_target_particles != 0 else 0
    fak = (n_tracks - n_matched_tracks) / n_matched_particles if n_matched_particles != 0 else 0

    return eff, signal_eff, dup, fak


def knn_eval(batch, hparams):

    edges = get_knn_graph(
        batch,
        k=hparams["knn_val"],
        r=hparams.get("r_max"),
        algorithm=hparams.get("knn_algorithm", "cu_knn"),
    )
    if hparams.get("node_filter"):
        edges = batch.filter_node_list[edges]

    y = get_target(edges, batch.hit_particle_id)
    w = get_weight(batch, edges, y, weighting_config=hparams.get("weighting"))
    tp = torch.sum(y == 1)
    target_tp = torch.sum((y == 1) & (w > 0))

    eff = (
        tp
        / get_number_of_true_edges(
            batch,
            reduction="sum",
            upper_bound=hparams["knn_val"],
            weighting_config=hparams.get("weighting"),
        )[1]
    )
    signal_eff = (
        target_tp
        / get_number_of_true_edges(
            batch,
            target="weight-based",
            reduction="sum",
            upper_bound=hparams["knn_val"],
            weighting_config=hparams.get("weighting"),
        )[1]
    )
    pur = tp / len(y)
    f1 = 2 * (eff * pur) / (eff + pur)

    return eff, signal_eff, pur, f1
