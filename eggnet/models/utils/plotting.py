from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os

from .utils import *
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from eggnet.utils.plotting import plot_1d_histogram
import pandas as pd
from pathlib import Path
from typing import List
from .utils import *
from eggnet.utils import nearest_neighboring
from eggnet.utils.mapping import get_node_target_mask
from eggnet.models.fast_walkthrough import FastWalkthrough
from eggnet.models.utils import cc_and_walk_utils, fast_walkthrough_utils
from eggnet.models.utils.walkthrough_diagnostics import build_track_particle_diagnostics
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

CUTOFF_PROPORTIONS = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]

_DML_SCAN_CACHE_DATA_KEYS = (
    "input_dir",
    "phi_segmented",
    "graph_fraction",
    "graph_adjustment_tol",
    "min_nodes",
    "max_nodes",
    "graph_fraction_adjustment_method",
    "max_possible_width",
    "hard_cuts",
    "double_metric_learning",
)

_DML_SCAN_CACHE_RECONSTRUCTION_KEYS = (
    "walkthrough_model",
    "score_cut_cc",
    "initial_edge_radius",
    "score_cut_walk",
    "cc_only",
    "reuse_hits",
    "walk_mode",
    "lookback",
    "knn_algorithm",
)


def _build_dml_eps_values(eval_config):
    configured_eps_values = eval_config.get("eps_values")
    if configured_eps_values is None:
        eps_scan_start = float(eval_config.get("eps_scan_start", 0.05))
        eps_scan_stop = float(eval_config.get("eps_scan_stop", 0.5))
        eps_scan_step = float(eval_config.get("eps_scan_step", 0.05))
        if eps_scan_step <= 0.0:
            raise ValueError("eps_scan_step must be positive.")
        if eps_scan_stop < eps_scan_start:
            raise ValueError("eps_scan_stop must be greater than or equal to eps_scan_start.")
        eps_values = np.arange(
            eps_scan_start,
            eps_scan_stop + (0.5 * eps_scan_step),
            eps_scan_step,
            dtype=np.float64,
        )
    else:
        eps_values = np.asarray(configured_eps_values, dtype=np.float64).reshape(-1)
        if eps_values.size == 0:
            raise ValueError("eps_values must contain at least one cutoff.")
        eps_values = np.unique(eps_values)
    eval_eps = float(eval_config["eps"])
    if not np.isclose(eps_values, eval_eps).any():
        eps_values = np.sort(
            np.concatenate([eps_values, np.asarray([eval_eps], dtype=np.float64)])
        )
    return eps_values


def _get_dml_efficiency_bins(eval_config, include_eta):
    if eval_config.get("pT_unit", "MeV") == "MeV":
        pt_min, pt_max = 1000, 50000
    else:
        pt_min, pt_max = 1, 50
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
    eta_bins = np.linspace(-4, 4) if include_eta else None
    return pt_bins, eta_bins


def _extract_target_particles(graph: Data, eval_config: dict, include_eta: bool):
    target_mask = get_node_target_mask(graph, eval_config.get("target_tracks", None))
    particle_info = [
        graph.hit_particle_id.long(),
        graph.hit_particle_pt.float(),
    ]
    if include_eta:
        particle_info.append(graph.hit_particle_eta.float())
    particles = torch.unique(torch.stack(particle_info, dim=0)[:, target_mask], dim=1)
    return target_mask, particles


def _match_dml_tracks(graph: Data, track_labels: torch.Tensor, target_mask: torch.Tensor, include_eta: bool):
    unique_labels, inverse, counts = torch.unique(
        track_labels,
        return_inverse=True,
        return_counts=True,
    )
    hit_track_length = counts[inverse]
    hit_track_info = torch.stack(
        [
            track_labels.long(),
            graph.hit_particle_id.long(),
            hit_track_length.long(),
            graph.hit_particle_pt.float(),
        ]
        + ([graph.hit_particle_eta.float()] if include_eta else []),
        dim=0,
    )

    unique_track_info, _, matched_hit_counts = torch.unique(
        hit_track_info,
        dim=1,
        return_counts=True,
        return_inverse=True,
    )
    matched_track_mask = (
        (unique_track_info[0] >= 0)
        & (matched_hit_counts / unique_track_info[2].float() > 0.5)
    )
    matched_track_particle_ids = unique_track_info[1][matched_track_mask]

    target_track_info = hit_track_info[:, target_mask]
    unique_target_track_info, _, matched_target_hit_counts = torch.unique(
        target_track_info,
        dim=1,
        return_counts=True,
        return_inverse=True,
    )
    matched_target_track_mask = (
        (unique_target_track_info[0] >= 0)
        & (matched_target_hit_counts / unique_target_track_info[2].float() > 0.5)
    )
    target_columns = [1, 3] + ([4] if include_eta else [])
    matched_target_tracks = unique_target_track_info[target_columns][
        :,
        matched_target_track_mask,
    ]
    matched_target_particles = torch.unique(matched_target_tracks, dim=1)

    return {
        "n_matched_particles": int(torch.unique(matched_track_particle_ids).numel()),
        "n_matched_tracks": int(matched_track_particle_ids.numel()),
        "n_matched_target_particles": int(matched_target_particles.shape[1]),
        "n_matched_target_tracks": int(matched_target_tracks.shape[1]),
        "n_tracks": int((unique_labels >= 0).sum().item()),
        "matched_target_particles": matched_target_particles,
    }


def _get_walkthrough_scan_hparams(hparams: dict, cutoff: float):
    base_cc_cut = float(hparams["score_cut_cc"])
    base_initial_edge_radius = float(hparams.get("initial_edge_radius", base_cc_cut))
    base_walk_min = float(hparams.get("score_cut_walk", {}).get("min", base_cc_cut))
    base_walk_add = float(hparams.get("score_cut_walk", {}).get("add", base_walk_min))

    scan_hparams = dict(hparams)
    scan_hparams["score_cut_cc"] = float(cutoff)
    scan_hparams["initial_edge_radius"] = max(
        float(cutoff) + (base_initial_edge_radius - base_cc_cut),
        float(cutoff),
    )
    walk_min = max(float(cutoff) + (base_walk_min - base_cc_cut), 0.0)
    walk_add = max(float(cutoff) + (base_walk_add - base_cc_cut), 0.0)
    scan_hparams["score_cut_walk"] = {
        "min": walk_min,
        "add": min(walk_add, walk_min),
    }
    scan_hparams["save_graph"] = False
    scan_hparams["save_walkthrough_graphs"] = False
    scan_hparams["enable_profiling"] = False
    walkthrough_output_dir = scan_hparams.get("walkthrough_output_dir")
    if walkthrough_output_dir is not None:
        scan_hparams["stage_dir"] = walkthrough_output_dir
    return scan_hparams


def _run_walkthrough_reconstruction(graph: Data, hparams: dict):
    walkthrough = FastWalkthrough(hparams)
    walkthrough.eval()
    graph_cuda = graph.to("cuda")
    score_name = "track_edge_scores_inverted"
    edge_index = walkthrough._get_initial_edge_index(graph_cuda)
    filtered_graph = fast_walkthrough_utils.filter_graph(
        graph_cuda,
        edge_index,
        score_name,
        1 - hparams["score_cut_cc"],
    )
    filtered_graph = cc_and_walk_utils.remove_cycles(filtered_graph)

    all_tracks = {}
    all_tracks["cc"], walk_input_graph = fast_walkthrough_utils.get_simple_path(
        filtered_graph,
        profile_target=None,
        profiling_enabled=False,
        include_structure_metadata=False,
    )
    if not hparams.get("cc_only", False):
        all_tracks["walk"] = fast_walkthrough_utils.walk_through(
            walk_input_graph,
            score_name,
            1 - hparams["score_cut_walk"]["min"],
            1 - hparams["score_cut_walk"]["add"],
            hparams.get("reuse_hits", False),
            hparams.get("walk_mode", 0),
            hparams.get("lookback", False),
            profile_target=None,
            profiling_enabled=False,
            include_structure_metadata=False,
        )

    reconstructed_graph = graph.clone().cpu()
    cc_and_walk_utils.add_track_labels(reconstructed_graph, all_tracks)
    reconstructed_graph.reco_tracks = cc_and_walk_utils.join_track_lists(all_tracks)
    return reconstructed_graph


def _can_reuse_saved_walkthrough_graph(graph: Data, hparams: dict, cutoff: float):
    return (
        np.isclose(float(cutoff), float(hparams["score_cut_cc"]))
        and hasattr(graph, "hit_track_labels")
        and getattr(graph, "hit_track_labels") is not None
    )


def _get_dml_reconstructed_graph(graph: Data, hparams: dict, cutoff: float):
    if _can_reuse_saved_walkthrough_graph(graph, hparams, cutoff):
        return graph
    scan_hparams = _get_walkthrough_scan_hparams(hparams, float(cutoff))
    return _run_walkthrough_reconstruction(graph, scan_hparams)


def _get_dataset_size(dataset):
    try:
        return len(dataset)
    except TypeError:
        return None


def _get_dml_knn_class(hparams: dict):
    return getattr(
        nearest_neighboring,
        hparams.get("knn_algorithm", "cu_knn"),
    )()


def _get_dml_track_edges_and_distances(graph: Data, hparams: dict, knn_class=None):
    if knn_class is None:
        knn_class = _get_dml_knn_class(hparams)
    graph_cuda = graph.to("cuda")
    track_edges = knn_class.get_graph(
        graph_cuda,
        1,
        use_double_metric_learning=True,
    )
    distances = get_edge_distances(graph_cuda, edges=track_edges)
    return graph_cuda, track_edges, distances.detach().cpu()


def _get_fake_edges_within_radius(graph: Data, radius=1.5):
    src = graph.src_embedding.detach().cpu().numpy()
    tgt = graph.tgt_embedding.detach().cpu().numpy()
    if src.size == 0 or tgt.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=graph.src_embedding.device)
    pairs = cKDTree(src).sparse_distance_matrix(
        cKDTree(tgt),
        radius,
        output_type="ndarray",
    )
    if pairs.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=graph.src_embedding.device)

    cand_src = pairs["i"].astype(np.int64, copy=False)
    cand_tgt = pairs["j"].astype(np.int64, copy=False)
    n_tgt = int(tgt.shape[0])
    cand_ids = cand_src * n_tgt + cand_tgt

    true_edges = graph.track_edges.detach().cpu().numpy()
    if true_edges.size == 0:
        fake_src = cand_src
        fake_tgt = cand_tgt
    else:
        true_ids = (
            true_edges[0].astype(np.int64, copy=False) * n_tgt
            + true_edges[1].astype(np.int64, copy=False)
        )
        fake_mask = ~np.isin(cand_ids, true_ids, assume_unique=False)
        fake_src = cand_src[fake_mask]
        fake_tgt = cand_tgt[fake_mask]

    if fake_src.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=graph.src_embedding.device)
    fake_edges = np.vstack((fake_src, fake_tgt))
    return torch.as_tensor(fake_edges, dtype=torch.long, device=graph.src_embedding.device)


def _hist_payload(distance_values):
    hist, bins = np.histogram(distance_values, bins=100)
    err = np.zeros_like(hist)
    ymax = float(hist.max()) if hist.size else 0.0
    return hist, bins, err, ymax


def _plot_distance_histogram_from_values(
    true_distance_values,
    output_dir: Path | List[str],
    plot_false_edges=False,
    seperate_plots=False,
    plot_suffix=None,
    fake_distance_values=None,
):
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    def _plot_and_save(distance_values, title, output_path, color="black", label="Edge distance"):
        hist, bins, err, ymax = _hist_payload(distance_values)
        ylim = (0.0, ymax * 1.1 if ymax > 0.0 else 1.0)
        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            r"$L_2$ Distance",
            "Count",
            ylim,
            label,
            logy=True,
            tightlayout=False,
            color=color,
        )
        for prop in CUTOFF_PROPORTIONS:
            cutoff = _find_cutoff(prop, hist, bins)
            ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.0)
            ax.text(
                cutoff,
                0.95,
                f"{prop*100:0.1f}% @ {cutoff:0.2f}",
                rotation=90,
                va="top",
                ha="right",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(output_path)
        plt.clf()
        print("INFO: Saved distance plot to " + str(output_path))

    def _plot_efficiency_purity(true_values, fake_values):
        true_sorted = np.sort(true_values)
        fake_sorted = np.sort(fake_values)
        if true_sorted.size == 0:
            return

        max_dist = float(max(true_sorted[-1], fake_sorted[-1] if fake_sorted.size else true_sorted[-1]))
        bins = np.linspace(0.0, max_dist, 101, dtype=np.float64)
        cutoffs = bins[1:]

        tp = np.searchsorted(true_sorted, cutoffs, side="right").astype(np.float64)
        fp = np.searchsorted(fake_sorted, cutoffs, side="right").astype(np.float64)

        total_true = float(true_sorted.size)
        efficiency = tp / total_true
        purity = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0.0)
        err = np.zeros_like(efficiency)

        fig, ax = plot_1d_histogram(
            efficiency,
            bins,
            err,
            r"$L_2$ Distance Cutoff",
            "Value",
            (0.0, 1.02),
            "Efficiency",
            logy=False,
            tightlayout=False,
            color="black",
            fmt="o",
        )
        plot_1d_histogram(
            purity,
            bins,
            err,
            r"$L_2$ Distance Cutoff",
            "Value",
            (0.0, 1.02),
            "Purity",
            canvas=(fig, ax),
            logy=False,
            tightlayout=False,
            color="red",
            fmt="x",
        )
        true_hist_for_cut, true_bins_for_cut = np.histogram(true_values, bins=100)
        for prop in CUTOFF_PROPORTIONS:
            cutoff = _find_cutoff(prop, true_hist_for_cut, true_bins_for_cut)
            ax.axvline(cutoff, color="gray", linestyle=":", linewidth=1.0)
            ax.text(
                cutoff,
                0.95,
                f"{prop*100:0.1f}% @ {cutoff:0.2f}",
                rotation=90,
                va="top",
                ha="right",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlabel(r"$L_2$ Distance Cutoff", ha="right", x=0.95, fontsize=14)
        ax.set_ylabel("Value", ha="right", y=0.95, fontsize=14)
        ax.set_title("Edge selection efficiency and purity vs distance cutoff")
        ax.legend()
        plt.tight_layout()
        metrics_output_path = output_dir.joinpath(
            f"distance_eff_purity_{plot_suffix}.png" if plot_suffix else "distance_eff_purity.png"
        )
        fig.savefig(metrics_output_path)
        plt.clf()
        print("INFO: Saved efficiency/purity plot to " + str(metrics_output_path))

    output_path = output_dir.joinpath(f"distance_hist_{plot_suffix}.png" if plot_suffix else "distance_hist.png")

    if not plot_false_edges:
        _plot_and_save(
            true_distance_values,
            "Distance between embeddings for true edges",
            output_path,
            color="black",
            label="True edge distance",
        )
        return

    if fake_distance_values is None:
        fake_distance_values = np.asarray([], dtype=np.float64)
    _plot_efficiency_purity(true_distance_values, fake_distance_values)

    if seperate_plots:
        _plot_and_save(
            true_distance_values,
            "Distance between embeddings for true edges",
            output_path,
            color="black",
            label="True edge distance",
        )
        fake_output_path = output_dir.joinpath(
            f"distance_hist_fake_{plot_suffix}.png" if plot_suffix else "distance_hist_fake.png"
        )
        _plot_and_save(
            fake_distance_values,
            "Distance between embeddings for fake edges",
            fake_output_path,
            color="red",
            label="Fake edge distance",
        )
        return

    true_hist, true_bins, true_err, true_ymax = _hist_payload(true_distance_values)
    fake_hist, fake_bins, fake_err, fake_ymax = _hist_payload(fake_distance_values)
    ymax = max(true_ymax, fake_ymax)
    ylim = (0.0, ymax * 1.1 if ymax > 0.0 else 1.0)
    fig, ax = plot_1d_histogram(
        true_hist,
        true_bins,
        true_err,
        r"$L_2$ Distance",
        "Count",
        ylim,
        "True edge distance",
        logy=True,
        tightlayout=False,
        color="black",
        fmt="o",
    )
    plot_1d_histogram(
        fake_hist,
        fake_bins,
        fake_err,
        r"$L_2$ Distance",
        "Count",
        ylim,
        "Fake edge distance",
        canvas=(fig, ax),
        logy=True,
        tightlayout=False,
        color="red",
        fmt="x",
    )
    for prop in CUTOFF_PROPORTIONS:
        cutoff = _find_cutoff(prop, true_hist, true_bins)
        ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("Distance between embeddings for true and fake edges")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.clf()
    print("INFO: Saved distance plot to " + str(output_path))


def _collect_dml_distance_values(dataset, hparams: dict, plot_false_edges=False):
    true_distance_values = []
    fake_distance_values = [] if plot_false_edges else None
    for graph in dataset:
        graph_cpu = graph.cpu()
        distances = get_edge_distances(graph_cpu, edges=None).detach().cpu().numpy()
        true_distance_values.append(distances)
        if plot_false_edges:
            graph_cuda = graph_cpu.to("cuda")
            fake_edges = _get_fake_edges_within_radius(graph_cuda, radius=1.5)
            fake_distances = get_edge_distances(graph_cuda, edges=fake_edges).detach().cpu().numpy()
            fake_distance_values.append(fake_distances)

    true_distance_values = (
        np.concatenate(true_distance_values)
        if true_distance_values
        else np.asarray([], dtype=np.float64)
    )
    if plot_false_edges:
        fake_distance_values = (
            np.concatenate(fake_distance_values)
            if fake_distance_values
            else np.asarray([], dtype=np.float64)
        )
    return true_distance_values, fake_distance_values


def _get_dml_dataset_cutoffs(dataset, hparams: dict):
    true_distance_values, _ = _collect_dml_distance_values(dataset, hparams, plot_false_edges=False)
    hist, bins = np.histogram(true_distance_values, bins=100)
    cutoffs = [_find_cutoff(prop, hist, bins) for prop in CUTOFF_PROPORTIONS]
    return cutoffs


def get_dml_eval_scan_data(graph: Data, eval_config: dict, hparams: dict):
    include_eta = bool(eval_config.get("plot_eta", False))
    graph_cpu = graph.cpu()
    target_mask, particles = _extract_target_particles(
        graph_cpu,
        eval_config,
        include_eta=include_eta,
    )
    eps_values = _build_dml_eps_values(eval_config)

    pt_bins, eta_bins = _get_dml_efficiency_bins(eval_config, include_eta)

    particles_pt_hist = np.histogram(particles[1].cpu().numpy(), bins=pt_bins)[0]
    particles_eta_hist = (
        np.histogram(particles[2].cpu().numpy(), bins=eta_bins)[0]
        if include_eta
        else None
    )

    rows = []
    matched_target_particles_fixed = None
    eval_eps = float(eval_config["eps"])

    for eps_value in eps_values:
        scan_hparams = _get_walkthrough_scan_hparams(hparams, float(eps_value))
        reconstructed_graph = _run_walkthrough_reconstruction(graph_cpu, scan_hparams)
        match_data = _match_dml_tracks(
            reconstructed_graph,
            reconstructed_graph.hit_track_labels,
            target_mask,
            include_eta=include_eta,
        )
        rows.append(
            {
                "eps": float(eps_value),
                "n_particles": int(particles.shape[1]),
                "n_matched_particles": match_data["n_matched_particles"],
                "n_matched_tracks": match_data["n_matched_tracks"],
                "n_matched_target_particles": match_data["n_matched_target_particles"],
                "n_matched_target_tracks": match_data["n_matched_target_tracks"],
                "n_tracks": match_data["n_tracks"],
            }
        )
        if np.isclose(eps_value, eval_eps):
            matched_target_particles_fixed = match_data["matched_target_particles"]

    if matched_target_particles_fixed is None:
        reconstructed_graph = _run_walkthrough_reconstruction(
            graph_cpu,
            _get_walkthrough_scan_hparams(hparams, eval_eps),
        )
        matched_target_particles_fixed = _match_dml_tracks(
            reconstructed_graph,
            reconstructed_graph.hit_track_labels,
            target_mask,
            include_eta=include_eta,
        )["matched_target_particles"]

    matched_target_particles_pt_hist = np.histogram(
        matched_target_particles_fixed[1].cpu().numpy(),
        bins=pt_bins,
    )[0]
    matched_target_particles_eta_hist = (
        np.histogram(
            matched_target_particles_fixed[2].cpu().numpy(),
            bins=eta_bins,
        )[0]
        if include_eta
        else None
    )

    eps_data = pd.DataFrame(rows)
    n_particles = eps_data["n_particles"].to_numpy(dtype=np.float64)
    n_matched_target_particles = eps_data["n_matched_target_particles"].to_numpy(dtype=np.float64)
    n_tracks = eps_data["n_tracks"].to_numpy(dtype=np.float64)
    eps_data["eff"] = np.divide(
        n_matched_target_particles,
        n_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_particles > 0.0,
    )
    eps_data["dup"] = np.divide(
        eps_data["n_matched_target_tracks"].to_numpy(dtype=np.float64) - n_matched_target_particles,
        n_matched_target_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_matched_target_particles > 0.0,
    )
    eps_data["fak"] = np.divide(
        n_tracks - eps_data["n_matched_tracks"].to_numpy(dtype=np.float64),
        n_tracks,
        out=np.zeros_like(n_tracks),
        where=n_tracks > 0.0,
    )

    return (
        eps_data,
        particles_pt_hist,
        matched_target_particles_pt_hist,
        particles_eta_hist,
        matched_target_particles_eta_hist,
        pt_bins,
        eta_bins,
    )


def get_dml_fixed_eps_eval_data(dataset, eval_config: dict, hparams: dict):
    include_eta = bool(eval_config.get("plot_eta", False))
    eval_eps = float(eval_config["eps"])
    pt_bins, eta_bins = _get_dml_efficiency_bins(eval_config, include_eta)

    particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    matched_target_particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    particles_eta_hist = (
        np.histogram([], bins=eta_bins)[0]
        if include_eta
        else None
    )
    matched_target_particles_eta_hist = (
        np.histogram([], bins=eta_bins)[0]
        if include_eta
        else None
    )

    totals = {
        "n_particles": 0,
        "n_matched_particles": 0,
        "n_matched_tracks": 0,
        "n_matched_target_particles": 0,
        "n_matched_target_tracks": 0,
        "n_tracks": 0,
    }

    dataset_size = _get_dataset_size(dataset)
    progress = tqdm(
        dataset,
        total=dataset_size,
        desc=f"DML fixed-eps eval (eps={eval_eps:g})",
        unit="event",
        dynamic_ncols=True,
    )
    for graph in progress:
        graph_cpu = graph.cpu()
        target_mask, particles = _extract_target_particles(
            graph_cpu,
            eval_config,
            include_eta=include_eta,
        )
        particles_pt_hist += np.histogram(
            particles[1].cpu().numpy(),
            bins=pt_bins,
        )[0]
        if include_eta:
            particles_eta_hist += np.histogram(
                particles[2].cpu().numpy(),
                bins=eta_bins,
            )[0]

        reconstructed_graph = _get_dml_reconstructed_graph(
            graph_cpu,
            hparams,
            eval_eps,
        )
        match_data = _match_dml_tracks(
            reconstructed_graph,
            reconstructed_graph.hit_track_labels,
            target_mask,
            include_eta=include_eta,
        )
        totals["n_particles"] += int(particles.shape[1])
        totals["n_matched_particles"] += int(match_data["n_matched_particles"])
        totals["n_matched_tracks"] += int(match_data["n_matched_tracks"])
        totals["n_matched_target_particles"] += int(match_data["n_matched_target_particles"])
        totals["n_matched_target_tracks"] += int(match_data["n_matched_target_tracks"])
        totals["n_tracks"] += int(match_data["n_tracks"])

        matched_target_particles = match_data["matched_target_particles"]
        matched_target_particles_pt_hist += np.histogram(
            matched_target_particles[1].cpu().numpy(),
            bins=pt_bins,
        )[0]
        if include_eta:
            matched_target_particles_eta_hist += np.histogram(
                matched_target_particles[2].cpu().numpy(),
                bins=eta_bins,
            )[0]

    fixed_eps_data = pd.DataFrame(
        [
            {
                "eps": eval_eps,
                **totals,
            }
        ]
    )
    n_particles = fixed_eps_data["n_particles"].to_numpy(dtype=np.float64)
    n_matched_target_particles = fixed_eps_data["n_matched_target_particles"].to_numpy(dtype=np.float64)
    n_tracks = fixed_eps_data["n_tracks"].to_numpy(dtype=np.float64)
    fixed_eps_data["eff"] = np.divide(
        n_matched_target_particles,
        n_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_particles > 0.0,
    )
    fixed_eps_data["dup"] = np.divide(
        fixed_eps_data["n_matched_target_tracks"].to_numpy(dtype=np.float64) - n_matched_target_particles,
        n_matched_target_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_matched_target_particles > 0.0,
    )
    fixed_eps_data["fak"] = np.divide(
        n_tracks - fixed_eps_data["n_matched_tracks"].to_numpy(dtype=np.float64),
        n_tracks,
        out=np.zeros_like(n_tracks),
        where=n_tracks > 0.0,
    )

    return (
        fixed_eps_data,
        particles_pt_hist,
        matched_target_particles_pt_hist,
        particles_eta_hist,
        matched_target_particles_eta_hist,
        pt_bins,
        eta_bins,
    )


def get_dml_eval_scan_dataset_data(
    dataset,
    eval_config: dict,
    hparams: dict,
    include_fixed_eps: bool = False,
    dataset_name: str = "valset",
):
    eps_values = _build_dml_eps_values(eval_config)
    eval_eps = float(eval_config["eps"])
    include_eta = include_fixed_eps and bool(eval_config.get("plot_eta", False))
    rows = {
        float(eps_value): {
            "eps": float(eps_value),
            "n_particles": 0,
            "n_matched_particles": 0,
            "n_matched_tracks": 0,
            "n_matched_target_particles": 0,
            "n_matched_target_tracks": 0,
            "n_tracks": 0,
        }
        for eps_value in eps_values
    }
    fixed_eps_data = None
    if include_fixed_eps:
        pt_bins, eta_bins = _get_dml_efficiency_bins(eval_config, include_eta)
        particles_pt_hist = np.histogram([], bins=pt_bins)[0]
        matched_target_particles_pt_hist = np.histogram([], bins=pt_bins)[0]
        particles_eta_hist = (
            np.histogram([], bins=eta_bins)[0]
            if include_eta
            else None
        )
        matched_target_particles_eta_hist = (
            np.histogram([], bins=eta_bins)[0]
            if include_eta
            else None
        )
        fixed_eps_totals = {
            "n_particles": 0,
            "n_matched_particles": 0,
            "n_matched_tracks": 0,
            "n_matched_target_particles": 0,
            "n_matched_target_tracks": 0,
            "n_tracks": 0,
        }

    cache_path = _get_dml_scan_cache_path(
        hparams.get("output_dir", eval_config.get("output_dir")),
        dataset_name,
        hparams,
        eval_config,
    )
    _ensure_dml_scan_cache_header(
        cache_path,
        dataset_name,
        hparams,
        eval_config,
    )
    _merge_legacy_child_dml_scan_caches(
        cache_path,
        dataset_name,
        hparams,
        eval_config,
    )
    cache_entries = _load_dml_scan_cache(cache_path)
    if cache_entries:
        print(
            f"INFO: Loaded {len(cache_entries)} cached DML cutoff reconstructions "
            f"from {cache_path}"
        )

    dataset_size = _get_dataset_size(dataset)
    total_reconstructions = None
    if dataset_size is not None:
        total_reconstructions = dataset_size * len(eps_values)
    with tqdm(
        total=total_reconstructions,
        desc=f"DML cutoff scan ({len(eps_values)} cutoffs)",
        unit="reco",
        dynamic_ncols=True,
    ) as progress:
        for event_index, graph in enumerate(dataset):
            graph_cpu = graph.cpu()
            event_id = _extract_event_id(graph_cpu, event_index)
            target_mask, particles = _extract_target_particles(
                graph_cpu,
                eval_config,
                include_eta=include_eta,
            )
            n_particles = int(particles.shape[1])
            if include_fixed_eps:
                particles_pt_hist += np.histogram(
                    particles[1].cpu().numpy(),
                    bins=pt_bins,
                )[0]
                if include_eta:
                    particles_eta_hist += np.histogram(
                        particles[2].cpu().numpy(),
                        bins=eta_bins,
                    )[0]
            for eps_value in eps_values:
                eps_value = float(eps_value)
                collect_fixed_eps = include_fixed_eps and np.isclose(
                    eps_value,
                    eval_eps,
                )
                cache_key = (event_id, eps_value)
                cached_entry = cache_entries.get(cache_key)
                if cached_entry is not None and collect_fixed_eps:
                    if not _fixed_eps_payload_matches(
                        cached_entry.get("fixed_eps_payload"),
                        eval_config,
                        include_eta,
                    ):
                        cached_entry = None

                if cached_entry is None:
                    reconstructed_graph = _get_dml_reconstructed_graph(
                        graph_cpu,
                        hparams,
                        eps_value,
                    )
                    match_data = _match_dml_tracks(
                        reconstructed_graph,
                        reconstructed_graph.hit_track_labels,
                        target_mask,
                        include_eta=collect_fixed_eps and include_eta,
                    )
                    fixed_eps_payload = None
                    if collect_fixed_eps:
                        matched_target_particles = match_data["matched_target_particles"]
                        fixed_eps_payload = {
                            "signature": _get_fixed_eps_payload_signature(
                                eval_config,
                                include_eta,
                            ),
                            "matched_target_particles_pt_hist": np.histogram(
                                matched_target_particles[1].cpu().numpy(),
                                bins=pt_bins,
                            )[0].astype(np.int64).tolist(),
                            "matched_target_particles_eta_hist": (
                                np.histogram(
                                    matched_target_particles[2].cpu().numpy(),
                                    bins=eta_bins,
                                )[0].astype(np.int64).tolist()
                                if include_eta
                                else None
                            ),
                        }
                    cached_entry = {
                        "event_id": event_id,
                        "eps": eps_value,
                        "n_particles": n_particles,
                        "n_matched_particles": int(match_data["n_matched_particles"]),
                        "n_matched_tracks": int(match_data["n_matched_tracks"]),
                        "n_matched_target_particles": int(match_data["n_matched_target_particles"]),
                        "n_matched_target_tracks": int(match_data["n_matched_target_tracks"]),
                        "n_tracks": int(match_data["n_tracks"]),
                        "fixed_eps_payload": fixed_eps_payload,
                    }
                    cache_entries[cache_key] = cached_entry
                    _append_dml_scan_cache_entry(cache_path, cached_entry)

                row = rows[eps_value]
                row["n_particles"] += n_particles
                row["n_matched_particles"] += int(cached_entry["n_matched_particles"])
                row["n_matched_tracks"] += int(cached_entry["n_matched_tracks"])
                row["n_matched_target_particles"] += int(cached_entry["n_matched_target_particles"])
                row["n_matched_target_tracks"] += int(cached_entry["n_matched_target_tracks"])
                row["n_tracks"] += int(cached_entry["n_tracks"])
                if collect_fixed_eps:
                    fixed_eps_totals["n_particles"] += n_particles
                    fixed_eps_totals["n_matched_particles"] += int(cached_entry["n_matched_particles"])
                    fixed_eps_totals["n_matched_tracks"] += int(cached_entry["n_matched_tracks"])
                    fixed_eps_totals["n_matched_target_particles"] += int(cached_entry["n_matched_target_particles"])
                    fixed_eps_totals["n_matched_target_tracks"] += int(cached_entry["n_matched_target_tracks"])
                    fixed_eps_totals["n_tracks"] += int(cached_entry["n_tracks"])
                    fixed_eps_payload = cached_entry["fixed_eps_payload"]
                    matched_target_particles_pt_hist += np.asarray(
                        fixed_eps_payload["matched_target_particles_pt_hist"],
                        dtype=np.int64,
                    )
                    if include_eta:
                        matched_target_particles_eta_hist += np.asarray(
                            fixed_eps_payload["matched_target_particles_eta_hist"],
                            dtype=np.int64,
                        )
                progress.update(1)

    eps_data = pd.DataFrame([rows[float(eps_value)] for eps_value in eps_values])
    n_particles = eps_data["n_particles"].to_numpy(dtype=np.float64)
    n_matched_target_particles = eps_data["n_matched_target_particles"].to_numpy(dtype=np.float64)
    n_tracks = eps_data["n_tracks"].to_numpy(dtype=np.float64)
    eps_data["eff"] = np.divide(
        n_matched_target_particles,
        n_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_particles > 0.0,
    )
    eps_data["dup"] = np.divide(
        eps_data["n_matched_target_tracks"].to_numpy(dtype=np.float64) - n_matched_target_particles,
        n_matched_target_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_matched_target_particles > 0.0,
    )
    eps_data["fak"] = np.divide(
        n_tracks - eps_data["n_matched_tracks"].to_numpy(dtype=np.float64),
        n_tracks,
        out=np.zeros_like(n_tracks),
        where=n_tracks > 0.0,
    )
    if not include_fixed_eps:
        return eps_data

    fixed_eps_data = pd.DataFrame(
        [
            {
                "eps": eval_eps,
                **fixed_eps_totals,
            }
        ]
    )
    n_particles = fixed_eps_data["n_particles"].to_numpy(dtype=np.float64)
    n_matched_target_particles = fixed_eps_data["n_matched_target_particles"].to_numpy(
        dtype=np.float64
    )
    n_tracks = fixed_eps_data["n_tracks"].to_numpy(dtype=np.float64)
    fixed_eps_data["eff"] = np.divide(
        n_matched_target_particles,
        n_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_particles > 0.0,
    )
    fixed_eps_data["dup"] = np.divide(
        fixed_eps_data["n_matched_target_tracks"].to_numpy(dtype=np.float64)
        - n_matched_target_particles,
        n_matched_target_particles,
        out=np.zeros_like(n_matched_target_particles),
        where=n_matched_target_particles > 0.0,
    )
    fixed_eps_data["fak"] = np.divide(
        n_tracks - fixed_eps_data["n_matched_tracks"].to_numpy(dtype=np.float64),
        n_tracks,
        out=np.zeros_like(n_tracks),
        where=n_tracks > 0.0,
    )

    return (
        eps_data,
        (
            fixed_eps_data,
            particles_pt_hist,
            matched_target_particles_pt_hist,
            particles_eta_hist,
            matched_target_particles_eta_hist,
            pt_bins,
            eta_bins,
        ),
    )


def _to_output_dir_path(output_dir):
    if isinstance(output_dir, list):
        return Path(*output_dir)
    if isinstance(output_dir, Path):
        return output_dir
    return Path(output_dir)


def _stable_cache_value(value):
    if isinstance(value, dict):
        return {str(k): _stable_cache_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_cache_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalize_config_file_identity(config_file):
    if not isinstance(config_file, dict):
        return config_file
    return {
        "sha256": config_file.get("sha256"),
    }


def _normalize_checkpoint_identity(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    return {
        "path": checkpoint.get("path"),
        "exists": checkpoint.get("exists"),
        "size": checkpoint.get("size"),
        "mtime_ns": checkpoint.get("mtime_ns"),
    }


def _get_infer_metadata_path(walkthrough_output_dir, dataset_name):
    return Path(walkthrough_output_dir) / f"infer_metadata_{dataset_name}.json"


def _load_infer_metadata(walkthrough_output_dir, dataset_name):
    if walkthrough_output_dir is None:
        return None

    metadata_path = _get_infer_metadata_path(walkthrough_output_dir, dataset_name)
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"WARNING: Failed to load infer metadata from {metadata_path}")
        return None


def _normalize_infer_metadata_for_cache(metadata):
    data_settings = metadata.get("data_settings", {})
    if isinstance(data_settings, dict):
        data_settings = {
            key: value
            for key, value in data_settings.items()
            if key != "data_split"
        }

    return {
        "schema_version": metadata.get("schema_version"),
        "dataset_name": metadata.get("dataset_name"),
        "config_file": _normalize_config_file_identity(metadata.get("config_file")),
        "checkpoint": _normalize_checkpoint_identity(metadata.get("checkpoint")),
        "data_settings": data_settings,
        "reconstruction_settings": metadata.get("reconstruction_settings"),
    }


def _normalize_cache_identity_for_compare(identity):
    if not isinstance(identity, dict):
        return identity

    if identity.get("source") == "infer_metadata":
        return {
            "source": "infer_metadata",
            "metadata": _normalize_infer_metadata_for_cache(
                identity.get("metadata", {})
            ),
        }

    if identity.get("source") == "hparams_fallback":
        return {
            "source": "hparams_fallback",
            "data_settings": identity.get("data_settings"),
            "reconstruction_settings": identity.get("reconstruction_settings"),
        }

    return identity


def _build_dml_scan_cache_identity(hparams, dataset_name):
    walkthrough_output_dir = hparams.get("walkthrough_output_dir")
    metadata = _load_infer_metadata(walkthrough_output_dir, dataset_name)
    if metadata is not None:
        return {
            "source": "infer_metadata",
            "metadata": _normalize_infer_metadata_for_cache(metadata),
        }

    return {
        "source": "hparams_fallback",
        "data_settings": {
            key: _stable_cache_value(hparams.get(key))
            for key in _DML_SCAN_CACHE_DATA_KEYS
            if key in hparams
        },
        "reconstruction_settings": {
            key: _stable_cache_value(hparams.get(key))
            for key in _DML_SCAN_CACHE_RECONSTRUCTION_KEYS
            if key in hparams
        },
    }


def _get_fixed_eps_payload_signature(eval_config, include_eta):
    return {
        "pT_unit": eval_config.get("pT_unit", "MeV"),
        "include_eta": bool(include_eta),
    }


def _fixed_eps_payload_matches(payload, eval_config, include_eta):
    if payload is None:
        return False
    payload_signature = payload.get("signature")
    if payload_signature is None:
        return False
    return payload_signature == _get_fixed_eps_payload_signature(eval_config, include_eta)


def _get_dml_scan_cache_root_dir(output_dir):
    output_dir = _to_output_dir_path(output_dir)
    if output_dir.name.startswith("first_") and output_dir.name.endswith("_events"):
        return output_dir.parent
    return output_dir


def _get_dml_scan_cache_path(
    output_dir,
    dataset_name,
    hparams,
    eval_config,
):
    output_dir = _get_dml_scan_cache_root_dir(output_dir)
    cache_signature = {
        "schema_version": 3,
        "dataset_name": dataset_name,
        "target_tracks": eval_config.get("target_tracks"),
        "trackML_data": bool(eval_config.get("trackML_data", False)),
        "identity": _build_dml_scan_cache_identity(hparams, dataset_name),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return output_dir / f"dml_cutoff_scan_cache_{dataset_name}_{cache_key}.jsonl"


def _build_dml_scan_cache_header(cache_path, dataset_name, hparams, eval_config):
    return {
        "record_type": "metadata",
        "schema_version": 1,
        "cache_format": "dml_cutoff_scan_cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(cache_path),
        "dataset_name": dataset_name,
        "shared_cache_root": str(cache_path.parent),
        "output_dir": (
            str(_to_output_dir_path(hparams.get("output_dir")))
            if hparams.get("output_dir") is not None
            else None
        ),
        "walkthrough_output_dir": (
            str(_to_output_dir_path(hparams.get("walkthrough_output_dir")))
            if hparams.get("walkthrough_output_dir") is not None
            else None
        ),
        "eval_eps": float(eval_config["eps"]),
        "eps_values": [float(value) for value in _build_dml_eps_values(eval_config)],
        "target_tracks": eval_config.get("target_tracks"),
        "trackML_data": bool(eval_config.get("trackML_data", False)),
        "pT_unit": eval_config.get("pT_unit", "MeV"),
        "cache_identity": _build_dml_scan_cache_identity(hparams, dataset_name),
    }


def _write_jsonl_line(file_obj, record):
    file_obj.write(json.dumps(record, sort_keys=True))
    file_obj.write("\n")


def _ensure_dml_scan_cache_header(cache_path, dataset_name, hparams, eval_config):
    header = _build_dml_scan_cache_header(
        cache_path,
        dataset_name,
        hparams,
        eval_config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        with open(cache_path, "w", encoding="utf-8") as f:
            _write_jsonl_line(f, header)
            f.flush()
            os.fsync(f.fileno())
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        existing_lines = f.readlines()

    first_record = None
    for raw_line in existing_lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            first_record = json.loads(line)
        except json.JSONDecodeError:
            first_record = None
        break

    if (
        isinstance(first_record, dict)
        and first_record.get("record_type") == "metadata"
    ):
        return

    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        _write_jsonl_line(f, header)
        for raw_line in existing_lines:
            f.write(raw_line)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, cache_path)


def _read_dml_scan_cache_file(cache_path):
    header = None
    entries = {}
    if cache_path is None or not cache_path.exists():
        return header, entries

    with open(cache_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: Ignoring malformed DML cutoff cache entry at "
                    f"{cache_path}:{line_number}"
                )
                continue

            if isinstance(record, dict) and record.get("record_type") == "metadata":
                if header is None:
                    header = record
                continue

            if "event_id" not in record or "eps" not in record:
                print(
                    f"WARNING: Ignoring incomplete DML cutoff cache entry at "
                    f"{cache_path}:{line_number}"
                )
                continue
            entries[(str(record["event_id"]), float(record["eps"]))] = record

    return header, entries


def _append_dml_scan_cache_entries(cache_path, entries):
    if not entries:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for entry in entries:
            _write_jsonl_line(f, entry)
        f.flush()
        os.fsync(f.fileno())


def _merge_legacy_child_dml_scan_caches(cache_path, dataset_name, hparams, eval_config):
    shared_cache_root = cache_path.parent
    current_identity = _normalize_cache_identity_for_compare(
        _build_dml_scan_cache_identity(hparams, dataset_name)
    )
    _, shared_entries = _read_dml_scan_cache_file(cache_path)
    missing_entries = []

    for legacy_path in sorted(
        shared_cache_root.glob(f"first_*_events/dml_cutoff_scan_cache_{dataset_name}_*.jsonl")
    ):
        if legacy_path == cache_path:
            continue

        legacy_header, legacy_entries = _read_dml_scan_cache_file(legacy_path)
        if legacy_header is None:
            continue

        legacy_identity = _normalize_cache_identity_for_compare(
            legacy_header.get("cache_identity")
        )
        if legacy_identity != current_identity:
            continue

        for cache_key, entry in legacy_entries.items():
            if cache_key in shared_entries:
                continue
            shared_entries[cache_key] = entry
            missing_entries.append(entry)

    _append_dml_scan_cache_entries(cache_path, missing_entries)


def _load_dml_scan_cache(cache_path):
    _, cache_entries = _read_dml_scan_cache_file(cache_path)
    return cache_entries


def _append_dml_scan_cache_entry(cache_path, entry):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _save_plot_figure(fig, output_dir, filename, message):
    output_dir = _to_output_dir_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"INFO: Saved {message} to {output_path}")


def _to_numpy_1d(values, dtype=None):
    if values is None:
        if dtype is None:
            return np.asarray([], dtype=np.float64)
        return np.asarray([], dtype=dtype)
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.atleast_1d(array)


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def _graph_scalar_to_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _graph_scalar_to_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _diagnostic_dict_to_dataframe(diag_dict):
    if not diag_dict:
        return pd.DataFrame()
    rows = {}
    for key, values in diag_dict.items():
        if torch.is_tensor(values):
            rows[key] = values.detach().cpu().numpy()
        else:
            rows[key] = np.asarray(values, dtype=object)
    return pd.DataFrame(rows)


def _append_event_id_column(frame, event_id):
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["event_id"] = event_id
    return frame


def _extract_event_id(graph, fallback_index):
    event_attr = getattr(graph, "event_id", None)
    if event_attr is None:
        return str(fallback_index)
    if torch.is_tensor(event_attr):
        if event_attr.numel() == 0:
            return str(fallback_index)
        first_value = event_attr.flatten()[0]
        try:
            return str(int(first_value.item()))
        except Exception:
            return str(first_value)
    if isinstance(event_attr, (list, tuple)):
        if not event_attr:
            return str(fallback_index)
        return str(event_attr[0])
    return str(event_attr)


def _get_profile_metric(profile_metadata, key, default=0):
    return profile_metadata.get(key, default)


def _merge_edge_diagnostics(aggregate_edge_summary, edge_summary):
    if not edge_summary:
        return

    bins = _to_numpy_1d(edge_summary.get("bins"), dtype=np.float64)
    if aggregate_edge_summary["bins"] is None:
        aggregate_edge_summary["bins"] = bins
    for stage_name in ("cc", "walk"):
        stage_summary = edge_summary.get(stage_name, {})
        for class_name in ("true_kept", "true_dropped", "fake_kept", "fake_dropped"):
            class_summary = stage_summary.get(class_name)
            if class_summary is None:
                continue
            aggregate_class = aggregate_edge_summary["stages"][stage_name][class_name]
            aggregate_class["count"] += int(class_summary.get("count", 0))
            aggregate_class["score_sum"] += float(class_summary.get("score_sum", 0.0))
            aggregate_class["score_sum_sq"] += float(class_summary.get("score_sum_sq", 0.0))
            aggregate_class["hist"] += _to_numpy_1d(
                class_summary.get("hist"),
                dtype=np.int64,
            )


def aggregate_dml_event_data(dataset, eval_config):
    assignment_bins = {
        "r": np.linspace(0.0, 1200.0, 61, dtype=np.float64),
        "z": np.linspace(-3200.0, 3200.0, 81, dtype=np.float64),
        "eta": np.linspace(-4.0, 4.0, 41, dtype=np.float64),
    }
    assignment_profiles = {
        name: {
            "bins": bins,
            "total": np.zeros(len(bins) - 1, dtype=np.int64),
            "assigned": np.zeros(len(bins) - 1, dtype=np.int64),
        }
        for name, bins in assignment_bins.items()
    }
    stage_timing = defaultdict(list)
    event_rows = []
    track_lengths = []
    track_frames = []
    particle_frames = []
    branch_multiplicities = []
    branch_score_gaps = []
    branch_threshold = None
    edge_summary = {
        "bins": None,
        "stages": {
            stage_name: {
                class_name: {
                    "count": 0,
                    "score_sum": 0.0,
                    "score_sum_sq": 0.0,
                    "hist": np.zeros(120, dtype=np.int64),
                }
                for class_name in ("true_kept", "true_dropped", "fake_kept", "fake_dropped")
            }
            for stage_name in ("cc", "walk")
        },
    }

    for event_index, graph in enumerate(dataset):
        graph = graph.cpu()
        event_id = _extract_event_id(graph, event_index)
        num_nodes = _graph_scalar_to_int(graph.num_nodes)
        reco_tracks = list(getattr(graph, "reco_tracks", []))
        n_tracks = len(reco_tracks)
        graph_edge_index = getattr(graph, "edge_index", None)
        default_input_edges = (
            int(graph_edge_index.shape[1])
            if graph_edge_index is not None and hasattr(graph_edge_index, "shape")
            else 0
        )
        track_lengths.extend(len(track) for track in reco_tracks)

        hit_track_labels = getattr(graph, "hit_track_labels", None)
        if hit_track_labels is None:
            hit_track_labels = torch.full((num_nodes,), -1, dtype=torch.long)
        else:
            hit_track_labels = hit_track_labels.long().cpu()
        assigned_mask = hit_track_labels >= 0

        for axis_name, attr_name in (
            ("r", "hit_r"),
            ("z", "hit_z"),
            ("eta", "hit_particle_eta"),
        ):
            if not hasattr(graph, attr_name):
                continue
            values = _to_numpy_1d(getattr(graph, attr_name), dtype=np.float64)
            assignment_profiles[axis_name]["total"] += np.histogram(
                values,
                bins=assignment_profiles[axis_name]["bins"],
            )[0]
            assignment_profiles[axis_name]["assigned"] += np.histogram(
                values[assigned_mask.numpy()],
                bins=assignment_profiles[axis_name]["bins"],
            )[0]

        target_mask, particles = _extract_target_particles(
            graph,
            eval_config,
            include_eta=False,
        )
        match_data = _match_dml_tracks(
            graph,
            hit_track_labels,
            target_mask,
            include_eta=False,
        )
        n_particles = int(particles.shape[1])
        n_matched_particles = int(match_data["n_matched_particles"])
        n_matched_tracks = int(match_data["n_matched_tracks"])
        n_matched_target_particles = int(match_data["n_matched_target_particles"])
        n_matched_target_tracks = int(match_data["n_matched_target_tracks"])
        event_eff = (
            n_matched_target_particles / n_particles if n_particles > 0 else 0.0
        )
        event_dup = (
            (n_matched_target_tracks - n_matched_target_particles) / n_matched_target_particles
            if n_matched_target_particles > 0
            else 0.0
        )
        event_fak = (n_tracks - n_matched_tracks) / n_tracks if n_tracks > 0 else 0.0

        profile_metadata = getattr(graph, "profile_metadata", {})
        profiling = getattr(graph, "profiling", {})
        for stage_name, elapsed in profiling.items():
            stage_timing[stage_name].append(float(elapsed))

        hit_reco_method = np.asarray(
            getattr(graph, "hit_reco_method", np.full(num_nodes, -1, dtype=object)),
            dtype=object,
        )
        method_hit_counts = {
            "cc": int(np.sum(hit_reco_method == "cc")),
            "walk": int(np.sum(hit_reco_method == "walk")),
            "unassigned": int(np.sum(hit_track_labels.numpy() < 0)),
        }

        track_diag_dict = getattr(graph, "track_diagnostics", None)
        particle_diag_dict = getattr(graph, "particle_reco_diagnostics", None)
        if not track_diag_dict or not particle_diag_dict:
            track_diag_dict, particle_diag_dict = build_track_particle_diagnostics(
                graph,
                target_tracks=eval_config.get("target_tracks", None),
            )
        track_diag_frame = _append_event_id_column(
            _diagnostic_dict_to_dataframe(track_diag_dict),
            event_id,
        )
        particle_diag_frame = _append_event_id_column(
            _diagnostic_dict_to_dataframe(particle_diag_dict),
            event_id,
        )
        if not track_diag_frame.empty:
            track_frames.append(track_diag_frame)
        if not particle_diag_frame.empty:
            particle_frames.append(particle_diag_frame)

        method_track_counts = {
            "cc": int(
                _get_profile_metric(
                    profile_metadata,
                    "fast_walkthrough.simple_path.num_tracks",
                    (track_diag_frame["reco_method"] == "cc").sum() if not track_diag_frame.empty else 0,
                )
            ),
            "walk": int(
                _get_profile_metric(
                    profile_metadata,
                    "fast_walkthrough.walk.num_tracks",
                    (track_diag_frame["reco_method"] == "walk").sum() if not track_diag_frame.empty else 0,
                )
            ),
        }

        edge_diag_summary = getattr(graph, "edge_diagnostics_summary", None)
        if edge_diag_summary:
            _merge_edge_diagnostics(edge_summary, edge_diag_summary)

        walk_branch_diagnostics = getattr(graph, "walk_branch_diagnostics", None)
        if walk_branch_diagnostics:
            branch_threshold = float(
                walk_branch_diagnostics.get("ambiguity_gap_threshold", 0.0)
            )
            branch_multiplicities.append(
                _to_numpy_1d(
                    walk_branch_diagnostics.get("branch_multiplicities"),
                    dtype=np.int64,
                )
            )
            branch_score_gaps.append(
                _to_numpy_1d(
                    walk_branch_diagnostics.get("branch_score_gaps"),
                    dtype=np.float64,
                )
            )

        event_rows.append(
            {
                "event_id": event_id,
                "num_nodes": num_nodes,
                "n_tracks": n_tracks,
                "assigned_hits": int(assigned_mask.sum().item()),
                "unassigned_hits": int((~assigned_mask).sum().item()),
                "assigned_fraction": float(assigned_mask.float().mean().item()) if num_nodes > 0 else 0.0,
                "time_taken": _graph_scalar_to_float(
                    getattr(
                        graph,
                        "time_taken",
                        profiling.get("fast_walkthrough.total", 0.0),
                    )
                ),
                "n_particles": n_particles,
                "n_matched_particles": n_matched_particles,
                "n_matched_tracks": n_matched_tracks,
                "n_matched_target_particles": n_matched_target_particles,
                "n_matched_target_tracks": n_matched_target_tracks,
                "eff": event_eff,
                "dup": event_dup,
                "fak": event_fak,
                "simple_path_tracks": method_track_counts["cc"],
                "walk_tracks": method_track_counts["walk"],
                "cc_hit_count": method_hit_counts["cc"],
                "walk_hit_count": method_hit_counts["walk"],
                "unassigned_hit_count": method_hit_counts["unassigned"],
                "input_nodes": int(_get_profile_metric(profile_metadata, "fast_walkthrough.input.num_nodes", num_nodes)),
                "input_edges": int(_get_profile_metric(profile_metadata, "fast_walkthrough.input.num_edges", default_input_edges)),
                "initial_edges": int(_get_profile_metric(profile_metadata, "fast_walkthrough.initial_edge_index.num_edges", 0)),
                "filtered_nodes": int(_get_profile_metric(profile_metadata, "fast_walkthrough.filtered.num_nodes", num_nodes)),
                "filtered_edges": int(_get_profile_metric(profile_metadata, "fast_walkthrough.filtered.num_edges", 0)),
                "acyclic_nodes": int(_get_profile_metric(profile_metadata, "fast_walkthrough.acyclic.num_nodes", num_nodes)),
                "acyclic_edges": int(_get_profile_metric(profile_metadata, "fast_walkthrough.acyclic.num_edges", 0)),
                "walk_input_nodes": int(_get_profile_metric(profile_metadata, "fast_walkthrough.walk_input.num_nodes", num_nodes)),
                "walk_input_edges": int(_get_profile_metric(profile_metadata, "fast_walkthrough.walk_input.num_edges", 0)),
                "num_branch_points": int(
                    walk_branch_diagnostics.get("num_branch_points", 0)
                    if walk_branch_diagnostics
                    else 0
                ),
                "num_ambiguous_choices": int(
                    walk_branch_diagnostics.get("num_ambiguous_choices", 0)
                    if walk_branch_diagnostics
                    else 0
                ),
            }
        )

    return {
        "events": pd.DataFrame(event_rows),
        "track_lengths": np.asarray(track_lengths, dtype=np.int64),
        "assignment_profiles": assignment_profiles,
        "stage_timing": {name: np.asarray(values, dtype=np.float64) for name, values in stage_timing.items()},
        "track_diagnostics": pd.concat(track_frames, ignore_index=True) if track_frames else pd.DataFrame(),
        "particle_diagnostics": pd.concat(particle_frames, ignore_index=True) if particle_frames else pd.DataFrame(),
        "edge_diagnostics": edge_summary,
        "branch_diagnostics": {
            "ambiguity_gap_threshold": branch_threshold,
            "branch_multiplicities": np.concatenate(branch_multiplicities) if branch_multiplicities else np.asarray([], dtype=np.int64),
            "branch_score_gaps": np.concatenate(branch_score_gaps) if branch_score_gaps else np.asarray([], dtype=np.float64),
        },
    }


def _plot_profile_from_counts(ax, bins, numerator, denominator, ylabel):
    centers = 0.5 * (bins[1:] + bins[:-1])
    values = _safe_ratio(numerator, denominator)
    ax.plot(centers, values, marker="o", color="black")
    ax.set_xlabel(ax.get_xlabel())
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)


def plot_track_length_distribution(aggregate_data, output_dir, output_suffix):
    track_lengths = aggregate_data["track_lengths"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if track_lengths.size > 0:
        bins = np.arange(1, track_lengths.max() + 2, dtype=np.int64)
        ax.hist(track_lengths, bins=bins, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Reconstructed track length [hits]")
    ax.set_ylabel("Count")
    ax.set_title("Track length distribution")
    _save_plot_figure(fig, output_dir, f"track_length_distribution_{output_suffix}.png", "track length distribution")


def plot_hit_assignment_fraction(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    assigned_hits = int(events["assigned_hits"].sum()) if not events.empty else 0
    unassigned_hits = int(events["unassigned_hits"].sum()) if not events.empty else 0
    total_hits = assigned_hits + unassigned_hits

    fig, ax = plt.subplots(figsize=(6, 5))
    fractions = _safe_ratio(
        np.asarray([assigned_hits, unassigned_hits], dtype=np.float64),
        np.asarray([total_hits, total_hits], dtype=np.float64),
    )
    ax.bar(["Assigned", "Unassigned"], fractions, color=["tab:green", "tab:red"])
    ax.set_ylabel("Fraction of hits")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Hit assignment fraction")
    _save_plot_figure(fig, output_dir, f"hit_assignment_fraction_{output_suffix}.png", "hit assignment fraction plot")

    axis_labels = {
        "r": r"$r$ [mm]",
        "z": r"$z$ [mm]",
        "eta": r"$\eta$",
    }
    for axis_name, profile in aggregate_data["assignment_profiles"].items():
        fig, ax = plt.subplots(figsize=(7, 5))
        centers = 0.5 * (profile["bins"][1:] + profile["bins"][:-1])
        values = _safe_ratio(profile["assigned"], profile["total"])
        ax.plot(centers, values, marker="o", linestyle="-", color="black")
        ax.set_xlabel(axis_labels[axis_name])
        ax.set_ylabel("Assigned-hit fraction")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Hit assignment vs {axis_name}")
        _save_plot_figure(
            fig,
            output_dir,
            f"hit_assignment_vs_{axis_name}_{output_suffix}.png",
            f"hit assignment vs {axis_name} plot",
        )


def plot_reco_method_breakdown(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    hit_counts = np.asarray(
        [
            events["cc_hit_count"].sum() if not events.empty else 0,
            events["walk_hit_count"].sum() if not events.empty else 0,
            events["unassigned_hit_count"].sum() if not events.empty else 0,
        ],
        dtype=np.float64,
    )
    track_counts = np.asarray(
        [
            events["simple_path_tracks"].sum() if not events.empty else 0,
            events["walk_tracks"].sum() if not events.empty else 0,
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        ["cc", "walk", "unassigned"],
        _safe_ratio(hit_counts, np.full_like(hit_counts, hit_counts.sum())),
        color=["tab:blue", "tab:orange", "tab:red"],
    )
    axes[0].set_ylabel("Fraction of hits")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Hit reconstruction method")
    axes[1].bar(
        ["cc", "walk"],
        _safe_ratio(track_counts, np.full_like(track_counts, track_counts.sum())),
        color=["tab:blue", "tab:orange"],
    )
    axes[1].set_ylabel("Fraction of tracks")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Track reconstruction method")
    _save_plot_figure(fig, output_dir, f"reco_method_breakdown_{output_suffix}.png", "reco method breakdown plot")


def plot_tracks_per_event(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not events.empty:
        ax.hist(events["n_tracks"], bins=30, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Tracks per event")
    ax.set_ylabel("Count")
    ax.set_title("Tracks per event")
    _save_plot_figure(fig, output_dir, f"tracks_per_event_{output_suffix}.png", "tracks per event plot")


def plot_time_vs_occupancy(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not events.empty:
        num_nodes = events["num_nodes"].to_numpy(dtype=np.float64)
        time_taken = events["time_taken"].to_numpy(dtype=np.float64)

        # Keep the main panel readable by clipping only the visual range.
        # Outliers are still indicated explicitly at the cap line.
        if len(time_taken) >= 20:
            q1, q3 = np.percentile(time_taken, [25, 75])
            iqr = q3 - q1
            whisker_cap = q3 + 3.0 * iqr
            percentile_cap = np.percentile(time_taken, 99)
            display_cap = min(max(whisker_cap, percentile_cap), time_taken.max())
        else:
            display_cap = time_taken.max()
        outlier_mask = time_taken > display_cap
        displayed_times = np.minimum(time_taken, display_cap)

        ax.scatter(num_nodes, displayed_times, s=12, alpha=0.35, color="tab:blue")
        if np.any(outlier_mask):
            ax.scatter(
                num_nodes[outlier_mask],
                np.full(np.sum(outlier_mask), display_cap),
                s=28,
                marker="^",
                color="tab:red",
                alpha=0.85,
                label="Capped outlier",
            )
            ax.axhline(display_cap, color="tab:red", linestyle=":", linewidth=1.0)
            ax.text(
                0.98,
                0.98,
                (
                    f"{int(np.sum(outlier_mask))} outlier events above {display_cap:.2f} s\n"
                    f"max = {time_taken.max():.2f} s"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        if len(events) >= 4:
            bins = np.linspace(num_nodes.min(), num_nodes.max(), min(16, len(events)) + 1)
            digitized = np.digitize(num_nodes, bins[1:-1], right=False)
            bin_centers = []
            median_times = []
            for bin_index in range(len(bins) - 1):
                mask = digitized == bin_index
                if not np.any(mask):
                    continue
                bin_centers.append(0.5 * (bins[bin_index] + bins[bin_index + 1]))
                median_times.append(np.median(time_taken[mask]))
            if bin_centers:
                ax.plot(bin_centers, np.minimum(median_times, display_cap), color="black", linewidth=2, label="Median")
        if np.any(outlier_mask):
            ax.set_ylim(top=display_cap * 1.05)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left")
    ax.set_xlabel("Event occupancy [hits]")
    ax.set_ylabel("Walkthrough time [s]")
    ax.set_title("Walkthrough time vs occupancy")
    _save_plot_figure(fig, output_dir, f"walkthrough_time_vs_num_nodes_{output_suffix}.png", "time vs occupancy plot")


def plot_stage_timing_breakdown(aggregate_data, output_dir, output_suffix):
    stage_timing = aggregate_data["stage_timing"]
    stage_names = sorted(stage_timing)
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(stage_names)), 5))
    if stage_names:
        means = np.asarray([stage_timing[name].mean() for name in stage_names], dtype=np.float64)
        medians = np.asarray([np.median(stage_timing[name]) for name in stage_names], dtype=np.float64)
        p95 = np.asarray([np.percentile(stage_timing[name], 95) for name in stage_names], dtype=np.float64)
        x = np.arange(len(stage_names), dtype=np.float64)
        width = 0.26
        ax.bar(x - width, means, width=width, label="mean")
        ax.bar(x, medians, width=width, label="median")
        ax.bar(x + width, p95, width=width, label="p95")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_names, rotation=45, ha="right")
    ax.set_ylabel("Time [s]")
    ax.set_title("Walkthrough stage timing")
    ax.legend()
    _save_plot_figure(fig, output_dir, f"walkthrough_stage_timing_{output_suffix}.png", "stage timing breakdown plot")


def plot_graph_reduction_waterfall(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    stages = [
        ("input", "input_nodes", "input_edges"),
        ("initial", "input_nodes", "initial_edges"),
        ("filtered", "filtered_nodes", "filtered_edges"),
        ("acyclic", "acyclic_nodes", "acyclic_edges"),
        ("walk_input", "walk_input_nodes", "walk_input_edges"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if not events.empty:
        x = np.arange(len(stages), dtype=np.float64)
        axes[0].plot(x, [events[node_col].mean() for _, node_col, _ in stages], marker="o", color="tab:blue")
        axes[1].plot(x, [events[edge_col].mean() for _, _, edge_col in stages], marker="o", color="tab:orange")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([stage for stage, _, _ in stages], rotation=30, ha="right")
    axes[0].set_ylabel("Mean nodes")
    axes[0].set_title("Node count through walkthrough")
    axes[1].set_ylabel("Mean edges")
    axes[1].set_title("Edge count through walkthrough")
    _save_plot_figure(fig, output_dir, f"graph_reduction_waterfall_{output_suffix}.png", "graph reduction waterfall plot")


def plot_efficiency_vs_occupancy(aggregate_data, output_dir, output_suffix):
    events = aggregate_data["events"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if not events.empty:
        num_nodes = events["num_nodes"].to_numpy(dtype=np.float64)
        bins = np.linspace(num_nodes.min(), num_nodes.max(), min(16, len(events)) + 1)
        digitized = np.digitize(num_nodes, bins[1:-1], right=False)
        centers = []
        eff_values = []
        dup_values = []
        fak_values = []
        for bin_index in range(len(bins) - 1):
            mask = digitized == bin_index
            if not np.any(mask):
                continue
            subset = events.loc[mask]
            n_particles = subset["n_particles"].sum()
            n_matched_target_particles = subset["n_matched_target_particles"].sum()
            n_matched_target_tracks = subset["n_matched_target_tracks"].sum()
            n_tracks = subset["n_tracks"].sum()
            centers.append(0.5 * (bins[bin_index] + bins[bin_index + 1]))
            eff_values.append(
                n_matched_target_particles / n_particles if n_particles > 0 else 0.0
            )
            dup_values.append(
                (n_matched_target_tracks - n_matched_target_particles) / n_matched_target_particles
                if n_matched_target_particles > 0
                else 0.0
            )
            fak_values.append(
                (n_tracks - subset["n_matched_tracks"].sum()) / n_tracks
                if n_tracks > 0
                else 0.0
            )
        ax.plot(centers, eff_values, marker="o", label="eff")
        ax.plot(centers, dup_values, marker="s", label="dup")
        ax.plot(centers, fak_values, marker="^", label="fak")
    ax.set_xlabel("Event occupancy [hits]")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Track metrics vs occupancy")
    ax.legend()
    _save_plot_figure(fig, output_dir, f"track_metrics_vs_num_nodes_{output_suffix}.png", "efficiency vs occupancy plot")


def plot_track_purity_distribution(aggregate_data, output_dir, output_suffix):
    track_diag = aggregate_data["track_diagnostics"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not track_diag.empty:
        ax.hist(
            track_diag["dominant_particle_fraction"].to_numpy(dtype=np.float64),
            bins=np.linspace(0.0, 1.0, 21),
            color="tab:blue",
            alpha=0.85,
        )
    ax.set_xlabel("Dominant-particle fraction")
    ax.set_ylabel("Tracks")
    ax.set_title("Track purity distribution")
    _save_plot_figure(fig, output_dir, f"track_purity_distribution_{output_suffix}.png", "track purity distribution plot")


def plot_track_completeness_distribution(aggregate_data, output_dir, output_suffix):
    particle_diag = aggregate_data["particle_diagnostics"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not particle_diag.empty:
        if "is_target_particle" in particle_diag:
            particle_diag = particle_diag[particle_diag["is_target_particle"].astype(bool)]
        ax.hist(
            particle_diag["best_track_completeness"].to_numpy(dtype=np.float64),
            bins=np.linspace(0.0, 1.0, 21),
            color="tab:green",
            alpha=0.85,
        )
    ax.set_xlabel("Best-track completeness")
    ax.set_ylabel("Truth particles")
    ax.set_title("Track completeness distribution")
    _save_plot_figure(fig, output_dir, f"track_completeness_distribution_{output_suffix}.png", "track completeness distribution plot")


def plot_split_merge_summary(aggregate_data, output_dir, output_suffix):
    track_diag = aggregate_data["track_diagnostics"]
    particle_diag = aggregate_data["particle_diagnostics"]
    many_truth_to_one_reco = int((track_diag["n_contributing_particles"] > 1).sum()) if not track_diag.empty else 0
    one_truth_to_many_reco = int((particle_diag["n_matched_tracks"] > 1).sum()) if not particle_diag.empty else 0
    clean_one_to_one = 0
    if not track_diag.empty and not particle_diag.empty:
        particle_match_counts = (
            particle_diag.set_index("particle_id")["n_matched_tracks"].to_dict()
        )
        clean_mask = []
        for _, row in track_diag.iterrows():
            dominant_particle_id = int(row["dominant_particle_id"])
            clean_mask.append(
                int(row["n_contributing_particles"]) == 1
                and dominant_particle_id in particle_match_counts
                and int(particle_match_counts[dominant_particle_id]) == 1
            )
        clean_one_to_one = int(np.sum(clean_mask))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["1 truth -> many reco", "many truth -> 1 reco", "clean 1-to-1"],
        [one_truth_to_many_reco, many_truth_to_one_reco, clean_one_to_one],
        color=["tab:orange", "tab:red", "tab:green"],
    )
    ax.set_ylabel("Count")
    ax.set_title("Split / merge summary")
    ax.tick_params(axis="x", rotation=20)
    _save_plot_figure(fig, output_dir, f"split_merge_summary_{output_suffix}.png", "split merge summary plot")


def plot_purity_vs_length(aggregate_data, output_dir, output_suffix):
    track_diag = aggregate_data["track_diagnostics"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not track_diag.empty:
        hist = ax.hist2d(
            track_diag["n_hits"].to_numpy(dtype=np.float64),
            track_diag["dominant_particle_fraction"].to_numpy(dtype=np.float64),
            bins=[np.linspace(0.5, max(track_diag["n_hits"].max() + 0.5, 2.5), 25), np.linspace(0.0, 1.0, 21)],
            cmap="Blues",
        )
        fig.colorbar(hist[3], ax=ax, label="Tracks")
    ax.set_xlabel("Track length [hits]")
    ax.set_ylabel("Dominant-particle fraction")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Track purity vs length")
    _save_plot_figure(fig, output_dir, f"track_purity_vs_length_{output_suffix}.png", "track purity vs length plot")


def plot_edge_score_diagnostics(aggregate_data, output_dir, output_suffix):
    edge_diag = aggregate_data["edge_diagnostics"]
    if edge_diag["bins"] is None:
        print("INFO: Skipping edge score diagnostics plot because no edge diagnostics were found.")
        return

    bins = edge_diag["bins"]
    centers = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {
        "true_kept": "tab:green",
        "true_dropped": "tab:blue",
        "fake_kept": "tab:red",
        "fake_dropped": "tab:orange",
    }
    for axis, stage_name in zip(axes, ("cc", "walk")):
        for class_name, color in colors.items():
            hist = edge_diag["stages"][stage_name][class_name]["hist"]
            if np.sum(hist) == 0:
                continue
            axis.plot(centers, hist, label=class_name, color=color)
        axis.set_title(stage_name.upper())
        axis.set_xlabel("Edge score")
    axes[0].set_ylabel("Edge count")
    axes[1].legend()
    _save_plot_figure(fig, output_dir, f"edge_score_diagnostics_{output_suffix}.png", "edge score diagnostics plot")


def plot_branching_diagnostics(aggregate_data, output_dir, output_suffix):
    branch_diag = aggregate_data["branch_diagnostics"]
    multiplicities = branch_diag["branch_multiplicities"]
    score_gaps = branch_diag["branch_score_gaps"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if multiplicities.size > 0:
        axes[0].hist(
            multiplicities,
            bins=np.arange(1, multiplicities.max() + 2, dtype=np.int64),
            color="tab:blue",
            alpha=0.85,
        )
    axes[0].set_xlabel("Branch multiplicity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Walk branching multiplicity")
    if score_gaps.size > 0:
        axes[1].hist(score_gaps, bins=20, color="tab:orange", alpha=0.85)
    axes[1].set_xlabel("Top-2 score gap")
    axes[1].set_ylabel("Count")
    threshold = branch_diag.get("ambiguity_gap_threshold", None)
    if threshold is not None:
        axes[1].axvline(float(threshold), color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Walk branching score gaps")
    _save_plot_figure(fig, output_dir, f"walk_branching_diagnostics_{output_suffix}.png", "branching diagnostics plot")

def plot_distance_histogram(graph:Data, output_dir:Path|List[str], plot_false_edges=False, seperate_plots=False, plot_suffix=None):
    """
    Distance between src and tgt embeddings for every true edge, 
    plotted by frequency (y axis)
    if "graph" is of type Data: runs on only the true edges
    else if "graph" is a list of edges
    """
    if isinstance(graph, Data):
        distances = get_edge_distances(graph, edges=None)
        distance_values = distances.detach().cpu().numpy()
        fake_distance_values = None
        if plot_false_edges:
            fake_edges = _get_fake_edges_within_radius(graph.to("cuda"), radius=1.5)
            fake_distances = get_edge_distances(graph.to("cuda"), edges=fake_edges)
            fake_distance_values = fake_distances.detach().cpu().numpy()
    else:
        distance_values, fake_distance_values = _collect_dml_distance_values(
            graph,
            {},
            plot_false_edges=plot_false_edges,
        )

    _plot_distance_histogram_from_values(
        distance_values,
        output_dir,
        plot_false_edges=plot_false_edges,
        seperate_plots=seperate_plots,
        plot_suffix=plot_suffix,
        fake_distance_values=fake_distance_values,
    )
        

def _find_cutoff(proportion:float, hist, bins):
    """
    Find the value such that proportion *100 percent of the values in 
    the histogram are less than or equal to the cutoff.
    """
    total = np.sum(hist)
    if total == 0:
        return bins[0] if len(bins) else 0.0
    if proportion <= 0.0:
        return bins[0]
    if proportion >= 1.0:
        return bins[-1]
    target = proportion * total
    idx = np.searchsorted(np.cumsum(hist), target, side="left")
    cutoff_idx = idx + 1
    if cutoff_idx >= len(bins):
        cutoff_idx = len(bins) - 1
    return bins[cutoff_idx]

def plot_cc(graph:Data, output_dir:Path|List[str], hparams:dict = {}, output_suffix=None,):
    """ Currently only works with Double Metric Learning Pipeline !!!"""
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if isinstance(graph, Data):
        cutoffs = None
        graphs = [graph]
    else:
        graphs = graph
        cutoffs = _get_dml_dataset_cutoffs(graphs, hparams)
    if cutoffs is None:
        _, track_edges, distances_cpu = _get_dml_track_edges_and_distances(graphs[0], hparams, _get_dml_knn_class(hparams))
        hist, bins = np.histogram(distances_cpu.numpy(), bins=100)
        cutoffs = [_find_cutoff(prop, hist, bins) for prop in CUTOFF_PROPORTIONS]
        graphs = [graphs[0]]

    knn_class = _get_dml_knn_class(hparams)
    n_components = np.zeros(len(cutoffs), dtype=np.int64)
    for event in graphs:
        _, track_edges, distances_cpu = _get_dml_track_edges_and_distances(event, hparams, knn_class)
        if track_edges.numel() == 0:
            continue
        for i, cutoff in enumerate(cutoffs):
            mask = distances_cpu < cutoff
            if not mask.any().item():
                continue
            edge_index = track_edges[:, mask]
            if edge_index.numel() == 0:
                continue
            nodes, inv = torch.unique(edge_index, return_inverse=True)
            edge_index = inv.view(edge_index.shape)
            adj_matrix = to_scipy_sparse_matrix(
                edge_index, num_nodes=int(nodes.numel())
            )
            n_comp, _ = connected_components(
                csgraph=adj_matrix, directed=True, connection="weak"
            )
            n_components[i] += int(n_comp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cutoffs, n_components, color="black", linestyle="-")
    ax.set_xlabel("Cutoff", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Connected components", ha="right", y=0.95, fontsize=14)
    ax.set_title("Connected components vs cutoff")
    plt.tight_layout()
    output_path = output_dir.joinpath(f"sparsity_hist_{output_suffix}.png" if output_suffix else "distance_hist.png")
    fig.savefig(output_path)
    print("INFO: Saved sparsity plot to " + str(output_path))
    plt.clf() # good manners

def plot_simple_graphs(graph:Data, output_dir:Path|List[str], hparams:dict = {}, output_suffix=None,):
    """Plot number of simple-path connected components vs distance cutoff."""
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if isinstance(graph, Data):
        cutoffs = None
        graphs = [graph]
    else:
        graphs = graph
        cutoffs = _get_dml_dataset_cutoffs(graphs, hparams)
    if cutoffs is None:
        _, track_edges, distances_cpu = _get_dml_track_edges_and_distances(graphs[0], hparams, _get_dml_knn_class(hparams))
        hist, bins = np.histogram(distances_cpu.numpy(), bins=100)
        cutoffs = [_find_cutoff(prop, hist, bins) for prop in CUTOFF_PROPORTIONS]
        graphs = [graphs[0]]

    knn_class = _get_dml_knn_class(hparams)
    simple_components = np.zeros(len(cutoffs), dtype=np.int64)
    for event in graphs:
        graph_cpu = event.cpu()
        _, track_edges, distances_cpu = _get_dml_track_edges_and_distances(event, hparams, knn_class)
        if track_edges.numel() == 0:
            continue
        track_edges_cpu = track_edges.detach().cpu()
        hit_id = graph_cpu.hit_id.detach().cpu() if hasattr(graph_cpu, "hit_id") else torch.arange(int(graph_cpu.num_nodes))
        for i, cutoff in enumerate(cutoffs):
            mask = distances_cpu < cutoff
            if not mask.any().item():
                continue
            edge_index = track_edges_cpu[:, mask]
            if edge_index.numel() == 0:
                continue
            filtered_graph = Data(
                edge_index=edge_index,
                track_edges=edge_index,
                num_nodes=int(graph_cpu.num_nodes),
                hit_id=hit_id,
            )
            simple_paths, _ = get_simple_path(filtered_graph)
            simple_components[i] += len(simple_paths)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cutoffs, simple_components, color="black", linestyle="-")
    ax.set_xlabel("Cutoff", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Simple-path components", ha="right", y=0.95, fontsize=14)
    ax.set_title("Simple-path components vs cutoff")
    plt.tight_layout()
    output_path = output_dir.joinpath(f"simple_path_hist_{output_suffix}.png" if output_suffix else "simple_path_hist.png")
    fig.savefig(output_path)
    print("INFO: Saved simple-path plot to " + str(output_path))
    plt.clf() # good manners

def plot_cutoff_efficiency(graph:Data, output_dir:Path|List[str], hparams:dict = {}, output_suffix=None,):
    """Plot particle-track capture efficiency vs distance cutoff."""
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if isinstance(graph, Data):
        cutoffs = None
        graphs = [graph]
    else:
        graphs = graph
        cutoffs = _get_dml_dataset_cutoffs(graphs, hparams)
    if cutoffs is None:
        _, _, distances_cpu = _get_dml_track_edges_and_distances(graphs[0], hparams, _get_dml_knn_class(hparams))
        hist, bins = np.histogram(distances_cpu.numpy(), bins=100)
        cutoffs = [_find_cutoff(prop, hist, bins) for prop in CUTOFF_PROPORTIONS]
        graphs = [graphs[0]]

    knn_class = _get_dml_knn_class(hparams)
    matched_particles = np.zeros(len(cutoffs), dtype=np.float64)
    total_particles = np.zeros(len(cutoffs), dtype=np.float64)
    for event in graphs:
        graph_cpu = event.cpu()
        if not hasattr(graph_cpu, "hit_particle_id"):
            raise AttributeError("plot_efficiency requires graph.hit_particle_id")
        _, track_edges, distances_cpu = _get_dml_track_edges_and_distances(event, hparams, knn_class)
        hit_particle_id = graph_cpu.hit_particle_id.detach().cpu().long()
        valid_hit_mask = hit_particle_id != 0
        if not valid_hit_mask.any().item():
            continue

        valid_particle_ids = hit_particle_id[valid_hit_mask]
        _, hit_particle_idx = torch.unique(valid_particle_ids, return_inverse=True)
        n_particles = int(hit_particle_idx.max().item()) + 1
        total_hits_per_particle = torch.bincount(
            hit_particle_idx, minlength=n_particles
        ).float()
        node_particle_idx = torch.full((int(graph_cpu.num_nodes),), -1, dtype=torch.long)
        node_particle_idx[valid_hit_mask] = hit_particle_idx
        if track_edges.numel() == 0:
            total_particles += n_particles
            continue
        track_edges_cpu = track_edges.detach().cpu()
        sort_idx = torch.argsort(distances_cpu)
        sorted_edges = track_edges_cpu[:, sort_idx]
        sorted_distances = distances_cpu[sort_idx].numpy()

        node_captured = torch.zeros(int(graph_cpu.num_nodes), dtype=torch.bool)
        captured_hits_per_particle = torch.zeros(n_particles, dtype=torch.float32)
        edge_ptr = 0

        for i, cutoff in enumerate(cutoffs):
            next_ptr = int(np.searchsorted(sorted_distances, cutoff, side="left"))
            if next_ptr > edge_ptr:
                new_nodes = torch.unique(
                    sorted_edges[:, edge_ptr:next_ptr].reshape(-1)
                )
                new_nodes = new_nodes[~node_captured[new_nodes]]
                if new_nodes.numel() > 0:
                    node_captured[new_nodes] = True
                    new_particle_idx = node_particle_idx[new_nodes]
                    new_particle_idx = new_particle_idx[new_particle_idx >= 0]
                    if new_particle_idx.numel() > 0:
                        captured_hits_per_particle += torch.bincount(
                            new_particle_idx, minlength=n_particles
                        ).float()
                edge_ptr = next_ptr

            captured_tracks = captured_hits_per_particle > (0.5 * total_hits_per_particle)
            matched_particles[i] += float(captured_tracks.sum().item())
            total_particles[i] += float(n_particles)

    efficiencies = np.divide(
        matched_particles,
        total_particles,
        out=np.zeros_like(matched_particles),
        where=total_particles > 0.0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cutoffs, efficiencies, color="black", linestyle="-")
    ax.set_xlabel("Cutoff", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Track efficiency", ha="right", y=0.95, fontsize=14)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Track efficiency vs cutoff")
    plt.tight_layout()
    output_path = output_dir.joinpath(
        f"track_efficiency_cutoff_{output_suffix}.png"
        if output_suffix
        else "track_efficiency_cutoff.png"
    )
    fig.savefig(output_path)
    print("INFO: Saved track efficiency plot to " + str(output_path))
    plt.clf()  # good manners
