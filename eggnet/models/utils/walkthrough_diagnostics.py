import numpy as np
import torch

from eggnet.utils.mapping import get_node_target_mask, get_target_ordered


EDGE_SCORE_BINS = np.linspace(-2.0, 1.0, 121, dtype=np.float64)
DEFAULT_BRANCH_AMBIGUITY_GAP = 0.05


def _empty_tensor(dtype=torch.float32):
    return torch.empty((0,), dtype=dtype)


def _ensure_cpu_graph(graph):
    return graph.cpu() if hasattr(graph, "cpu") else graph


def _build_particle_property_lookup(graph, target_tracks=None):
    graph = _ensure_cpu_graph(graph)
    particle_id = graph.hit_particle_id.long().cpu()
    valid_mask = particle_id != 0
    target_mask = get_node_target_mask(graph, target_tracks).cpu()
    unique_particle_ids = torch.unique(particle_id[valid_mask])

    particle_lookup = {}
    target_particle_ids = set(torch.unique(particle_id[target_mask & valid_mask]).tolist())

    for pid in unique_particle_ids.tolist():
        particle_mask = particle_id == pid
        first_index = int(torch.nonzero(particle_mask, as_tuple=False)[0].item())
        particle_lookup[pid] = {
            "pt": float(graph.hit_particle_pt[first_index]),
            "eta": float(graph.hit_particle_eta[first_index]) if hasattr(graph, "hit_particle_eta") else 0.0,
            "n_true_hits": int(particle_mask.sum().item()),
            "is_target_particle": pid in target_particle_ids,
        }

    return particle_lookup


def build_track_particle_diagnostics(graph, target_tracks=None):
    graph = _ensure_cpu_graph(graph)
    if not hasattr(graph, "hit_track_labels"):
        return {}, {}

    particle_lookup = _build_particle_property_lookup(graph, target_tracks=target_tracks)
    hit_track_labels = graph.hit_track_labels.long().cpu()
    hit_particle_id = graph.hit_particle_id.long().cpu()
    assigned_track_ids = torch.unique(hit_track_labels[hit_track_labels >= 0]).tolist()
    hit_reco_method = getattr(graph, "hit_reco_method", None)

    track_rows = []
    track_purity_lookup = {}

    for track_id in assigned_track_ids:
        track_mask = hit_track_labels == track_id
        n_hits = int(track_mask.sum().item())
        track_particle_ids = hit_particle_id[track_mask]
        valid_particle_ids = track_particle_ids[track_particle_ids != 0]
        if valid_particle_ids.numel() > 0:
            unique_particle_ids, counts = torch.unique(valid_particle_ids, return_counts=True)
            dominant_index = int(torch.argmax(counts).item())
            dominant_particle_id = int(unique_particle_ids[dominant_index].item())
            dominant_hits = int(counts[dominant_index].item())
            dominant_fraction = dominant_hits / n_hits if n_hits > 0 else 0.0
            n_contributing_particles = int(unique_particle_ids.numel())
            particle_properties = particle_lookup[dominant_particle_id]
            particle_pt = particle_properties["pt"]
            particle_eta = particle_properties["eta"]
            is_target_track = particle_properties["is_target_particle"]
        else:
            dominant_particle_id = 0
            dominant_fraction = 0.0
            n_contributing_particles = 0
            particle_pt = 0.0
            particle_eta = 0.0
            is_target_track = False

        if hit_reco_method is not None:
            method_values = np.asarray(hit_reco_method)[track_mask.numpy()]
            reco_method = str(method_values[0]) if len(method_values) > 0 else "unknown"
        else:
            reco_method = "unknown"

        track_rows.append(
            {
                "track_id": track_id,
                "n_hits": n_hits,
                "dominant_particle_id": dominant_particle_id,
                "dominant_particle_fraction": dominant_fraction,
                "n_contributing_particles": n_contributing_particles,
                "is_target_track": is_target_track,
                "particle_pt": particle_pt,
                "particle_eta": particle_eta,
                "reco_method": reco_method,
            }
        )
        track_purity_lookup[track_id] = dominant_fraction

    particle_rows = []
    positive_particle_ids = sorted(particle_lookup.keys())
    for particle_id in positive_particle_ids:
        particle_mask = hit_particle_id == particle_id
        reco_track_ids = hit_track_labels[particle_mask]
        reco_track_ids = reco_track_ids[reco_track_ids >= 0]
        unique_reco_track_ids = torch.unique(reco_track_ids)

        best_track_id = -1
        n_reco_hits_best_track = 0
        best_track_purity = 0.0
        best_track_completeness = 0.0
        if unique_reco_track_ids.numel() > 0:
            best_hits = -1
            for track_id in unique_reco_track_ids.tolist():
                hits_in_track = int((reco_track_ids == track_id).sum().item())
                if hits_in_track > best_hits:
                    best_hits = hits_in_track
                    best_track_id = track_id
            n_reco_hits_best_track = best_hits
            best_track_purity = float(track_purity_lookup.get(best_track_id, 0.0))
            best_track_completeness = (
                n_reco_hits_best_track / particle_lookup[particle_id]["n_true_hits"]
                if particle_lookup[particle_id]["n_true_hits"] > 0
                else 0.0
            )

        particle_rows.append(
            {
                "particle_id": particle_id,
                "n_true_hits": particle_lookup[particle_id]["n_true_hits"],
                "n_reco_hits_best_track": n_reco_hits_best_track,
                "n_matched_tracks": int(unique_reco_track_ids.numel()),
                "best_track_id": best_track_id,
                "best_track_purity": best_track_purity,
                "best_track_completeness": best_track_completeness,
                "particle_pt": particle_lookup[particle_id]["pt"],
                "particle_eta": particle_lookup[particle_id]["eta"],
                "is_target_particle": particle_lookup[particle_id]["is_target_particle"],
            }
        )

    def _rows_to_dict(rows, string_fields=None):
        string_fields = set(string_fields or [])
        if not rows:
            return {}
        out = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            if key in string_fields:
                out[key] = list(values)
            elif isinstance(values[0], bool):
                out[key] = torch.as_tensor(values, dtype=torch.bool)
            elif isinstance(values[0], int):
                out[key] = torch.as_tensor(values, dtype=torch.long)
            else:
                out[key] = torch.as_tensor(values, dtype=torch.float32)
        return out

    return (
        _rows_to_dict(track_rows, string_fields={"reco_method"}),
        _rows_to_dict(particle_rows),
    )


def _build_original_node_index_lookup(graph):
    hit_ids = graph.hit_id.long().cpu()
    sorted_hit_ids, sorted_indices = torch.sort(hit_ids)
    return sorted_hit_ids, sorted_indices


def _map_compact_edges_to_original_indices(original_graph, compact_graph):
    if compact_graph is None or compact_graph.edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sorted_hit_ids, sorted_indices = _build_original_node_index_lookup(original_graph)
    compact_edge_hit_ids = compact_graph.hit_id.long().cpu()[compact_graph.edge_index.long().cpu()]
    positions = torch.searchsorted(sorted_hit_ids, compact_edge_hit_ids)
    valid = (positions < sorted_hit_ids.numel()) & (sorted_hit_ids[positions] == compact_edge_hit_ids)
    if not bool(valid.all().item()):
        raise RuntimeError("Failed to map compact walkthrough edges back to the original graph.")
    return sorted_indices[positions]


def _compute_edge_class_stats(scores, truth_mask, keep_mask, bins):
    stats = {}
    class_masks = {
        "true_kept": truth_mask & keep_mask,
        "true_dropped": truth_mask & ~keep_mask,
        "fake_kept": ~truth_mask & keep_mask,
        "fake_dropped": ~truth_mask & ~keep_mask,
    }
    for class_name, class_mask in class_masks.items():
        class_scores = scores[class_mask]
        class_scores_np = class_scores.detach().cpu().numpy() if class_scores.numel() > 0 else np.asarray([], dtype=np.float64)
        hist, _ = np.histogram(class_scores_np, bins=bins)
        stats[class_name] = {
            "count": int(class_scores.numel()),
            "score_sum": float(class_scores_np.sum()) if class_scores_np.size else 0.0,
            "score_sum_sq": float(np.square(class_scores_np).sum()) if class_scores_np.size else 0.0,
            "hist": torch.as_tensor(hist, dtype=torch.long),
        }
    return stats


def build_edge_diagnostics_summary(
    graph,
    initial_edge_index,
    cc_threshold,
    walk_input_graph=None,
    walk_pruned_graph=None,
    bins=EDGE_SCORE_BINS,
):
    graph = _ensure_cpu_graph(graph)
    initial_edge_index = initial_edge_index.detach().cpu()
    if initial_edge_index.numel() == 0:
        empty_hist = torch.zeros(len(bins) - 1, dtype=torch.long)
        empty_stage = {
            key: {
                "count": 0,
                "score_sum": 0.0,
                "score_sum_sq": 0.0,
                "hist": empty_hist.clone(),
            }
            for key in ("true_kept", "true_dropped", "fake_kept", "fake_dropped")
        }
        return {
            "bins": torch.as_tensor(bins, dtype=torch.float32),
            "cc": empty_stage,
            "walk": empty_stage,
        }

    src = graph.src_embedding[initial_edge_index[0]]
    tgt = graph.tgt_embedding[initial_edge_index[1]]
    cc_scores = (1.0 - torch.sqrt(torch.sum((src - tgt) ** 2, dim=-1))).float().cpu()
    cc_keep_mask = cc_scores > float(cc_threshold)
    cc_truth_mask = get_target_ordered(initial_edge_index, graph.track_edges.long().cpu()) > 0

    summary = {
        "bins": torch.as_tensor(bins, dtype=torch.float32),
        "cc": _compute_edge_class_stats(cc_scores, cc_truth_mask, cc_keep_mask, bins),
    }

    if walk_input_graph is None or walk_pruned_graph is None:
        summary["walk"] = {
            key: {
                "count": 0,
                "score_sum": 0.0,
                "score_sum_sq": 0.0,
                "hist": torch.zeros(len(bins) - 1, dtype=torch.long),
            }
            for key in ("true_kept", "true_dropped", "fake_kept", "fake_dropped")
        }
        return summary

    walk_input_edges = _map_compact_edges_to_original_indices(graph, walk_input_graph)
    walk_pruned_edges = _map_compact_edges_to_original_indices(graph, walk_pruned_graph)
    walk_scores = walk_input_graph["track_edge_scores_inverted"].detach().cpu().float()
    walk_truth_mask = get_target_ordered(walk_input_edges, graph.track_edges.long().cpu()) > 0

    if walk_pruned_edges.numel() == 0:
        walk_keep_mask = torch.zeros(walk_input_edges.shape[1], dtype=torch.bool)
    else:
        max_node = int(
            max(
                walk_input_edges.max().item(),
                walk_pruned_edges.max().item(),
                graph.track_edges.max().item() if graph.track_edges.numel() > 0 else 0,
            )
        ) + 1
        walk_input_ids = walk_input_edges[0].long() * max_node + walk_input_edges[1].long()
        walk_pruned_ids = walk_pruned_edges[0].long() * max_node + walk_pruned_edges[1].long()
        walk_pruned_ids_sorted, _ = torch.sort(walk_pruned_ids)
        pos = torch.searchsorted(walk_pruned_ids_sorted, walk_input_ids)
        walk_keep_mask = (pos < walk_pruned_ids_sorted.numel()) & (
            walk_pruned_ids_sorted[pos] == walk_input_ids
        )

    summary["walk"] = _compute_edge_class_stats(walk_scores, walk_truth_mask, walk_keep_mask, bins)
    return summary


def build_walk_branch_diagnostics(pruned_graph, ambiguity_gap_threshold=DEFAULT_BRANCH_AMBIGUITY_GAP):
    if pruned_graph is None or pruned_graph.edge_index.numel() == 0:
        return {
            "num_branch_points": 0,
            "num_ambiguous_choices": 0,
            "ambiguity_gap_threshold": float(ambiguity_gap_threshold),
            "branch_multiplicities": _empty_tensor(dtype=torch.long),
            "branch_score_gaps": _empty_tensor(dtype=torch.float32),
        }

    edge_index = pruned_graph.edge_index.detach().cpu().long()
    edge_scores = pruned_graph["track_edge_scores_inverted"].detach().cpu().float()
    out_degree = torch.bincount(edge_index[0], minlength=int(pruned_graph.num_nodes))
    branch_nodes = torch.nonzero(out_degree > 1, as_tuple=False).flatten()
    branch_multiplicities = out_degree[branch_nodes].long()

    score_gaps = []
    for node_index in branch_nodes.tolist():
        node_scores = edge_scores[edge_index[0] == node_index]
        top_scores, _ = torch.sort(node_scores, descending=True)
        if top_scores.numel() >= 2:
            score_gaps.append(float(top_scores[0] - top_scores[1]))

    branch_score_gaps = torch.as_tensor(score_gaps, dtype=torch.float32)
    num_ambiguous_choices = int((branch_score_gaps < float(ambiguity_gap_threshold)).sum().item())
    return {
        "num_branch_points": int(branch_nodes.numel()),
        "num_ambiguous_choices": num_ambiguous_choices,
        "ambiguity_gap_threshold": float(ambiguity_gap_threshold),
        "branch_multiplicities": branch_multiplicities,
        "branch_score_gaps": branch_score_gaps,
    }
