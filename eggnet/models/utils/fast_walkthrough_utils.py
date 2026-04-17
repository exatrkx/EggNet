# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_max
from numba import njit, types
from numba.typed import Dict, List

from eggnet.utils.timing import profile_section, set_profile_metadata


def _record_graph_profile_metadata(
    profile_target,
    prefix,
    graph,
    include_structure=False,
):
    if profile_target is None:
        return

    num_nodes = int(graph.num_nodes)
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        num_edges = 0
    elif hasattr(edge_index, "shape") and len(edge_index.shape) >= 2:
        num_edges = int(edge_index.shape[1])
    else:
        num_edges = 0
    set_profile_metadata(profile_target, f"{prefix}.num_nodes", num_nodes)
    set_profile_metadata(profile_target, f"{prefix}.num_edges", num_edges)

    if not include_structure or num_edges == 0 or num_nodes == 0:
        return

    edge_index = edge_index.cpu()
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    set_profile_metadata(
        profile_target,
        f"{prefix}.max_out_degree",
        int(out_degree.max().item()),
    )
    set_profile_metadata(
        profile_target,
        f"{prefix}.branching_nodes",
        int((out_degree > 1).sum().item()),
    )


def _empty_graph_like(graph, score_name):
    output_device = graph.hit_id.device
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=output_device)
    empty_node_index = torch.empty((0,), dtype=torch.long, device=output_device)
    out_graph = Data(
        edge_index=empty_edge_index,
        hit_id=graph.hit_id[empty_node_index],
        num_nodes=0,
        hit_r=graph.hit_r[empty_node_index],
        hit_z=graph.hit_z[empty_node_index],
    )
    out_graph[score_name] = torch.empty(
        (0,),
        dtype=graph[score_name].dtype if score_name in graph else graph.hit_r.dtype,
        device=output_device,
    )
    return out_graph


def _compact_graph_from_edge_mask(graph, edge_mask, score_name):
    output_device = graph.hit_id.device
    if edge_mask.numel() == 0 or not bool(edge_mask.any().item()):
        return _empty_graph_like(graph, score_name)

    kept_edge_index = graph.edge_index[:, edge_mask]
    kept_edge_scores = graph[score_name][edge_mask].float().to(output_device)
    kept_nodes, inverse = torch.unique(
        kept_edge_index.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    compact_edge_index = inverse.view(2, -1).to(output_device)

    out_graph = Data(
        edge_index=compact_edge_index,
        hit_id=graph.hit_id[kept_nodes],
        num_nodes=int(kept_nodes.numel()),
        hit_r=graph.hit_r[kept_nodes],
        hit_z=graph.hit_z[kept_nodes],
    )
    out_graph[score_name] = kept_edge_scores
    return out_graph


def _compact_graph_from_node_mask(graph, node_mask, score_name):
    output_device = graph.hit_id.device
    kept_nodes = torch.nonzero(node_mask, as_tuple=False).flatten()
    if kept_nodes.numel() == 0:
        return _empty_graph_like(graph, score_name)

    node_lookup = torch.full(
        (int(graph.num_nodes),),
        -1,
        dtype=torch.long,
        device=node_mask.device,
    )
    node_lookup[kept_nodes] = torch.arange(
        kept_nodes.numel(),
        device=node_mask.device,
    )
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    compact_edge_index = node_lookup[graph.edge_index[:, edge_mask]].to(output_device)

    out_graph = Data(
        edge_index=compact_edge_index,
        hit_id=graph.hit_id[kept_nodes],
        num_nodes=int(kept_nodes.numel()),
        hit_r=graph.hit_r[kept_nodes],
        hit_z=graph.hit_z[kept_nodes],
    )
    out_graph[score_name] = graph[score_name][edge_mask].float().to(output_device)
    return out_graph


def filter_graph(graph, edge_index, score_name, threshold):
    if edge_index.device != graph.src_embedding.device:
        edge_index_for_score = edge_index.to(graph.src_embedding.device)
    else:
        edge_index_for_score = edge_index

    if edge_index_for_score.numel() == 0:
        edge_scores = torch.empty(
            (0,),
            dtype=graph.src_embedding.dtype,
            device=edge_index_for_score.device,
        )
    else:
        src = graph.src_embedding[edge_index_for_score[0]]
        tgt = graph.tgt_embedding[edge_index_for_score[1]]
        distances = torch.sqrt(torch.sum((src - tgt) ** 2, dim=-1))
        edge_scores = 1 - distances if "inverted" in score_name else distances

    mask = edge_scores > threshold
    scored_graph = Data(
        edge_index=edge_index.to(graph.hit_id.device),
        hit_id=graph.hit_id,
        num_nodes=int(graph.num_nodes),
        hit_r=graph.hit_r,
        hit_z=graph.hit_z,
    )
    scored_graph[score_name] = edge_scores.float().to(graph.hit_id.device)
    return _compact_graph_from_edge_mask(scored_graph, mask.to(graph.hit_id.device), score_name)


def process_components(graph, labels, large_component_mask, simple_path_components, score_name):
    device = graph.edge_index.device
    in_degrees = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    out_degrees = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    in_degrees.index_add_(
        0,
        graph.edge_index[1],
        torch.ones(graph.num_edges, dtype=torch.long, device=device),
    )
    out_degrees.index_add_(
        0,
        graph.edge_index[0],
        torch.ones(graph.num_edges, dtype=torch.long, device=device),
    )

    simple_path_mask = simple_path_components[labels]
    large_component_simple_path_mask = simple_path_mask & large_component_mask
    large_component_complex_path_mask = ~simple_path_mask & large_component_mask

    subgraph_simple_paths = _compact_graph_from_node_mask(
        graph,
        large_component_simple_path_mask,
        score_name,
    )
    subgraph_complex_paths = _compact_graph_from_node_mask(
        graph,
        large_component_complex_path_mask,
        score_name,
    )

    return subgraph_simple_paths, subgraph_complex_paths


def labels_to_lists(labels, hit_ids):
    unique_labels = torch.unique(labels)
    return [hit_ids[labels == label].tolist() for label in unique_labels]


def get_simple_path(
    graph,
    profile_target=None,
    profiling_enabled=True,
    include_structure_metadata=False,
):
    from scipy.sparse.csgraph import connected_components
    from torch_geometric.utils import to_scipy_sparse_matrix

    if profile_target is None:
        profile_target = graph

    with profile_section(
        profile_target,
        "fast_walkthrough.simple_path.to_sparse",
        enabled=profiling_enabled,
    ):
        adj_matrix = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)

    with profile_section(
        profile_target,
        "fast_walkthrough.simple_path.connected_components",
        enabled=profiling_enabled,
    ):
        n_components, labels = connected_components(
            csgraph=adj_matrix, directed=True, connection="weak"
        )
    set_profile_metadata(
        profile_target,
        "fast_walkthrough.simple_path.n_components",
        int(n_components),
    )
    labels = torch.from_numpy(labels).long().to(graph.edge_index.device)

    with profile_section(
        profile_target,
        "fast_walkthrough.simple_path.large_components",
        enabled=profiling_enabled,
    ):
        component_sizes = torch.bincount(labels, minlength=n_components)
        large_component_mask = component_sizes[labels] >= 3
        in_degrees = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
        out_degrees = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
        in_degrees_max = scatter_max(
            in_degrees,
            labels,
            dim=0,
            dim_size=n_components,
        )[0]
        out_degrees_max = scatter_max(
            out_degrees,
            labels,
            dim=0,
            dim_size=n_components,
        )[0]
        simple_path_components = (in_degrees_max <= 1) & (out_degrees_max <= 1)
        large_component_count = int((component_sizes >= 3).sum().item())
    set_profile_metadata(
        profile_target,
        "fast_walkthrough.simple_path.large_components",
        large_component_count,
    )

    with profile_section(
        profile_target,
        "fast_walkthrough.simple_path.process_components",
        enabled=profiling_enabled,
    ):
        subgraph_simple_paths, subgraph_rest = process_components(
            graph,
            labels,
            large_component_mask,
            simple_path_components,
            score_name="track_edge_scores_inverted",
        )
    _record_graph_profile_metadata(
        profile_target,
        "fast_walkthrough.simple_path.simple_subgraph",
        subgraph_simple_paths,
        include_structure=include_structure_metadata,
    )
    _record_graph_profile_metadata(
        profile_target,
        "fast_walkthrough.simple_path.rest_subgraph",
        subgraph_rest,
        include_structure=include_structure_metadata,
    )

    with profile_section(
        profile_target,
        "fast_walkthrough.simple_path.labels_to_lists",
        enabled=profiling_enabled,
    ):
        simple_path_node_mask = large_component_mask & simple_path_components[labels]
        simple_path_lists = labels_to_lists(
            labels[simple_path_node_mask],
            graph.hit_id[simple_path_node_mask],
        )
    set_profile_metadata(
        profile_target,
        "fast_walkthrough.simple_path.cc_tracks",
        int(len(simple_path_lists)),
    )

    return simple_path_lists, subgraph_rest


@njit
def topological_sort_numba(numba_edges):
    in_degree = Dict.empty(key_type=types.int64, value_type=types.int64)
    for node in numba_edges:
        if node not in in_degree:
            in_degree[node] = 0
        for neighbor in numba_edges[node]:
            if neighbor not in in_degree:
                in_degree[neighbor] = 1
            else:
                in_degree[neighbor] += 1

    queue = List()
    for node in in_degree:
        if in_degree[node] == 0:
            queue.append(node)

    result = List()
    while queue:
        node = queue.pop(0)
        result.append(node)

        if node in numba_edges:
            for neighbor in numba_edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    # TODO: Handle this later
    # if len(result) != len(numba_edges):
    #     return None  # Graph has a cycle

    return result


def topological_sort_graph(graph, numba_edges):
    sorted_hit_ids = topological_sort_numba(numba_edges)
    return sorted_hit_ids


def resolve_ambiguities(tracks, max_ambi_hits):
    # 1. Order the optimized tracks by their length, in descending order
    sorted_tracks = sorted(tracks, key=lambda x: len(x), reverse=True)

    # 2. Iterate through each track, keeping a dict of hit_ids, and remove any hit ids from tracks if they have already been used more than once
    used_hit_ids = {}
    resolved_tracks = []

    for track in sorted_tracks:
        updated_track = [
            hit_id for hit_id in track if used_hit_ids.get(hit_id, 0) < max_ambi_hits
        ]
        resolved_tracks.append(updated_track)
        for hit_id in updated_track:
            used_hit_ids[hit_id] = used_hit_ids.get(hit_id, 0) + 1

    return resolved_tracks


def walk_through(
    graph,
    score_name,
    th_min,
    th_add,
    allow_node_reuse,
    mode,
    lookback=False,
    profile_target=None,
    profiling_enabled=True,
    include_structure_metadata=False,
):
    if profile_target is None:
        profile_target = graph

    with profile_section(
        profile_target,
        "fast_walkthrough.walk.max_add_cuts",
        enabled=profiling_enabled,
    ):
        graph = max_add_cuts(graph, score_name, th_min, th_add, lookback=lookback)
    _record_graph_profile_metadata(
        profile_target,
        "fast_walkthrough.walk.graph",
        graph,
        include_structure=include_structure_metadata,
    )
    with profile_section(
        profile_target,
        "fast_walkthrough.walk.convert_to_numba",
        enabled=profiling_enabled,
    ):
        numba_edges = convert_pyg_graph_to_numba(graph, score_name)
    with profile_section(
        profile_target,
        "fast_walkthrough.walk.topological_sort",
        enabled=profiling_enabled,
    ):
        sorted_hit_ids = topological_sort_graph(graph, numba_edges=numba_edges)
    set_profile_metadata(
        profile_target,
        "fast_walkthrough.walk.sorted_hit_ids",
        int(len(sorted_hit_ids)),
    )
    with profile_section(
        profile_target,
        "fast_walkthrough.walk.get_tracks",
        enabled=profiling_enabled,
    ):
        tracks = get_tracks(
            numba_edges,
            sorted_hit_ids,
            allow_node_reuse,
            mode,
        )
    set_profile_metadata(
        profile_target,
        "fast_walkthrough.walk.tracks",
        int(len(tracks)),
    )
    return tracks


def max_add_cuts(graph, score_name, th_min, th_add, lookback=False):
    edge_scores = graph[score_name]
    edge_index = graph.edge_index

    mask_min = edge_scores > th_min
    mask_add = edge_scores > th_add

    out, argmax = scatter_max(edge_scores, edge_index[0], dim=0)
    mask_max = torch.zeros_like(mask_min, dtype=torch.bool)
    mask_max[argmax[out >= th_min]] = True

    final_mask = mask_max | mask_add

    if lookback:
        # to select edges that are not connected to the start node
        not_first_mask = torch.isin(edge_index[0], edge_index[1])
        # to select the incoming edge with the highest score for each junction
        out, argmax = scatter_max(edge_scores, edge_index[1], dim=0)
        mask_imcoming_max = torch.zeros_like(mask_min, dtype=torch.bool)
        mask_imcoming_max[argmax[argmax < len(mask_imcoming_max)]] = True
        # either not the first, or has the highest score
        final_mask = (not_first_mask | mask_imcoming_max) & final_mask

    return _compact_graph_from_edge_mask(graph, final_mask, score_name)


@njit
def find_longest_path(complete_paths):
    longest_path = List.empty_list(types.int64)
    max_length = 0
    for path in complete_paths:
        if len(path) > max_length:
            max_length = len(path)
            longest_path.clear()
            longest_path.extend(path)
    return longest_path


@njit
def find_most_likely_local_path(complete_paths, complete_branching_scores):
    # Compare the paths with their branching score in order
    best_path = List.empty_list(types.int64)
    best_score = List.empty_list(types.float64)
    for path, branching_score in zip(complete_paths, complete_branching_scores):
        if branching_score > best_score:
            best_score.clear()
            best_score.extend(branching_score)
            best_path.clear()
            best_path.extend(path)
    return best_path


@njit
def process_sorted_nodes(sorted_hit_ids, numba_edges, allow_node_reuse, mode):
    tracks = List()
    used_nodes = Dict.empty(key_type=types.int64, value_type=types.boolean)
    for hit_id in sorted_hit_ids:
        if hit_id in used_nodes:
            continue

        complete_paths, complete_branching_scores = find_paths(hit_id, numba_edges, used_nodes, allow_node_reuse, mode)

        if complete_paths:
            if mode == 0:
                # resolved_path = find_longest_path(complete_paths)
                resolved_path = complete_paths[0] # they should all have the same length, so we just select the first one
            elif mode == 1:
                raise NotImplementedError("Most likely path not implemented yet")
            elif mode == 2:
                resolved_path = find_most_likely_local_path(complete_paths, complete_branching_scores)

            if len(resolved_path) > 1:
                tracks.append(resolved_path)
                for node in resolved_path:
                    used_nodes[node] = True

    return tracks


def get_tracks(numba_edges, sorted_hit_ids, allow_node_reuse, mode):
    numba_sorted_hit_ids = List(sorted_hit_ids)
    tracks = process_sorted_nodes(numba_sorted_hit_ids, numba_edges, allow_node_reuse, mode)
    return tracks


@njit
def find_paths(start_node, edges, used_nodes, allow_node_reuse, mode):
    paths = List()
    paths.append(List([start_node]))
    complete_paths = List()
    if mode >= 0:
        # for mode 1 and 2, we need to keep track of the branching scores
        branching_scores = List()
        branching_scores.append(List([0.0]))
        complete_branching_scores = List()

    while len(paths) > 0:
        if mode == 2:
            old_complete_paths = complete_paths.copy() # for mode 2, we also need to keep the second longest paths
            old_complete_branching_scores = complete_branching_scores.copy()
            complete_branching_scores.clear()
        if mode != 1:
            complete_paths.clear() # we only need to keep the longest paths
        # BFS approach to iterate all the possible paths
        for i in range(len(paths)):
            path = paths.pop(0)
            if mode > 0:
                branching_score = branching_scores.pop(0)
            current_node = path[-1]

            if current_node not in edges:
                complete_paths.append(path)
                if mode > 0:
                    complete_branching_scores.append(branching_score)
                continue

            num_branches = 0
            for neighbor in edges[current_node]:
                if not allow_node_reuse and neighbor in used_nodes:
                    continue
                num_branches += 1
                new_path = path.copy()
                new_path.append(neighbor)
                paths.append(new_path)
            if num_branches == 0:
                # if all neighbors are already used, we consider the path complete
                complete_paths.append(path)
                if mode > 0:
                    complete_branching_scores.append(branching_score)
                continue
            if mode > 0:
                for neighbor in edges[current_node]:
                    if not allow_node_reuse and neighbor in used_nodes:
                        continue
                    new_branching_score = branching_score.copy()
                    if num_branches > 1:
                        # add bracnching scores if more than one next nodes
                        new_branching_score.append(edges[current_node][neighbor])
                    branching_scores.append(new_branching_score)

    if mode == 2:
        complete_paths.extend(old_complete_paths)
        complete_branching_scores.extend(old_complete_branching_scores)

    return complete_paths, complete_branching_scores


inner_dict_type = types.DictType(types.int64, types.float64)
outer_dict_type = types.DictType(types.int64, inner_dict_type)


@njit
def pyg_to_dict_numba(
    edge_index_src: np.ndarray,
    edge_index_dst: np.ndarray,
    edge_attr: np.ndarray,
    hit_ids: np.ndarray,
):
    edge_index_src = edge_index_src.astype(np.int64)
    edge_index_dst = edge_index_dst.astype(np.int64)
    edge_attr = edge_attr.astype(np.float64)
    hit_ids = hit_ids.astype(np.int64)

    numba_edges = Dict.empty(key_type=types.int64, value_type=inner_dict_type)

    for i in range(len(edge_index_src)):
        src = hit_ids[edge_index_src[i]]
        dst = hit_ids[edge_index_dst[i]]
        attr = edge_attr[i]

        if src not in numba_edges:
            numba_edges[src] = Dict.empty(
                key_type=types.int64, value_type=types.float64
            )

        numba_edges[src][dst] = attr

    return numba_edges


def convert_pyg_graph_to_numba(pyg_graph, score_name):
    edge_index = pyg_graph.edge_index.detach().cpu().numpy()
    edge_index_src = edge_index[0]
    edge_index_dst = edge_index[1]
    edge_attr = pyg_graph[score_name].detach().cpu().numpy()
    hit_ids = pyg_graph.hit_id.detach().cpu().numpy()

    numba_edges = pyg_to_dict_numba(edge_index_src, edge_index_dst, edge_attr, hit_ids)

    return numba_edges
