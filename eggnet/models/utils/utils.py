import torch.nn as nn
####
import torch
import numpy as np
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.data import Data
from torch_scatter import scatter_max
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix
from typing import Callable
# from numba import njit, types
# from numba.typed import Dict, List



def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,  # TODO : change name to hidden_layer_norm while ensuring backward compatibility
    output_layer_norm=False,
    batch_norm=False,  # TODO : change name to hidden_batch_norm while ensuring backward compatibility
    output_batch_norm=False,
    input_dropout=0,
    hidden_dropout=0,
    track_running_stats=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:  # hidden_layer_norm
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:  # hidden_batch_norm
            layers.append(
                nn.BatchNorm1d(
                    sizes[i + 1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(hidden_activation())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(
                nn.BatchNorm1d(
                    sizes[-1],
                    eps=6e-05,
                    track_running_stats=track_running_stats,
                    affine=True,
                )  # TODO : Set BatchNorm and LayerNorm parameters in config file ?
            )
        layers.append(output_activation())
    return nn.Sequential(*layers)

# max_distance = 0.14;

def get_edge_distances(graph, edges=None):
    if edges is None:
        edges = graph.track_edges
    if edges.numel() == 0:
        return torch.empty((0,)).to(graph.src_embedding.device)
    src = graph.src_embedding[edges[0]]
    tgt = graph.tgt_embedding[edges[1]]
    return torch.sqrt(torch.sum((src - tgt) ** 2, dim=-1))
     
def filter_edges(graph, edge_index_key:str, critereon:Callable):
    """
    Remove edges from the graph that have a score below the given threshold
    And remove nodes that become isolated after the edge filtering
    score_name :  the name of the datakey corresponding to the GNN score
    critereon : function with 1 argument that takes in array of distances and outputs a bit mask
    """
    edge_distances = get_edge_distances(graph)
    keep = critereon(edge_distances)
    transform = RemoveIsolatedNodes()
    graph.track_edges_filtered = graph.track_edges.copy()[:, keep]
    graph.track_edges_filtered = transform(graph.track_edges_filtered)
        
def remove_cycles(graph, edge_index_key:str):
    """
    Remove cycles from the graph, simply by pointing all edges outwards
    """
    R = graph.hit_r**2 + graph.hit_z**2
    edge_flip_mask = (R[graph.get(edge_index_key)[0]] > R[graph.get(edge_index_key)[1]]) | (
        (R[graph.get(edge_index_key)[0]] == R[graph.get(edge_index_key)[1]])
        & (graph.get(edge_index_key)[0] > graph.get(edge_index_key)[1])
    )
    graph.get(edge_index_key)[:, edge_flip_mask] = graph.get(edge_index_key)[:, edge_flip_mask].flip(0)
    return graph

def get_simple_path(graph:Data):
    """
    Find all conencted components that form a singly linked "simple path". 
    """
    adj_matrix = to_scipy_sparse_matrix(graph.track_edges, num_nodes=graph.num_nodes)
    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=True, connection="weak"
    )
    labels = torch.from_numpy(labels).long()
    graph.labels = labels

    unique_labels, counts = torch.unique(labels, return_counts=True)
    large_component_labels = unique_labels[counts >= 3]

    subgraph_simple_paths, subgraph_rest = process_components(
        graph, labels, large_component_labels
    )

    simple_path_lists = labels_to_lists(subgraph_simple_paths)

    return simple_path_lists, subgraph_rest

def process_components(graph:Data, labels, large_component_labels):
    """
    Separate larger graph components into simple path-like subgraphs and the
    remaining complex parts.
    """
    in_degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
    out_degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
    in_degrees.index_add_(
        0, graph.edge_index[1], torch.ones(graph.num_edges, dtype=torch.long)
    )
    out_degrees.index_add_(
        0, graph.edge_index[0], torch.ones(graph.num_edges, dtype=torch.long)
    )

    large_component_mask = torch.isin(labels, large_component_labels)
    small_component_mask = ~large_component_mask

    in_degrees_max = scatter_max(in_degrees, labels, dim=0)[0]
    out_degrees_max = scatter_max(out_degrees, labels, dim=0)[0]

    simple_path_components = (in_degrees_max <= 1) & (out_degrees_max <= 1)

    simple_path_mask = simple_path_components[labels]

    large_component_simple_path_mask = simple_path_mask & large_component_mask
    large_component_complex_path_mask = ~simple_path_mask & large_component_mask

    assert torch.all(
        large_component_simple_path_mask
        | large_component_complex_path_mask
        | small_component_mask
        == torch.ones_like(labels, dtype=torch.bool)
    ), "Categorization is not complete and mutually exclusive"

    subgraph_simple_paths = graph.subgraph(large_component_simple_path_mask)
    subgraph_complex_paths = graph.subgraph(large_component_complex_path_mask)

    return subgraph_simple_paths, subgraph_complex_paths

def labels_to_lists(simple_path_graph):
    """
    """
    labels = simple_path_graph.labels
    hit_ids = simple_path_graph.hit_id

    unique_labels, counts = torch.unique(labels, return_counts=True)
    mask = labels.unsqueeze(0) == unique_labels.unsqueeze(1)
    grouped_hit_ids = hit_ids.unsqueeze(0).expand(len(unique_labels), -1)[mask]
    result = torch.split(grouped_hit_ids, counts.tolist())
    result = [track.tolist() for track in result]

    return result

import networkx as nx
def topological_sort_graph(G):
    """
    Sort Topologcially the graph such node u appears befroe v if the connection is u->v
    This ordering is valid only if the graph has no directed cycles
    """
    H = nx.DiGraph()
    # Add nodes w/o any features attached
    # maybe this is not needed given line 48?
    H.add_nodes_from(nx.topological_sort(G))

    # put it after the add nodes
    H.add_edges_from(G.edges(data=True))
    sorted_nodes = []

    # Add corresponding nodes features
    for i in list(nx.topological_sort(G)):
        sorted_nodes.append((i, G.nodes[i]))
    H.add_nodes_from(sorted_nodes)

    return H


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

def walkthrough(graph:Data):
    """
    Determine a directed, singular graph from each connected component
    """
    sorted_hit_ids = topological_sort_graph(graph, )