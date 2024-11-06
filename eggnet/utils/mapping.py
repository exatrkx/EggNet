from enum import Enum

import torch
from torch_scatter import scatter


def get_target(edges, hit_particle_id):
    """
    Return truth labels for all edges.
    """
    y = torch.ones(edges.shape[1], device=edges.device) * (-1)
    y[
        (hit_particle_id[edges[0]] == hit_particle_id[edges[1]])
        & (hit_particle_id[edges[0]] != 0)
    ] = 1
    return y


def get_weight(batch, edges, y, weighting_config):
    """
    Return edge weights based on the specified weighting configuration.
    """
    w = torch.ones(edges.shape[1], device=edges.device)
    if not weighting_config:
        return w
    if (
        "true_default" in weighting_config
        and weighting_config["true_default"] is not None
    ):
        w[y == 1] = weighting_config["true_default"]
    if (
        "fake_default" in weighting_config
        and weighting_config["fake_default"] is not None
    ):
        w[y == -1] = weighting_config["fake_default"]
    if (
        "conditional_weighting" in weighting_config
        and weighting_config["conditional_weighting"] is not None
    ):
        for weight_spec in weighting_config["conditional_weighting"]:
            graph_mask = get_edge_target_mask(batch, edges, target_tracks=weight_spec["conditions"], y=y)

            w[graph_mask] = weight_spec["weight"]

    return w


def get_condition_lambda(condition_key, condition_val):
    condition_dict = {
        "is": lambda event: event[condition_key] == condition_val,
        "is_not": lambda event: event[condition_key] != condition_val,
        "in": lambda event: torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "not_in": lambda event: ~torch.isin(
            event[condition_key],
            torch.tensor(condition_val[1], device=event[condition_key].device),
        ),
        "within": lambda event: (condition_val[1][0] <= event[condition_key].float())
        & (event[condition_key].float() <= condition_val[1][1]),
        "not_within": lambda event: ~(
            (condition_val[1][0] <= event[condition_key].float())
            & (event[condition_key].float() <= condition_val[1][1])
        ),
    }

    if isinstance(condition_val, bool):
        return lambda event: event[condition_key] == condition_val
    elif isinstance(condition_val, list) and not isinstance(condition_val[0], str):
        return lambda event: (condition_val[0] <= event[condition_key].float()) & (
            event[condition_key].float() <= condition_val[1]
        )
    elif isinstance(condition_val, list):
        return condition_dict[condition_val[0]]
    else:
        raise ValueError(f"Condition {condition_val} not recognised")


def get_edge_target_mask(event, edges, target_tracks=None, y=None):
    """
    Get the masking for the target edges (edges associated with target particles).
    target_tracks is the config specifying the selections of target particles.
    """
    if y is None:
        graph_mask = torch.ones_like(edges[0], dtype=torch.bool)
    else:
        graph_mask = y == 1

    if target_tracks:
        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            value_mask = condition_lambda(event)
            graph_mask = graph_mask & value_mask[edges[0]]

    return graph_mask


def get_node_target_mask(event, target_tracks=None):
    """
    Get the masking for the target hits (hits associated with target particles).
    target_tracks is the config specifying the selections of target particles.
    """
    graph_mask = event.hit_particle_id != 0

    if target_tracks:
        for condition_key, condition_val in target_tracks.items():
            condition_lambda = get_condition_lambda(condition_key, condition_val)
            value_mask = condition_lambda(event)
            graph_mask = graph_mask & value_mask

    return graph_mask


def get_number_of_true_edges(
    batch,
    target=None,
    target_tracks=None,
    reduction=None,
    upper_bound=None,
    weighting_config=None,
):
    """
    Get the number of true edges for each node. Can sum them up if reduction is sum.
    Can also return the upper bound of the number of true edges one can possibly reconstruct with a KNN (if upper_bound is given).
    Can calculate only for the target particles. Target particles can be specified based on weighting config ("weight-based"), or selections ("mask-based").
    """
    if target is None:
        signal_mask = torch.ones_like(batch.hit_particle_nhits, dtype=torch.bool)
    elif target == "weight-based":
        if (
            "true_default" not in weighting_config
            or weighting_config["true_default"] != 0
        ):
            signal_mask = torch.ones_like(batch.hit_particle_nhits, dtype=torch.bool)
        else:
            signal_mask = torch.zeros_like(batch.hit_particle_nhits, dtype=torch.bool)

        if (
            "conditional_weighting" in weighting_config
            and weighting_config["conditional_weighting"]
        ):
            for weight_spec in weighting_config["conditional_weighting"]:
                graph_mask = get_node_target_mask(batch, target_tracks=weight_spec["conditions"])

                if weight_spec["weight"] == 0:
                    signal_mask &= ~graph_mask
                else:
                    signal_mask |= graph_mask
    elif target == "mask-based":
        signal_mask = get_node_target_mask(batch, target_tracks=target_tracks)
    hit_t = (
        batch.hit_particle_nhits[(batch.hit_particle_nhits > 1) & (signal_mask)] - 1
    )
    if upper_bound is not None:
        max_hit_t = torch.min(
            torch.stack(
                [
                    hit_t,
                    torch.full(hit_t.shape, upper_bound, device=hit_t.device),
                ]
            ),
            dim=0,
        )[0]
    if reduction == "sum":
        hit_t = hit_t.sum()
        if upper_bound is not None:
            max_hit_t = max_hit_t.sum()
    if upper_bound is not None:
        return hit_t, max_hit_t
    else:
        return hit_t


class VariableType(Enum):
    NODE_LIKE = "node-like"
    EDGE_LIKE = "edge-like"
    TRACK_LIKE = "track-like"
    OTHER = "other"


def get_variable_type(variable_name: str):
    if variable_name.startswith("hit_"):
        return VariableType.NODE_LIKE
    elif variable_name.startswith("edge_"):
        return VariableType.EDGE_LIKE
    elif variable_name.startswith("track_"):
        return VariableType.TRACK_LIKE
    else:
        return VariableType.OTHER


def map_tensor_handler(
    input_tensor: torch.Tensor,
    output_type: VariableType,
    input_type: VariableType,
    truth_map: torch.Tensor = None,
    edge_index: torch.Tensor = None,
    track_edges: torch.Tensor = None,
    num_nodes: int = None,
    num_edges: int = None,
    num_track_edges: int = None,
    aggr: str = None,
):
    """
    A general function to handle arbitrary maps of one tensor type to another
    Types are "node-like", "edge-like" and "track-like".
    - node-like: The input tensor is of the same size as the 
        number of nodes in the graph
    - edge-like: The input tensor is of the same size as the 
        number of edges in the graph, that is, the *constructed* graph
    - track-like: The input tensor is of the same size as the 
        number of true track edges in the event, that is, the *truth* graph

    To visualize:
                    (n)
                     ^
                    / \
      edge_to_node /   \ track_to_node
                  /     \
                 /       \
                /         \
               /           \
              /             \
node_to_edge /               \ node_to_track
            /                 \
           v     edge_to_track v
          (e) <-------------> (t)
            track_to_edge

    Args:
        input_tensor (torch.Tensor): The input tensor to be mapped
        output_type (str): The type of the output tensor. 
            One of "node-like", "edge-like" or "track-like"
        input_type (str, optional): The type of the input tensor. 
            One of "node-like", "edge-like" or "track-like". Defaults to None,
            and will try to infer the type from the input tensor, if num_nodes
            and/or num_edges are provided.
        truth_map (torch.Tensor, optional): The truth map tensor. 
            Defaults to None. Used for mappings to/from track-like tensors.
        num_nodes (int, optional): The number of nodes in the graph. 
            Defaults to None. Used for inferring the input type.
        num_edges (int, optional): The number of edges in the graph. 
            Defaults to None. Used for inferring the input type.
        num_track_edges (int, optional): The number of track edges in the graph 
            Defaults to None. Used for inferring the input type.
    """

    if num_track_edges is None and truth_map is not None:
        num_track_edges = truth_map.shape[0]
    if num_track_edges is None and track_edges is not None:
        num_track_edges = track_edges.shape[1]
    if num_edges is None and edge_index is not None:
        num_edges = edge_index.shape[1]
    if input_type == output_type:
        return input_tensor

    input_args = {
        "truth_map": truth_map,
        "edge_index": edge_index,
        "track_edges": track_edges,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_track_edges": num_track_edges,
        "aggr": aggr,
    }

    mapping_functions = {
        (VariableType.NODE_LIKE, VariableType.EDGE_LIKE): map_nodes_to_edges,
        (VariableType.EDGE_LIKE, VariableType.NODE_LIKE): map_edges_to_nodes,
        (VariableType.NODE_LIKE, VariableType.TRACK_LIKE): map_nodes_to_tracks,
        (VariableType.TRACK_LIKE, VariableType.NODE_LIKE): map_tracks_to_nodes,
        (VariableType.EDGE_LIKE, VariableType.TRACK_LIKE): map_edges_to_tracks,
        (VariableType.TRACK_LIKE, VariableType.EDGE_LIKE): map_tracks_to_edges,
    }
    if (input_type, output_type) not in mapping_functions:
        raise ValueError(f"Mapping from {input_type} to {output_type} not supported")

    return mapping_functions[(input_type, output_type)](input_tensor, **input_args)


def map_nodes_to_edges(
    nodelike_input: torch.Tensor, edge_index: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to an edge-like tensor. If the aggregation is None, this is simply done by sending node values to the edges, thus returning a tensor of shape (2, num_edges).
    If the aggregation is not None, the node values are aggregated to the edges, and the resulting tensor is of shape (num_edges,).
    """

    if aggr is None:
        return nodelike_input[edge_index]

    edgelike_tensor = nodelike_input[edge_index]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(edgelike_tensor, dim=0)


def map_edges_to_nodes(
    edgelike_input: torch.Tensor,
    edge_index: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map an edge-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending edge values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the edge values are aggregated to the nodes at the destination node (edge_index[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=edgelike_input.dtype, device=edgelike_input.device
        )
        nodelike_output[edge_index] = edgelike_input
        return nodelike_output

    return scatter(
        edgelike_input, edge_index[1], dim=0, dim_size=num_nodes, reduce=aggr
    )


def map_nodes_to_tracks(
    nodelike_input: torch.Tensor, track_edges: torch.Tensor, aggr: str = None, **kwargs
):
    """
    Map a node-like tensor to a track-like tensor. If the aggregation is None, this is simply done by sending node values to the tracks, thus returning a tensor of shape (2, num_track_edges).
    If the aggregation is not None, the node values are aggregated to the tracks, and the resulting tensor is of shape (num_track_edges,).
    """

    if aggr is None:
        return nodelike_input[track_edges]

    tracklike_tensor = nodelike_input[track_edges]
    torch_aggr = getattr(torch, aggr)
    return torch_aggr(tracklike_tensor, dim=0)


def map_tracks_to_nodes(
    tracklike_input: torch.Tensor,
    track_edges: torch.Tensor,
    aggr: str = None,
    num_nodes: int = None,
    **kwargs,
):
    """
    Map a track-like tensor to a node-like tensor. If the aggregation is None, this is simply done by sending track values to the nodes, thus returning a tensor of shape (num_nodes,).
    If the aggregation is not None, the track values are aggregated to the nodes at the destination node (track_edges[1]), and the resulting tensor is of shape (num_nodes,).
    """

    if num_nodes is None:
        num_nodes = int(track_edges.max().item() + 1)

    if aggr is None:
        nodelike_output = torch.zeros(
            num_nodes, dtype=tracklike_input.dtype, device=tracklike_input.device
        )
        nodelike_output[track_edges] = tracklike_input
        return nodelike_output

    return scatter(
        tracklike_input.repeat(2),
        torch.cat([track_edges[0], track_edges[1]]),
        dim=0,
        dim_size=num_nodes,
        reduce=aggr,
    )


def map_tracks_to_edges(
    tracklike_input: torch.Tensor,
    truth_map: torch.Tensor,
    num_edges: int = None,
    **kwargs,
):
    """
    Map an track-like tensor to a edge-like tensor. This is done by sending the track value through the truth map, where the truth map is >= 0. Note that where truth_map == -1,
    the true edge has not been constructed in the edge_index. In that case, the value is set to NaN.
    """

    if num_edges is None:
        num_edges = int(truth_map.max().item() + 1)
    edgelike_output = torch.zeros(
        num_edges, dtype=tracklike_input.dtype, device=tracklike_input.device
    )
    if num_edges == 0:
        return edgelike_output
    edgelike_output[truth_map[truth_map >= 0]] = tracklike_input[truth_map >= 0]
    edgelike_output[truth_map[truth_map == -1]] = float("nan")
    return edgelike_output


def map_edges_to_tracks(
    edgelike_input: torch.Tensor, truth_map: torch.Tensor, **kwargs
):
    """
    TODO: Implement this. I don't think it is a meaningful operation, but it is needed for completeness.
    """
    raise NotImplementedError(
        "This is not a meaningful operation, but it is needed for completeness"
    )
