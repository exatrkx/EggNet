import os
from pathlib import Path
import warnings
import random
import math
import logging
from enum import Enum

import torch

from eggnet.utils.mapping import (
    get_variable_type,
    VariableType,
    get_condition_lambda,
    map_tensor_handler,
)


def load_datafiles_in_dir(input_dir, data_name=None, data_num=None):
    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
    if len(data_files) == 0:
        warnings.warn(f"No data files found in {input_dir}")
    if data_num is not None:
        assert len(data_files) == data_num, (
            f"Number of data files found ({len(data_files)}) is less than the number"
            f" requested ({data_num})"
        )

    return data_files


def infer_num_nodes(graph):
    """
    Ensure the num_nodes is set properly
    """

    if "num_nodes" not in graph or graph.num_nodes is None:
        assert "hit_id" in graph, "No node features found in graph"
        graph.num_nodes = graph.hit_id.shape[0]


class NodeCountStatus(Enum):
    UNINIT = 0
    UNDER = 1
    OVER = 2
    GOOD = 3


def handle_hard_node_cuts(
    event, hard_cuts_config, min_nodes=0, max_nodes=None, edges=False, tracks=False
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    node_like_feature = [
        event[feature] for feature in event.keys() if get_variable_type(feature) == VariableType.NODE_LIKE
    ][0]
    node_mask = torch.ones_like(node_like_feature, dtype=torch.bool)

    # TODO: Refactor this to simply trim the true tracks and check which nodes are in the true tracks
    for condition_key, condition_val in hard_cuts_config.items():
        assert (
            condition_key in event.keys()
        ), f"Condition key {condition_key} not found in event keys"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = map_tensor_handler(
            value_mask,
            output_type=VariableType.NODE_LIKE,
            input_type=get_variable_type(condition_key),
            track_edges=event.track_edges,
            num_nodes=node_like_feature.shape[0],
            num_track_edges=event.track_edges.shape[1],
        )
        node_mask = node_mask * node_val_mask

    if node_mask.sum() < min_nodes:
        return NodeCountStatus.UNDER
    if max_nodes and node_mask.sum() > max_nodes:
        return NodeCountStatus.OVER

    logging.info(
        f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} /"
        f" {node_mask.shape[0]}"
    )

    # TODO: Refactor the below to use the remap_from_mask function
    num_nodes = event.num_nodes
    for feature in event.keys():
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == VariableType.NODE_LIKE
        ):
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in event.keys():
        if (
            isinstance(event[feature], torch.Tensor)
            and get_variable_type(feature) == VariableType.TRACK_LIKE
        ):
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]
    event.num_nodes = node_mask.sum()

    if edges:
        edge_mask = node_mask[event.edge_index]
        edge_mask = edge_mask[0] & edge_mask[1]
        for feature in event.keys():
            if feature == "edge_index":
                event.edge_index = event.edge_index.T[edge_mask].T
                event.edge_index = node_lookup[event.edge_index]
            elif (
                isinstance(event[feature], torch.Tensor)
                and get_variable_type(feature) == VariableType.EDGE_LIKE
            ):
                event[feature] = event[feature][edge_mask]

    return NodeCountStatus.GOOD


def apply_hard_cuts(event, hparams, stage):
    """
    Apply hard cuts to the event. This is implemented by
    1. Finding which true edges are from tracks that pass the hard cut.
    2. Pruning the input graph to only include nodes that are connected to these edges.
    """

    if hparams.get("hard_cuts") or (
        hparams.get("phi_segmented") and stage == "fit"  # self.data_name == "trainset"
    ):
        hard_cut_finished = NodeCountStatus.UNINIT

        graph_fraction = hparams.get("graph_fraction", 0.1)

        # If search range changes by less than 1 degree (~ 0.02 rad), give up
        graph_adjustment_tol = hparams.get("graph_adjustment_tol", 0.02)

        phi_mid = math.pi * 2 * random.random()

        phi_width_low = 0
        phi_width_high = math.pi * 2
        phi_width = math.pi * 2 * graph_fraction

        min_nodes = hparams.get("min_nodes", 20)
        max_nodes = hparams.get("max_nodes")

        while (
            hard_cut_finished != NodeCountStatus.GOOD
            and phi_width_high - phi_width_low >= graph_adjustment_tol
        ):
            hard_cuts = hparams.get(
                "hard_cuts", {}
            )  # TODO: Is this needed in the loop?

            phi_low = phi_mid - phi_width / 2
            phi_high = phi_mid + phi_width / 2

            phi_low = (phi_low + math.pi) % (math.pi * 2) - math.pi
            phi_high = (phi_high + math.pi) % (math.pi * 2) - math.pi
            # Another way to adjust phi_high (might be better)
            # if phi_high > math.pi:
            #     phi_high -= math.pi * 2

            if phi_low < phi_high:
                phi_range = [phi_low, phi_high]
            else:
                phi_range = ["not_within", [phi_high, phi_low]]

            if hparams.get("phi_segmented") and stage == "fit":
                hard_cuts["hit_phi"] = phi_range

            hard_cut_finished = handle_hard_node_cuts(
                event, hard_cuts, min_nodes, max_nodes
            )

            if hard_cut_finished != NodeCountStatus.GOOD:
                if (
                    hparams.get("graph_fraction_adjustment_method", "binary_search")
                    == "binary_search"
                ):
                    if hard_cut_finished == NodeCountStatus.UNDER:  # Grow
                        phi_width_low = phi_width
                    elif hard_cut_finished == NodeCountStatus.OVER:  # Shrink
                        phi_width_high = phi_width
                        if hparams.get("max_possible_width"):
                            min_nodes = max_nodes * 0.95
                    phi_width = (phi_width_low + phi_width_high) / 2
                elif hparams.get("graph_fraction_adjustment_method") == "resample":
                    phi_mid = math.pi * 2 * random.random()

        uni, inv_idx, count = torch.unique(
            event.hit_particle_id, return_counts=True, return_inverse=True
        )
        event.hit_particle_nhits = count[inv_idx]
