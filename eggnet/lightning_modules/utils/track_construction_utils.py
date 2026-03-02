import torch_geometric
import torch
import tqdm
from torch_geometric.data import Data as PygData
import warnings
from typing import List, Dict
import os, sys
import numpy as np
import pandas as pd
import csv

def test_data(self, stage):
    """
    Test the data to ensure it is of the right format and loaded correctly.
    """
    required_features = [
        "hit_x",
        "edge_index",
        "track_edges",
        "track_to_edge_map",
        "edge_y",
    ]
    optional_features = [
        "track_particle_id",
        "track_particle_nhits",
        "track_particle_primary",
        "track_particle_pdgId",
        "hit_region",
        "hit_id",
        "track_particle_pt",
        "track_particle_radius",
        "track_particle_eta",
    ]

    # Test only non empty data set
    datasets = [
        getattr(self, data_name)
        for data_name in ["trainset", "valset", "testset"]
        if hasattr(self, data_name)
    ]

    run_data_tests(datasets, required_features, optional_features)
def run_data_tests(datasets: List, required_features, optional_features):
    for dataset in datasets:
        if dataset is None or len(dataset) == 0:
            warnings.warn(
                "Found an empty dataset. Please check if this is not intended."
            )
            continue
        sample_event = dataset[0]
        assert sample_event is not None, "No data loaded"
        # Check that the event is the latest PyG format
        try:
            _ = len(sample_event)
        except RuntimeError:
            warnings.warn(
                "Data is not in the latest PyG format, so will be converted on-the-fly."
                " Consider re-saving the data in latest PyG Data type."
            )
            if dataset is not None:
                for i, event in enumerate(dataset):
                    dataset[i] = _convert_to_latest_pyg_format(event)

        for feature in required_features:
            assert feature in sample_event or f"x_{feature}" in sample_event, (
                f"Feature [{feature}] not found in data, this is REQUIRED. Features"
                f" found: {_get_pyg_data_keys(sample_event)}"
            )

        missing_optional_features = [
            feature for feature in optional_features if feature not in sample_event
        ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

        # Check that the number of nodes is compatible with the edge indexing
        if "edge_index" in _get_pyg_data_keys(sample_event) and "x" in _get_pyg_data_keys(
            sample_event
        ):
            assert (
                sample_event.x.shape[0] >= sample_event.edge_index.max().item() + 1
            ), (
                "Number of nodes is not compatible with the edge indexing. Possibly an"
                " earlier stage has removed nodes, but not updated the edge indexing."
            )
def _convert_to_latest_pyg_format(event):
    """
    Convert the data to the latest PyG format.
    """
    return PygData.from_dict(event.__dict__)

def _get_pyg_data_keys(event: PygData):
    """
    Get the keys of the pyG data object.
    """
    if torch_geometric.__version__ < "2.4.0":
        return event.keys
    else:
        return event.keys()
            
def distance_true_edges(self):
    """
    """
    dataset = self.testset if self.testset is not None else self.valset or self.trainset
    total_sum = None
    total_count = 0
    with torch.no_grad():
        for event in dataset:
            edges = event.track_edges
            if edges.numel() == 0:
                continue
            src = event.src_embedding[edges[0]]
            tgt = event.tgt_embedding[edges[1]]
            d = torch.sqrt(torch.sum((tgt - src) ** 2, dim=-1) + 1e-12)
            total_sum = d.sum() if total_sum is None else total_sum + d.sum()
            total_count += d.numel()
    if total_count == 0:
        return 0.0
    return (total_sum / total_count).item()

def apply_target_conditions(self, event, target_tracks):
    """
    Apply the target conditions to the event. This is used for the evaluation stage.
    Target_tracks is a list of dictionaries, each of which contains the conditions to be applied to the event.
    """
    passing_tracks = torch.ones(event.track_to_edge_map.shape[0], dtype=torch.bool)

    for key, values in target_tracks.items():
        if isinstance(values, list):
            # passing_tracks = passing_tracks & (values[0] <= event[key]).bool() & (event[key] <= values[1]).bool()
            passing_tracks = (
                passing_tracks
                * (values[0] <= event[key].float())
                * (event[key].float() <= values[1])
            )
        else:
            passing_tracks = passing_tracks * (event[key] == values)

    event.target_mask = passing_tracks

def save_tracks(self, graph, tracks, output_dir):
    tracks_dir = os.path.join(
        self.hparams["stage_dir"], f"{os.path.basename(output_dir)}_tracks"
    )
    os.makedirs(tracks_dir, exist_ok=True)

    if self.hparams.get("save_tracks_as_csv", False):
        _delimiter = ","
        filename = self.event_prefix
        if self.hparams.get("athena_csv_format", False):
            # In Athena, the GNNTrackReader needs CSV track file with the following name : "<prefix>_RUNNUMBER_EVTNUMBER.csv".
            # By default, the <prefix> is set to "track" in Athena, so if you want to produce track files with the equivalent filename, set 'event_prefix' in yaml config file to : "track_RUNNUMBER".
            # If you want to use a specific prefix, you have to define it in both ACORN and Athena as follows:
            # in ACORN yaml config file with 'event_prefix' : "<any_prefix_you_want>_RUNNUMBER" (The run number is mandatory, otherwise it will not work in Athena.)
            # in Athena with 'csvPrefix' : "<any_prefix_you_want>" (without the RUNNUMBER, it is already handled by Athena.)

            filename += f"{graph.event_id[0].lstrip('0')}.csv"
        else:
            filename += f"event{graph.event_id[0]}.csv"
    else:
        _delimiter = " "
        filename = f"{self.event_prefix}event{graph.event_id[0]}.txt"

    output_file = os.path.join(tracks_dir, filename)
    with open(output_file, "w", newline="") as f:
        csv.writer(f, delimiter=_delimiter).writerows(tracks)

def save_graph(self, graph, output_dir):

    torch.save(
        graph,
        os.path.join(
            output_dir, f"{self.event_prefix}event{graph.event_id[0]}.pyg"
        ),
    )
    return graph

# ------------- MATCHING UTILS ----------------


def load_reconstruction_df(graph):
    """Load the reconstructed tracks from a file."""
    if hasattr(graph, "hit_id"):
        hit_id = graph.hit_id
    else:
        hit_id = torch.arange(graph.num_nodes)

    reco_df = pd.DataFrame({"hit_id": hit_id, "track_id": graph.hit_track_labels})

    node_id = graph.track_edges.reshape(-1)
    pids = graph.track_particle_id.repeat(2)
    pid_df = pd.DataFrame({"hit_id": node_id, "particle_id": pids})
    pid_df.drop_duplicates(subset=["hit_id", "particle_id"], inplace=True)

    # Merge the two dataframes
    reco_df = reco_df.merge(pid_df, on="hit_id", how="outer")
    reco_df.fillna({"track_id": -1, "particle_id": 0}, inplace=True)  # Fill NaN values
    return reco_df


def load_particles_df(graph, sel_conf: dict):
    """Load the particles from a file."""
    # Get the particle dataframe

    # By default have only particle pt
    cols = {
        "particle_id": graph.track_particle_id,
        "pt": graph.track_particle_pt,
    }
    if "track_particle_eta" in graph:
        cols["eta"] = graph.track_particle_eta

    # Add more variable if needed for th fiducial selection
    for var in sel_conf:
        if var not in cols:
            if var == "n_true_hits":
                # Specific case: not embedded in graphs but added in the dataframe later on
                # So we ignore it at this stage
                continue
            cols[var] = graph[var]

    # particles_df = pd.DataFrame({"particle_id": graph.particle_id,
    #                              "pt": graph.pt, "eta_particle": graph.eta_particle,
    #                              "pdgId": graph.pdgId, "radius": graph.radius,
    #                              "primary": graph.primary})

    particles_df = pd.DataFrame(cols)

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=["particle_id"])

    return particles_df



# ------------- PLOTTING UTILS ----------------

def rearrange_by_distance(event, edge_index):
    if "hit_R" not in _get_pyg_data_keys(event):
        warnings.warn(
            "hit_R not found in the event, calculating it from hit_r and hit_z"
        )
        assert "hit_r" in _get_pyg_data_keys(event) and "hit_z" in _get_pyg_data_keys(
            event
        ), "event must contain R or contain r and z"
        event.hit_R = event.hit_r**2 + event.hit_z**2

    # flip edges that are pointing inward
    edge_mask = (event.hit_R[edge_index[0]] > event.hit_R[edge_index[1]]) | (
        (event.hit_R[edge_index[0]] == event.hit_R[edge_index[1]])
        & (edge_index[0] > edge_index[1])
    )
    edge_index[:, edge_mask] = edge_index[:, edge_mask].flip(0)

    return edge_index
