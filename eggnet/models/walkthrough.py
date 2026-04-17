import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import process_time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from eggnet.utils import nearest_neighboring

from .utils import cc_and_walk_utils


def _debug_print(hparams, message):
    """Print a debug message when track-building debug is enabled."""
    if hparams.get("track_build_debug", False):
        print(f"[Walkthrough] {message}", flush=True)


def _save_graph(graph, output_dir, hparams):
    """Persist a reconstructed event graph atomically."""
    if not hasattr(graph, "config") or graph.config is None:
        graph.config = []
    graph.config.append(hparams)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"event{graph.event_id[0]}.pyg")
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        torch.save(graph.cpu(), tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return graph


class Walkthrough(nn.Module):
    """Build tracks with the networkx walkthrough pipeline."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gpu_available = torch.cuda.is_available()
        self.cc_only = self.hparams.get("cc_only", False)
        self.knn: nearest_neighboring.abstract_knn = getattr(
            nearest_neighboring, hparams.get("knn_algorithm", "cu_knn")
        )()

        self.log = logging.getLogger("TrackBuilding")
        log_level = hparams.get("log_level", "WARNING").upper()
        self.log.setLevel(logging._nameToLevel.get(log_level, logging.WARNING))

    def _get_initial_edge_index(self, graph):
        """Recover candidate edges from the learned embeddings."""
        if graph.src_embedding.device.type == "cuda":
            knn_graph = graph
        else:
            if not self.gpu_available:
                raise RuntimeError(
                    "Walkthrough requires CUDA to recover initial edges from embeddings."
                )
            knn_graph = graph.to("cuda", non_blocking=True)

        edge_index = self.knn.get_graph(
            knn_graph,
            1,
            use_double_metric_learning=True,
        )
        return edge_index.to(graph.hit_id.device)

    def _build_scored_graph(self, graph, edge_index, score_name):
        """Build a PyG graph with candidate edges and walkthrough scores."""
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

        output_device = graph.hit_id.device
        scored_graph = Data(
            edge_index=edge_index.to(output_device),
            hit_id=graph.hit_id,
            num_nodes=int(graph.num_nodes),
            hit_r=graph.hit_r,
            hit_z=graph.hit_z,
        )
        scored_graph[score_name] = edge_scores.float().to(output_device)
        return scored_graph

    def _build_tracks_one_evt_impl(self, graph, output_dir):
        """Build tracks for a single event graph."""
        event_id = graph.event_id[0] if hasattr(graph, "event_id") else "unknown"
        start_time = process_time()

        threshold = self.hparams["score_cut_cc"]
        inverted_threshold = 1 - threshold
        inverted_score_name = "track_edge_scores_inverted"
        edge_index = self._get_initial_edge_index(graph)
        scored_graph = self._build_scored_graph(graph, edge_index, inverted_score_name)
        scored_graph = cc_and_walk_utils.remove_cycles(scored_graph)

        walk_graph = cc_and_walk_utils.filter_graph(
            scored_graph, inverted_score_name, inverted_threshold
        )
        walk_graph = cc_and_walk_utils.topological_sort_graph(walk_graph)

        all_trks = {}
        all_trks["cc"] = cc_and_walk_utils.get_simple_path(walk_graph)

        if not self.cc_only:
            all_trks["walk"] = cc_and_walk_utils.walk_through(
                walk_graph,
                inverted_score_name,
                1 - self.hparams["score_cut_walk"]["min"],
                1 - self.hparams["score_cut_walk"]["add"],
            )

        if self.hparams.get("save_graph", True):
            cc_and_walk_utils.add_track_labels(graph, all_trks)

        tracks = cc_and_walk_utils.join_track_lists(all_trks)
        graph.reco_tracks = tracks
        graph.time_taken = process_time() - start_time

        if self.hparams.get("save_walkthrough_graphs", True):
            _save_graph(graph, output_dir, self.hparams)

        _debug_print(
            self.hparams,
            f"event={event_id} tracks={len(tracks)} time={graph.time_taken:.3f}s",
        )
        return event_id

    def _build_tracks_one_evt_from_path(self, event_path, output_dir):
        """Load one event from disk and reconstruct its tracks."""
        try:
            graph = torch.load(event_path, map_location=torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load event file: {event_path}") from exc
        return self._build_tracks_one_evt_impl(graph, output_dir)

    def _build_tracks_one_evt(self, graph, output_dir):
        """Reconstruct tracks for an in-memory event graph."""
        return self._build_tracks_one_evt_impl(graph, output_dir)

    def build_tracks(self, dataset, data_name):
        """Run walkthrough track building over a dataset split."""
        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)

        max_workers = self.hparams["max_workers"] if "max_workers" in self.hparams else 1
        assert isinstance(max_workers, int) and max_workers > 0

        dataset_size = len(dataset) if hasattr(dataset, "__len__") else "unknown"
        _debug_print(
            self.hparams,
            f"split={data_name} events={dataset_size} workers={max_workers}",
        )
        if max_workers != 1:
            print(
                "INFO: Reconstruction progress counts completed events; the first "
                "update may be delayed while the initial worker batch is still running.",
                flush=True,
            )

        if max_workers != 1 and hasattr(dataset, "input_paths"):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._build_tracks_one_evt_from_path,
                        event_path,
                        output_dir,
                    ): event_path
                    for event_path in dataset.input_paths
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Reconstructing tracks for {data_name} data",
                ):
                    future.result()
        else:
            if max_workers != 1 and not hasattr(dataset, "input_paths"):
                self.log.warning(
                    "Dataset does not expose input_paths; using single-worker track building."
                )
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                self._build_tracks_one_evt(event, output_dir=output_dir)

    def save_graph(self, graph, output_dir):
        """Save a reconstructed graph using the configured metadata."""
        return _save_graph(graph, output_dir, self.hparams)

    def forward(self, dataset, data_name):
        """Build tracks for the provided dataset split."""
        self.build_tracks(dataset, data_name)
