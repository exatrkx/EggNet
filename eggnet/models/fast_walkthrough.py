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

import logging
import os
import __main__
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import current_process
from multiprocessing import get_context
from time import perf_counter

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import cc_and_walk_utils, fast_walkthrough_utils
from .utils.walkthrough_diagnostics import (
    DEFAULT_BRANCH_AMBIGUITY_GAP,
    build_edge_diagnostics_summary,
    build_track_particle_diagnostics,
    build_walk_branch_diagnostics,
)
from eggnet.utils import nearest_neighboring
from eggnet.utils.timing import add_profile_time, profile_section, set_profile_metadata


def _debug_print(hparams, message):
    """Write a debug message to the walkthrough debug log when enabled."""
    if not hparams.get("track_build_debug", False):
        return

    log_path = hparams.get("track_build_debug_log")
    if not log_path:
        log_dir = (
            hparams.get("stage_dir")
            or hparams.get("walkthrough_output_dir")
            or hparams.get("output_dir")
            or os.getcwd()
        )
        log_path = os.path.join(log_dir, "fast_walkthrough_debug.log")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} [FastWalkthrough pid={os.getpid()}] {message}\n")


def _graph_size_summary(graph):
    """Return a compact node/edge summary for debug logging."""
    num_nodes = int(getattr(graph, "num_nodes", 0))
    edge_index = getattr(graph, "edge_index", None)
    num_edges = (
        int(edge_index.shape[1])
        if edge_index is not None and hasattr(edge_index, "shape") and len(edge_index.shape) >= 2
        else 0
    )
    return f"nodes={num_nodes} edges={num_edges}"


def _format_profile_summary(profile):
    if not profile:
        return ""
    items = sorted(profile.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}={value:.3f}s" for name, value in items)


def _record_graph_profile_metadata(
    graph,
    prefix,
    target=None,
    include_structure=False,
):
    if target is None:
        target = graph

    num_nodes = int(graph.num_nodes)
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        num_edges = 0
    elif hasattr(edge_index, "shape") and len(edge_index.shape) >= 2:
        num_edges = int(edge_index.shape[1])
    else:
        num_edges = 0
    set_profile_metadata(target, f"{prefix}.num_nodes", num_nodes)
    set_profile_metadata(target, f"{prefix}.num_edges", num_edges)

    if not include_structure or num_edges == 0 or edge_index is None:
        return

    edge_index = edge_index.cpu()
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    set_profile_metadata(target, f"{prefix}.max_out_degree", int(out_degree.max().item()))
    set_profile_metadata(
        target,
        f"{prefix}.branching_nodes",
        int((out_degree > 1).sum().item()),
    )


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


def _walkthrough_output_path(output_dir, event_name):
    """Return the walkthrough output path for an event filename or id."""
    event_name = str(event_name)
    if not event_name.endswith(".pyg"):
        event_name = f"event{event_name}.pyg"
    return os.path.join(output_dir, os.path.basename(event_name))


_FAST_WALKTHROUGH_WORKER = None


def _get_worker_index():
    """Best-effort stable worker index for process-pool children."""
    process = current_process()
    identity = getattr(process, "_identity", None)
    if identity:
        return max(int(identity[0]) - 1, 0)

    name = getattr(process, "name", "")
    if "-" in name:
        suffix = name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return max(int(suffix) - 1, 0)

    return os.getpid()


def _assign_worker_cuda_device(hparams):
    """Pin each walkthrough worker to one visible CUDA device."""
    if not torch.cuda.is_available():
        return None

    visible_device_count = torch.cuda.device_count()
    if visible_device_count <= 0:
        return None

    configured_devices = hparams.get("walkthrough_cuda_devices")
    if configured_devices is None:
        candidate_devices = list(range(visible_device_count))
    else:
        candidate_devices = [int(device) for device in configured_devices]
        if not candidate_devices:
            return None

    worker_index = _get_worker_index()
    device_index = candidate_devices[worker_index % len(candidate_devices)]
    torch.cuda.set_device(device_index)
    return device_index


def _configure_track_build_worker():
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    try:
        from numba import set_num_threads as numba_set_num_threads

        numba_set_num_threads(1)
    except Exception:
        pass


def _init_fast_walkthrough_worker(hparams):
    global _FAST_WALKTHROUGH_WORKER
    _configure_track_build_worker()
    assigned_device = _assign_worker_cuda_device(hparams)
    _debug_print(
        hparams,
        "worker_init "
        f"pid={os.getpid()} worker_index={_get_worker_index()} "
        f"cuda_device={assigned_device}",
    )
    _FAST_WALKTHROUGH_WORKER = FastWalkthrough(hparams)
    _FAST_WALKTHROUGH_WORKER.assigned_cuda_device = assigned_device
    _FAST_WALKTHROUGH_WORKER.eval()


def _run_fast_walkthrough_worker(event_path, output_dir):
    if _FAST_WALKTHROUGH_WORKER is None:
        raise RuntimeError("FastWalkthrough worker was not initialized.")
    return _FAST_WALKTHROUGH_WORKER._build_tracks_one_evt_from_path(
        event_path,
        output_dir,
    )


def _can_use_process_pool():
    main_file = getattr(__main__, "__file__", None)
    return isinstance(main_file, str) and len(main_file) > 0


def _shutdown_executor_now(executor, futures=None):
    """Cancel pending work and tear down an executor without blocking on workers."""
    if futures is not None:
        for future in futures:
            future.cancel()

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


def _terminate_process_pool(executor, futures=None):
    """Forcefully stop worker processes so Ctrl-C does not hang during exit."""
    _shutdown_executor_now(executor, futures=futures)

    process_map = getattr(executor, "_processes", None) or {}
    processes = list(process_map.values())
    for process in processes:
        if process is None:
            continue
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            pass

    for process in processes:
        if process is None:
            continue
        try:
            process.join(timeout=0.2)
        except Exception:
            pass

    for process in processes:
        if process is None:
            continue
        try:
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        except Exception:
            pass

    for process in processes:
        if process is None:
            continue
        try:
            process.join(timeout=0.2)
        except Exception:
            pass


class FastWalkthrough(nn.Module):
    def __init__(self, hparams):
        """Initialize the fast walkthrough track builder."""
        super().__init__()
        self.hparams = hparams
        self.cc_only = self.hparams.get("cc_only", False)

        self.log = logging.getLogger("TrackBuilding")
        log_level = hparams.get("log_level", "WARNING").upper()
        self.log.setLevel(logging._nameToLevel.get(log_level, logging.WARNING))
        self.enable_profiling = self.hparams.get("enable_profiling", True)
        self.enable_structural_profile_metadata = self.hparams.get(
            "enable_structural_profile_metadata",
            False,
        )
        self.gpu_available = torch.cuda.is_available()

    def _get_graph_for_initial_edge_index(self, graph):
        """Use CUDA embeddings for the initial neighbor search when available."""
        if graph.src_embedding.device.type == "cuda":
            return graph
        if not self.gpu_available:
            return graph
        return graph.to("cuda", non_blocking=True)

    def _get_initial_edge_index(self, graph):
        """Recover initial candidate edges with an FRNN-style radius query."""
        initial_edge_radius = float(
            self.hparams.get("initial_edge_radius", self.hparams["score_cut_cc"])
        )
        initial_edge_max_neighbors = int(
            self.hparams.get("initial_edge_max_neighbors", 64)
        )
        edge_backend = self.hparams.get("initial_edge_backend", "FRNN")

        edge_index = nearest_neighboring.build_edges(
            query=graph.tgt_embedding,
            database=graph.src_embedding,
            r_max=initial_edge_radius,
            k_max=initial_edge_max_neighbors,
            backend=edge_backend,
        )
        edge_index = edge_index.flip(0).contiguous()
        if edge_index.numel() > 0:
            max_index = int(edge_index.max().item())
            if max_index >= int(graph.num_nodes):
                raise RuntimeError(
                    "Recovered candidate edges reference nodes outside the current "
                    f"event: max edge index {max_index}, num_nodes={int(graph.num_nodes)}."
                )
        return edge_index.to(graph.hit_id.device)

    def _build_tracks_one_evt_impl(self, graph, output_dir):
        """Build tracks for a single event graph."""
        event_id = graph.event_id[0] if hasattr(graph, "event_id") else "unknown"
        start_time = perf_counter()
        debug_enabled = bool(self.hparams.get("track_build_debug", False))
        save_walkthrough_diagnostics = bool(
            self.hparams.get("save_walkthrough_diagnostics", False)
        )
        _debug_print(
            self.hparams,
            "event_start "
            f"id={event_id} pid={os.getpid()} "
            f"device={getattr(getattr(graph, 'hit_id', None), 'device', 'unknown')} "
            f"{_graph_size_summary(graph)}",
        )
        if self.enable_profiling:
            _record_graph_profile_metadata(
                graph,
                "fast_walkthrough.input",
                include_structure=self.enable_structural_profile_metadata,
            )

        threshold = self.hparams["score_cut_cc"]
        inverted_threshold = 1 - threshold
        inverted_score_name = "track_edge_scores_inverted"
        initial_edge_radius = float(
            self.hparams.get("initial_edge_radius", self.hparams["score_cut_cc"])
        )
        initial_edge_max_neighbors = int(
            self.hparams.get("initial_edge_max_neighbors", 64)
        )
        initial_edge_backend = self.hparams.get("initial_edge_backend", "FRNN")
        graph_move_start = perf_counter()
        graph_for_initial_edges = self._get_graph_for_initial_edge_index(graph)
        if debug_enabled:
            _debug_print(
                self.hparams,
                "stage=prepare_initial_edge_index "
                f"event={event_id} time={perf_counter() - graph_move_start:.3f}s "
                f"device={graph_for_initial_edges.src_embedding.device} "
                f"{_graph_size_summary(graph_for_initial_edges)}",
            )
        initial_edge_start = perf_counter()
        with profile_section(
            graph,
            "fast_walkthrough.initial_edge_index",
            enabled=self.enable_profiling,
        ):
            if self.enable_profiling:
                set_profile_metadata(
                    graph,
                    "fast_walkthrough.initial_edge_index.radius",
                    initial_edge_radius,
                )
                set_profile_metadata(
                    graph,
                    "fast_walkthrough.initial_edge_index.max_neighbors",
                    initial_edge_max_neighbors,
                )
                set_profile_metadata(
                    graph,
                    "fast_walkthrough.initial_edge_index.backend",
                    initial_edge_backend,
                )
            edge_index = self._get_initial_edge_index(graph_for_initial_edges)
        if debug_enabled:
            num_initial_edges = int(edge_index.shape[1]) if edge_index.ndim >= 2 else 0
            _debug_print(
                self.hparams,
                "stage=initial_edge_index "
                f"event={event_id} time={perf_counter() - initial_edge_start:.3f}s "
                f"edges={num_initial_edges} backend={initial_edge_backend} "
                f"radius={initial_edge_radius} k={initial_edge_max_neighbors}",
            )
        if self.enable_profiling:
            set_profile_metadata(
                graph,
                "fast_walkthrough.initial_edge_index.num_edges",
                int(edge_index.shape[1]),
            )
        filter_start = perf_counter()
        with profile_section(
            graph,
            "fast_walkthrough.filter_graph",
            enabled=self.enable_profiling,
        ):
            filtered_graph = fast_walkthrough_utils.filter_graph(
                graph_for_initial_edges,
                edge_index,
                inverted_score_name,
                inverted_threshold,
            )
        if debug_enabled:
            _debug_print(
                self.hparams,
                "stage=filter_graph "
                f"event={event_id} time={perf_counter() - filter_start:.3f}s "
                f"{_graph_size_summary(filtered_graph)} threshold={inverted_threshold:.4f}",
            )
        edge_index = edge_index.to(graph.hit_id.device)
        if filtered_graph.hit_id.device.type != "cpu":
            filtered_graph = filtered_graph.cpu()
        if self.enable_profiling:
            _record_graph_profile_metadata(
                graph=filtered_graph,
                prefix="fast_walkthrough.filtered",
                target=graph,
                include_structure=self.enable_structural_profile_metadata,
            )
        remove_cycles_start = perf_counter()
        with profile_section(
            graph,
            "fast_walkthrough.remove_cycles",
            enabled=self.enable_profiling,
        ):
            filtered_graph = cc_and_walk_utils.remove_cycles(filtered_graph)
        if debug_enabled:
            _debug_print(
                self.hparams,
                "stage=remove_cycles "
                f"event={event_id} time={perf_counter() - remove_cycles_start:.3f}s "
                f"{_graph_size_summary(filtered_graph)}",
            )
        if self.enable_profiling:
            _record_graph_profile_metadata(
                graph=filtered_graph,
                prefix="fast_walkthrough.acyclic",
                target=graph,
                include_structure=self.enable_structural_profile_metadata,
            )

        all_trks = {}
        walk_input_graph = None
        walk_pruned_graph = None
        simple_path_start = perf_counter()
        with profile_section(
            graph,
            "fast_walkthrough.simple_path",
            enabled=self.enable_profiling,
        ):
            all_trks["cc"], walk_input_graph = fast_walkthrough_utils.get_simple_path(
                filtered_graph,
                profile_target=graph,
                profiling_enabled=self.enable_profiling,
                include_structure_metadata=self.enable_structural_profile_metadata,
            )
        if debug_enabled:
            _debug_print(
                self.hparams,
                "stage=simple_path "
                f"event={event_id} time={perf_counter() - simple_path_start:.3f}s "
                f"cc_tracks={len(all_trks['cc'])} {_graph_size_summary(walk_input_graph)}",
            )
        if self.enable_profiling:
            _record_graph_profile_metadata(
                graph=walk_input_graph,
                prefix="fast_walkthrough.walk_input",
                target=graph,
                include_structure=self.enable_structural_profile_metadata,
            )

        if not self.hparams.get("cc_only", False):
            walk_min_threshold = 1 - self.hparams["score_cut_walk"]["min"]
            walk_add_threshold = 1 - self.hparams["score_cut_walk"]["add"]
            if save_walkthrough_diagnostics:
                walk_pruned_graph = fast_walkthrough_utils.max_add_cuts(
                    walk_input_graph.clone(),
                    inverted_score_name,
                    walk_min_threshold,
                    walk_add_threshold,
                    lookback=self.hparams.get("lookback", False),
                )
                if debug_enabled:
                    _debug_print(
                        self.hparams,
                        "stage=max_add_cuts_diagnostics "
                        f"event={event_id} {_graph_size_summary(walk_pruned_graph)}",
                    )
            walk_start = perf_counter()
            with profile_section(
                graph,
                "fast_walkthrough.walk",
                enabled=self.enable_profiling,
            ):
                all_trks["walk"] = fast_walkthrough_utils.walk_through(
                    walk_input_graph,
                    inverted_score_name,
                    walk_min_threshold,
                    walk_add_threshold,
                    self.hparams.get("reuse_hits", False),
                    self.hparams.get("walk_mode", 0),
                    self.hparams.get("lookback", False),
                    profile_target=graph,
                    profiling_enabled=self.enable_profiling,
                    include_structure_metadata=self.enable_structural_profile_metadata,
                )
            if debug_enabled:
                _debug_print(
                    self.hparams,
                    "stage=walk "
                    f"event={event_id} time={perf_counter() - walk_start:.3f}s "
                    f"walk_tracks={len(all_trks['walk'])} "
                    f"min={walk_min_threshold:.4f} add={walk_add_threshold:.4f}",
                )
            if self.enable_profiling:
                set_profile_metadata(
                    graph,
                    "fast_walkthrough.walk.num_tracks",
                    int(len(all_trks["walk"])),
                )

        if self.hparams.get("save_graph", True) or save_walkthrough_diagnostics:
            add_labels_start = perf_counter()
            with profile_section(
                graph,
                "fast_walkthrough.add_track_labels",
                enabled=self.enable_profiling,
            ):
                cc_and_walk_utils.add_track_labels(graph, all_trks)
            if debug_enabled:
                _debug_print(
                    self.hparams,
                    "stage=add_track_labels "
                    f"event={event_id} time={perf_counter() - add_labels_start:.3f}s",
                )

        join_tracks_start = perf_counter()
        with profile_section(
            graph,
            "fast_walkthrough.join_track_lists",
            enabled=self.enable_profiling,
        ):
            tracks = cc_and_walk_utils.join_track_lists(all_trks)
        if debug_enabled:
            _debug_print(
                self.hparams,
                "stage=join_track_lists "
                f"event={event_id} time={perf_counter() - join_tracks_start:.3f}s "
                f"total_tracks={len(tracks)}",
            )
        if self.hparams.get("resolve_ambiguities", False) and self.hparams.get(
            "reuse_hits", False
        ):
            resolve_start = perf_counter()
            with profile_section(
                graph,
                "fast_walkthrough.resolve_ambiguities",
                enabled=self.enable_profiling,
            ):
                tracks = fast_walkthrough_utils.resolve_ambiguities(
                    tracks, self.hparams.get("max_ambi_hits", 2)
                )
            if debug_enabled:
                _debug_print(
                    self.hparams,
                    "stage=resolve_ambiguities "
                    f"event={event_id} time={perf_counter() - resolve_start:.3f}s "
                    f"total_tracks={len(tracks)}",
                )

        graph.reco_tracks = tracks
        if self.enable_profiling:
            set_profile_metadata(
                graph,
                "fast_walkthrough.simple_path.num_tracks",
                int(len(all_trks["cc"])),
            )
            set_profile_metadata(graph, "fast_walkthrough.total_tracks", int(len(tracks)))
        graph.time_taken = perf_counter() - start_time
        if self.enable_profiling:
            add_profile_time(
                graph,
                "fast_walkthrough.total",
                graph.time_taken,
            )

        if save_walkthrough_diagnostics:
            diagnostics_start = perf_counter()
            target_tracks = self.hparams.get("target_tracks", None)
            track_diag, particle_diag = build_track_particle_diagnostics(
                graph,
                target_tracks=target_tracks,
            )
            graph.track_diagnostics = track_diag
            graph.particle_reco_diagnostics = particle_diag
            graph.edge_diagnostics_summary = build_edge_diagnostics_summary(
                graph,
                edge_index,
                inverted_threshold,
                walk_input_graph=walk_input_graph,
                walk_pruned_graph=walk_pruned_graph,
            )
            graph.walk_branch_diagnostics = build_walk_branch_diagnostics(
                walk_pruned_graph,
                ambiguity_gap_threshold=float(
                    self.hparams.get(
                        "walk_branch_ambiguity_gap",
                        DEFAULT_BRANCH_AMBIGUITY_GAP,
                    )
                ),
            )
            if debug_enabled:
                _debug_print(
                    self.hparams,
                    "stage=walkthrough_diagnostics "
                    f"event={event_id} time={perf_counter() - diagnostics_start:.3f}s",
                )

        if self.hparams.get("save_walkthrough_graphs", True):
            save_graph_start = perf_counter()
            with profile_section(
                graph,
                "fast_walkthrough.save_graph",
                enabled=self.enable_profiling,
            ):
                _save_graph(graph, output_dir, self.hparams)
            if debug_enabled:
                _debug_print(
                    self.hparams,
                    "stage=save_graph "
                    f"event={event_id} time={perf_counter() - save_graph_start:.3f}s "
                    f"path={output_dir}",
                )

        profile_summary = _format_profile_summary(
            getattr(graph, "profiling", None)
        )
        _debug_print(
            self.hparams,
            f"event={event_id} tracks={len(tracks)} time={graph.time_taken:.3f}s"
            + (f" profile[{profile_summary}]" if profile_summary else ""),
        )
        return event_id

    def _build_tracks_one_evt_from_path(self, event_path, output_dir):
        """Load one event from disk and reconstruct its tracks."""
        load_start = perf_counter()
        try:
            graph = torch.load(event_path, map_location=torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load event file: {event_path}") from exc
        _debug_print(
            self.hparams,
            "event_load "
            f"path={os.path.basename(event_path)} time={perf_counter() - load_start:.3f}s "
            f"{_graph_size_summary(graph)}",
        )
        if self.enable_profiling:
            add_profile_time(
                graph,
                "fast_walkthrough.load_graph",
                perf_counter() - load_start,
            )
        return self._build_tracks_one_evt_impl(graph, output_dir)

    def _build_tracks_one_evt(self, graph, output_dir):
        """Reconstruct tracks for an in-memory event graph."""
        return self._build_tracks_one_evt_impl(graph, output_dir)

    def _should_reuse_walkthrough_output(self):
        """Reuse walkthrough outputs only when explicitly enabled."""
        return bool(self.hparams.get("reuse_walkthrough_output", False))

    def build_tracks(self, dataset, data_name):
        """Run fast walkthrough track building over a dataset split."""
        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)

        requested_workers = (
            self.hparams["max_workers"] if "max_workers" in self.hparams else 1
        )
        assert isinstance(requested_workers, int) and requested_workers > 0
        parallel_backend = self.hparams.get("parallel_backend")
        if parallel_backend is None:
            parallel_backend = "process" if requested_workers > 1 else "serial"
        use_process_pool = parallel_backend == "process"
        if use_process_pool and not _can_use_process_pool():
            parallel_backend = "thread"
            use_process_pool = False
        process_worker_cap = self.hparams.get("max_process_workers")
        if process_worker_cap is None and use_process_pool:
            process_worker_cap = 8
        if process_worker_cap is not None:
            assert isinstance(process_worker_cap, int) and process_worker_cap > 0
            max_workers = min(requested_workers, process_worker_cap)
        else:
            max_workers = requested_workers
        reuse_outputs = (
            self.hparams.get("save_walkthrough_graphs", True)
            and self._should_reuse_walkthrough_output()
        )

        dataset_size = len(dataset) if hasattr(dataset, "__len__") else "unknown"
        _debug_print(
            self.hparams,
            f"split={data_name} events={dataset_size} workers={max_workers} "
            f"backend={parallel_backend} reuse_outputs={reuse_outputs}",
        )
        print(
            "INFO: FastWalkthrough using "
            f"{max_workers} worker(s) for {data_name} with {parallel_backend} backend.",
            flush=True,
        )
        if max_workers != requested_workers:
            print(
                "INFO: Capping FastWalkthrough workers from "
                f"{requested_workers} to {max_workers}; set max_process_workers "
                "to override.",
                flush=True,
            )
        if max_workers != 1:
            print(
                "INFO: Reconstruction progress counts completed events; the first "
                "update may be delayed while the initial worker batch is still running.",
                flush=True,
            )

        if hasattr(dataset, "input_paths"):
            input_paths = list(dataset.input_paths)
            pending_paths = input_paths
            pending_indices = list(range(len(input_paths)))
            if reuse_outputs:
                pending_indices = [
                    idx
                    for idx, event_path in enumerate(input_paths)
                    if not os.path.isfile(
                        _walkthrough_output_path(
                            output_dir, os.path.basename(event_path)
                        )
                    )
                ]
                pending_paths = [input_paths[idx] for idx in pending_indices]
                skipped = len(input_paths) - len(pending_paths)
                print(
                    "INFO: Reusing existing walkthrough outputs for "
                    f"{data_name}; skipping {skipped} completed event(s), "
                    f"processing {len(pending_paths)} remaining.",
                    flush=True,
                )
                _debug_print(
                    self.hparams,
                    f"split={data_name} pending_paths={len(pending_paths)} skipped={skipped}",
                )

            if max_workers != 1:
                if use_process_pool:
                    executor = ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=get_context(
                            self.hparams.get("process_start_method", "spawn")
                        ),
                        initializer=_init_fast_walkthrough_worker,
                        initargs=(dict(self.hparams),),
                    )
                    futures = {}
                    progress = None
                    try:
                        futures = {
                            executor.submit(
                                _run_fast_walkthrough_worker,
                                event_path,
                                output_dir,
                            ): event_path
                            for event_path in pending_paths
                        }
                        progress = tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"Reconstructing tracks for {data_name} data",
                        )
                        for future in progress:
                            future.result()
                    except KeyboardInterrupt:
                        if progress is not None:
                            progress.close()
                        print(
                            "INFO: Interrupted FastWalkthrough; terminating worker "
                            "processes immediately.",
                            flush=True,
                        )
                        _terminate_process_pool(executor, futures=futures)
                        os._exit(130)
                    except Exception:
                        if progress is not None:
                            progress.close()
                        _terminate_process_pool(executor, futures=futures)
                        raise
                    else:
                        if progress is not None:
                            progress.close()
                        executor.shutdown(wait=True)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                self._build_tracks_one_evt_from_path,
                                event_path,
                                output_dir,
                            ): event_path
                            for event_path in pending_paths
                        }
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"Reconstructing tracks for {data_name} data",
                        ):
                            future.result()
            else:
                for event_idx in tqdm(
                    pending_indices,
                    desc=f"Reconstructing tracks for {data_name} data",
                ):
                    self._build_tracks_one_evt(
                        dataset[event_idx], output_dir=output_dir
                    )
        else:
            if max_workers != 1:
                self.log.warning(
                    "Dataset does not expose input_paths; using single-worker track building."
                )
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                if reuse_outputs and os.path.isfile(
                    _walkthrough_output_path(output_dir, event.event_id[0])
                ):
                    continue
                self._build_tracks_one_evt(event, output_dir=output_dir)

    def save_graph(self, graph, output_dir):
        """Save a reconstructed graph using the configured metadata."""
        return _save_graph(graph, output_dir, self.hparams)

    def forward(self, dataset, data_name):
        """Build tracks for the provided dataset split."""
        self.build_tracks(dataset, data_name)
