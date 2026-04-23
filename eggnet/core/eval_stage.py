import os

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

from eggnet import lightning_modules
from eggnet.utils.cluster import cluster_and_match
from eggnet.utils.plotting import (
    plot_computing_time,
    plot_eff_fixed_eps,
    plot_eff_vs_eps,
    plot_walkthrough_eff_vs_cutoff,
)
from eggnet.utils.slurm import submit_to_slurm
from eggnet.models.utils.plotting import (
    aggregate_dml_event_data,
    get_dml_fixed_eps_eval_data,
    get_dml_eval_scan_dataset_data,
    get_dml_eval_scan_data,
    plot_branching_diagnostics,
    plot_distance_histogram,
    plot_cc,
    plot_edge_score_diagnostics,
    plot_efficiency_vs_occupancy,
    plot_graph_reduction_waterfall,
    plot_hit_assignment_fraction,
    plot_reco_method_breakdown,
    plot_stage_timing_breakdown,
    plot_simple_graphs,
    plot_cutoff_efficiency,
    plot_split_merge_summary,
    plot_time_vs_occupancy,
    plot_track_completeness_distribution,
    plot_track_length_distribution,
    plot_track_purity_distribution,
    plot_tracks_per_event,
    plot_purity_vs_length,
)


UNSUPPORTED_DML_PLOT_FLAGS = (
    "plot_computing_time",
    "plot_parameter_predict",
    "plot_resolution_PT",
    "plot_resolution_ETA",
)

DML_AGGREGATE_PLOT_HANDLERS = (
    ("plot_track_length_distribution", plot_track_length_distribution),
    ("plot_hit_assignment_fraction", plot_hit_assignment_fraction),
    ("plot_reco_method_breakdown", plot_reco_method_breakdown),
    ("plot_tracks_per_event", plot_tracks_per_event),
    ("plot_time_vs_occupancy", plot_time_vs_occupancy),
    ("plot_stage_timing_breakdown", plot_stage_timing_breakdown),
    ("plot_graph_reduction_waterfall", plot_graph_reduction_waterfall),
    ("plot_efficiency_vs_occupancy", plot_efficiency_vs_occupancy),
    ("plot_track_purity_distribution", plot_track_purity_distribution),
    ("plot_track_completeness_distribution", plot_track_completeness_distribution),
    ("plot_split_merge_summary", plot_split_merge_summary),
    ("plot_purity_vs_length", plot_purity_vs_length),
    ("plot_edge_score_diagnostics", plot_edge_score_diagnostics),
    ("plot_branching_diagnostics", plot_branching_diagnostics),
)


def _resolve_walkthrough_output_dir(config, base_output_dir):
    explicit_output_dir = config.get("walkthrough_output_dir")
    if explicit_output_dir:
        return explicit_output_dir
    walkthrough_subdir = config.get("walkthrough_subdir", "walkthrough")
    return os.path.join(base_output_dir, walkthrough_subdir)


def _enabled_eval_flags(eval_config, flag_names):
    return [flag for flag in flag_names if bool(eval_config.get(flag, False))]


def _validate_dml_eval_config(eval_config):
    enabled_legacy_flags = _enabled_eval_flags(eval_config, UNSUPPORTED_DML_PLOT_FLAGS)
    if enabled_legacy_flags:
        flags = ", ".join(enabled_legacy_flags)
        raise ValueError(
            "double_metric_learning eval does not support these plot flags: "
            f"{flags}. Disable them in the eval config or run a "
            "non-DML eval."
        )


def _get_dataset_index(dataset_name):
    return {
        "trainset": 0,
        "valset": 1,
        "testset": 2,
    }[dataset_name]


def eval(config_file, eval_config_file, output_dir, accelerator, dataset, max_events, slurm):

    if slurm:
        eval_slurm(
            config_file,
            eval_config_file,
            output_dir,
            accelerator,
            dataset,
            max_events,
        )
        return

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if output_dir is not None:
        config["output_dir"] = output_dir
        if "walkthrough_output_dir" not in config:
            config["walkthrough_output_dir"] = _resolve_walkthrough_output_dir(
                config,
                output_dir,
            )
    with open(eval_config_file, "r") as f:
        eval_config = yaml.load(f, Loader=yaml.FullLoader)
    plot_output_dir = config["output_dir"]
    eval_config["output_dir"] = plot_output_dir

    data_config = dict(config)
    if max_events is not None:
        data_split = list(data_config["data_split"])
        dataset_index = _get_dataset_index(dataset)
        current_limit = int(data_split[dataset_index])
        if current_limit > 0:
            data_split[dataset_index] = min(current_limit, max_events)
        else:
            data_split[dataset_index] = max_events
        data_config["data_split"] = data_split
        print(f"Limiting {dataset} eval to {data_split[dataset_index]} files")
    if config.get("double_metric_learning", False):
        _validate_dml_eval_config(eval_config)
        data_config["output_dir"] = _resolve_walkthrough_output_dir(
            config,
            plot_output_dir,
        )
        if not os.path.isdir(data_config["output_dir"]):
            raise FileNotFoundError(
                "Double-metric-learning eval expects walkthrough outputs in "
                f"{data_config['output_dir']}. Run infer first to regenerate them."
            )

    base_model = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))(data_config)
    base_model.setup(stage="test", datasets=[dataset])
    data = getattr(base_model, dataset)

    if config.get("double_metric_learning", False):
        if (eval_config.get("plot_distance_histogram")):
            plot_distance_histogram(
                data,
                eval_config["output_dir"], 
                plot_false_edges=True,
                seperate_plots=False,
                plot_suffix=dataset)
        if (eval_config.get("plot_cc")):
            plot_cc(data, eval_config["output_dir"], hparams=config, output_suffix=dataset)
        if (eval_config.get("plot_simple_path_purity")):
            plot_simple_graphs(data, eval_config["output_dir"], hparams=config, output_suffix=dataset)
        if (eval_config.get("plot_cutoff_efficiency")):
            plot_cutoff_efficiency(data, eval_config["output_dir"], hparams=config, output_suffix=dataset)

        dml_plot_eta_enabled = bool(eval_config.get("plot_eta", False))
        dml_plot_eff_vs_eps_enabled = bool(eval_config.get("plot_eff_vs_eps", False))
        dml_plot_eff_fixed_eps_enabled = bool(eval_config.get("plot_eff_fixed_eps", False))
        dml_fixed_eps_results = None
        if dml_plot_eff_vs_eps_enabled:
            if dml_plot_eff_fixed_eps_enabled or dml_plot_eta_enabled:
                dml_eps_data, dml_fixed_eps_results = get_dml_eval_scan_dataset_data(
                    data,
                    eval_config,
                    config,
                    include_fixed_eps=True,
                )
            else:
                dml_eps_data = get_dml_eval_scan_dataset_data(data, eval_config, config)
            plot_walkthrough_eff_vs_cutoff(
                dml_eps_data,
                eval_config,
            )

        if dml_plot_eff_fixed_eps_enabled or dml_plot_eta_enabled:
            if dml_fixed_eps_results is None:
                dml_fixed_eps_results = get_dml_fixed_eps_eval_data(
                    data,
                    eval_config,
                    config,
                )
            (
                dml_fixed_eps_data,
                dml_particles_pt_hist,
                dml_matched_target_particles_pt_hist,
                dml_particles_eta_hist,
                dml_matched_target_particles_eta_hist,
                dml_pt_bins,
                dml_eta_bins,
            ) = dml_fixed_eps_results
            selection_subtext = f"Walkthrough distance cutoff (d={eval_config['eps']})"
            if dml_plot_eff_fixed_eps_enabled:
                plot_eff_fixed_eps(
                    dml_matched_target_particles_pt_hist,
                    dml_particles_pt_hist,
                    dml_fixed_eps_data,
                    eval_config,
                    dml_pt_bins,
                    f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]",
                    logx=True,
                    filename="track_efficiency_pt.png",
                    selection_subtext=selection_subtext,
                )
            if dml_plot_eta_enabled and dml_eta_bins is not None:
                plot_eff_fixed_eps(
                    dml_matched_target_particles_eta_hist,
                    dml_particles_eta_hist,
                    dml_fixed_eps_data,
                    eval_config,
                    dml_eta_bins,
                    r"$\eta$",
                    logx=False,
                    filename="track_efficiency_eta.png",
                    selection_subtext=selection_subtext,
                )

        enabled_aggregate_handlers = [
            (flag_name, handler)
            for flag_name, handler in DML_AGGREGATE_PLOT_HANDLERS
            if bool(eval_config.get(flag_name, False))
        ]
        if enabled_aggregate_handlers:
            aggregate_data = aggregate_dml_event_data(data, eval_config)
            for _, handler in enabled_aggregate_handlers:
                handler(aggregate_data, eval_config["output_dir"], dataset)
    else:
        plot_eta = bool(eval_config.get("plot_eta", True))
        plot_eff_vs_eps_enabled = bool(eval_config.get("plot_eff_vs_eps", True))
        plot_eff_fixed_eps_enabled = bool(eval_config.get("plot_eff_fixed_eps", True))
        plot_computing_time_enabled = bool(eval_config.get("plot_computing_time", True))

        eps_data = pd.DataFrame({
            "eps": np.arange(0.05, 0.51, 0.05),
            "n_particles": 0,
            "n_matched_particles": 0,
            "n_matched_tracks": 0,
            "n_matched_target_particles": 0,
            "n_matched_target_tracks": 0,
            "n_tracks": 0,
        })

        if plot_computing_time_enabled:
            time_data = pd.DataFrame({
                "num_nodes": [],
                "eggnet": [],
                "knn": [],
                "dbscan": [],
            })

        if eval_config.get("pT_unit", "MeV") == "MeV":
            pt_min, pt_max = 1000, 50000
        else:
            pt_min, pt_max = 1, 50
        pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)

        particles_pt_hist = np.histogram([], bins=pt_bins)[0]
        matched_target_particles_pt_hist = np.histogram([], bins=pt_bins)[0]
        if plot_eta:
            eta_bins = np.linspace(-4, 4)
            particles_eta_hist = np.histogram([], bins=eta_bins)[0]
            matched_target_particles_eta_hist = np.histogram([], bins=eta_bins)[0]

        for event in tqdm(data):
            event = event.to(accelerator)

            for eps_i in eps_data.eps:
                eps_data_i, particles_pt_hist_i, matched_target_particles_pt_hist_i, particles_eta_hist_i, matched_target_particles_eta_hist_i = cluster_and_match(
                    event,
                    eps_i,
                    eval_config,
                    time_yes=plot_computing_time_enabled and eps_i == eval_config["eps"],
                )

                eps_data[eps_data.eps == eps_i] = eps_data[eps_data.eps == eps_i].to_numpy() + eps_data_i.to_numpy()

                if eps_i == eval_config["eps"]:
                    particles_pt_hist += particles_pt_hist_i
                    matched_target_particles_pt_hist += matched_target_particles_pt_hist_i
                    if plot_eta:
                        particles_eta_hist += particles_eta_hist_i
                        matched_target_particles_eta_hist += matched_target_particles_eta_hist_i

            if plot_computing_time_enabled:
                num_nodes = event["num_nodes"]
                if hasattr(num_nodes, "cpu"):
                    num_nodes = num_nodes.cpu()
                time_data = pd.concat([time_data, pd.DataFrame({
                    "num_nodes": [num_nodes],
                    "eggnet": [event["BaseModule.forward"]],
                    "knn": [event[f"{config.get('knn_algorithm', 'cu_knn')}.get_graph"]],
                    "dbscan": [event["cluster"]],
                })])

        # check metric!!
        eps_data["eff"] = eps_data.n_matched_target_particles / eps_data.n_particles
        eps_data["dup"] = (
            eps_data.n_matched_target_tracks - eps_data.n_matched_target_particles
        ) / eps_data.n_matched_target_particles
        eps_data["fak"] = np.divide(
            eps_data.n_tracks - eps_data.n_matched_tracks,
            eps_data.n_tracks,
            out=np.zeros_like(eps_data.n_tracks, dtype=np.float64),
            where=eps_data.n_tracks > 0,
        )

        if plot_computing_time_enabled:
            time_data["gnn"] = time_data["eggnet"] - time_data["knn"]
            time_data["total"] = time_data["eggnet"] + time_data["dbscan"]

        if plot_eff_vs_eps_enabled:
            plot_eff_vs_eps(eps_data, eval_config)
        if plot_eff_fixed_eps_enabled:
            plot_eff_fixed_eps(matched_target_particles_pt_hist, particles_pt_hist, eps_data, eval_config, pt_bins, f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]", logx=True, filename="track_efficiency_pt.png")
        if plot_eff_fixed_eps_enabled and plot_eta:
            plot_eff_fixed_eps(matched_target_particles_eta_hist, particles_eta_hist, eps_data, eval_config, eta_bins, r"$\eta$", logx=False, filename="track_efficiency_eta.png")
        if plot_computing_time_enabled:
            plot_computing_time(time_data, eval_config)


def eval_slurm(config_file, eval_config_file, output_dir, accelerator, dataset, max_events):

    command = (
        (f"eggnet eval {config_file} {eval_config_file}") +
        (f" --output_dir {output_dir}" if output_dir else "") +
        (f" --accelerator {accelerator}" if accelerator else "") +
        (f" --dataset {dataset}" if dataset else "") +
        (f" --max-events {max_events}" if max_events else "")
    )

    submit_to_slurm(command, accelerator, 1, 1, gpu_memory=40)
