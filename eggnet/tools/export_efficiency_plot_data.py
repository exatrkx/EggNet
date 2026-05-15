import json
import os
import tempfile
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from eggnet import lightning_modules
from eggnet.core import infer_stage
from eggnet.models.utils.plotting import get_dml_fixed_eps_eval_data
from eggnet.utils.cluster import cluster_and_match
from eggnet.utils.plotting import get_ratio


def _resolve_walkthrough_output_dir(config, base_output_dir):
    explicit_output_dir = config.get("walkthrough_output_dir")
    if explicit_output_dir:
        return explicit_output_dir
    walkthrough_subdir = config.get("walkthrough_subdir", "walkthrough")
    return os.path.join(base_output_dir, walkthrough_subdir)


def _get_dataset_index(dataset_name):
    return {
        "trainset": 0,
        "valset": 1,
        "testset": 2,
    }[dataset_name]


def _get_efficiency_bins(eval_config, include_eta):
    if eval_config.get("pT_unit", "MeV") == "MeV":
        pt_min, pt_max = 1000, 50000
    else:
        pt_min, pt_max = 1, 50
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 10)
    eta_bins = np.linspace(-4, 4) if include_eta else None
    return pt_bins, eta_bins


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {str(k): _json_ready(v) for k, v in value.to_dict(orient="list").items()}
    if isinstance(value, pd.Series):
        return {str(k): _json_ready(v) for k, v in value.to_dict().items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _build_fixed_eps_payload(
    matched_target_particles_hist,
    particles_hist,
    fixed_eps_data,
    eval_config,
    bins,
    *,
    xlabel,
    logx,
    filename,
    selection_subtext,
):
    hist, err = get_ratio(matched_target_particles_hist, particles_hist)
    fixed_row = fixed_eps_data.iloc[0]
    return {
        "plot_kind": "fixed_eps_efficiency",
        "output_filename": filename,
        "xlabel": xlabel,
        "ylabel": "Track Efficiency",
        "logx": bool(logx),
        "ylim": _json_ready(eval_config.get("ylim", [0.0, 1.04])),
        "selection_subtext": selection_subtext,
        "fixed_eps_metrics": _json_ready(fixed_row),
        "bin_edges": _json_ready(bins),
        "bin_centers": _json_ready((bins[1:] + bins[:-1]) / 2.0),
        "bin_half_widths": _json_ready((bins[1:] - bins[:-1]) / 2.0),
        "total_counts": _json_ready(particles_hist),
        "matched_target_counts": _json_ready(matched_target_particles_hist),
        "efficiency": _json_ready(hist),
        "efficiency_err_low": _json_ready(err[0]),
        "efficiency_err_high": _json_ready(err[1]),
    }


def _compute_non_dml_fixed_eps_data(data, eval_config, accelerator):
    plot_eta = bool(eval_config.get("plot_eta", True))
    eval_eps = float(eval_config["eps"])
    pt_bins, eta_bins = _get_efficiency_bins(eval_config, include_eta=plot_eta)

    totals = {
        "eps": eval_eps,
        "n_particles": 0,
        "n_matched_particles": 0,
        "n_matched_tracks": 0,
        "n_matched_target_particles": 0,
        "n_matched_target_tracks": 0,
        "n_tracks": 0,
    }
    particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    matched_target_particles_pt_hist = np.histogram([], bins=pt_bins)[0]
    particles_eta_hist = np.histogram([], bins=eta_bins)[0] if plot_eta else None
    matched_target_particles_eta_hist = (
        np.histogram([], bins=eta_bins)[0] if plot_eta else None
    )

    for event in tqdm(
        data,
        desc=f"Non-DML fixed-eps export (eps={eval_eps:g})",
        unit="event",
        dynamic_ncols=True,
    ):
        event = event.to(accelerator)
        (
            eps_data_i,
            particles_pt_hist_i,
            matched_target_particles_pt_hist_i,
            particles_eta_hist_i,
            matched_target_particles_eta_hist_i,
        ) = cluster_and_match(
            event,
            eval_eps,
            eval_config,
            time_yes=False,
        )
        totals["n_particles"] += int(eps_data_i["n_particles"].iloc[0])
        totals["n_matched_particles"] += int(eps_data_i["n_matched_particles"].iloc[0])
        totals["n_matched_tracks"] += int(eps_data_i["n_matched_tracks"].iloc[0])
        totals["n_matched_target_particles"] += int(
            eps_data_i["n_matched_target_particles"].iloc[0]
        )
        totals["n_matched_target_tracks"] += int(eps_data_i["n_matched_target_tracks"].iloc[0])
        totals["n_tracks"] += int(eps_data_i["n_tracks"].iloc[0])

        particles_pt_hist += particles_pt_hist_i
        matched_target_particles_pt_hist += matched_target_particles_pt_hist_i
        if plot_eta:
            particles_eta_hist += particles_eta_hist_i
            matched_target_particles_eta_hist += matched_target_particles_eta_hist_i

    fixed_eps_data = pd.DataFrame([totals])
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
        fixed_eps_data.reset_index(drop=True),
        particles_pt_hist,
        matched_target_particles_pt_hist,
        particles_eta_hist,
        matched_target_particles_eta_hist,
        pt_bins,
        eta_bins,
    )


def _prepare_eval_inputs(config_file, eval_config_file, output_dir, dataset, max_events):
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
    eval_config["output_dir"] = config["output_dir"]

    data_config = dict(config)
    if max_events is not None:
        data_split = list(data_config["data_split"])
        dataset_index = _get_dataset_index(dataset)
        current_limit = int(data_split[dataset_index])
        data_split[dataset_index] = (
            min(current_limit, max_events) if current_limit > 0 else max_events
        )
        data_config["data_split"] = data_split
        print(f"Limiting {dataset} export to {data_split[dataset_index]} events")

    # Test-time exports load cached inference graphs from output_dir. Those
    # graphs already reflect any inference-time hard cuts, so applying them
    # again here can both distort the metrics and break on 2D node features
    # such as cached embeddings.
    data_config["hard_cuts"] = None

    if config.get("double_metric_learning", False):
        data_config["output_dir"] = _resolve_walkthrough_output_dir(
            config,
            config["output_dir"],
        )
        if not os.path.isdir(data_config["output_dir"]):
            raise FileNotFoundError(
                "Double-metric-learning export expects walkthrough outputs in "
                f"{data_config['output_dir']}. Run inference first or pass --run-infer."
            )

    return config, eval_config, data_config


def _prepare_export_infer_config(config_file):
    """Disable walkthrough extras that are irrelevant to fixed-eps JSON export."""
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not config.get("double_metric_learning", False):
        return config_file, None

    infer_config = dict(config)
    infer_config["save_walkthrough_diagnostics"] = False
    infer_config["enable_profiling"] = False
    infer_config["enable_structural_profile_metadata"] = False

    fd, infer_config_path = tempfile.mkstemp(
        prefix="eggnet_export_infer_",
        suffix=".yaml",
    )
    os.close(fd)
    with open(infer_config_path, "w") as f:
        yaml.safe_dump(infer_config, f, sort_keys=False)

    return infer_config_path, infer_config_path


def export_efficiency_plot_data(
    config_file,
    eval_config_file,
    checkpoint=None,
    output_path=None,
    output_dir=None,
    dataset="valset",
    accelerator="cuda",
    devices=None,
    num_nodes=None,
    max_events=None,
    run_infer=False,
    reuse_inference_output=False,
):
    if run_infer:
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --run-infer is enabled.")
        infer_config_file, temp_infer_config = _prepare_export_infer_config(config_file)
        try:
            infer_stage.infer(
                config_file=infer_config_file,
                checkpoint=checkpoint,
                output_dir=output_dir,
                dataset=(dataset,),
                max_events=max_events,
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                reuse_inference_output=reuse_inference_output,
                slurm=False,
            )
        finally:
            if temp_infer_config is not None:
                try:
                    os.remove(temp_infer_config)
                except FileNotFoundError:
                    pass
        if not infer_stage._is_global_zero_process():
            print(
                "INFO: Skipping export on nonzero distributed rank; global rank 0 "
                "will write the JSON output.",
                flush=True,
            )
            return None

    config, eval_config, data_config = _prepare_eval_inputs(
        config_file,
        eval_config_file,
        output_dir,
        dataset,
        max_events,
    )

    base_model = getattr(
        lightning_modules,
        config.get("base_model", "NodeEncoding"),
    )(data_config)
    base_model.setup(stage="test", datasets=[dataset])
    data = getattr(base_model, dataset)

    is_dml = bool(config.get("double_metric_learning", False))
    include_eta = bool(eval_config.get("plot_eta", True))
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "config_file": os.path.abspath(config_file),
            "eval_config_file": os.path.abspath(eval_config_file),
            "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
            "dataset": dataset,
            "output_dir": os.path.abspath(eval_config["output_dir"]),
            "double_metric_learning": is_dml,
            "run_infer": bool(run_infer),
            "reuse_inference_output": bool(reuse_inference_output),
            "accelerator": accelerator,
            "devices": devices,
            "num_nodes": num_nodes,
            "max_events": max_events,
            "eval_eps": float(eval_config["eps"]),
            "plot_eta": include_eta,
            "pT_unit": eval_config.get("pT_unit", "MeV"),
            "trackML_data": bool(eval_config.get("trackML_data", False)),
        },
        "plots": {},
    }

    if is_dml:
        (
            fixed_eps_data,
            particles_pt_hist,
            matched_target_particles_pt_hist,
            particles_eta_hist,
            matched_target_particles_eta_hist,
            pt_bins,
            eta_bins,
        ) = get_dml_fixed_eps_eval_data(data, eval_config, config)
        selection_subtext = f"Walkthrough distance cutoff (d={eval_config['eps']})"
        payload["plots"]["eff_fixed_eps_pt"] = _build_fixed_eps_payload(
            matched_target_particles_pt_hist,
            particles_pt_hist,
            fixed_eps_data,
            eval_config,
            pt_bins,
            xlabel=f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]",
            logx=True,
            filename="track_efficiency_pt.png",
            selection_subtext=selection_subtext,
        )
        if include_eta and eta_bins is not None:
            payload["plots"]["eff_fixed_eps_eta"] = _build_fixed_eps_payload(
                matched_target_particles_eta_hist,
                particles_eta_hist,
                fixed_eps_data,
                eval_config,
                eta_bins,
                xlabel=r"$\eta$",
                logx=False,
                filename="track_efficiency_eta.png",
                selection_subtext=selection_subtext,
            )
    else:
        (
            fixed_eps_data,
            particles_pt_hist,
            matched_target_particles_pt_hist,
            particles_eta_hist,
            matched_target_particles_eta_hist,
            pt_bins,
            eta_bins,
        ) = _compute_non_dml_fixed_eps_data(data, eval_config, accelerator)
        selection_subtext = (
            r"DBSCAN ($\epsilon$"
            + f"={eval_config['eps']}, min_samples=3)"
        )
        payload["plots"]["eff_fixed_eps_pt"] = _build_fixed_eps_payload(
            matched_target_particles_pt_hist,
            particles_pt_hist,
            fixed_eps_data,
            eval_config,
            pt_bins,
            xlabel=f"$p_T$ [{eval_config.get('pT_unit', 'MeV')}]",
            logx=True,
            filename="track_efficiency_pt.png",
            selection_subtext=selection_subtext,
        )
        if include_eta and eta_bins is not None:
            payload["plots"]["eff_fixed_eps_eta"] = _build_fixed_eps_payload(
                matched_target_particles_eta_hist,
                particles_eta_hist,
                fixed_eps_data,
                eval_config,
                eta_bins,
                xlabel=r"$\eta$",
                logx=False,
                filename="track_efficiency_eta.png",
                selection_subtext=selection_subtext,
            )

    if output_path is None:
        filename = f"efficiency_plot_data_{dataset}.json"
        output_path = os.path.join(eval_config["output_dir"], filename)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"INFO: Saved efficiency plot data to {output_path}")
    return output_path


@click.command("export-efficiency-data")
@click.argument("config_file")
@click.argument("eval_config_file")
@click.option(
    "--checkpoint",
    "-c",
    default=None,
    help="Checkpoint to use when --run-infer is enabled.",
)
@click.option(
    "--output-path",
    default=None,
    help="Path to the JSON export file. Defaults to <output_dir>/efficiency_plot_data_<dataset>.json.",
)
@click.option(
    "--output_dir",
    "-o",
    default=None,
    help="Directory containing inference outputs and where the export file should be written.",
)
@click.option(
    "--dataset",
    "-d",
    default="valset",
    type=click.Choice(["trainset", "valset", "testset"]),
    help="Dataset split to evaluate.",
)
@click.option(
    "--accelerator",
    "-a",
    default="cuda",
    type=click.Choice(["cuda", "cpu"]),
    help="Execution device. Non-DML clustering currently expects CUDA.",
)
@click.option(
    "--devices",
    "-dv",
    default=None,
    type=int,
    help="Device count to use if --run-infer is enabled.",
)
@click.option(
    "--num_nodes",
    "-n",
    default=None,
    type=int,
    help="Node count to use if --run-infer is enabled.",
)
@click.option(
    "--max-events",
    "max_events",
    default=None,
    type=click.IntRange(min=1),
    help="Limit export to the first N events from the selected dataset.",
)
@click.option(
    "--run-infer/--no-run-infer",
    default=False,
    help="Run inference first before exporting plot data.",
)
@click.option(
    "--reuse_inference_output/--no-reuse_inference_output",
    default=False,
    help="Reuse cached inference outputs when --run-infer is enabled.",
)
def export_efficiency_data_command(**kwargs):
    return export_efficiency_plot_data(**kwargs)


if __name__ == "__main__":
    export_efficiency_data_command()
