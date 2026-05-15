import yaml
import os

import torch
from pytorch_lightning import Trainer

from eggnet import lightning_modules, models
from eggnet.utils.slurm import submit_to_slurm


_RUNTIME_DATASET_HPARAM_KEYS = (
    "input_dir",
    "data_split",
    "phi_segmented",
    "graph_fraction",
    "graph_adjustment_tol",
    "min_nodes",
    "max_nodes",
    "graph_fraction_adjustment_method",
    "max_possible_width",
)


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


def _apply_max_events_limit(config, datasets, max_events):
    if max_events is None:
        return

    data_split = list(config["data_split"])
    for dataset_name in datasets:
        dataset_index = _get_dataset_index(dataset_name)
        current_limit = int(data_split[dataset_index])
        data_split[dataset_index] = (
            min(current_limit, max_events) if current_limit > 0 else max_events
        )
        print(
            f"Limiting {dataset_name} inference to {data_split[dataset_index]} events"
        )
    config["data_split"] = data_split


def _apply_runtime_dataset_overrides(base_model, config):
    """Use inference-time dataset settings instead of checkpoint defaults."""
    for key in _RUNTIME_DATASET_HPARAM_KEYS:
        if key in config:
            base_model._hparams[key] = config[key]

    # Treat the inference config as authoritative for hard cuts so that
    # removing or nulling the key disables checkpoint-time filtering.
    base_model._hparams["hard_cuts"] = config.get("hard_cuts")


def _resolve_runtime_execution_overrides(base_model, config, accelerator, devices, num_nodes):
    """Apply execution settings consistently across the runtime config and hparams."""
    accelerator = accelerator or config.get(
        "accelerator", base_model._hparams.get("accelerator", "cuda")
    )
    devices = devices if devices is not None else config.get(
        "devices", base_model._hparams.get("devices", 1)
    )
    num_nodes = num_nodes if num_nodes is not None else config.get(
        "num_nodes", base_model._hparams.get("num_nodes", 1)
    )

    config["accelerator"] = accelerator
    config["devices"] = devices
    config["num_nodes"] = num_nodes
    base_model._hparams["accelerator"] = accelerator
    base_model._hparams["devices"] = devices
    base_model._hparams["num_nodes"] = num_nodes

    return accelerator, devices, num_nodes


def _is_global_zero_process():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def infer(
    config_file,
    checkpoint,
    output_dir,
    dataset,
    max_events,
    accelerator,
    devices,
    num_nodes,
    reuse_inference_output=False,
    slurm=False,
):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    track_build_debug = bool(config.get("track_build_debug", False))
    validate_reused_inference_output = bool(
        config.get("validate_reused_inference_output", False)
    )
    use_walkthrough = bool(config.get("double_metric_learning", False))
    dataset = list(dataset) if dataset else None
    target_datasets = dataset or ["trainset", "valset", "testset"]
    _apply_max_events_limit(config, target_datasets, max_events)

    base_model_class = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))
    base_model = base_model_class.load_from_checkpoint(checkpoint)
    _apply_runtime_dataset_overrides(base_model, config)
    if output_dir is not None:
        base_model._hparams["output_dir"] = output_dir
        config["output_dir"] = output_dir
    accelerator, devices, num_nodes = _resolve_runtime_execution_overrides(
        base_model,
        config,
        accelerator,
        devices,
        num_nodes,
    )
    if dataset is not None:
        base_model._hparams["predict_datasets"] = dataset
    elif "predict_datasets" in base_model._hparams:
        del base_model._hparams["predict_datasets"]
    base_model._hparams["reuse_inference_output"] = reuse_inference_output

    if slurm:
        infer_slurm(
            config,
            config_file,
            checkpoint,
            output_dir,
            dataset,
            max_events,
            accelerator,
            devices,
            num_nodes,
            reuse_inference_output,
        )
        return

    with torch.inference_mode():
        run_predict_pass = True
        if use_walkthrough and reuse_inference_output and not validate_reused_inference_output:
            run_predict_pass = False

        if run_predict_pass:
            base_model.setup(stage="predict", datasets=dataset)
            trainer = Trainer(
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                logger=False,
            )
            if reuse_inference_output:
                print(
                    "INFO: Reusing cached inference outputs; the initial predict "
                    "progress bar reflects dataset loading and cache checks, not "
                    "embedding compute.",
                    flush=True,
                )
            trainer.predict(base_model)
        elif reuse_inference_output:
            print(
                "INFO: Reusing cached inference outputs and skipping the initial "
                "predict pass; loading cached graphs directly for walkthrough.",
                flush=True,
            )

        if use_walkthrough:
            if not _is_global_zero_process():
                if track_build_debug:
                    print(
                        "[infer] skipping walkthrough on nonzero distributed rank",
                        flush=True,
                    )
                return
            walkthrough_output_dir = _resolve_walkthrough_output_dir(
                config,
                base_model._hparams["output_dir"],
            )
            if track_build_debug:
                print(
                    f"[infer] walkthrough -> {walkthrough_output_dir}",
                    flush=True,
                )
            # Load predicted graphs from output_dir, either after a fresh predict pass
            # or directly from cache when reusing existing inference outputs.
            base_model.setup(stage="test", datasets=dataset)

            # walkthrough:torch.nn.Module = getattr(models, config.get("walkthrough_model", "FastWalkthrough"))
            # assert walkthrough is not None
            # walkthrough.build_tracks()

            track_builder_name = config.get("walkthrough_model", "FastWalkthrough")
            track_builder_class: torch.nn.Module = getattr(models, track_builder_name, None)
            if track_builder_class is None:
                raise ValueError(f"Unknown track-building model: {track_builder_name}")

            track_builder_hparams = dict(base_model._hparams)
            track_builder_hparams.update(config)
            if output_dir is not None:
                track_builder_hparams["output_dir"] = output_dir
            track_builder_hparams["walkthrough_output_dir"] = walkthrough_output_dir
            track_builder_hparams["stage_dir"] = walkthrough_output_dir

            track_builder = track_builder_class(track_builder_hparams)
            track_builder.eval()

            selected_data = []
            for data_name in ("trainset", "valset", "testset"):
                data_iterable = getattr(base_model, data_name, None)
                if data_iterable is None:
                    continue
                if dataset and data_name not in dataset:
                    continue
                selected_data.append((data_name, data_iterable))

            for data_name, data_iterable in selected_data:
                if track_build_debug:
                    print(f"[infer] build_tracks {data_name}", flush=True)
                track_builder.build_tracks(data_iterable, data_name)


def infer_slurm(
    config,
    config_file,
    checkpoint,
    output_dir,
    dataset,
    max_events,
    accelerator,
    devices,
    num_nodes,
    reuse_inference_output=False,
):

    if dataset:
        print([f" --dataset {d}" for d in dataset])

    command = (
        (f"eggnet infer {config_file} -c {checkpoint}") +
        (f" --output_dir {output_dir}" if output_dir else "") +
        ("".join([f" --dataset {d}" for d in dataset]) if dataset else "") +
        (f" --max-events {max_events}" if max_events else "") +
        (f" --accelerator {accelerator}" if accelerator else "") +
        (f" --devices {devices}" if devices else "") +
        (f" --num_nodes {num_nodes}" if num_nodes else "") +
        (" --reuse_inference_output" if reuse_inference_output else "")
    )
    accelerator = config["accelerator"]
    devices = config["devices"]
    num_nodes = config["num_nodes"]

    submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=40)
