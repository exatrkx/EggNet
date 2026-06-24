import os
import torch
import yaml

from eggnet import lightning_modules
from eggnet.utils.loading import get_stage_module, get_trainer
from eggnet.utils.slurm import submit_to_slurm

# hyperparamers that we wish to override with the YAML file
_RUNTIME_TRAINING_HPARAM_KEYS = (
    "input_dir",
    "data_split",
    "phi_segmented",
    "graph_fraction",
    "graph_adjustment_tol",
    "min_nodes",
    "max_nodes",
    "graph_fraction_adjustment_method",
    "max_possible_width",
    "num_workers",
    "max_epochs",
)


def _validate_cuda_training_config(config, config_file, slurm):
    accelerator = config.get("accelerator")
    if accelerator != "cuda":
        raise ValueError(
            f"eggnet train requires CUDA. Set `accelerator: cuda` in "
            f"{config_file}. Got {accelerator!r}."
        )
    if not slurm and not torch.cuda.is_available():
        raise RuntimeError(
            "eggnet train requires a visible CUDA device for local runs. "
            "Launch from a GPU node or submit with `--slurm`."
        )


def _apply_runtime_training_overrides(base_model, config):
    """Use training-time runtime settings instead of checkpoint defaults."""
    for key in _RUNTIME_TRAINING_HPARAM_KEYS:
        if key in config:
            base_model._hparams[key] = config[key]

    # Treat the training config as authoritative for hard cuts so that
    # removing or nulling the key disables checkpoint-time filtering.
    base_model._hparams["hard_cuts"] = config.get("hard_cuts")


def train(
    config_file, checkpoint, checkpoint_resume_dir, load_only_model_parameters, slurm
):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    _validate_cuda_training_config(config, config_file, slurm)

    if slurm:
        train_slurm(config, config_file, checkpoint, checkpoint_resume_dir, load_only_model_parameters)
        return

    base_model_class = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))

    os.makedirs(config["output_dir"], exist_ok=True)

    base_model, ckpt_config, default_root_dir, checkpoint = get_stage_module(
        config,
        base_model_class,
        checkpoint_path=checkpoint,
        checkpoint_resume_dir=checkpoint_resume_dir,
    )
    _apply_runtime_training_overrides(base_model, config)
    trainer = get_trainer(config, default_root_dir)
    if load_only_model_parameters:
        trainer.fit(base_model)
    else:
        trainer.fit(base_model, ckpt_path=checkpoint)


def train_slurm(config, config_file, checkpoint, checkpoint_resume_dir, load_only_model_parameters):

    command = (
        (f"eggnet train {config_file}") +
        (f" -c {checkpoint}" if checkpoint else "") +
        (f" --checkpoint_resume_dir {checkpoint_resume_dir}" if checkpoint_resume_dir else "") +
        (" --load_only_model_parameters" if load_only_model_parameters else "")
    )
    accelerator = config["accelerator"]
    devices = config["devices"]
    num_nodes = config["num_nodes"]

    submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=80)
