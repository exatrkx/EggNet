import os
import yaml

from eggnet import lightning_modules
from eggnet.utils.loading import get_stage_module, get_trainer
from eggnet.utils.slurm import submit_to_slurm


def train(
    config_file, checkpoint, checkpoint_resume_dir, load_only_model_parameters, slurm
):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
        (f" --load_only_model_parameters {load_only_model_parameters}" if load_only_model_parameters else "")
    )
    accelerator = config["accelerator"]
    devices = config["devices"]
    num_nodes = config["num_nodes"]

    submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=80)
