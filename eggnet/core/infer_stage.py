import yaml

import torch
from pytorch_lightning import Trainer

from eggnet import lightning_modules
from eggnet.utils.slurm import submit_to_slurm


def infer(config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, slurm, dataset_path):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    base_model_class = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))
    base_model = base_model_class.load_from_checkpoint(checkpoint)
    if output_dir is not None:
        base_model._hparams["output_dir"] = output_dir
    if accelerator is None:
        accelerator = base_model._hparams.get("accelerator", "cuda")
    if devices is None:
        devices = base_model._hparams.get("devices", 1)
    if num_nodes is None:
        num_nodes = base_model._hparams.get("num_nodes", 1)

    if slurm:
        infer_slurm(config, config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, dataset_path)
        return

    base_model.dataset_path = dataset_path
    base_model.setup(stage="predict", datasets=dataset)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
    )

    with torch.inference_mode():
        trainer.predict(base_model)


def infer_slurm(config, config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, dataset_path):

    if dataset:
        print([f" --dataset {d}" for d in dataset])

    command = (
        (f"eggnet infer {config_file} -c {checkpoint}") +
        (f" --output_dir {output_dir}" if output_dir else "") +
        ("".join([f" --dataset {d}" for d in dataset]) if dataset else "") +
        (f" --accelerator {accelerator}" if accelerator else "") +
        (f" --devices {devices}" if devices else "") +
        (f" --num_nodes {num_nodes}" if num_nodes else "") +
        (f" --dataset_path {dataset_path}" if dataset_path else "")
    )
    accelerator = config["accelerator"]
    devices = config["devices"]
    num_nodes = config["num_nodes"]

    submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=40)
