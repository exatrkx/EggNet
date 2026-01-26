import yaml

import torch
from pytorch_lightning import Trainer

from eggnet import lightning_modules
from eggnet.utils.slurm import submit_to_slurm


def infer(config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, slurm, input_dir, data_split):

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
    if data_split is not None: 
        # Attempt to override data_split if specified. 
        # Might not work due to bug in setup (see base_module.py)
        data_split_list = [int(x) for x in data_split.split()]
        print(f"INFO: Using data_split: {data_split_list}")
        if len(data_split_list) != 3:
            raise ValueError("data_split argument must be three integers separated by whitespace and be a string.\n"
                             "Did you forget to include quotation marks around the integer values?")
        config["data_split"] = data_split_list # Hopefully this patches
        base_model._hparams["data_split"] = data_split_list

    if slurm:
        infer_slurm(config, config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, input_dir, data_split)
        return

    # pass overrides to setup via state
    base_model.input_dir = input_dir
    base_model.datasets = dataset
    base_model.setup(stage="predict")

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
    )

    with torch.inference_mode():
        trainer.predict(base_model)


def infer_slurm(config, config_file, checkpoint, output_dir, dataset, accelerator, devices, num_nodes, input_dir, data_split):

    if dataset:
        print([f" --dataset {d}" for d in dataset])

    command = (
        (f"eggnet infer {config_file} -c {checkpoint}") +
        (f" --output_dir {output_dir}" if output_dir else "") +
        ("".join([f" --dataset {d}" for d in dataset]) if dataset else "") +
        (f" --accelerator {accelerator}" if accelerator else "") +
        (f" --devices {devices}" if devices else "") +
        (f" --num_nodes {num_nodes}" if num_nodes else "") +
        (f" --input_dir {input_dir}" if input_dir else "") +
        (f" --data_split {data_split}" if data_split else "")
    )
    accelerator = config["accelerator"]
    devices = config["devices"]
    num_nodes = config["num_nodes"]

    submit_to_slurm(command, accelerator, devices, num_nodes, gpu_memory=40)
