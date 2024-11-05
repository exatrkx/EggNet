import yaml

import torch
from pytorch_lightning import Trainer

from eggnet import lightning_modules


def infer(config_file, checkpoint, output_dir, datasets, accelerator, devices, num_nodes):

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    base_model_class = getattr(lightning_modules, config.get("base_model", "NodeEncoding"))
    base_model = base_model_class.load_from_checkpoint(checkpoint)
    if output_dir is not None:
        base_model._hparams["output_dir"] = output_dir
    base_model.setup(stage="predict", datasets=datasets)

    if accelerator is None:
        accelerator = base_model._hparams.get("accelerator", "cuda")
    if devices is None:
        devices = base_model._hparams.get("devices", 1)
    if num_nodes is None:
        num_nodes = base_model._hparams.get("num_nodes", 1)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
    )

    with torch.inference_mode():
        trainer.predict(base_model)
