import os

import torch
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader

from eggnet import datasets, models, loss
from ..utils.utils import get_optimizers
from eggnet.utils.timing import time_function


class BaseModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.save_hyperparameters(hparams)

        self.trainset, self.valset, self.testset = None, None, None
        self.dataset_class = getattr(datasets, hparams.get("dataset", "GraphDataset"))
        self.model = getattr(models, hparams.get("model", "EggNet"))(hparams)
        self.loss_fn = getattr(loss, hparams.get("loss", "contrastive"))(hparams)

    @time_function
    def forward(self, batch, time_yes=False, **kwargs):
        return self.model(batch, time_yes=time_yes, **kwargs)

    def setup(self, stage="fit", datasets=None):
        if datasets is None:
            datasets = ["trainset", "valset", "testset"]
        if stage == "fit":
            datasets = ["trainset", "valset"]
            precision = "medium"
        else:
            precision = "high"
        if stage == "test":
            data_dir = self.hparams["output_dir"]
        else:
            data_dir = self.hparams["input_dir"]
        self.load_data(data_dir, stage, datasets)
        torch.set_float32_matmul_precision(precision)

    def load_data(self, input_dir, stage, datasets=["trainset", "valset", "testset"]):
        for data_name, data_num in zip(
            ["trainset", "valset", "testset"], self.hparams["data_split"]
        ):
            if data_num > 0 and data_name in datasets:
                dataset = self.dataset_class(
                    input_dir,
                    data_name,
                    data_num,
                    stage,
                    self.hparams,
                )
                setattr(self, data_name, dataset)

        print(
            f"Loaded {len(self.trainset) if self.trainset else 0} training events,"
            f" {len(self.valset) if self.valset else 0} validation events and {len(self.testset) if self.testset else 0} testing"
            " events"
        )

    def train_dataloader(self):
        if self.trainset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][0]
        )
        return DataLoader(self.trainset, batch_size=1, num_workers=num_workers)

    def val_dataloader(self):
        if self.valset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][1]
        )
        return DataLoader(self.valset, batch_size=1, num_workers=num_workers)

    def test_dataloader(self):
        if self.testset is None:
            return None
        num_workers = (
            16
            if (
                "num_workers" not in self.hparams or self.hparams["num_workers"] is None
            )
            else self.hparams["num_workers"][2]
        )
        return DataLoader(self.testset, batch_size=1, num_workers=num_workers)

    def predict_dataloader(self):
        """
        Load the prediction sets (which is a list of the three datasets)
        """
        dataloaders = [
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader(),
        ]
        dataloaders = [
            dataloader for dataloader in dataloaders if dataloader is not None
        ]
        return dataloaders

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(self.parameters(), self.hparams)
        return optimizer, scheduler

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))

    def save_graph(self, event, data_name):
        event.config.append(self.hparams)
        os.makedirs(os.path.join(self.hparams["output_dir"], data_name), exist_ok=True)
        torch.save(
            event.cpu(),
            os.path.join(
                self.hparams["output_dir"], data_name, f"event{event.event_id[0]}.pyg"
            ),
        )
