import os

import torch

from eggnet.utils.nearest_neighboring import get_knn_graph
from eggnet.utils.mapping import get_target, get_weight, get_number_of_true_edges
from .base_module import BaseModule


class NodeEncoding(BaseModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def training_step(self, batch, batch_idx):

        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch)
        else:
            batch.hit_embedding = self(batch)

        ls = self.loss_fn(batch)

        self.log_dict(
            {
                "train_loss": ls,
                # "train_signal_loss": signal_loss,
                # "train_knn_loss": knn_loss,
                # "train_random_loss": random_loss,
            },
            batch_size=1,
        )

        return ls

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch)
        else:
            batch.hit_embedding = self(batch)

        edges = get_knn_graph(
            batch,
            k=self.hparams["knn_val"],
            r=self.hparams.get("r_max"),
            algorithm=self.hparams.get("knn_algorithm", "cu_knn"),
        )
        if self.hparams.get("node_filter"):
            edges = batch.filter_node_list[edges]

        y = get_target(edges, batch.hit_particle_id)
        w = get_weight(batch, edges, y, weighting_config=self.hparams.get("weighting"))
        tp = torch.sum(y == 1)
        target_tp = torch.sum((y == 1) & (w > 0))

        eff = (
            tp
            / get_number_of_true_edges(
                batch,
                reduction="sum",
                upper_bound=self.hparams["knn_val"],
                weighting_config=self.hparams.get("weighting"),
            )[1]
        )
        signal_eff = (
            target_tp
            / get_number_of_true_edges(
                batch,
                target="weight-based",
                reduction="sum",
                upper_bound=self.hparams["knn_val"],
                weighting_config=self.hparams.get("weighting"),
            )[1]
        )
        pur = tp / len(y)
        # f1 = 2 * (eff * pur) / (eff + pur)

        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log_dict(
            {
                "lr": current_lr,
                "val_eff": eff,
                "val_signal_eff": signal_eff,
                "val_pur": pur,
            },
            batch_size=1,
        )
        # print("validation step end", torch.cuda.max_memory_allocated(device="cuda"))

        return eff

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 0:
            return

        dataset = self.predict_dataloader()[dataloader_idx].dataset
        if os.path.isfile(
            os.path.join(
                self.hparams["output_dir"],
                dataset.data_name,
                f"event{batch.event_id[0]}.pyg",
            )
        ):
            return 0

        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch, time_yes=True)
        else:
            batch.hit_embedding = self(batch, time_yes=True)

        dataset.unscale_features(batch)

        self.save_graph(batch, dataset.data_name)

        return 0
