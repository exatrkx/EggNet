import torch.nn as nn

from .utils.utils import signal_loss, knn_loss, random_loss
from eggnet.utils.timing import time_function


class contrastive(nn.Module):
    """
    Naive contrastive loss
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

    @time_function
    def forward(self, batch):

        signal_ls = signal_loss(
            batch,
            self.hparams["margin"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )
        knn_ls = knn_loss(
            batch,
            self.hparams["margin"],
            self.hparams["knn_loss"],
            r=self.hparams.get("r_max"),
            algorithm=self.hparams.get("knn_algorithm", "cu_knn"),
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )
        random_ls = random_loss(
            batch,
            self.hparams["margin"],
            self.hparams["randomisation"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )
        loss = signal_ls + knn_ls + random_ls

        return loss
