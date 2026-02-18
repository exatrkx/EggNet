import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from eggnet.utils import nearest_neighboring
from .utils.utils import make_mlp


class Walkthrough(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.knn:nearest_neighboring.abstract_knn = getattr(
            nearest_neighboring, hparams.get("knn_algorithm", "cu_knn")
        )()
        
    def forward(self, batch):
        self.knn.get_graph(batch, k=1, use_double_metric_learning=True)
        ...
        
    def track_construction(self, batch):
        ...