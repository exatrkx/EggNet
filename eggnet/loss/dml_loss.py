import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import hinge_loss
from eggnet.utils.timing import time_function
from eggnet.utils.mapping import get_node_weight
from eggnet.utils import nearest_neighboring
from .contrastive import *


class DML_Loss(nn.Module):
    """
    Deep Metric Learning loss
    """
    def __init__(self, hparams):
        raise NameError("This module is obsolete. Use 'Contrastive' or 'ObjectCondensation' instead")
        super().__init__()
        print("DEBUG: DML Loss module initiated")

        self.hparams = hparams
        self.contrastive = Contrastive(hparams)
        # In theory, we only really need to apply the Contrastive loss to both embeddings

    @time_function
    def forward(self, batch):
        # We don't need to do nothin' else :|
        return self.contrastive(batch)