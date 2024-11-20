import torch
import torch.nn as nn
# from torch_scatter import scatter_max
from object_condensation.pytorch.losses import condensation_loss_tiger

# from .utils.utils import hinge_loss  # , beta_loss, vertex_loss
from eggnet.utils.timing import time_function


class ObjectCondensation(nn.Module):
    """
    Object condensation loss as described in https://arxiv.org/abs/2002.03605
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

    @time_function
    def forward(self, batch):

        res = condensation_loss_tiger(beta=batch.hit_score, x=batch.hit_embedding, object_id=batch.hit_particle_id, weights=None, q_min=self.hparams["qmin"], noise_threshold=0, max_n_rep=100000, torch_compile=False)
        res["loss"] = res["attractive"] + res["repulsive"] + res["coward"] + (0 if torch.isnan(res["noise"]).item() else res["noise"]) * self.hparams["sb"]
        return res

#         regression_ls = 0  # TO BE IMPLEMENTED
#         beta_ls = beta_loss(
#             batch,
#             self.hparams["sb"],
#         )
#         vertex_ls = vertex_loss(
#             batch,
#             self.hparams["qmin"],
#         )
#         loss = regression_ls + (beta_ls + vertex_ls) * self.hparams.get("sc", 1)

#         return loss


# def beta_loss(batch, sb=1):
#     hit_particle_idx = torch.unique(batch.hit_particle_id, sorted=False, return_inverse=True)[1]
#     beta_alpha_k, argmax = scatter_max(batch.hit_score.flatten(), hit_particle_idx)
#     alpha_mask = argmax < len(batch.hit_id)
#     noise_mask = batch.hit_particle_id == 0
#     return (1 - beta_alpha_k[alpha_mask]).sum() / (alpha_mask).sum() + batch.hit_score[noise_mask].sum() / noise_mask.sum() * sb


# def vertex_loss(batch, qmin):
#     hit_particle_idx = torch.unique(batch.hit_particle_id, sorted=False, return_inverse=True)[1]
#     beta_alpha_k, argmax = scatter_max(batch.hit_score.flatten(), hit_particle_idx)
#     alpha_mask = argmax < len(batch.hit_id)
#     q = torch.arctanh(batch.hit_score)**2 + qmin
#     ls = 0
#     for alpha in argmax[alpha_mask]:
#         edges = torch.stack([torch.full((len(batch.hit_id),), alpha), torch.arange(len(batch.hit_id))]).to(batch.hit_id.device)
#         ls += hinge_loss(batch, edges, 1, f=q * q[alpha])
#     return ls / len(batch.hit_id)
