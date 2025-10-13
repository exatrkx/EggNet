import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import hinge_loss
from eggnet.utils.timing import time_function
from eggnet.utils.mapping import get_node_weight, map_tracks_to_nodes
from eggnet.utils import nearest_neighboring


class Contrastive(nn.Module):
    """
    Naive contrastive loss
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.signal_loss = signal_contrastive_loss(hparams)
        self.knn_loss = knn_contrastive_loss(hparams)
        self.random_loss = random_contrastive_loss(hparams)
        if hparams.get("predict_track_parameters"):
            self.parameter_estimation_loss = parameter_estimation_loss(hparams)

    @time_function
    def forward(self, batch):

        res = {}

        res["signal_loss"] = self.signal_loss(
            batch,
            self.hparams["margin"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )["loss"]
        res["knn_loss"] = self.knn_loss(
            batch,
            self.hparams["margin"],
            self.hparams["knn_loss"],
            r=self.hparams.get("r_max_loss"),
            algorithm=self.hparams.get("knn_algorithm_loss", "cu_knn"),
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )["loss"]
        res["random_loss"] = self.random_loss(
            batch,
            self.hparams["margin"],
            self.hparams["randomisation"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
        )["loss"]
        if self.hparams.get("predict_track_parameters"):
            ret = self.parameter_estimation_loss(batch)
            res["parameter_estimation_loss"] = ret["loss"]
            res.update({k:v for k, v in ret.items() if k != "loss"})  # Add other keys from the parameter estimation loss
        res["loss"] = res["signal_loss"] + res["knn_loss"] + res["random_loss"]
        
        if self.hparams.get("predict_track_parameters"):
            res["loss"] += res["parameter_estimation_loss"]
        return res


class WeightedContrastive(nn.Module):
    """
    Weighted contrastive loss
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.signal_loss = signal_contrastive_loss(hparams)
        self.knn_loss = knn_contrastive_loss(hparams)
        self.random_loss = random_contrastive_loss(hparams)
        if hparams.get("predict_track_parameters"):
            self.parameter_estimation_loss = parameter_estimation_loss(hparams)

    @time_function
    def forward(self, batch):

        res = {}

        res["signal_loss"] = self.signal_loss(
            batch,
            self.hparams["margin"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
            node_score=True,
        )["loss"]
        res["knn_loss"] = self.knn_loss(
            batch,
            self.hparams["margin"],
            self.hparams["knn_loss"],
            r=self.hparams.get("r_max_loss"),
            algorithm=self.hparams.get("knn_algorithm_loss", "cu_knn"),
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
            node_score=True,
        )["loss"]
        res["random_loss"] = self.random_loss(
            batch,
            self.hparams["margin"],
            self.hparams["randomisation"],
            node_filter=self.hparams.get("node_filter"),
            weighting_config=self.hparams.get("weighting"),
            node_score=True,
        )["loss"]
        w = get_node_weight(batch, self.hparams.get("weighting"))
        res["beta_loss"] = F.binary_cross_entropy_with_logits(batch.hit_score.flatten(), (batch.hit_particle_id != 0).float(), reduction="sum", weight=w) / w.sum()
        
        if self.hparams.get("predict_track_parameters"):
            ret = self.parameter_estimation_loss(batch)
            res["parameter_estimation_loss"] = ret["loss"]
            res.update({k:v for k, v in ret.items() if k != "loss"})  # Add other keys from the parameter estimation loss
        
        res["loss"] = res["signal_loss"] + res["knn_loss"] + res["random_loss"]
        if self.hparams.get("predict_track_parameters"):
            res["loss"] += res["parameter_estimation_loss"]
            
        return res


class signal_contrastive_loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

    @time_function
    def forward(self, batch, margin, node_filter=False, weighting_config=None, node_score=False):

        res = {}
        res["loss"] = hinge_loss(
            batch,
            batch.track_edges,
            margin,
            y=torch.ones(batch.track_edges.shape[1], device=batch.track_edges.device),
            node_filter=node_filter,
            weighting_config=weighting_config,
            node_score=node_score,
        )
        return res


class knn_contrastive_loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.knn = getattr(nearest_neighboring, hparams.get("knn_algorithm_loss", "cu_knn"))()

    @time_function
    def forward(self, batch, margin, k, r=None, algorithm="cu_knn", node_filter=False, weighting_config=None, node_score=False):

        res = {}
        edges = self.knn.get_graph(
            batch,
            k=k,
            r=self.hparams.get("r_max_train"),
        )
        if node_filter:
            edges = batch.filter_node_list[edges]
        res["loss"] = hinge_loss(
            batch, edges, margin, node_filter=node_filter, weighting_config=weighting_config, node_score=node_score
        )
        return res


class random_contrastive_loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

    @time_function
    def forward(self, batch, margin, randomisation, node_filter=False, weighting_config=None, node_score=False):

        res = {}
        edges = torch.randint(
            0,
            batch.hit_id.shape[0],
            (2, randomisation),
            device=batch.hit_id.device,
        )
        res["loss"] = hinge_loss(
            batch, edges, margin, node_filter=node_filter, weighting_config=weighting_config, node_score=node_score
        )
        return res

class parameter_estimation_loss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        assert self.hparams.get("predict_track_parameters")
        if self.hparams.get("parameter_loss_confidence", False):
            self.lam:float = self.hparams.get("parameter_loss_confidence_lambda", 1.0)
            self.beta:float = self.hparams.get("parameter_loss_confidence_beta", 1.0)
            self.dlambda = self.hparams.get("parameter_loss_confidence_delta_lambda", 0.01)
            print(f"DEBUG: Using confidence. \nlam={self.lam} \nbeta={self.beta} \ndlam={self.dlambda}")
        if self.hparams.get("parameter_loss_normalize", False) \
            and not self.hparams.get("parameter_loss_confidence", False):
            norm_method = self.hparams.get("parameter_loss_norm_method", "")
            print(f"DEBUG: Using normalization method {norm_method}")
        
    @time_function
    def forward(self, batch):
        use_lambda = self.hparams.get("parameter_loss_confidence_lambda", False)
        res = {}
        x = batch.hit_parameters
        y = batch.get("hit_charge_pt_ratio") * self.hparams.get("parameter_loss_scale", 1.0) # Use std or range to scale maybe globally 
        y = y.to(x.dtype).to(x.device)  # Ensure y has the same dtype as x. y is usually a float64/double tensor, while x is usually a float16 tensor
        if self.hparams.get("parameter_loss_normalize", False) \
            and not self.hparams.get("parameter_loss_confidence", False):
            y *= batch["normalization_constant"]
            res["normalization_factor"] = batch["normalization_constant"]

        # https://bharathpbhat.github.io/2021/04/04/getting-confidence-estimates-from-neural-networks.html#learned-confidence
        # This source recommends using log(p(y|x)) ∝ -log(σ) - (-1/2 * np.square((y - µ)/σ)) which would technically we
        # minimizing the negative log likelihood of the Gaussian distro. For now we will use MSE with 
        if self.hparams.get("parameter_loss_confidence", False):
            # If confidence is provided, apply a penalty to the loss based on the confidence
            sigma:torch.Tensor = batch.get("hit_parameter_confidence", None)
            penalty = torch.log(1 + (sigma + 1e-6)**2) if sigma is not None else None  # Add a ssall constant to avoid log(0)
            assert sigma.shape == x.shape, f"Confidence shape must match hit_parameters shape\n\n{sigma.shape=}\n{x.shape=}" 
            assert penalty.shape == x.shape, f"Confidence shape must match hit_parameters shape\n\n{penalty.shape=}\n{x.shape=}"
            assert torch.min(penalty).item() > 0
            assert torch.min(sigma).item() > 0
            if sigma is not None:
                sigma = torch.maximum(sigma, torch.fill(torch.empty_like(sigma), 1e-6))
                res["uncertainty_ave"] = sigma.mean()
                res["uncertainty_std"] = sigma.std()
                res["penalty_ave"] = penalty.mean()
                res["penalty_std"] = penalty.std()
                if use_lambda:
                    res["penalty_lambda"] = self.lam
                    res["loss"] = self.lam * penalty + torch.square((y - x)) / sigma**2
                    if penalty.mean() < self.beta:
                        self.lam += self.dlambda
                    elif penalty.mean() > self.beta:
                        self.lam -= self.dlambda
                    self.lam = max(self.lam, self.dlambda)  # Ensure lambda is not zero or negative
                else:
                    res["loss"] = penalty + torch.square((y - x)) / sigma**2
        else:
            res["loss"] = torch.square(y - x) # fix mean over mean

                
            
        res["loss"] = res["loss"].mean()  # Average over all hits and devide by the number of hits
        return res