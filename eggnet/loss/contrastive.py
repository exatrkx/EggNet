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
        print("DEBUG: Initialize Contrastive Loss Module")

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
            res["parameter_estimation_loss"] = self.parameter_estimation_loss(
                batch
            )["loss"]
        res["loss"] = res["signal_loss"] + res["knn_loss"] + res["random_loss"]
        
        if self.hparams.get("predict_track_parameters"):
            res["loss"] += res["parameter_estimation_loss"]
        # print("DEBUG: Contrastive Loss", res["loss"].item())
        return res


class WeightedContrastive(nn.Module):
    """
    Weighted contrastive loss
    """
    def __init__(self, hparams):
        super().__init__()
        print("DEBUG: Initialize Weighted Contrastive Loss Module")

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
            res["parameter_estimation_loss"] = self.parameter_estimation_loss(
                batch
            )["loss"]
        
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
        
    @time_function
    def forward(self, batch):
        res = {}
        x = batch.hit_parameters
        y = batch.get("hit_charge_pt_ratio") * 1000 # Scale up to make numbers nicer
        y = y.to(x.dtype).to(x.device)  # Ensure y has the same dtype as x. y is usually a float64/double tensor, while x is usually a float16 tensor
        # y = map_tracks_to_nodes(y, batch.track_edges).float().to(device=x.device) # BUG: Not quite the same shape, usually +/- 1-10 
        assert max(x.shape) == max(y.shape), f"Shape mismatch: {x.shape} vs {y.shape}.\n hit_size: {batch.hit_particle_id.shape[0]} track_size: {batch.track_particle_id.shape[0]}\n event_id: {batch.event_id}"
        res["loss"] = F.mse_loss(x, y, reduction="mean")
        return res