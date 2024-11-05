import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import torch.nn as nn

# import pytorch_pfn_extras as ppe

from torch.utils.checkpoint import checkpoint

from eggnet.utils.nearest_neighboring import get_knn_graph
from .utils.utils import make_mlp


class EggNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # Construct the MLP architecture
        in_channels = len(hparams["node_features"])

        self.node_encoder = make_mlp(
            in_channels,
            [hparams["encoder_hidden"]] * (hparams["n_encoder_layers"] - 1)
            + [hparams["node_rep_dim"]],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        if hparams["n_iters"] > 0:
            self.edge_networks = nn.ModuleList(
                [
                    make_mlp(
                        (hparams["node_rep_dim"] * 2)
                        if i % hparams["n_gnns_per_iter"] == 0
                        else (hparams["node_rep_dim"] * 2 + hparams["edge_rep_dim"]),
                        [hparams["edge_hidden"]] * (hparams["n_edge_layers"] - 1)
                        + [hparams["edge_rep_dim"] + 1],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation=hparams["hidden_activation"],
                    )
                    for i in range(
                        (1 if hparams["recurrent"] else hparams["n_iters"])
                        * (
                            2
                            if hparams["recurrent_gnn"]
                            else hparams["n_gnns_per_iter"]
                        )
                    )
                ]
            )

        self.node_network_0 = make_mlp(
            hparams["node_rep_dim"],
            [hparams["node_0_hidden"]] * (hparams["n_node_0_layers"] - 1)
            + [hparams["node_rep_dim"]],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["hidden_activation"],
        )

        if hparams["n_iters"] > 0:
            self.node_networks = nn.ModuleList(
                [
                    make_mlp(
                        hparams["node_rep_dim"] + hparams["edge_rep_dim"],
                        [hparams["node_hidden"]] * (hparams["n_node_layers"] - 1)
                        + [hparams["node_rep_dim"]],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation=hparams["hidden_activation"],
                    )
                    for i in range(
                        (1 if hparams["recurrent"] else hparams["n_iters"])
                        * (
                            1
                            if hparams["recurrent_gnn"]
                            else hparams["n_gnns_per_iter"]
                        )
                    )
                ]
            )

        self.node_decoders = nn.ModuleList(
            [
                make_mlp(
                    hparams["node_rep_dim"],
                    [hparams["decoder_hiden"]] * (hparams["n_decoder_layers"] - 1)
                    + [hparams["node_pspace_dim"]],
                    layer_norm=hparams["layernorm"],
                    batch_norm=hparams["batchnorm"],
                    hidden_activation=hparams["hidden_activation"],
                    output_activation=hparams["output_activation"],
                )
                for i in range(1 if hparams["recurrent"] else (hparams["n_iters"] + 1))
            ]
        )

        if hparams.get("node_filter"):
            self.node_filters = nn.ModuleList(
                [
                    make_mlp(
                        hparams["node_rep_dim"],
                        [hparams["node_filter_hiden"]]
                        * (hparams["n_node_filter_layers"] - 1)
                        + [1],
                        layer_norm=hparams["layernorm"],
                        batch_norm=hparams["batchnorm"],
                        hidden_activation=hparams["hidden_activation"],
                        output_activation="Sigmoid",
                    )
                    for i in range(
                        1 if hparams["recurrent"] else (hparams["n_iters"] + 1)
                    )
                ]
            )

        # ppe.cuda.use_torch_mempool_in_cupy()

    def build_edges(self, batch, x, i, time_yes=False):
        x = self.node_decoders[0 if self.hparams["recurrent"] else i](x).detach()
        if self.hparams["embedding_norm"]:
            batch.hit_embedding = F.normalize(x)

        k = (
            self.hparams["knn_train"]
            if type(self.hparams["knn_train"]) is int
            else self.hparams["knn_train"][i]
        )

        return get_knn_graph(
            batch,
            k=k,
            time_yes=time_yes,
            r=self.hparams.get("r_max"),
            algorithm=self.hparams.get("knn_algorithm", "cu_knn"),
        )

    def forward(self, batch, time_yes=False, **kwargs):
        x = torch.stack(
            [batch[feature] for feature in self.hparams["node_features"]], dim=-1
        ).float()

        assert len(x) > 0, "Input node size == 0!!"

        if self.hparams.get("checkpoint", False):
            v = checkpoint(self.node_encoder, x, use_reentrant=False)
            x = checkpoint(self.node_network_0, v, use_reentrant=False)
        else:
            v = self.node_encoder(x)
            x = self.node_network_0(v)
        if self.hparams.get("node_filter"):
            filter_node_list = torch.arange(len(x), device=self.device)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_iters"]):
            # node filter
            if self.hparams.get("node_filter"):
                if self.hparams.get("checkpoint", False):
                    node_score = checkpoint(
                        self.node_filters[0 if self.hparams["recurrent"] else i],
                        x,
                        use_reentrant=False,
                    )
                else:
                    node_score = self.node_filters[
                        0 if self.hparams["recurrent"] else i
                    ](x)
                node_mask = (node_score > self.hparams["node_filter_cut"][i]).flatten()
                x = x[node_mask]
                v = v[node_mask]
                filter_node_list = filter_node_list[node_mask]
            # KNN
            if self.hparams.get("checkpoint", False):
                start, end = checkpoint(
                    self.build_edges,
                    batch,
                    x,
                    i,
                    use_reentrant=False,
                )
            else:
                start, end = self.build_edges(batch, x, i, time_yes=time_yes)
            if self.hparams.get("recycle_node_representation", True):
                x = v
            if self.hparams.get("checkpoint", False):
                x = checkpoint(self.gat, x, start, end, i, use_reentrant=False)
            else:
                x = self.gat(x, start, end, i)

        if self.hparams.get("checkpoint", False):
            x = checkpoint(self.node_decoders[-1], x, use_reentrant=False)
        else:
            x = self.node_decoders[-1](x)
        if self.hparams["embedding_norm"]:
            x = F.normalize(x)
        if self.hparams.get("node_filter"):
            return x, filter_node_list
        else:
            return x

    def gat(self, x, start, end, i):
        e = None
        for j in range(self.hparams["n_gnns_per_iter"]):
            x, e = self.message_passing(e, x, start, end, i, j)
        return x

    def message_passing(self, e, x, start, end, i, j):

        e = torch.cat([x[start], x[end]] if j == 0 else [x[start], x[end], e], dim=-1)

        e = self.edge_networks[
            (
                0
                if self.hparams["recurrent"]
                else (
                    i
                    * (
                        1
                        if self.hparams["recurrent_gnn"]
                        else self.hparams["n_gnns_per_iter"]
                    )
                )
            )
            + (min(1, j) if self.hparams["recurrent_gnn"] else j)
        ](e)
        w = e[:, -1:]
        w = softmax(w, end)
        e = e[:, :-1]

        # Node
        w = scatter_add(e * w, end, dim=0, dim_size=x.shape[0])
        # w = scatter_mean(e, end, dim=0, dim_size=x.shape[0])
        x = torch.cat([x, w], dim=1)
        x = self.node_networks[
            (
                0
                if self.hparams["recurrent"]
                else (
                    i
                    * (
                        1
                        if self.hparams["recurrent_gnn"]
                        else self.hparams["n_gnns_per_iter"]
                    )
                )
            )
            + (0 if self.hparams["recurrent_gnn"] else j)
        ](x)

        return x, e
