import torch
from torch_geometric.nn import knn_graph, radius_graph
from cuml.neighbors import NearestNeighbors
import cupy

from eggnet.utils.timing import time_function


def cu_knn_graph(x, k, loop=False, cosine=False, r=None):
    if not loop:
        k += 1
    with cupy.cuda.Device(x.device.index):
        x_cu = cupy.from_dlpack(x.detach())
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(x_cu)
        d, graph_idxs = knn.kneighbors(x_cu)
        graph_idxs = torch.from_dlpack(graph_idxs)
        if r:
            d = torch.from_dlpack(d)
    ind = (
        torch.arange(graph_idxs.shape[0], device=x.device)
        .unsqueeze(1)
        .expand(graph_idxs.shape)
    )
    graph = torch.stack([graph_idxs.flatten(), ind.flatten()], dim=0)
    if r:
        graph = graph[:, d.flatten() <= r]
    if not loop:
        return graph[:, graph[0] != graph[1]]
    else:
        return graph


@time_function
def get_knn_graph(batch, k, r=None, algorithm="cu_knn", node_filter=False):

    if algorithm == "cu_knn":
        edges = cu_knn_graph(batch.hit_embedding.detach(), k=k, cosine=False, loop=False, r=r)
    elif algorithm == "radius_graph":
        edges = radius_graph(batch.hit_embedding.detach(), r=r, loop=False, max_num_neighbors=k)
    else:
        edges = knn_graph(batch.hit_embedding.detach(), k=k, cosine=False, loop=False)

    return edges
