import torch
from torch_geometric.nn import knn_graph, radius_graph, radius
from cuml.neighbors import NearestNeighbors
import cupy
import faiss
from abc import abstractmethod, ABC
from threading import local

from eggnet.utils.timing import time_function

try:
    import frnn

    FRNN_AVAILABLE = True
except ImportError:
    frnn = None
    FRNN_AVAILABLE = False

if not torch.cuda.is_available():
    FRNN_AVAILABLE = False


def build_edges(
    query,
    database,
    indices=None,
    r_max=1.0,
    k_max=10,
    return_indices=False,
    backend="FRNN",
):
    """
    Build bipartite edges from `query` to `database`.

    The returned edge list follows the ACORN convention: row 0 indexes the
    query points and row 1 indexes the database points.
    """
    use_frnn = (
        str(backend).upper() == "FRNN"
        and FRNN_AVAILABLE
        and query.device.type == "cuda"
        and database.device.type == "cuda"
    )

    if use_frnn:
        dists, idxs, _, _ = frnn.frnn_grid_points(
            points1=query.unsqueeze(0),
            points2=database.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )
        idxs = idxs.squeeze(0).int()
        ind = (
            torch.arange(idxs.shape[0], device=query.device)
            .repeat(idxs.shape[1], 1)
            .T.int()
        )
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    else:
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (
        (edge_list, dists, idxs, ind)
        if (return_indices and use_frnn)
        else edge_list
    )


def cu_knn_graph(x, k, loop=False, cosine=False, r=None, use_double_metric_learning=False):
    """
    An function equivalent to knn_graph but based on cuml.neighbors.NearestNeighbors with GPU implementation.
    In addition, the function supports specification of a max radius (the radius cut is applied after k neighbors are found).
    """
    if not loop:
        k += 1
    with cupy.cuda.Device(x.device.index):
        knn = NearestNeighbors(n_neighbors=k)
        x_cu = cupy.from_dlpack(x.detach())
        if use_double_metric_learning:
            src = x_cu[0]
            tgt = x_cu[1]
            knn.fit(src)
            d, graph_idxs = knn.kneighbors(tgt)
        else: 
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
    if not loop: #self looping
        return graph[:, graph[0] != graph[1]]
    else:
        return graph

class abstract_knn(ABC):
    @abstractmethod
    @time_function
    def get_graph(self, batch, 
                  k, r=None, node_filter=False, 
                  loop=False, use_double_metric_learning=False
                  ) -> torch.Tensor:
        ...
        
class cu_knn(abstract_knn):
    """
    cuml.knn graph
    """

    def __init__(self):
        self._thread_local = local()

    def _get_knn(self):
        knn = getattr(self._thread_local, "knn", None)
        if knn is None:
            knn = NearestNeighbors()
            self._thread_local.knn = knn
        return knn

    @time_function
    def get_graph(self, batch, k, r=None, node_filter=False, loop=False, use_double_metric_learning=False):
        assert k is not None
        if not loop:
            k += 1
        knn = self._get_knn()
        if use_double_metric_learning:
            dml_device = batch.tgt_embedding.device  
            with cupy.cuda.Device(dml_device.index): 
                tgt_cu = cupy.from_dlpack(batch.tgt_embedding.detach()) #TYDO somehow this throws an error since tg
                src_cu = cupy.from_dlpack(batch.src_embedding.detach())
                knn.fit(src_cu)
                d, graph_idxs = knn.kneighbors(tgt_cu)
                graph_idxs = torch.from_dlpack(graph_idxs)
        else:
            # assert(0), "Code not supposed to reach here"
            with cupy.cuda.Device(batch.hit_embedding.device.index):
                x_cu = cupy.from_dlpack(batch.hit_embedding.detach())
                knn.fit(x_cu)
                d, graph_idxs = knn.kneighbors(x_cu)
                graph_idxs = torch.from_dlpack(graph_idxs)
        if r:
            d = torch.from_dlpack(d)
        device = dml_device if use_double_metric_learning else batch.hit_embedding.device
        ind = (
            torch.arange(graph_idxs.shape[0], device=device)
            .unsqueeze(1)
            .expand(graph_idxs.shape)
        )
        graph = torch.stack([graph_idxs.flatten(), ind.flatten()], dim=0) # TYDO: Broken -- Type error expected Tensor as element 0 in arg 0, but got ndarray
        if r:
            graph = graph[:, d.flatten() <= r]
        if not loop: # self-loop
            return graph[:, graph[0] != graph[1]]
        else:
            return graph


class faiss_knn(abstract_knn):
    """
    cuml.knn graph
    """

    def __init__(self):
        self.faiss_res = faiss.StandardGpuResources()
        self.flat_config = faiss.GpuIndexFlatConfig()

    @time_function
    def get_graph(self, batch, k, r=None, node_filter=False, loop=False, use_double_metric_learning=False):
        if use_double_metric_learning:
            raise NotImplementedError("faiss_knn does not support use_double_metric_learning yet.")
        if not loop:
            k += 1
        self.flat_config.device = batch.hit_embedding.device.index
        gpu_index = faiss.GpuIndexFlatL2(self.faiss_res, batch.hit_embedding.shape[1], self.flat_config)
        embedding_array = batch.hit_embedding.detach().cpu().numpy()
        gpu_index.add(embedding_array)  # add vectors to the index - has to be np.array
        d, graph_idxs = gpu_index.search(embedding_array, k)  # has to be np.array
        graph_idxs = torch.from_numpy(graph_idxs).to(batch.hit_embedding.device)
        ind = (
            torch.arange(graph_idxs.shape[0], device=batch.hit_embedding.device)
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


class pyg_rnn:
    """
    pytorch_geoemtric radius graph
    """

    def __init__(self):
        return

    @time_function
    def get_graph(self, batch, k, r, node_filter=False, loop=False):
        return radius_graph(batch.hit_embedding, r=r, loop=loop, max_num_neighbors=k)


class pyg_knn:
    """
    pytorch_geoemtric radius graph
    """

    def __init__(self):
        return

    @time_function
    def get_graph(self, batch, k, r=None, node_filter=False, loop=False):
        return knn_graph(batch.hit_embedding, k=k, cosine=False, loop=loop)
