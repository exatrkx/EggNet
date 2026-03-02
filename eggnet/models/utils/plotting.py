from .utils import *
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from eggnet.utils.plotting import plot_1d_histogram
from pathlib import Path
from typing import List
from .utils import *
from eggnet.utils import nearest_neighboring

def plot_distance_histogram(graph:Data, output_dir:Path|List[str], plot_suffix=None):
    """
    Distance between src and tgt embeddings for every true edge, 
    plotted by frequency (y axis)
    if "graph" is of type Data: runs on only the true edges
    else if "graph" is a list of edges
    """
    #TYDO: ALso plot fake edges
    if not isinstance(output_dir, Path): 
        output_dir = Path(output_dir)
    distances = get_edge_distances(graph)
    distance_values = distances.detach().cpu().numpy()
    hist, bins = np.histogram(distance_values, bins=100)
    err = np.zeros_like(hist)
    ymax = float(hist.max()) if hist.size else 0.0
    ylim = (0.0, ymax * 1.1 if ymax > 0.0 else 1.0)
    fig, ax = plot_1d_histogram(
        hist,
        bins,
        err,
        r"$L_2$ Distance",
        "Count",
        ylim,
        "Edge distance",
        logy=True,
        tightlayout=False,
    )
    for prop in [0.9, 0.95, 0.99, 0.999]:
        cutoff = _find_cutoff(prop, hist, bins)
        ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.0)
        ax.text(
            cutoff,
            0.95,
            f"{prop*100:0.1f}% @ {cutoff:0.2f}",
            rotation=90,
            va="top",
            ha="right",
            transform=ax.get_xaxis_transform(),
        )
    ax.set_title(f"Distance between embeddings for true edges")
    plt.tight_layout() # do it *now* **AFTER** we added the title 
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    output_path = output_dir.joinpath(f"distance_hist_{plot_suffix}.png" if plot_suffix else "distance_hist.png")
    fig.savefig(output_path)
    plt.clf()
    print("INFO: Saved distance plot to " + str(output_path))
        

def _find_cutoff(proportion:float, hist, bins):
    """
    Find the value such that proportion *100 percent of the values in 
    the histogram are less than or equal to the cutoff.
    """
    total = np.sum(hist)
    if total == 0:
        return bins[0] if len(bins) else 0.0
    if proportion <= 0.0:
        return bins[0]
    if proportion >= 1.0:
        return bins[-1]
    target = proportion * total
    idx = np.searchsorted(np.cumsum(hist), target, side="left")
    cutoff_idx = idx + 1
    if cutoff_idx >= len(bins):
        cutoff_idx = len(bins) - 1
    return bins[cutoff_idx]

def plot_cc(graph:Data, output_dir:Path|List[str], hparams:dict = {}, output_suffix=None,):
    """ Currently only works with Double Metric Learning Pipeline !!!"""
    if isinstance(output_dir, list):
        output_dir = Path(*output_dir)
    elif not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    knn_class:nearest_neighboring.abstract_knn = getattr(nearest_neighboring, hparams.get("knn_algorithm", "cu_knn"))()
    graph = graph.to("cuda")
    track_edges = knn_class.get_graph(
        graph,
        1,
        use_double_metric_learning=True,
    )
    distances = get_edge_distances(graph, edges = track_edges)
    distances_cpu = distances.detach().cpu()
    distance_values = distances_cpu.numpy()
    hist, bins = np.histogram(distance_values, bins=100)
    cutoffs = [_find_cutoff(prop, hist, bins) for prop in [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]]
    n_components = []
    if track_edges.numel() == 0:
        n_components = [0] * len(cutoffs)
    else:
        for cutoff in cutoffs:
            mask = distances_cpu < cutoff
            if not mask.any().item():
                n_components.append(0)
                continue
            edge_index = track_edges[:, mask]
            if edge_index.numel() == 0:
                n_components.append(0)
                continue
            nodes, inv = torch.unique(edge_index, return_inverse=True)
            edge_index = inv.view(edge_index.shape)
            adj_matrix = to_scipy_sparse_matrix(
                edge_index, num_nodes=int(nodes.numel())
            )
            n_comp, _ = connected_components(
                csgraph=adj_matrix, directed=True, connection="weak"
            )
            n_components.append(int(n_comp))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cutoffs, n_components, color="black", linestyle="-")
    ax.set_xlabel("Cutoff", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Connected components", ha="right", y=0.95, fontsize=14)
    ax.set_title("Connected components vs cutoff")
    plt.tight_layout()
    output_path = output_dir.joinpath(f"sparsity_hist_{output_suffix}.png" if output_suffix else "distance_hist.png")
    fig.savefig(output_path)
    print("INFO: Saved sparsity plot to " + str(output_path))
    plt.clf() # good manners

# TYDO: Plot Efficiency vs Cutoff