import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.utils import hinge_loss
from eggnet.utils.timing import time_function
from eggnet.utils.mapping import get_node_weight
from eggnet.utils import nearest_neighboring
from .contrastive import *

# from https://gitlab.cern.ch/gnn4itkteam/acorn/-/blob/dev/acorn/stages/track_building/models/walkthrough.py
import os
import logging
import pandas as pd
import torch
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils import to_networkx
import networkx as nx
from itertools import chain
from functools import partial


class DML_Loss(nn.Module):
    """
    Deep Metric Learning loss
    """
    def __init__(self, hparams):
        super().__init__()
        print("DEBUG: DML Loss module initiated")

        self.hparams = hparams
        self.contrastive = Contrastive(hparams)
        # In theory, we only really need to apply the Contrastive loss to both embeddings
    @staticmethod
    def find_all_paths(start, G=None, ending_nodes=None):
        return list(
            chain.from_iterable(
                [
                    list(nx.all_simple_paths(G, start, end))
                    for end in ending_nodes
                    if nx.has_path(G, start, end)
                ]
            )
        )

    @staticmethod
    def find_shortest_paths(start, G=None, ending_nodes=None):
        return [
            nx.shortest_path(G, start, end)
            for end in ending_nodes
            if nx.has_path(G, start, end)
        ]
        
    def build_tracks(self, dataset, data_name):
        """
        Given a set of scored graphs, and a score cut, build tracks from graphs by:
        1. Applying the score cut to the graph
        2. Running walkthrough path method as a partial method

        """

        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving tracks to {output_dir}")

        for graph in tqdm(dataset):
            # Apply score cut
            edge_mask = graph.edge_scores > self.hparams["score_cut"]

            # Convert to sparse scipy array
            new_graph = graph.clone()
            new_graph.edge_index = new_graph.edge_index[:, edge_mask]
            new_graph.edge_scores = new_graph.edge_scores[edge_mask]

            # Convert to networkx graph
            G = to_networkx(new_graph, to_undirected=False)
            G.remove_nodes_from(list(nx.isolates(G)))

            starting_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            ending_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

            workers = 32  # TODO: Remove this hardcoded value

            # Make partial method for multiprocessing
            find_paths_partial = partial(
                self.find_shortest_paths, G=G, ending_nodes=ending_nodes
            )

            # Run multiprocessing
            with Pool(workers) as p:
                paths = list(p.map(find_paths_partial, starting_nodes))

            track_df = pd.DataFrame(
                {
                    "hit_id": list(chain.from_iterable(paths)),
                    "track_id": list(
                        chain.from_iterable([[i] * len(p) for i, p in enumerate(paths)])
                    ),
                }
            )

            # Remove duplicates on hit_id: TODO: In very near future, handle multiple tracks through the same hit!
            track_df = track_df.drop_duplicates(subset="hit_id")

            hit_id = track_df.hit_id
            track_id = track_df.track_id

            track_id_tensor = torch.ones(len(graph.hit_x), dtype=torch.long) * -1
            track_id_tensor[hit_id.values] = torch.from_numpy(track_id.values)

            graph.hit_track_labels = track_id_tensor
            # TYDO what is this?
            # if not self.hparams.get("variable_with_prefix"):
            #     graph = remove_variable_name_prefix_in_pyg(graph)
            torch.save(graph, os.path.join(output_dir, f"event{graph.event_id[0]}.pyg"))


    @time_function
    def forward(self, batch):

        res = {}
        tgt_dict:dict = self.contrastive(batch) #tydo: this doesn't care about which embedding we pick, always defaults to 'hit_embedding'
        src_dict:dict = self.contrastive(batch)
        tgt_dict = {'tgt_' + key: value for key, value in tgt_dict.items()}
        src_dict = {'src_' + key: value for key, value in src_dict.items()}
        res.update(tgt_dict)
        res.update(src_dict)
        res["loss"] = tgt_dict["tgt_loss"] + src_dict["src_loss"]

        return res