import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_scatter import scatter_add
import torch.nn as nn

import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

from torch.utils.checkpoint import checkpoint

from eggnet.utils import nearest_neighboring
from .utils.utils import *
from .utils.plotting import *
# https://gitlab.cern.ch/gnn4itkteam/acorn/-/tree/dev/acorn/stages/track_building/models?ref_type=heads

class Walkthrough(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.testmode = True
        self.hparams = hparams
        self.knn:nearest_neighboring.abstract_knn = getattr(
            nearest_neighboring, hparams.get("knn_algorithm", "cu_knn")
        )()
        self.gpu_available = torch.cuda.is_available()
        self.cc_only = self.hparams.get("cc_only", False)
        print(f"DEBUG: Walkthrough Module Initiated")
        
    
    def _build_tracks_one_evt(self, graph, output_dir):
        print(f"INFO: {type(graph)}=")
        # maybe sweep this value and make plot eff vs radius to find score_cut. DML from Acorn uses 0.14
        # use dynamic calculation? nudging the value to get to a target sparsity
        # filter_graph
        with torch.no_grad():
            if (self.hparams.get("filter_graph", True)):
                # remove edges between nodes that arn't "good" enough, i.e., the src and tgt embeddings aren't close enough
                filter_edges()
            if (self.hparams.get("remove_cycles", False)):
                remove_cycles(graph, edge_index_key="track_edges_filtered")
            simple_tracks, rest_subgraph = get_simple_path(graph)
            if not self.cc_only:
                walkthrough_tracks = walkthrough(
                    
                )
            
    
    #@ Override    
    def build_tracks(self, dataset, data_name):
                
        output_dir = os.path.join(self.hparams["stage_dir"], data_name)
        os.makedirs(output_dir, exist_ok=True)
        self.log.info(f"Saving tracks to {output_dir}")

        max_workers = (
            self.hparams["max_workers"] if "max_workers" in self.hparams else None
        )
        if self.testmode:
            max_workers = 1 # TYDO remove eventually
            print("DEBUG: OVERRIDING MAX_WORKERS = 1")
        if max_workers != 1:
            process_map(
                partial(self._build_tracks_one_evt, output_dir=output_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Reconstructing tracks for {data_name} data",
            )
        else:
            for event in tqdm(
                dataset, desc=f"Reconstructing tracks for {data_name} data"
            ):
                self._build_tracks_one_evt(event, output_dir=output_dir)
                
