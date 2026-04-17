import os
import torch

from .base_module import BaseModule
from .utils.utils import cluster_eval, knn_eval
from eggnet.utils.timing import add_profile_time, profile_section
from time import perf_counter


class NodeEncoding(BaseModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.enable_profiling = self.hparams.get("enable_profiling", True)

    def training_step(self, batch, batch_idx):
        
        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch)
        elif self.hparams.get("double_metric_learning"):
            batch.src_embedding, batch.tgt_embedding = self(batch)
        else:
            batch.hit_embedding = self(batch)
        
        res = self.loss_fn(batch)
        self.log_dict(
            {f"train_{metric}": res[metric] for metric in self.hparams.get("train_metric", ["loss"])},
            batch_size=1,
        )

        return res["loss"]

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch)
        elif self.hparams.get("double_metric_learning"):
            batch.src_embedding, batch.tgt_embedding = self(batch) # tydo: Is this necessary? The batch field seems to already be assigned to in the forward pass
        else:
            batch.hit_embedding = self(batch)
        current_lr = self.optimizers().param_groups[0]["lr"]
        if self.hparams.get("no_cluster_eval") or self.hparams.get("double metric learning"): #TODO Fix
            # TYDO: Figure out what eff is and log and stuff with knn=1
            # eff, signal_eff, dup, fak = 0, 0, 0, 0 #bandaid
            # self.log_dict(
            #     {
            #         "lr": current_lr,
            #         "val_eff": eff,
            #         "val_signal_eff": signal_eff,
            #         "val_fak": fak,
            #         "val_dup": dup,
            #     },
            #     batch_size=1,
            #     sync_dist=True,
            # )
            # We want to output the graph sparsity 
            _, _, pur, _ = knn_eval(batch, self.hparams, k=1, ordering=False)
            
            
            self.log_dict(
                {"val_purity": pur,},
                batch_size=1,
                sync_dist=True,
            )
        else:
            eff, signal_eff, dup, fak = cluster_eval(batch, self.hparams)
            self.log_dict(
                {
                    "lr": current_lr,
                    "val_eff": eff,
                    "val_signal_eff": signal_eff,
                    "val_fak": fak,
                    "val_dup": dup,
                },
                batch_size=1,
                sync_dist=True,
            )
        # print("validation step end", torch.cuda.max_memory_allocated(device="cuda"))


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 0:
            return

        step_start = perf_counter()
        dataset = self.predict_dataloader()[dataloader_idx].dataset
        reuse_inference_output = bool(
            self.hparams.get("reuse_inference_output", False)
        )
        output_path = os.path.join(
            self.hparams["output_dir"],
            dataset.data_name,
            f"event{batch.event_id[0]}.pyg",
        )
        if os.path.isfile(output_path):
            if self.hparams.get("double_metric_learning"):
                if reuse_inference_output:
                    try:
                        existing_graph = torch.load(output_path, map_location="cpu")
                    except Exception:
                        print(
                            f"Unreadable output graph at {output_path}; "
                            "recomputing and overwriting it.",
                            flush=True,
                        )
                        try:
                            os.remove(output_path)
                        except FileNotFoundError:
                            pass
                    else:
                        if hasattr(existing_graph, "src_embedding") and hasattr(
                            existing_graph, "tgt_embedding"
                        ):
                            return 0
                        print(
                            f"Output graph at {output_path} is missing embeddings; "
                            "recomputing and overwriting it.",
                            flush=True,
                        )
            elif reuse_inference_output:
                return 0
        if self.hparams.get("node_filter"):
            with profile_section(
                batch,
                "predict_step.embed",
                enabled=self.enable_profiling,
            ):
                batch.hit_embedding, batch.filter_node_list = self(
                    batch, time_yes=self.enable_profiling
                )
        elif self.hparams.get("double_metric_learning"):
            with profile_section(
                batch,
                "predict_step.embed",
                enabled=self.enable_profiling,
            ):
                batch.src_embedding, batch.tgt_embedding = self(
                    batch, time_yes=self.enable_profiling
                ) # tydo: Is this necessary? The batch field seems to already be assigned to in the forward pass
        else:
            with profile_section(
                batch,
                "predict_step.embed",
                enabled=self.enable_profiling,
            ):
                batch.hit_embedding = self(batch, time_yes=self.enable_profiling)

        with profile_section(
            batch,
            "predict_step.unscale_features",
            enabled=self.enable_profiling,
        ):
            dataset.unscale_features(batch)
        if self.enable_profiling:
            add_profile_time(
                batch,
                "predict_step.total",
                perf_counter() - step_start,
            )

        with profile_section(
            batch,
            "predict_step.save_graph",
            enabled=self.enable_profiling,
        ):
            self.save_graph(batch, dataset.data_name)

        return 0
