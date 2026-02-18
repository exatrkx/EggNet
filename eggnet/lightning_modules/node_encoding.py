import os

from .base_module import BaseModule
from .utils.utils import cluster_eval, knn_eval, knn_eval_from_graph, get_knn_graph


class NodeEncoding(BaseModule):
    def __init__(self, hparams):
        super().__init__(hparams)

    def training_step(self, batch, batch_idx):
        
        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch)
        elif self.hparams.get("double_metric_learning"):
            batch.src_embedding, batch.tgt_embedding = self(batch)
        else:
            batch.hit_embedding = self(batch)
        
        res = self.loss_fn(batch)
        if self.hparams.get("double_metric_learning"):
            # we use new prefixes that need to be logged correctly. 
            for prefix in ["tgt", "src"]:
                self.log_dict(
                    {f"{prefix}_train_{metric}": res[prefix+'_'+metric] for metric in self.hparams.get("train_metric", ["loss"])},
                    batch_size=1,
                )
        else:
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
            _, _, pur, _ = knn_eval(batch, self.hparams)
            
            self.log_dict(
                {"purity": pur,},
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

        return eff

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 0:
            return

        dataset = self.predict_dataloader()[dataloader_idx].dataset
        if os.path.isfile(
            os.path.join(
                self.hparams["output_dir"],
                dataset.data_name,
                f"event{batch.event_id[0]}.pyg",
            )
        ):
            return 0

        if self.hparams.get("node_filter"):
            batch.hit_embedding, batch.filter_node_list = self(batch, time_yes=True)
        else:
            batch.hit_embedding = self(batch, time_yes=True)

        dataset.unscale_features(batch)

        self.save_graph(batch, dataset.data_name)

        return 0
