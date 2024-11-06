import torch
from torch_geometric.data import Dataset

from .utils.utils import load_datafiles_in_dir, infer_num_nodes, apply_hard_cuts


class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(
        self,
        input_dir,
        data_name=None,
        num_events=None,
        stage="fit",
        hparams=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **kwargs,
    ):
        super().__init__(input_dir, transform, pre_transform, pre_filter)

        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage

        self.input_paths = load_datafiles_in_dir(
            self.input_dir, self.data_name, self.num_events
        )
        self.input_paths.sort()  # We sort here for reproducibility

    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event_path = self.input_paths[idx]
        event = torch.load(event_path, map_location=torch.device("cpu"))
        self.preprocess_event(event)

        return event

    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """

        infer_num_nodes(event)
        apply_hard_cuts(event, self.hparams, self.stage)
        if self.stage != "test":
            self.scale_features(event)
        if "hit_module_index" in event.keys():
            event["hit_module_id"] = event.pop("hit_module_index")

    def scale_features(self, event):
        """
        Handle feature scaling for the event
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys(), f"Feature {feature} not found in event"
                event[feature] = event[feature] / self.hparams["node_scales"][i]

    def unscale_features(self, event):
        """
        Unscale features when doing prediction
        """

        if (
            self.hparams is not None
            and "node_scales" in self.hparams.keys()
            and "node_features" in self.hparams.keys()
        ):
            assert isinstance(
                self.hparams["node_scales"], list
            ), "Feature scaling must be a list of ints or floats"
            for i, feature in enumerate(self.hparams["node_features"]):
                assert feature in event.keys(), f"Feature {feature} not found in event"
                event[feature] = event[feature] * self.hparams["node_scales"][i]
