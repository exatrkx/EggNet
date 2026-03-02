import numpy as np
import torch
from typing import Dict
from .base_module import BaseModule
import os, sys
from . import utils
import pandas as pd
from tqdm import tqdm


class TrackBuildingStage(BaseModule):
    def __init__(self, hparams, get_logger=True):
        super().__init__()

    def build_tracks(self, dataset, data_name):
        """
        OVERWRITE THIS
        Build the track candidates using the track building algorithm. This is the only function that needs to be overwritten by the child class.
        """
        raise NotImplementedError("You must implement the \'build_tracks\' method!")