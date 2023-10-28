from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence, Dict

from astra.torch.al.acquisitions.base import Acquisition

# Base class for query strategies


class Strategy(ABC, nn.Module):
    def __init__(
        self,
        acquisitions: Sequence[Acquisition],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """Base class for query strategies

        Args:
            acquisitions: A sequence of acquisition functions.
            inputs: A tensor of inputs.
            outputs: A tensor of outputs.
        """
        super().__init__()
        assert isinstance(inputs, torch.Tensor), f"inputs must be a torch.Tensor, got {type(inputs)}"
        assert isinstance(outputs, torch.Tensor), f"outputs must be a torch.Tensor, got {type(outputs)}"

        self.inputs = inputs
        self.outputs = outputs
        self.n_classes = len(torch.unique(outputs))
        self.dataset = TensorDataset(self.inputs, self.outputs)
        self.dummy_param = nn.Parameter(torch.zeros(1))

        if not isinstance(acquisitions, Sequence):
            acquisitions = [acquisitions]
        self.acquisitions = {acq.__class__.__name__: acq for acq in acquisitions}

    @property
    def device(self):
        return self.dummy_param.device

    @abstractmethod
    def query(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass")
