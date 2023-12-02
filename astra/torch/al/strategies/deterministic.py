import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al.strategies.base import Strategy
from astra.torch.al.acquisitions.base import DeterministicAcquisition
from astra.torch.al.errors import AcquisitionMismatchError

from typing import Sequence, Dict, Union


class DeterministicStrategy(Strategy):
    def __init__(
        self,
        acquisitions: Union[DeterministicAcquisition, Sequence[DeterministicAcquisition]],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """Base class for query strategies

        Args:
            acquisitions: A sequence of acquisition functions.
            inputs: A tensor of inputs.
            outputs: A tensor of outputs.
        """
        super().__init__(acquisitions, inputs, outputs)

        for name, acquisition in self.acquisitions.items():
            if not isinstance(acquisition, DeterministicAcquisition):
                raise AcquisitionMismatchError(DeterministicAcquisition, acquisition)

    def query(
        self,
        net: nn.Module,
        pool_indices: torch.Tensor,
        context_indices: torch.Tensor = None,
        n_query_samples: int = 1,
        n_mc_samples: int = 10,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo query strategy

        Args:
            net: A neural network with dropout layers.
            pool_indices: The indices of the pool set.
            context_indices: This argument is ignored.
            n_query_samples: Number of samples to query.
            n_mc_samples: This argument is ignored.
            batch_size: Batch size for the data loader.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
        assert isinstance(pool_indices, torch.Tensor), f"pool_indices must be a torch.Tensor, got {type(pool_indices)}"

        if batch_size is None:
            batch_size = len(pool_indices)

        data_loader = DataLoader(self.dataset[pool_indices])

        # put the model on eval mode
        net.eval()
        
        x = data_loader.dataset[0]
        y = data_loader.dataset[1]
        x = x.permute(0, 3, 1, 2)
        logits = net(x)  # (pool_dim, n_classes)
        
        best_indices = {}
        for acq_name, acquisition in self.acquisitions.items():
            scores = acquisition.acquire_scores(logits)
            index = torch.topk(scores, n_query_samples).indices
            selected_indices = pool_indices[index.cpu()]
            best_indices[acq_name] = selected_indices

        return best_indices
