import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al.strategies.base import Strategy
from astra.torch.al.acquisitions.base import MCAcquisition
from astra.torch.al.errors import AcquisitionMismatchError

from typing import Sequence, Dict, Union


class MCStrategy(Strategy):
    def __init__(
        self,
        acquisitions: Union[MCAcquisition, Sequence[MCAcquisition]],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """Monte Carlo Dropout query strategy. See https://arxiv.org/abs/1506.02142 (Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning)

        Args:
            acquisitions: A sequence of acquisition functions.
            inputs: A tensor of inputs.
            outputs: A tensor of outputs.
        """
        super().__init__(acquisitions, inputs, outputs)

        for name, acquisition in self.acquisitions.items():
            if not isinstance(acquisition, MCAcquisition):
                raise AcquisitionMismatchError(MCAcquisition, acquisition)

    def query(
        self,
        net: nn.Module,
        pool_indices: torch.Tensor,
        context_indices: torch.Tensor = None,
        n_query_samples: int = 1,
        n_mc_samples: int = 10,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            net: A neural network with dropout layers.
            pool_indices: The indices of the pool set.
            context_indices: This argument is ignored.
            n_query_samples: Number of samples to query.
            n_mc_samples: Number of Monte Carlo samples to draw from the model.
            batch_size: Batch size for the data loader.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
        assert isinstance(pool_indices, torch.Tensor), f"pool_indices must be a torch.Tensor, got {type(pool_indices)}"

        if batch_size is None:
            batch_size = len(pool_indices)

        # Set n_repeats for x for vmap later
        repeats = [1] * (len(self.dataset.tensor[0].shape) + 1)  # +1 for mc dimension
        repeats[0] = n_mc_samples

        data_loader = DataLoader(self.dataset[pool_indices])

        # Put the model on train mode to enable dropout
        net.train()

        with torch.no_grad():
            logits_list = []
            for x, _ in data_loader:
                vx = x[np.newaxis, ...].repeat(*repeats).to(self.device)
                logits = torch.vmap(net, randomness="different")(vx)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=1)  # (n_mc_samples, pool_dim, n_classes)

            best_indices = {}
            for acq_name, acquisition in self.acquisitions.items():
                scores = acquisition.acquire_scores(logits)
                index = torch.topk(scores, n_query_samples).indices
                selected_indices = pool_indices[index]
                best_indices[acq_name] = selected_indices

        return best_indices
