import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al import Strategy

from typing import Sequence, Dict


class MCStrategy(Strategy):
    def query(
        self,
        net: nn.Module,
        pool_indices: Sequence[int],
        context_indices: Sequence[int] = None,
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
            n_mc_samples: Number of Monte Carlo samples to draw from the model.
            batch_size: Batch size for the data loader.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
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
                vx = x[np.newaxis, ...].repeat(*repeats).to(self.device())
                logits = torch.vmap(net, randomness="different")(vx)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=1)  # (n_mc_samples, pool_dim, n_classes)

            best_indices = {}
            for acq_name, acquisition in self.acquisitions.items():
                scores = acquisition.acquire_scores(logits)
                selected_indices = torch.topk(scores, n_query_samples).indices
                best_indices[acq_name] = selected_indices

        return best_indices
