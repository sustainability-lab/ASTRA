import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al import Strategy

from typing import Sequence, Dict


class EnsembleStrategy(Strategy):
    def query(
        self,
        net: Sequence[nn.Module],
        pool_indices: Sequence[int],
        context_indices: Sequence[int] = None,
        n_query_samples: int = 1,
        n_mc_samples: int = None,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Ensemble query strategy with multiple neural networks

        Args:
            net: A sequence of neural networks.
            pool_indices: The indices of the pool set.
            context_indices: This argument is ignored.
            n_query_samples: Number of samples to query.
            n_mc_samples: This argument is ignored.
            batch_size: Batch size for the data loader.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
        if not isinstance(net, Sequence):
            raise ValueError(f"net must be a sequence of nets, got {type(net)}")

        if batch_size is None:
            batch_size = len(pool_indices)

        data_loader = DataLoader(self.dataset[pool_indices])

        with torch.no_grad():
            logits_list = []
            for x, _ in data_loader:
                net_logits_list = []
                for model in net:
                    net_logits = model(x.to(self.device()))[np.newaxis, ...]
                    net_logits_list.append(net_logits)
                logits = torch.cat(net_logits_list, dim=0)  # (n_nets, batch_dim, n_classes)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=1)  # (n_nets, pool_dim, n_classes)

            best_indices = {}
            for acq_name, acquisition in self.acquisitions.items():
                scores = acquisition.acquire_scores(logits)
                selected_indices = torch.topk(scores, n_query_samples).indices
                best_indices[acq_name] = selected_indices

        return best_indices
