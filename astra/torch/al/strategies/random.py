import torch
import torch.nn as nn
from astra.torch.al import Strategy

from typing import Sequence, Dict


class RandomStrategy(Strategy):
    def query(
        self,
        net: nn.Module,
        pool_indices: Sequence[int],
        context_indices: Sequence[int] = None,
        n_query_samples: int = 1,
        n_mc_samples: int = 10,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Random query strategy

        Args:
            net: This argument is ignored.
            pool_indices: The indices of the pool set.
            context_indices: This argument is ignored.
            n_query_samples: Number of samples to query.
            n_mc_samples: This argument is used to match the interface of other strategies.
            batch_size: This argument is ignored.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
        # logits shape (n_mc_samples, pool_dim, n_classes)
        logits = torch.rand(n_mc_samples, len(pool_indices), self.n_classes)
        best_indices = {}
        for acq_name, acquisition in self.acquisitions.items():
            scores = acquisition.acquire_scores(logits)
            selected_indices = torch.topk(scores, n_query_samples).indices
            best_indices[acq_name] = selected_indices

        return best_indices
