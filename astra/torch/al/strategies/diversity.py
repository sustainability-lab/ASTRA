import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al import Strategy

from typing import Sequence, Dict


class DiversityStrategy(Strategy):
    def query(
        self,
        net: nn.Module,
        pool_indices: Sequence[int],
        context_indices: Sequence[int] = None,
        n_query_samples: int = 1,
        n_mc_samples: int = None,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Diversity query strategy with multiple neural networks

        Args:
            net: A neural network to extract features.
            pool_indices: The indices of the pool set.
            context_indices: The indices of the context set which is used to compute the diversity.
            n_query_samples: Number of samples to query.
            n_mc_samples: This argument is ignored.
            batch_size: Batch size for the data loader.

        Returns:
            best_indices: A dictionary of acquisition names and the corresponding best indices.
        """
        if batch_size is None:
            batch_size = len(pool_indices)

        data_loader = DataLoader(self.dataset[pool_indices])
        context_data_loader = DataLoader(self.dataset[context_indices])

        with torch.no_grad():
            # Get the features for the pool
            x_pool, y_pool = data_loader.dataset
            x_pool = x_pool.permute(0,3,1,2)
            pool_features = net(x_pool).cpu() # (pool_dim, feature_dim)

            # Get the features for the context
            x_context, y_context = context_data_loader.dataset
            x_context = x_context[0]
            y_context = y_context[0]
            x_context = x_context.permute(0,3,1,2)            
            context_features = net(x_context).cpu() # (context_dim, feature_dim)

            best_indices = {}
            for acq_name, acquisition in self.acquisitions.items():
                selected_indices = acquisition.acquire_scores(context_features, pool_features, n_query_samples)
                selected_indices = torch.tensor(selected_indices)#, device=self.device)
                best_indices[acq_name] = selected_indices
        return best_indices
