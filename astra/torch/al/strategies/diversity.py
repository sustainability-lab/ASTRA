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
        """Ensemble query strategy with multiple neural networks

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
            pool_features_list = []
            for x, _ in data_loader:
                pool_features = net(x)
                pool_features_list.append(pool_features)
            pool_features = torch.cat(pool_features_list, dim=0)  # (pool_dim, feature_dim)

            # Get the features for the context
            context_features_list = []
            for x, _ in context_data_loader:
                context_features = net(x)
                context_features_list.append(context_features)
            context_features = torch.cat(context_features_list, dim=0)  # (context_dim, feature_dim)

            best_indices = {}
            for acq_name, acquisition in self.acquisitions.items():
                scores = acquisition.acquire_scores(pool_features, context_features)
                selected_indices = torch.topk(scores, n_query_samples).indices
                best_indices[acq_name] = selected_indices

        return best_indices
