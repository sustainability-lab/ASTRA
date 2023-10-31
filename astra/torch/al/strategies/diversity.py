import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astra.torch.al.strategies.base import Strategy
from astra.torch.al.acquisitions.base import DiversityAcquisition
from astra.torch.al.errors import AcquisitionMismatchError

from typing import Sequence, Dict, Union


class DiversityStrategy(Strategy):
    def __init__(
        self,
        acquisitions: Union[DiversityAcquisition, Sequence[DiversityAcquisition]],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """Diversity query strategy to maximize the diversity of the selected samples.

        Args:
            acquisitions: A sequence of acquisition functions.
            inputs: A tensor of inputs.
            outputs: A tensor of outputs.
        """
        super().__init__(acquisitions, inputs, outputs)

        for name, acquisition in self.acquisitions.items():
            if not isinstance(acquisition, DiversityAcquisition):
                raise AcquisitionMismatchError(DiversityAcquisition, acquisition)

    def query(
        self,
        net: nn.Module,
        pool_indices: torch.Tensor,
        context_indices: torch.Tensor = None,
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
        assert isinstance(pool_indices, torch.Tensor), f"pool_indices must be a torch.Tensor, got {type(pool_indices)}"
        assert isinstance(
            context_indices, torch.Tensor
        ), f"context_indices must be a torch.Tensor, got {type(context_indices)}"

        if batch_size is None:
            batch_size = len(pool_indices)

        data_loader = DataLoader(self.dataset)

        # put model on eval mode
        net.eval()

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

            if not isinstance(pool_indices, list):
                # tolist() works for both numpy and torch tensors. It also work for tensors on GPU.
                pool_indices = pool_indices.tolist()
            if not isinstance(context_indices, list):
                context_indices = context_indices.tolist()

            for acq_name, acquisition in self.acquisitions.items():
                selected_indices = acquisition.acquire_scores(context_features, pool_features, n_query_samples)
                selected_indices = torch.tensor(selected_indices)#, device=self.device)
                best_indices[acq_name] = selected_indices
        return best_indices
