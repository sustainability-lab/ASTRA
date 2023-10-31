import torch
from astra.torch.al.acquisitions.base import RandomAcquisition


class UniformRandomAcquisition(RandomAcquisition):
    def acquire_scores(self, pool_indices: torch.Tensor):
        # shape (pool_dim, )
        return torch.rand_like(pool_indices.float())
