import torch
from base import RandomAcquisition


class Random(RandomAcquisition):
    def acquire_scores(self, logits: torch.Tensor):
        # logits shape (n_mc_samples, pool_dim, n_classes)
        return torch.rand(logits.shape[1])
