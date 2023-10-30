import torch
from astra.torch.al.aquisition.base import EnsembleAcquisition
from astra.torch.al.aquisition.base import MCAcquisition
# Ensemble and MC strategy for producing different model parameters

    
# maximum mean standard deviation aquisition function
class Mean_std(EnsembleAcquisition,MCAcquisition):
    def acquire_scores(self, logits: torch.Tensor) -> torch.Tensor:
        # Mean-STD acquisition function
        # (n_nets/n_mc_samples, pool_dim, n_classes) logits shape
        pool_num = logits.shape[1]
        assert len(logits.shape) == 3, "logits shape must be 3-Dimensional"
        std = torch.std(logits, dim=0) # standard deviation over model parameters, shape (pool_dim, n_classes)
        scores = torch.mean(std, dim=1) # mean over classes, shape (pool_dim)
        assert len(scores.shape) == 1 and scores.shape[0]==pool_num, "scores shape must be 1-Dimensional and must have length equal to that of pool dataset"
        return scores
