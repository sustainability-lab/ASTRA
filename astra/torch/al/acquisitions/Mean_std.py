import torch
from astra.torch.al.acquisitions.base import EnsembleAcquisition
from astra.torch.al.acquisitions.base import MCAcquisition
# Ensemble and MC strategy for producing different model parameters

    
# maximum mean standard deviation aquisition function
class MeanStd(EnsembleAcquisition,MCAcquisition):
    def acquire_scores(self, logits: torch.Tensor) -> torch.Tensor:
        # Mean-STD acquisition function
        # (n_nets/n_mc_samples, pool_dim, n_classes) logits shape
        assert len(logits.shape) == 3, "logits shape must be 3-Dimensional"
        pool_num = logits.shape[1]
        softmax_activation = torch.nn.Softmax(dim=2)
        prob = softmax_activation(logits)
        std = torch.std(prob,dim=0,unbiased=False)
        scores = torch.mean(std, dim=1) # mean over classes, shape (pool_dim)
        return scores
