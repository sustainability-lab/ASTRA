from astra.torch.al.acquisitions.base import MCAcquisition, EnsembleAcquisition
import torch
import torch.nn as nn

class BALDAcquisition(MCAcquisition, EnsembleAcquisition):
    """
    BALD acquisition function
    """

    def acquire_scores(self, logits: torch.Tensor):
        # logits shape (n_mc_samples, pool_dim, n_classes)
        if len(logits.shape) == 3:
            return self.bald_score(torch.mean(logits, dim=0))
        else:
            return self.bald_score(logits)

    def bald_score(self, logits: torch.Tensor):
        """
        Parameters: logits (pool_dim, n_classes)
        Returns: BALD score (pool_dim, )
        """
        # logits shape (pool_dim, n_classes)
        num_classes = logits.shape[1]
        log_softmax_logits = nn.LogSoftmax(dim=0)(logits)

        expected_entropy = -torch.sum(torch.mean(log_softmax_logits.exp(), dim=0) * log_softmax_logits.exp(), dim=1)
        predictive_entropy = torch.mean(-torch.sum(logits * log_softmax_logits.exp(), dim=1), dim=0)

        return expected_entropy - predictive_entropy

