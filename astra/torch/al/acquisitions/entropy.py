import torch
from astra.torch.al import MCAcquisition, EnsembleAcquisition, DeterministicAcquisition   

import torch.nn.functional as F


class EntropyAcquisition(MCAcquisition, EnsembleAcquisition, DeterministicAcquisition):
    def acquire_scores(self, logits: torch.Tensor):
        """for McAcquisition, EnsembleAcquisition"""
        if logits.dim() == 3:
            log_probs = F.log_softmax(logits, dim=2)

            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=2)

        else:
            """for DeterministicAcquisition"""

            log_probs = F.log_softmax(logits, dim=1)
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1)
        return torch.sum(entropy, dim=0)
