import torch 
from base import EnsembleAcquisition, MCAcquisition

class MaxEntropy(EnsembleAcquisition, MCAcquisition):
    def acquire_scores(self, logits: torch.Tensor):
        
        #calculate entropy for each pool datapoint for each model
        probs=torch.softmax(logits,dim=2)
        entropy=-torch.sum(probs*torch.log(probs),dim=2)


        score=torch.sum(entropy,dim=0)

        return score

      