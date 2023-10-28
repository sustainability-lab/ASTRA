import torch 
from base import EnsembleAcquisition, MCAcquisition

class MaxEntropy(EnsembleAcquisition, MCAcquisition):
    def acquire_scores(self, logits: torch.Tensor):
        
        #this is ensemble strategy


        if isinstance(self,EnsembleAcquisition):
            #calculate entropy for each pool datapoint for each model
            entropy=-torch.sum(logits*torch.log(logits),dim=2)

            score=torch.sum(entropy,dim=0)

            return score
        
        elif isinstance(self,MCAcquisition):
            #calculate entropy for each pool datapoint accross all Monte Carlo samples  
            entropy=-torch.sum(logits*torch.log(logits),dim=2)

            score=torch.mean(entropy,dim=0)

            return score
        
        else:
            raise NotImplementedError("Unknown acquisition strategy")
      