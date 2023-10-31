import torch
from astra.torch.al.acquisitions.base import DiversityAcquisition
from distil.active_learning_strategies.core_set import CoreSet, Strategy


class Furthest(DiversityAcquisition):
    def acquire_scores(self, labeled_embeddings, unlabeled_embeddings, n):
        """
        Parameters
        ----------
            labeled_embeddings: tensor([n_train, embedding_size])
                Embedding of the train data
            unlabeled_embeddings: tensor([n_pool, embedding_size])
                Embedding of the pool data
            n: int
                Number of data points to be selected
        Returns:
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_embeddings
        """
        strategy = Strategy(X=-1, Y=-1, unlabeled_x=-1, net=-1, handler=-1, nclasses=-1)
        idxs = CoreSet.furthest_first(
            strategy, X_set=labeled_embeddings, X=unlabeled_embeddings, n=n
        )
        return idxs

