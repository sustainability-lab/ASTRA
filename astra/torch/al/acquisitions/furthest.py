import torch
from astra.torch.al.acquisitions.base import DiversityAcquisition


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
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.full((unlabeled_embeddings.shape[0],), float("inf"))
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]
        idxs = []
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(
                unlabeled_embeddings, unlabeled_embeddings[[idx], :]
            )
            min_dist = torch.minimum(min_dist, dist_new_ctr[:, 0])
        return idxs
