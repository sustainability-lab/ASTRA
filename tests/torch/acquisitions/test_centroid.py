import torch
from torchvision.datasets import CIFAR10

from astra.torch.models import CNNClassifier
from astra.torch.al import Centroid, DiversityStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_centroid():
    data = CIFAR10(root="data", download=True, train=False)  # "test" for less data
    inputs = torch.tensor(data.data).permute(0, 3, 1, 2).float().to(device)
    outputs = torch.tensor(data.targets).long().to(device)

    # Meta parameters
    n_pool = 1000
    indices = torch.randperm(len(inputs))
    pool_indices = indices[:n_pool]
    train_indices = indices[n_pool:]
    n_query_samples = 10

    # Define the acquisition function
    acquisition = Centroid()
    strategy = DiversityStrategy(acquisition, inputs, outputs)
    
    # Put the strategy on the device
    strategy.to(device)
    
    # Define the model
    net = CNNClassifier(32, 3, 3, [4, 8], [2, 3], 10).to(device)

    # Feature extractor callable from the network
    feature_extractor = net.featurizer

    # Query the strategy
    best_indices = strategy.query(
        feature_extractor, pool_indices, train_indices, n_query_samples=n_query_samples
    )

    assert best_indices["Centroid"].shape == (n_query_samples,)

# test_centroid()