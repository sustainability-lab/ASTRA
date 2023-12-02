from astra.torch.al.acquisitions.entropy import EntropyAcquisition
from astra.torch.al.strategies.deterministic import DeterministicStrategy
from astra.torch.al.strategies.ensemble import EnsembleStrategy
from astra.torch.al.strategies.mc import MCStrategy
import pytest

import torch
from torchvision.datasets import CIFAR10

from astra.torch.models import CNN

# from astra.torch.al import UniformRandomAcquisition, RandomStrategy, EnsembleAcquisition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_entropy():
    data = CIFAR10(root="data", download=True, train=False)  # "test" for less data
    inputs = torch.tensor(data.data).float().to(device)
    outputs = torch.tensor(data.targets).long().to(device)

    # Meta parameters
    n_pool = 1000
    indices = torch.randperm(len(inputs))
    pool_indices = indices[:n_pool]
    train_indices = indices[n_pool:]
    n_query_samples = 10

    # # Define the acquisition function
    acquisition = EntropyAcquisition()
    strategy = DeterministicStrategy(acquisition, inputs, outputs)

    # Put the strategy on the device
    strategy.to(device)

    # Define the model
    net = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)

    # Query the strategy
    best_indices = strategy.query(net, pool_indices, n_query_samples=n_query_samples)
    print(best_indices)

    assert best_indices["EntropyAcquisition"].shape == (n_query_samples,)

    # For Ensemble Strategy
    strategy = EnsembleStrategy(acquisition, inputs, outputs)

    # Put the strategy on the device
    strategy.to(device)

    # Define the model
    net1 = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)
    net2 = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)
    net = [net1, net2]

    best_indices = strategy.query(net, pool_indices, n_query_samples=n_query_samples)
    print(best_indices)

    assert best_indices["EntropyAcquisition"].shape == (n_query_samples,)

    # # For MC Strategy
    strategy = MCStrategy(acquisition, inputs, outputs)
    strategy.to(device)
    # # Define the model
    net = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)
    best_indices = strategy.query(net, pool_indices, n_query_samples=n_query_samples)
    print(best_indices)
    assert best_indices["EntropyAcquisition"].shape == (n_query_samples,)
test_entropy()