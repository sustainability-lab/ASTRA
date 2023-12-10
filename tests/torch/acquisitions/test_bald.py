import pytest

import torch
from torchvision.datasets import CIFAR10

from astra.torch.models import CNN
from astra.torch.al import EnsembleAcquisition, MCAcquisition, MCStrategy, RandomAcquisition
from astra.torch.al.acquisitions.bald import BALDAcquisition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_bald():
    data = CIFAR10(root="data", download=True, train=False)  # "test" for less data
    inputs = torch.tensor(data.data).permute(0,3,1,2).float().to(device)
    outputs = torch.tensor(data.targets).long().to(device)

    # Meta parameters
    n_pool = 1000
    indices = torch.randperm(len(inputs))
    pool_indices = indices[:n_pool]
    train_indices = indices[n_pool:]
    n_query_samples = 10

    # Define the acquisition function
    acquisition = BALDAcquisition()
    strategy = MCStrategy(acquisition, inputs, outputs)

    # Put the strategy on the device
    strategy.to(device)

    # Define the model
    net = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)

    # Query the strategy
    best_indices = strategy.query(net, pool_indices, n_query_samples=n_query_samples)
    assert best_indices["BALDAcquisition"].shape == (n_query_samples,)