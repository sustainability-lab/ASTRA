import pytest
import torch
from torchvision.datasets import CIFAR10
from astra.torch.models import CNN
from astra.torch.al import (
    EntropyAcquisition,
    MCAcquisition,
    MCStrategy,
    EnsembleStrategy,
    DeterministicStrategy
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_entropy():
    data = CIFAR10(root="data", download=True, train=False)  # "test" for less data
    inputs = torch.tensor(data.data).float().to(device)
    outputs = torch.tensor(data.targets).long().to(device)
    
    #Meta parameters
    n_pool = 1000
    indices = torch.randperm(len(inputs))
    pool_indices = indices[:n_pool]
    train_indices = indices[n_pool:]
    n_query_samples = 10

        # Define the acquisition function
    acquisition = EntropyAcquisition()
    strategy = DeterministicStrategy(acquisition, inputs, outputs)
    strategy.to(device)
    net = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)
    best_indices = strategy.query(net, pool_indices, n_query_samples=n_query_samples)
    print(best_indices)
    assert best_indices['EntropyAcquisition'].shape==(n_query_samples,)














# def test_entropy():
#     # Create synthetic data
#     data_size = 200  # Total number of data points
#     num_features = 10  # Number of features
#     num_classes = 2  # Number of classes (binary classification)

#     # Generate random input data
#     inputs = torch.rand(data_size, num_features).to(device)

#     # Generate random output labels (binary: 0 or 1)
#     outputs = torch.randint(0, 2, (data_size,)).to(device)

#     # Meta parameters
#     n_pool = 100  # Number of data points in the pool set
#     indices = torch.randperm(data_size)
#     pool_indices = indices[:n_pool]  # Randomly select indices for the pool set
#     train_indices = indices[n_pool:]  # Indices for the training set
#     n_query_samples = 5  # Number of samples to query

#     # Define the acquisition function
#     acquisition = EntropyAcquisition()

#     # Create an instance of EnsembleStrategy
#     strategy = EnsembleStrategy(acquisition, inputs, outputs)
#     strategy.to(device)

#     # Define a simple neural network model for binary classification
#     class SimpleModel(nn.Module):
#         def __init__(self, input_size, hidden_size, output_size):
#             super(SimpleModel, self).__init__()
#             self.fc1 = nn.Linear(input_size, hidden_size)
#             self.fc2 = nn.Linear(hidden_size, output_size)

#         def forward(self, x):
#             x = torch.relu(self.fc1(x))
#             x = self.fc2(x)
#             return x

#     # Create two simple neural network models
#     net = SimpleModel(num_features, 16, num_classes).to(device)
#     net1 = SimpleModel(num_features, 16, num_classes).to(device)
    
#     # Query for the best indices using the strategy
#     best_indices = strategy.query([net, net1], pool_indices, n_query_samples=n_query_samples)
#     # best_indices=strategy.query(net, pool_indices, n_query_samples=n_query_samples)// for deterministic strategy

#     # Assert the shape of the queried indices
#     assert best_indices["EntropyAcquisition"].shape == (n_query_samples,) # Import your EntropyAcquisition class from your module
# import torch.nn.functional as F

# def test_entropy_acquisition():
#     # Create random logits (replace this with your actual logits)
#     # Random logits with shape (batch_size, num_classes)
#     logits = torch.rand(5, 4)

#     # Create an instance of your EntropyAcquisition
#     entropy_acquisition = EntropyAcquisition()

#     # Calculate entropy scores
#     scores = entropy_acquisition.acquire_scores(logits)

#     # Print the entropy scores
#     print("Entropy Scores:", scores)

#     # Check if the scores are scalar
#     assert isinstance(scores, torch.Tensor)
    

