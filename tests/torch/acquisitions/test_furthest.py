import torch
from torchvision.datasets import CIFAR10

from astra.torch.models import CNN

# from astra.torch.al.acquisitions.furthest import acquire_scores
from astra.torch.al import Furthest, DiversityStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_furthest():
    data = CIFAR10(root="data", download=True, train=False)  # "test" for less data
    inputs = torch.tensor(data.data).float().to(device)
    outputs = torch.tensor(data.targets).long().to(device)

    # Meta parameters
    n_pool = 1000
    indices = torch.randperm(len(inputs))
    pool_indices = indices[:n_pool]
    train_indices = indices[n_pool:]
    n_query_samples = 10

    # Define the acquisition function
    acquisition = Furthest()
    strategy = DiversityStrategy(acquisition, inputs, outputs)
    # Put the strategy on the device
    strategy.to(device)
    # Define the model
    net = CNN(32, 3, 3, [4, 8], [2, 3], 10).to(device)

    def extract_features(net):
        def feature_extractor(input_tensor):
            # Initialize features with the input tensor
            features = input_tensor

            # Apply each layer, activation, and max-pooling
            for layer in net.feature_extractor:
                features = layer(features)
                features = net.activation(features)
                features = net.max_pool(features)
            features = net.flatten(features)
            return features

        return feature_extractor

    # Create a feature extractor callable from the network
    feature_extractor = extract_features(net)

    # Query the strategy
    best_indices = strategy.query(
        feature_extractor, pool_indices, train_indices, n_query_samples=n_query_samples
    )

    print(best_indices)

    assert best_indices["Furthest"].shape == (n_query_samples,)


# test_furthest()
