import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import EfficientNet as _EfficientNet  # To prevent auto import
from torchvision.models.vision_transformer import VisionTransformer as _VisionTransformer  # To prevent auto import
from torchvision.models._api import WeightsEnum


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

## Hotfix for https://github.com/pytorch/vision/issues/7744#issuecomment-1757321451
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


def get_state_dict(self, *args, **kwargs):
    if "check_hash" in kwargs:
        kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict
################################################################################


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) with dropout and activation function."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim

        # One time initialization layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout, inplace=True)

        # Define input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Define hidden layers
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            hidden_layers.append(hidden_layer)
        self.hidden_layers = nn.ModuleList(hidden_layers)

        # Define output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x


class SIREN(MLP):
    """Sinusoidal Representation Network (SIREN). See https://arxiv.org/abs/2006.09661 (Implicit Neural Representations with Periodic Activation Functions by Sitzmann et al.)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 1,
        activation_scale: float = 30.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            output_dim,
            activation=lambda x: torch.sin(activation_scale * x),
            dropout=dropout,
        )

        # Initialize weights with SIREN initialization
        self._initialize(activation_scale)

    def _initialize(self, activation_scale):
        def first_layer_init(m):
            if hasattr(m, "weight"):
                input_size = m.weight.size(-1)
                m.weight.uniform_(-1 / input_size, 1 / input_size)

        def other_layer_init(m):
            if hasattr(m, "weight"):
                input_size = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / input_size) / activation_scale, np.sqrt(6 / input_size) / activation_scale
                )

        with torch.no_grad():
            self.input_layer.apply(first_layer_init)
            self.hidden_layers.apply(other_layer_init)
            self.output_layer.apply(other_layer_init)


class CNN(nn.Module):
    def __init__(
        self,
        image_dim: int,
        kernel_size: int,
        n_channels: int,
        conv_hidden_dims: list,
        dense_hidden_dims: list,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = kernel_size // 2

        # One time initialization layers
        self.flatten = nn.Flatten()
        self.activation = activation
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define feature extractor
        layers = []

        layers.append(nn.Conv2d(n_channels, conv_hidden_dims[0], kernel_size=kernel_size, padding=padding))

        # Define hidden layers
        for i in range(len(conv_hidden_dims) - 1):
            hidden_layer = nn.Conv2d(
                conv_hidden_dims[i],
                conv_hidden_dims[i + 1],
                kernel_size=kernel_size,
                padding=padding,
            )
            layers.append(hidden_layer)

        self.feature_extractor = nn.ModuleList(layers)

        # Define dense layers
        out_image_dim = image_dim // 2 ** len(conv_hidden_dims)
        output_in_dim = conv_hidden_dims[-1] * out_image_dim**2
        self.dense_layers = MLP(output_in_dim, dense_hidden_dims, output_dim, activation, dropout)

    def forward(self, x):
        # Feature extractor
        for layer in self.feature_extractor:
            x = layer(x)
            x = self.activation(x)
            x = self.max_pool(x)

        # Flatten
        x = self.flatten(x)

        # Dense layers
        x = self.dense_layers(x)

        return x


class EfficientNet(nn.Module):
    """EfficientNet. See https://arxiv.org/abs/1905.11946 (EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Tan et al.)"""

    def __init__(
        self,
        model: _EfficientNet = efficientnet_b0,
        weights: WeightsEnum = EfficientNet_B0_Weights.DEFAULT,
        output_dim: int = 2,
    ):
        super().__init__()

        self.transform = weights.transforms()
        self.efficientnet = model(weights=weights)
        self.efficientnet.classifier = nn.Linear(1280, output_dim)

    def forward(self, x):
        x = self.transform(x)
        x = self.efficientnet(x)
        return x


class ViT(nn.Module):
    """Vision transformer (ViT) with B_16 architecture. See https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al.)"""

    def __init__(
        self,
        model: _VisionTransformer = vit_b_16,
        weights: WeightsEnum = ViT_B_16_Weights.DEFAULT,
        output_dim: int = 2,
    ):
        super().__init__()

        self.transform = weights.transforms()
        self.vit = model(weights=weights)
        self.vit.heads = nn.Linear(768, output_dim)

    def forward(self, x):
        x = self.transform(x)
        x = self.vit(x)
        return x
