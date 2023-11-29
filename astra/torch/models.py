from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.efficientnet import (
    EfficientNet as _EfficientNet,
)  # To prevent users from importing EfficientNet from torchvision
from torchvision.models.vision_transformer import VisionTransformer as _VisionTransformer  # To prevent ...
from torchvision.models.resnet import ResNet as _ResNet  # To prevent ...
from torchvision.models.densenet import DenseNet as _DenseNet  # To prevent ...
from torchvision.models.alexnet import AlexNet as _AlexNet  # To prevent ...
from torchvision.models._api import WeightsEnum


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import alexnet, AlexNet_Weights

## Hotfix for https://github.com/pytorch/vision/issues/7744#issuecomment-1757321451
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

from typing import Tuple, Literal

from astra.torch.utils import get_model_device


def get_state_dict(self, *args, **kwargs):
    if "check_hash" in kwargs:
        kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict
################################################################################


# Base class for all ASTRA models
class AstraModel(nn.Module):
    def predict(self, X=None, dataloader=None, batch_size=None, eval_mode: bool = True, verbose=True):
        device = get_model_device(self)
        if eval_mode:
            self.eval()
        else:
            self.train()

        assert (X is not None) ^ (dataloader is not None), "Either X or dataloader should be provided, but not both."

        if X is not None:
            if batch_size is None:
                batch_size = len(X)

            preds = []
            pbar = range(0, len(X), batch_size)
            if verbose:
                pbar = tqdm(pbar)
            for i in pbar:
                pred = self(X[i : i + batch_size].to(device))
                preds.append(pred)
            pred = torch.cat(preds)

            return pred

        elif dataloader is not None:
            preds = []
            pbar = dataloader
            if verbose:
                pbar = tqdm(pbar)
            for batch_input, _ in pbar:
                batch_input = batch_input.to(device)
                pred = self(batch_input)
                preds.append(pred)
            pred = torch.cat(preds)

            return pred

        else:
            raise ValueError("Either X or dataloader should be provided. This should never happen. Contact Zeel.")


# Classifier and regressor base classes
class Classifier(AstraModel):
    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x

    def predict_class(self, X=None, dataloader=None, batch_size=None, eval_mode=True, verbose=True):
        logits = self.predict(X, dataloader, batch_size, eval_mode, verbose)
        return logits.argmax(dim=1)


class Regressor(AstraModel):
    def __init__(self, featurizer, regressor):
        super().__init__()
        self.featurizer = featurizer
        self.regressor = regressor

    def forward(self, x):
        x = self.featurizer(x)
        x = self.regressor(x)
        return x

    def rmse(self, X, y, batch_size=None, device=None, eval_mode: bool = True):
        y_pred = self.predict(X=X, batch_size=batch_size, device=device, eval_mode=eval_mode)

        if device is None:
            device = get_model_device(self)
        return F.mse_loss(y_pred, y.to(device), reduction="mean").sqrt().item()


# Multi-layer perceptron (MLP)
class MLP(AstraModel):
    """Multi-layer perceptron (MLP) with dropout and activation function."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # One time initialization layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Define input layer
        self.input_layer = nn.Linear(input_dim, self.hidden_dims[0])

        # Define hidden layers
        for i in range(1, len(self.hidden_dims)):
            hidden_layer = nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i])
            setattr(self, f"hidden_layer_{i}", hidden_layer)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            x = getattr(self, f"hidden_layer_{i}")(x)
            x = self.activation(x)
            x = self.dropout(x)

        return x


class MLPClassifier(Classifier):
    def __init__(self, input_dim, hidden_dims, n_classes=2, activation=nn.ReLU(), dropout=0.0):
        """Multi-Layer Perceptron (MLP) classifier.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): List of hidden dimensions.
            n_classes (int): Number of classes.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        featurizer = MLP(input_dim, hidden_dims, activation, dropout)
        classifier = nn.Linear(hidden_dims[-1], n_classes)
        super().__init__(featurizer, classifier)


class MLPRegressor(Regressor):
    def __init__(self, input_dim, hidden_dims, output_dim: int = 1, activation=nn.ReLU(), dropout=0.0):
        """Multi-Layer Perceptron (MLP) regressor.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): List of hidden dimensions.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        featurizer = MLP(input_dim, hidden_dims, activation, dropout)
        regressor = nn.Linear(hidden_dims[-1], output_dim)
        super().__init__(featurizer, regressor)


# SIREN Regressor
class SIRENRegressor(MLPRegressor):
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
            self.apply(other_layer_init)
            self.featurizer.input_layer.apply(first_layer_init)


# CNN Models
class CNN(AstraModel):
    def __init__(
        self,
        kernel_size: int,
        input_channels: int,
        hidden_dims: list,
        activation: nn.Module = nn.ReLU(),
        adaptive_pooling: bool = False,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims

        padding = kernel_size // 2

        # One time initialization layers
        self.activation = activation

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_layer = nn.Conv2d(input_channels, self.hidden_dims[0], kernel_size=kernel_size, padding=padding)

        # Define hidden layers
        for i in range(1, len(self.hidden_dims)):
            hidden_layer = nn.Conv2d(
                self.hidden_dims[i - 1],
                self.hidden_dims[i],
                kernel_size=kernel_size,
                padding=padding,
            )
            setattr(self, f"hidden_layer_{i}", hidden_layer)

        # Define aggregator
        if adaptive_pooling:
            aggregator = nn.AdaptiveAvgPool2d((1, 1))
        else:
            aggregator = nn.Identity()
        self.aggregator = aggregator

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.max_pool(x)

        for i in range(1, len(self.hidden_dims)):
            x = getattr(self, f"hidden_layer_{i}")(x)
            x = self.activation(x)
            x = self.max_pool(x)

        # Aggregator
        x = self.aggregator(x)

        # Flatten
        x = self.flatten(x)

        return x


class CNNClassifier(Classifier):
    def __init__(
        self,
        image_dims: Tuple[int, int],
        kernel_size: int,
        input_channels: int,
        conv_hidden_dims: list,
        dense_hidden_dims: list,
        n_classes: int = 2,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        adaptive_pooling: bool = False,
    ):
        """Convolutional Neural Network (CNN) classifier.

        Args:
            image_dim (int): Image dimension.
            kernel_size (int): Kernel size.
            n_channels (int): Number of input channels.
            hidden_dims (list): List of convolutional hidden dimensions.
            dense_hidden_dims (list): List of dense hidden dimensions.
            n_classes (int): Number of classes.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        featurizer = CNN(kernel_size, input_channels, conv_hidden_dims, activation, adaptive_pooling)
        if adaptive_pooling:
            mlp_input_dim = conv_hidden_dims[-1]
        else:
            out_image_dim0 = image_dims[0] // 2 ** len(conv_hidden_dims)
            out_image_dim1 = image_dims[1] // 2 ** len(conv_hidden_dims)
            mlp_input_dim = conv_hidden_dims[-1] * out_image_dim0 * out_image_dim1
        classifier = MLPClassifier(mlp_input_dim, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class CNNRegressor(Regressor):
    def __init__(
        self,
        image_dims: Tuple[int, int],
        kernel_size: int,
        input_channels: int,
        conv_hidden_dims: list,
        dense_hidden_dims: list,
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        adaptive_pooling: bool = False,
    ):
        """Convolutional Neural Network (CNN) regressor.

        Args:
            image_dim (int): Image dimension.
            kernel_size (int): Kernel size.
            n_channels (int): Number of input channels.
            hidden_dims (list): List of convolutional hidden dimensions.
            dense_hidden_dims (list): List of dense hidden dimensions.
            output_dim (int): Number of output features.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        featurizer = CNN(kernel_size, input_channels, conv_hidden_dims, activation, adaptive_pooling)
        if adaptive_pooling:
            mlp_input_dim = conv_hidden_dims[-1]
        else:
            out_image_dim0 = image_dims[0] // 2 ** len(conv_hidden_dims)
            out_image_dim1 = image_dims[1] // 2 ** len(conv_hidden_dims)
            mlp_input_dim = conv_hidden_dims[-1] * out_image_dim0 * out_image_dim1
        regressor = MLPRegressor(mlp_input_dim, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)


# ResNet
class ResNet(AstraModel):
    """ResNet. See https://arxiv.org/abs/1512.03385 (Deep Residual Learning for Image Recognition by He et al.)"""

    def __init__(
        self,
        model: _ResNet = resnet18,
        weights: WeightsEnum = ResNet18_Weights.DEFAULT,
        transform: bool = False,
    ):
        super().__init__()

        self.transform = weights.transforms() if transform else None
        self.resnet = model(weights=weights)
        self.resnet.fc = nn.Identity()
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.resnet(x)
        x = self.flatten(x)
        return x


class ResNetClassifier(Classifier):
    def __init__(
        self,
        model: _ResNet = resnet18,
        weights: WeightsEnum = ResNet18_Weights.DEFAULT,
        n_classes: int = 2,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """ResNet classifier.

        Args:
            model (_ResNet, optional): _description_. Defaults to resnet18.
            weights (WeightsEnum, optional): _description_. Defaults to ResNet18_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            n_classes (int, optional): _description_. Defaults to 1.
        """
        featurizer = ResNet(model, weights, transform)
        classifier = MLPClassifier(512, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class ResNetRegressor(Regressor):
    def __init__(
        self,
        model: _ResNet = resnet18,
        weights: WeightsEnum = ResNet18_Weights.DEFAULT,
        output_dim: int = 1,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """ResNet regressor.

        Args:
            model (_ResNet, optional): _description_. Defaults to resnet18.
            weights (WeightsEnum, optional): _description_. Defaults to ResNet18_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            output_dim (int, optional): _description_. Defaults to 1.
        """
        featurizer = ResNet(model, weights, transform)
        regressor = MLPRegressor(512, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)


# EfficientNet
class EfficientNet(AstraModel):
    """EfficientNet. See https://arxiv.org/abs/1905.11946 (EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks by Tan et al.)"""

    def __init__(
        self,
        model: _EfficientNet = efficientnet_b0,
        weights: WeightsEnum = EfficientNet_B0_Weights.DEFAULT,
        transform: bool = False,
    ):
        super().__init__()

        self.transform = weights.transforms() if transform else None
        self.efficientnet = model(weights=weights)
        self.efficientnet.classifier = nn.Identity()
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.efficientnet(x)
        x = self.flatten(x)
        return x


class EfficientNetClassifier(Classifier):
    def __init__(
        self,
        model: _EfficientNet = efficientnet_b0,
        weights: WeightsEnum = EfficientNet_B0_Weights.DEFAULT,
        n_classes: int = 2,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """EfficientNet classifier.

        Args:
            model (_EfficientNet, optional): _description_. Defaults to efficientnet_b0.
            weights (WeightsEnum, optional): _description_. Defaults to EfficientNet_B0_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            n_classes (int, optional): _description_. Defaults to 1.
        """
        featurizer = EfficientNet(model, weights, transform)
        # Efficient uses AdaptiveAvgPool2d so output_dim is always 1280
        classifier = MLPClassifier(1280, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class EfficientNetRegressor(Regressor):
    def __init__(
        self,
        model: _EfficientNet = efficientnet_b0,
        weights: WeightsEnum = EfficientNet_B0_Weights.DEFAULT,
        output_dim: int = 1,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """EfficientNet regressor.

        Args:
            model (_EfficientNet, optional): _description_. Defaults to efficientnet_b0.
            weights (WeightsEnum, optional): _description_. Defaults to EfficientNet_B0_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            output_dim (int, optional): _description_. Defaults to 1.
        """
        featurizer = EfficientNet(model, weights, transform)
        # Efficient uses AdaptiveAvgPool2d so output_dim is always 1280
        regressor = MLPRegressor(1280, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)


class DenseNet(AstraModel):
    """DenseNet. See https://arxiv.org/abs/1608.06993 (Densely Connected Convolutional Networks by Huang et al.)"""

    def __init__(
        self,
        model: _DenseNet = densenet121,
        weights: WeightsEnum = DenseNet121_Weights.DEFAULT,
        transform: bool = False,
    ):
        super().__init__()

        self.transform = weights.transforms() if transform else None
        self.densenet = model(weights=weights)
        self.densenet.classifier = nn.Identity()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.densenet(x)
        return x


class DenseNetClassifier(Classifier):
    def __init__(
        self,
        model: _DenseNet = densenet121,
        weights: WeightsEnum = DenseNet121_Weights.DEFAULT,
        n_classes: int = 2,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """DenseNet classifier.

        Args:
            model (_DenseNet, optional): _description_. Defaults to densenet121.
            weights (WeightsEnum, optional): _description_. Defaults to DenseNet121_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            n_classes (int, optional): _description_. Defaults to 1.
        """
        featurizer = DenseNet(model, weights, transform)
        # DenseNet uses AdaptiveAvgPool2d so output_dim is always 1024
        classifier = MLPClassifier(1024, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class DenseNetRegressor(Regressor):
    def __init__(
        self,
        model: _DenseNet = densenet121,
        weights: WeightsEnum = DenseNet121_Weights.DEFAULT,
        output_dim: int = 1,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """DenseNet regressor.

        Args:
            model (_DenseNet, optional): _description_. Defaults to densenet121.
            weights (WeightsEnum, optional): _description_. Defaults to DenseNet121_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            output_dim (int, optional): _description_. Defaults to 1.
        """
        featurizer = DenseNet(model, weights, transform)
        # DenseNet uses AdaptiveAvgPool2d so output_dim is always 1024
        regressor = MLPRegressor(1024, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)


class AlexNet(AstraModel):
    """AlexNet. See https://arxiv.org/abs/1404.5997 (ImageNet Classification with Deep Convolutional Neural Networks by Krizhevsky et al.)"""

    def __init__(
        self,
        model: _AlexNet = alexnet,
        weights: WeightsEnum = AlexNet_Weights.DEFAULT,
        transform: bool = False,
    ):
        super().__init__()

        self.transform = weights.transforms() if transform else None
        self.alexnet = model(weights=weights)
        self.alexnet.classifier._modules["6"] = nn.Identity()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.alexnet(x)
        return x


class AlexNetClassifier(Classifier):
    def __init__(
        self,
        model: _AlexNet = alexnet,
        weights: WeightsEnum = AlexNet_Weights.DEFAULT,
        n_classes: int = 2,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """AlexNet classifier.

        Args:
            model (_AlexNet, optional): _description_. Defaults to alexnet.
            weights (WeightsEnum, optional): _description_. Defaults to AlexNet_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            n_classes (int, optional): _description_. Defaults to 1.
        """
        featurizer = AlexNet(model, weights, transform)
        # AlexNet uses AdaptiveAvgPool2d so output_dim is always 4096
        classifier = MLPClassifier(4096, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class AlexNetRegressor(Regressor):
    def __init__(
        self,
        model: _AlexNet = alexnet,
        weights: WeightsEnum = AlexNet_Weights.DEFAULT,
        output_dim: int = 1,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """AlexNet regressor.

        Args:
            model (_AlexNet, optional): _description_. Defaults to alexnet.
            weights (WeightsEnum, optional): _description_. Defaults to AlexNet_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            output_dim (int, optional): _description_. Defaults to 1.
        """
        featurizer = AlexNet(model, weights, transform)
        # AlexNet uses AdaptiveAvgPool2d so output_dim is always 4096
        regressor = MLPRegressor(4096, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)


class ViT(AstraModel):
    def __init__(
        self,
        model: _VisionTransformer = vit_b_16,
        weights: WeightsEnum = ViT_B_16_Weights.DEFAULT,
        transform: bool = False,
    ):
        """Vision transformer (ViT). See https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al.)"""
        super().__init__()

        self.transform = weights.transforms() if transform else None
        self.vit = model(weights=weights)
        self.vit.heads = nn.Identity()
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.vit(x)
        x = self.flatten(x)
        return x


class ViTClassifier(Classifier):
    def __init__(
        self,
        model: _VisionTransformer = vit_b_16,
        weights: WeightsEnum = ViT_B_16_Weights.DEFAULT,
        n_classes: int = 2,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """Vision transformer (ViT) classifier.

        Args:
            model (_VisionTransformer, optional): _description_. Defaults to vit_b_16.
            weights (WeightsEnum, optional): _description_. Defaults to ViT_B_16_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            n_classes (int, optional): _description_. Defaults to 1.
        """
        featurizer = ViT(model, weights, transform)
        # ViT uses fixed input size of 224 so output_dim is always 768
        classifier = MLPClassifier(768, dense_hidden_dims, n_classes, activation, dropout)
        super().__init__(featurizer, classifier)


class ViTRegressor(Regressor):
    def __init__(
        self,
        model: _VisionTransformer = vit_b_16,
        weights: WeightsEnum = ViT_B_16_Weights.DEFAULT,
        output_dim: int = 1,
        transform: bool = False,
        dense_hidden_dims: list = [512],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """Vision transformer (ViT) regressor.

        Args:
            model (_VisionTransformer, optional): _description_. Defaults to vit_b_16.
            weights (WeightsEnum, optional): _description_. Defaults to ViT_B_16_Weights.DEFAULT.
            transform (bool, optional): _description_. Defaults to False.
            dense_hidden_dims (list, optional): _description_. Defaults to [512].
            activation (nn.Module, optional): _description_. Defaults to nn.ReLU().
            output_dim (int, optional): _description_. Defaults to 1.
        """
        featurizer = ViT(model, weights, transform)
        # ViT uses fixed input size of 224 so output_dim is always 768
        regressor = MLPRegressor(768, dense_hidden_dims, output_dim, activation, dropout)
        super().__init__(featurizer, regressor)
