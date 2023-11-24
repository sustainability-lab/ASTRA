from copy import deepcopy
import os
import numpy as np
import torch
from torchvision import datasets, transforms
import warnings


def prerequisite(f):
    if "TORCH_HOME" not in os.environ:
        os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")
        warnings.warn(f"TORCH_HOME not set, setting it to {os.environ['TORCH_HOME']}")
    return f


@prerequisite
def load_mnist():
    mnist_train = datasets.MNIST(root=f"{os.environ['TORCH_HOME']}/data", train=True, download=True)
    mnist_test = datasets.MNIST(root=f"{os.environ['TORCH_HOME']}/data", train=False, download=True)

    train_images = mnist_train.data.float() / 255
    train_labels = mnist_train.targets.long()

    test_images = mnist_test.data.float() / 255
    test_labels = mnist_test.targets.long()

    images = torch.cat([train_images, test_images], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)

    mnist_train.data = images
    mnist_train.targets = labels

    def repr(self):
        return f"""
MNIST Dataset
length of dataset: {len(self)}
shape of images: {self.data.shape[1:]}
len of classes: {len(self.classes)}
classes: {self.classes}
dtype of images: {self.data.dtype}
dtype of labels: {self.targets.dtype}
range of image values: min={self.data.min()}, max={self.data.max()}
"""

    mnist_train.__class__.__repr__ = repr

    return mnist_train


@prerequisite
def load_cifar_10():
    cfar_10_train = datasets.CIFAR10(root=f"{os.environ['TORCH_HOME']}/data", train=True, download=True)
    cfar_10_test = datasets.CIFAR10(root=f"{os.environ['TORCH_HOME']}/data", train=False, download=True)

    train_images = torch.tensor(cfar_10_train.data).float() / 255
    train_images = torch.einsum("nhwc->nchw", train_images)
    train_labels = torch.tensor(cfar_10_train.targets).long()

    test_images = torch.tensor(cfar_10_test.data).float() / 255
    test_images = torch.einsum("nhwc->nchw", test_images)
    test_labels = torch.tensor(cfar_10_test.targets).long()

    images = torch.cat([train_images, test_images], dim=0)
    labels = torch.cat([train_labels, test_labels], dim=0)

    cfar_10_train.data = images
    cfar_10_train.targets = labels

    def repr(self):
        return f"""
CIFAR-10 Dataset
length of dataset: {len(self)}
shape of images: {self.data.shape[1:]}
len of classes: {len(self.classes)}
classes: {self.classes}
dtype of images: {self.data.dtype}
dtype of labels: {self.targets.dtype}
range of image values: min={self.data.min()}, max={self.data.max()}
            """

    cfar_10_train.__class__.__repr__ = repr

    return cfar_10_train
