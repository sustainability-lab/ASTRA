import os
import numpy as np
from torchvision import datasets, transforms
import xarray as xr
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

    train_img = mnist_train.data.float().cpu().numpy()
    train_label = mnist_train.targets.float().cpu().numpy()

    test_img = mnist_test.data.float().cpu().numpy()
    test_label = mnist_test.targets.float().cpu().numpy()

    img = np.concatenate([train_img, test_img], axis=0)
    label = np.concatenate([train_label, test_label], axis=0)

    ds = xr.Dataset(
        {
            "img": (["sample", "channel", "x", "y"], img[:, None, :, :]),
            "label": (["sample"], label),
        },
        coords={
            "sample": np.arange(len(img)),
            "channel": [0],
            "x": np.arange(img.shape[1])[::-1],
            "y": np.arange(img.shape[2]),
        },
    )

    return ds, "MNIST"


@prerequisite
def load_cifar_10():
    cfar_10_train = datasets.CIFAR10(root=f"{os.environ['TORCH_HOME']}/data", train=True, download=True)
    cfar_10_test = datasets.CIFAR10(root=f"{os.environ['TORCH_HOME']}/data", train=False, download=True)

    train_img = cfar_10_train.data.transpose(0, 3, 1, 2).astype(np.float32) / 255
    train_label = np.array(cfar_10_train.targets).astype(np.float32)

    test_img = cfar_10_test.data.transpose(0, 3, 1, 2).astype(np.float32) / 255
    test_label = np.array(cfar_10_test.targets).astype(np.float32)

    img = np.concatenate([train_img, test_img], axis=0)
    label = np.concatenate([train_label, test_label], axis=0)

    ds = xr.Dataset(
        {
            "img": (["sample", "channel", "x", "y"], img),
            "label": (["sample"], label),
        },
        coords={
            "sample": range(len(img)),
            "channel": range(img.shape[1]),
            "x": np.arange(img.shape[2])[::-1],
            "y": np.arange(img.shape[3]),
        },
    )

    return ds, "CIFAR-10"
