import torch
from astra.torch.data import load_mnist, load_cifar_10


def test_load_mnist():
    dataset = load_mnist()
    assert dataset.data.shape == torch.Size([70000, 28, 28])
    assert dataset.targets.shape == torch.Size([70000])
    assert dataset.data.dtype == torch.float32
    assert dataset.targets.dtype == torch.int64
    assert dataset.classes == [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]


def test_load_cifar_10():
    dataset = load_cifar_10()
    assert dataset.data.shape == torch.Size([60000, 3, 32, 32])
    assert dataset.targets.shape == torch.Size([60000])
    assert dataset.data.dtype == torch.float32
    assert dataset.targets.dtype == torch.int64
    assert dataset.classes == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
