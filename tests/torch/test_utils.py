import pytest
import torch
import optree
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights

import astra.torch.models as models
import astra.torch.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "model, expected_size",
    [
        (models.MLPRegressor(input_dim=2, hidden_dims=[3, 4], output_dim=3), 40),
        (
            models.CNNClassifier(
                image_dim=23,
                kernel_size=3,
                input_channels=3,
                conv_hidden_dims=[2, 3],
                dense_hidden_dims=[4, 5],
                n_classes=4,
            ),
            466,
        ),
    ],
)
def test_count_params(model, expected_size):
    assert utils.count_params(model) == expected_size


def test_ravel_pytree():
    # Testing on most complex model
    model = models.ViTRegressor(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim=3)
    params = dict(model.named_parameters())
    flat_params, unravel_function = utils.ravel_pytree(params)
    unraveled_params = unravel_function(flat_params)

    # Assert structure is the same
    optree.tree_structure(params) == optree.tree_structure(unraveled_params)

    # Assert values are the same
    for k, v in params.items():
        assert torch.all(v == unraveled_params[k])


def test_train_fn():
    torch.manual_seed(0)
    inputs = torch.randn(10, 3, 224, 224)
    outputs = torch.randint(0, 3, (10,))
    lr = 1e-4
    n_epochs = 10

    model = models.CNNClassifier(224, 5, 3, [13, 14], [15, 16], n_classes=3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    iter_losses, epoch_losses = utils.train_fn(model, inputs.to(device), outputs.to(device), loss_fn, lr, n_epochs)

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs

    # without verbose
    iter_losses, epoch_losses = utils.train_fn(
        model, inputs.to(device), outputs.to(device), loss_fn, lr, n_epochs, verbose=False
    )

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs

    # wihout shuffle
    iter_losses, epoch_losses = utils.train_fn(
        model, inputs.to(device), outputs.to(device), loss_fn, lr, n_epochs, shuffle=False
    )

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs
