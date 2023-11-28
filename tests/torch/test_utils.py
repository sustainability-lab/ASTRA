import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from astra.torch.models import CNNClassifier, MLPRegressor
from astra.torch.utils import train_fn, count_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "model, expected_size",
    [
        (MLPRegressor(input_dim=2, hidden_dims=[3, 4], output_dim=3), 40),
        (
            CNNClassifier(
                image_dims=(23, 23),
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
    counts = count_params(model)
    assert counts["total_params"] == expected_size
    assert counts["trainable_params"] == expected_size
    assert counts["non_trainable_params"] == 0


def test_train_fn():
    torch.manual_seed(0)
    inputs = torch.randn(10, 3, 224, 224)
    outputs = torch.randint(0, 3, (10,))
    lr = 1e-4
    n_epochs = 10

    model = CNNClassifier((224, 224), 5, 3, [13, 14], [15, 16], n_classes=3).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    iter_losses, epoch_losses = train_fn(
        model, loss_fn, input=inputs.to(device), output=outputs.to(device), lr=lr, epochs=n_epochs
    )

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs

    # without verbose
    iter_losses, epoch_losses = train_fn(
        model, loss_fn, input=inputs.to(device), output=outputs.to(device), lr=lr, epochs=n_epochs, verbose=False
    )

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs

    # wihout shuffle
    iter_losses, epoch_losses = train_fn(
        model, loss_fn, input=inputs.to(device), output=outputs.to(device), lr=lr, epochs=n_epochs, shuffle=False
    )

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
    assert len(iter_losses) == n_epochs
    assert len(epoch_losses) == n_epochs


def test_train_fn_kl_minimize():
    class TrainableNormal(nn.Module):
        def __init__(self):
            super().__init__()
            self.loc = nn.Parameter(torch.tensor(0.0))
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, input):
            return dist.Normal(self.loc, self.scale)

    def loss_fn(model_output, output, target_dist):
        learned_distribution = model_output
        return dist.kl_divergence(learned_distribution, target_dist)

    model = TrainableNormal()
    target_dist = dist.Normal(2.0, 3.0)

    # Tune without optimizer
    iter_losses, epoch_losses = train_fn(
        model, loss_fn, input=None, output=None, lr=0.1, epochs=50, loss_fn_kwargs={"target_dist": target_dist}
    )
    assert (model.loc - target_dist.loc).abs() < 0.2
    assert (model.scale - target_dist.scale).abs() < 0.2

    # Tune with optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    iter_losses, epoch_losses = train_fn(
        model, loss_fn, input=None, output=None, epochs=50, loss_fn_kwargs={"target_dist": target_dist}, optimizer=optim
    )

    # Return state dict
    (iter_losses, epoch_losses), state_dict_list = train_fn(
        model,
        loss_fn,
        input=None,
        output=None,
        lr=0.01,
        epochs=50,
        loss_fn_kwargs={"target_dist": target_dist},
        return_state_dict=True,
    )
    locs = torch.tensor([state_dict["loc"] for state_dict in state_dict_list])
    scales = torch.tensor([state_dict["scale"] for state_dict in state_dict_list])


def test_train_fn_wihout_outputs():
    torch.manual_seed(0)
    input = torch.randn(10, 3)

    model = MLPRegressor(input_dim=3, hidden_dims=[4, 5], output_dim=1).to(device)

    def loss_fn(model_output, output):
        return model_output.mean()

    iter_losses, epoch_losses = train_fn(model, loss_fn, input.to(device), output=None, lr=1e-4, epochs=10)

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"


def test_train_fn_with_dataloader():
    torch.manual_seed(0)
    input = torch.randn(10, 3)
    output = torch.randn(10, 1)

    model = MLPRegressor(input_dim=3, hidden_dims=[4, 5], output_dim=1).to(device)

    def loss_fn(model_output, output):
        return model_output.mean()

    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(input, output)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    iter_losses, epoch_losses = train_fn(model, loss_fn, dataloader=dataloader, lr=1e-4, epochs=10)

    assert epoch_losses[-1] < epoch_losses[0], "Loss should decrease"
