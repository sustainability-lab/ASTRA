import torch
import torch.nn as nn
import numpy as np
from astra.torch.utils import train_fn
from astra.torch.models import AstraModel


class CustomModel(AstraModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.inp1_linear = nn.Linear(2, 1)

    def forward(self, x, inp1, fixed_bias):
        return self.linear(x) + self.inp1_linear(inp1) + fixed_bias


def custom_loss_fn(model_output, output, norm_factor):
    loss_fn = nn.MSELoss()
    loss_val = loss_fn(model_output, output)
    return loss_val / norm_factor


X = torch.randn(10, 2)
y = torch.randn(10, 1)
inp1 = torch.randn(10, 2)
bias = torch.randn(1)
norm_factor = torch.randn(1)

model = CustomModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
(iter_losses, epoch_losses), state_dict_history = train_fn(
    model,
    input=X,  # Can be None if model.forward() does not require input
    model_kwargs={"inp1": inp1, "fixed_bias": bias},
    output=y,  # Can be None if loss_fn does not require output
    loss_fn=custom_loss_fn,
    loss_fn_kwargs={"norm_factor": norm_factor},
    optimizer=optimizer,
    epochs=5,
    shuffle=True,
    verbose=True,
    return_state_dict=True,
)

print("Epoch_losses", np.array(epoch_losses).round(2))
