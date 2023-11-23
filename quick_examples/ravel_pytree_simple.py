import torch
import torch.nn as nn
from astra.torch.utils import ravel_pytree
import optree

model = nn.Sequential(*[nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1)])
params = dict(model.named_parameters())

flat_params, unravel_fn = ravel_pytree(params)
unraveled_params = unravel_fn(flat_params)  # returns the original params

print("Before")
print(params)
print("\nAfter ravel")
print(flat_params)
print("\nAfter unravel")
print(unraveled_params)
