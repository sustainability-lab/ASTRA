# ASTRA
"**A**I for **S**ustainability" **T**oolkit for **R**esearch and **A**nalysis. ASTRA (अस्त्र) means a "tool" or "a weapon" in Sanskrit.

![Python version](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
[![CI](https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml/badge.svg)](https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml)
[![Coverage Status](https://coveralls.io/repos/github/sustainability-lab/ASTRA/badge.svg?branch=main)](https://coveralls.io/github/sustainability-lab/ASTRA?branch=main)

# Install

Stable version:
```bash
pip install astra-lib
```

Latest version:
```bash
pip install git+https://github.com/sustainability-lab/ASTRA
```


# Contributing
Please go through the [contributing guidelines](CONTRIBUTING.md) before making a contribution.


# Useful Code Snippets

## Data
### Load Data
```python
from astra.torch.data import load_mnist, load_cifar_10
ds, ds_name = load_cifar_10()
```

## Models
### MLPs
```python
from astra.torch.models import MLP

mlp = MLP(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
```

### CNNs
```python
from astra.torch.models import CNN
cnn = CNN(image_dim=32, 
          kernel_size=5, 
          n_channels=3, 
          conv_hidden_dims=[32, 64], 
          dense_hidden_dims=[128, 64], 
          output_dim=10)
```

### EfficientNets
```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from astra.torch.models import EfficientNet

model = EfficientNet(efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, output_dim=10)
```

### ViT
```python
from torchvision.models import vit_b_16, ViT_B_16_Weights
from astra.torch.models import ViT

model = ViT(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim=10)
```

## Training
### Quick train a model
```python
from astra.torch.utils import train_fn
result = train_fn(model, inputs, outputs, loss_fn, lr, n_epochs, batch_size, enable_tqdm=True)
print(result.keys()) # dict_keys(['epoch_losses', 'iter_losses'])
```

## Adhoc
### Count number of parameters in a model
```python
from astra.torch.utils import count_params
n_params = count_params(mlp)
```

### Flatten/Unflatten the weights of a model
```python
import torch
from astra.torch.models import ViT
from torchvision.models import vit_b_16, ViT_B_16_Weights
from astra.torch.utils import ravel_pytree
import optree

model = ViT(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim=10)
params = dict(model.named_parameters())

flat_params, unravel_fn = ravel_pytree(params)
unraveled_params = unravel_fn(flat_params) # returns the original params

# check if the tree structure is preserved
assert optree.tree_structure(params) == optree.tree_structure(unraveled_params)

# check if the values are preserved
for before_leaf, after_leaf in zip(optree.tree_leaves(params), optree.tree_leaves(unraveled_params)):
    assert torch.all(before_leaf == after_leaf)
```
