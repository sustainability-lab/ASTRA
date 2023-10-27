# ASTRA
"**A**I for **S**ustainability" **T**oolkit for **R**esearch and **A**nalysis. ASTRA (अस्त्र) means "a weapon" in Sanskrit, Hindi and a few other Bharatiya languages.

![Python version](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
[![CI](https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml/badge.svg)](https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml)
[![Coverage Status](https://coveralls.io/repos/github/sustainability-lab/ASTRA/badge.svg?branch=main)](https://coveralls.io/github/sustainability-lab/ASTRA?branch=main)

# Install
```bash
pip install astra
```

# Useful Code Snippets

## Data
### Load Data
```python
from astra.torch.data import load_mnist, load_cifar_10
ds, ds_name = load_cifar_10()
```

## Models
### Initialize MLPs
```python
from astra.torch.models import MLP

mlp = MLP(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
```

### Initialize CNNs
```python
from astra.torch.models import CNN
CNN(image_dim=32, 
    kernel_size=5, 
    n_channels=3, 
    conv_hidden_dims=[32, 64], 
    dense_hidden_dims=[128, 64], 
    output_dim=10)
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