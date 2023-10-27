# SusML
![Python version](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Build](https://img.shields.io/github/actions/workflow/status/sustainability-lab/SusML/build.yml?label=build&logo=github)
![Tests](https://img.shields.io/github/actions/workflow/status/sustainability-lab/SusML/tests.yml?label=tests&logo=github)

# Useful Code Snippets

## Data
### Load Data
```python
from susml.torch.data import load_mnist, load_cifar_10
ds, ds_name = load_cifar_10()
```

## Models
### Initialize MLPs
```python
from susml.torch.models import MLP

mlp = MLP(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
```

### Initialize CNNs
```python
from susml.torch.models import CNN
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
from susml.torch.utils import train_fn
result = train(model, inputs, outputs, loss_fn, lr, n_epochs, batch_size, enable_tqdm=True)
print(result.keys()) # dict_keys(['epoch_losses', 'iter_losses'])
```

## Adhoc
### Count number of parameters in a model
```python
from susml.torch.utils import count_params
n_params = count_params(mlp)
```