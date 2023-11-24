<p align="center">
<img src="https://github.com/sustainability-lab/ASTRA/assets/59758528/d6a8e7ed-5368-4574-801e-76b273b56091" width="512">
</p>

<p align="center">
          <img src="https://img.shields.io/badge/Python-3.9%2B-brightgreen">
          <a href="https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml">
                    <img src="https://github.com/sustainability-lab/ASTRA/actions/workflows/CI.yml/badge.svg">
          </a>
          <a href="https://coveralls.io/github/sustainability-lab/ASTRA?branch=main">
                    <img src="https://coveralls.io/repos/github/sustainability-lab/ASTRA/badge.svg?branch=main">
          </a>
</p>

"**A**I for **S**ustainability" **T**oolkit for **R**esearch and **A**nalysis. ASTRA (अस्त्र) means a "tool" or "a weapon" in Sanskrit.

# Design Principles
Since `astra` is developed for research purposes, we'd try to adhere to these principles:

## What we will try to do:
1. Keep the API simple-to-use and standardized to enable quick prototyping via automated scripts.
2. Keep the API transparent to expose as many details as possilbe. Explicit should be preferred over implicit.
3. Keep the API flexible to allow users to stretch the limits of their experiments.

## What we will try to avoid:
4. We will try not to reduce code repeatation at expence of transparency, flexibility and performance. Too much abstraction often makes the API complex to understand and thus becomes hard to adapt for custom use cases.

## Examples
| Points | Example |
| --- | --- |
| 1 and 2 | We have exactly same arguments for all strategies in `astra.torch.al.strategies` to ease the automation but we explicitely mention in the docstrings if an argument is used or ignored for a strategy. |
| 2 | predict functions in `astra` by default put the model on `eval` mode but also allow to set `eval_mode` to `False`. This can be useful for techniques like [MC dropout](https://arxiv.org/abs/1506.02142).
| 3 | `train_fn` from `astra.torch.utils` works for all types of models and losses which may or may not be from `astra`.
| 4 | Though F1 score can be computed from precision and recall, we explicitely use F1 score formula to allow transparency and to avoid computing `TP` multiple times.

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

data = load_cifar_10()
print(data)

```
````python
Files already downloaded and verified
Files already downloaded and verified

CIFAR-10 Dataset
length of dataset: 60000
shape of images: torch.Size([3, 32, 32])
len of classes: 10
classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
dtype of images: torch.float32
dtype of labels: torch.int64
            


````

## Models
### MLPs
```python
from astra.torch.models import MLPRegressor

mlp = MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
print(mlp)

```
```python
MLPRegressor(
  (featurizer): MLP(
    (dropout): Dropout(p=0.1, inplace=True)
    (input_layer): Linear(in_features=100, out_features=128, bias=True)
    (hidden_layer_1): Linear(in_features=128, out_features=64, bias=True)
  )
  (regressor): Linear(in_features=64, out_features=10, bias=True)
)


```

### CNNs
```python
from astra.torch.models import CNNClassifier

cnn = CNNClassifier(
    image_dims=(32, 32),
    kernel_size=5,
    input_channels=3,
    conv_hidden_dims=[32, 64],
    dense_hidden_dims=[128, 64],
    n_classes=10,
)
print(cnn)

```
```python
CNNClassifier(
  (featurizer): CNN(
    (activation): ReLU()
    (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (input_layer): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (hidden_layer_1): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (aggregator): Identity()
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (classifier): MLPClassifier(
    (featurizer): MLP(
      (activation): ReLU()
      (dropout): Dropout(p=0.0, inplace=True)
      (input_layer): Linear(in_features=4096, out_features=128, bias=True)
      (hidden_layer_1): Linear(in_features=128, out_features=64, bias=True)
    )
    (classifier): Linear(in_features=64, out_features=10, bias=True)
  )
)


```

### EfficientNets
```python
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from astra.torch.models import EfficientNetClassifier

# Pretrained model
model = EfficientNetClassifier(model=efficientnet_b0, weights=EfficientNet_B0_Weights.DEFAULT, n_classes=10)
# OR without pretrained weights
# model = EfficientNetClassifier(model=efficientnet_b0, weights=None, n_classes=10)

x = torch.rand(10, 3, 224, 224)
out = model(x)
print(out.shape)

```
```python
torch.Size([10, 10])


```


### ViT
```python
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from astra.torch.models import ViTClassifier

model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)
x = torch.rand(10, 3, 224, 224)  # (batch_size, channels, h, w)
out = model(x)
print(out.shape)

```
```python
torch.Size([10, 10])


```


## Training
### Train Function Usage
```python
import torch
import torch.nn as nn
import numpy as np
from astra.torch.utils import train_fn
from astra.torch.models import CNNClassifier

torch.autograd.set_detect_anomaly(True)

X = torch.rand(100, 3, 28, 28)
y = torch.randint(0, 2, size=(200,)).reshape(100, 2).float()

model = CNNClassifier(
    image_dims=(28, 28), kernel_size=5, input_channels=3, conv_hidden_dims=[4], dense_hidden_dims=[2], n_classes=2
)

# Let train_fn do the optimization for you
iter_losses, epoch_losses = train_fn(
    model, input=X, output=y, loss_fn=nn.CrossEntropyLoss(), lr=0.1, epochs=5, verbose=False
)
print(np.array(epoch_losses).round(2))

# OR

# Define your own optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
iter_losses, epoch_losses = train_fn(
    model,
    input=X,
    output=y,
    loss_fn=nn.MSELoss(),
    optimizer=optimizer,
    verbose=False,
    epochs=5,
)
print(np.array(epoch_losses).round(2))

# Get the state_dict of the model at each epoch

(iter_losses, epoch_losses), state_dict_history = train_fn(
    model,
    input=X,
    output=y,
    loss_fn=nn.MSELoss(),
    lr=0.1,
    epochs=5,
    verbose=False,
    return_state_dict=True,
)
print(np.array(epoch_losses).round(2))

```
```python
[ 0.69 10.82  0.66  0.68  0.68]
[0.61 0.51 0.43 0.38 0.33]
[0.3  0.27 0.26 0.25 0.25]


```


### Advanced Usage
```python
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

```
```python
Epoch 1: -0.7875077128410339
Epoch 2: -0.8151381611824036
Epoch 3: -0.9291865229606628
Epoch 4: -1.9994704723358154
Epoch 5: -2.32348895072937
Epoch_losses [-0.79 -0.82 -0.93 -2.   -2.32]

  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00,  1.36it/s]100%|██████████| 1/1 [00:00<00:00,  1.36it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 581.17it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1076.84it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1198.37it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1208.38it/s]

```


## Others
### Count number of parameters in a model
```python
from astra.torch.utils import count_params
from astra.torch.models import MLPRegressor

mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)

n_params = count_params(mlp)
print(n_params)

```
```python
{'total_params': 58, 'trainable_params': 58, 'non_trainable_params': 0}


```

### Flatten/Unflatten the weights of a model

#### Simple Example
```python
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

```
```python
Before
{'0.weight': Parameter containing:
tensor([[-0.1172,  0.1053, -0.3732],
        [ 0.1407,  0.4086,  0.4325]], requires_grad=True), '0.bias': Parameter containing:
tensor([ 0.2276, -0.0026], requires_grad=True), '2.weight': Parameter containing:
tensor([[0.5320, 0.3535]], requires_grad=True), '2.bias': Parameter containing:
tensor([-0.7025], requires_grad=True)}

After ravel
tensor([ 0.2276, -0.0026, -0.1172,  0.1053, -0.3732,  0.1407,  0.4086,  0.4325,
        -0.7025,  0.5320,  0.3535], grad_fn=<CatBackward0>)

After unravel
{'0.weight': tensor([[-0.1172,  0.1053, -0.3732],
        [ 0.1407,  0.4086,  0.4325]], grad_fn=<ViewBackward0>), '0.bias': tensor([ 0.2276, -0.0026], grad_fn=<ViewBackward0>), '2.weight': tensor([[0.5320, 0.3535]], grad_fn=<ViewBackward0>), '2.bias': tensor([-0.7025], grad_fn=<ViewBackward0>)}


```

#### Advanced Example
```python
import torch
from astra.torch.models import ViTClassifier
from torchvision.models import vit_b_16, ViT_B_16_Weights
from astra.torch.utils import ravel_pytree
import optree

model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)
params = dict(model.named_parameters())

flat_params, unravel_fn = ravel_pytree(params)
unraveled_params = unravel_fn(flat_params)  # returns the original params

# check if the tree structure is preserved
assert optree.tree_structure(params) == optree.tree_structure(unraveled_params)

# check if the values are preserved
for before_leaf, after_leaf in zip(optree.tree_leaves(params), optree.tree_leaves(unraveled_params)):
    assert torch.all(before_leaf == after_leaf)

```
```python


```