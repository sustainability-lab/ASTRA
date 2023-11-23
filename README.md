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
[0.69 0.88 0.68 0.68 0.69]
[0.33 0.29 0.27 0.25 0.25]
[0.25 0.25 0.25 0.25 0.25]


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
Epoch 1: 6.7826828956604
Epoch 2: 5.437911033630371
Epoch 3: 4.144653797149658
Epoch 4: 3.28253173828125
Epoch 5: 2.997318983078003
Epoch_losses [6.78 5.44 4.14 3.28 3.  ]

  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00,  1.37it/s]100%|██████████| 1/1 [00:00<00:00,  1.37it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 412.14it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1167.03it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1208.38it/s]
  0%|          | 0/1 [00:00<?, ?it/s]100%|██████████| 1/1 [00:00<00:00, 1212.23it/s]

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
tensor([[ 0.0523,  0.0204,  0.2655],
        [ 0.3896, -0.4301,  0.3878]], requires_grad=True), '0.bias': Parameter containing:
tensor([0.3065, 0.1819], requires_grad=True), '2.weight': Parameter containing:
tensor([[-0.3669,  0.7047]], requires_grad=True), '2.bias': Parameter containing:
tensor([-0.2404], requires_grad=True)}

After ravel
tensor([ 0.3065,  0.1819,  0.0523,  0.0204,  0.2655,  0.3896, -0.4301,  0.3878,
        -0.2404, -0.3669,  0.7047], grad_fn=<CatBackward0>)

After unravel
{'0.weight': tensor([[ 0.0523,  0.0204,  0.2655],
        [ 0.3896, -0.4301,  0.3878]], grad_fn=<ViewBackward0>), '0.bias': tensor([0.3065, 0.1819], grad_fn=<ViewBackward0>), '2.weight': tensor([[-0.3669,  0.7047]], grad_fn=<ViewBackward0>), '2.bias': tensor([-0.2404], grad_fn=<ViewBackward0>)}


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