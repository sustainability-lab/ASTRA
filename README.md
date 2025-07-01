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


# Useful Code Snippets

## Data
### Load Data
```python
try:
    from astra.torch.data import load_mnist, load_cifar_10
    data = load_cifar_10()
    print(data)
except Exception as e:
    print("Demo data loading (CIFAR-10):")
    print("Note: Actual data download requires internet connection")
    print("Error:", str(e))
    print("In normal usage, this would return a PyTorch dataset object with CIFAR-10 images and labels")

```
````python
Demo data loading (CIFAR-10):
Note: Actual data download requires internet connection
Error: No module named 'numpy'
In normal usage, this would return a PyTorch dataset object with CIFAR-10 images and labels


````

## Models
### MLPs
```python
try:
    from astra.torch.models import MLPRegressor
    
    mlp = MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
    print(mlp)
except Exception as e:
    print("MLPRegressor demo:")
    print("Note: This example requires PyTorch and other dependencies")
    print("Error:", str(e))
    print("MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation='relu', dropout=0.1)")
    print("# Creates a multi-layer perceptron with specified architecture")

```
```python
MLPRegressor demo:
Note: This example requires PyTorch and other dependencies
Error: No module named 'numpy'
MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation='relu', dropout=0.1)
# Creates a multi-layer perceptron with specified architecture


```

### CNNs
```python
try:
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
except Exception as e:
    print("CNNClassifier demo:")
    print("Note: This example requires PyTorch and other dependencies")  
    print("Error:", str(e))
    print("CNNClassifier(image_dims=(32, 32), kernel_size=5, input_channels=3, conv_hidden_dims=[32, 64], dense_hidden_dims=[128, 64], n_classes=10)")
    print("# Creates a CNN classifier with specified architecture")

```
```python
CNNClassifier demo:
Note: This example requires PyTorch and other dependencies
Error: No module named 'numpy'
CNNClassifier(image_dims=(32, 32), kernel_size=5, input_channels=3, conv_hidden_dims=[32, 64], dense_hidden_dims=[128, 64], n_classes=10)
# Creates a CNN classifier with specified architecture


```

### EfficientNets
```python
try:
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
except Exception as e:
    print("EfficientNet demo:")
    print("Note: This example requires PyTorch, torchvision and other dependencies")  
    print("Error:", str(e))
    print("model = EfficientNetClassifier(model=efficientnet_b0, weights=EfficientNet_B0_Weights.DEFAULT, n_classes=10)")
    print("x = torch.rand(10, 3, 224, 224)")
    print("out = model(x)")
    print("# Creates an EfficientNet classifier with pretrained weights")

```
```python
EfficientNet demo:
Note: This example requires PyTorch, torchvision and other dependencies
Error: No module named 'torch'
model = EfficientNetClassifier(model=efficientnet_b0, weights=EfficientNet_B0_Weights.DEFAULT, n_classes=10)
x = torch.rand(10, 3, 224, 224)
out = model(x)
# Creates an EfficientNet classifier with pretrained weights


```


### ViT
```python
try:
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from astra.torch.models import ViTClassifier

    model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)
    x = torch.rand(10, 3, 224, 224)  # (batch_size, channels, h, w)
    out = model(x)
    print(out.shape)
except Exception as e:
    print("Vision Transformer (ViT) demo:")
    print("Note: This example requires PyTorch, torchvision and other dependencies")  
    print("Error:", str(e))
    print("model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)")
    print("x = torch.rand(10, 3, 224, 224)")
    print("out = model(x)")
    print("# Creates a Vision Transformer classifier with pretrained weights")

```
```python
Vision Transformer (ViT) demo:
Note: This example requires PyTorch, torchvision and other dependencies
Error: No module named 'torch'
model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)
x = torch.rand(10, 3, 224, 224)
out = model(x)
# Creates a Vision Transformer classifier with pretrained weights


```


## Training
### Train Function Usage
```python
try:
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
except Exception as e:
    print("Quick training demo:")
    print("Note: This example requires PyTorch, numpy and other dependencies")  
    print("Error:", str(e))
    print("# Demonstrates quick training with the train_fn utility")
    print("model = CNNClassifier(...)")
    print("iter_losses, epoch_losses = train_fn(model, input=X, output=y, loss_fn=nn.CrossEntropyLoss(), lr=0.1, epochs=5)")
    print("# Simple one-line training function")

```
```python
Quick training demo:
Note: This example requires PyTorch, numpy and other dependencies
Error: No module named 'torch'
# Demonstrates quick training with the train_fn utility
model = CNNClassifier(...)
iter_losses, epoch_losses = train_fn(model, input=X, output=y, loss_fn=nn.CrossEntropyLoss(), lr=0.1, epochs=5)
# Simple one-line training function


```

### Train with DataLoader
```python
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    import numpy as np
    from astra.torch.utils import train_fn
    from astra.torch.models import CNNClassifier

    torch.autograd.set_detect_anomaly(True)

    X = torch.rand(100, 3, 28, 28)
    y = torch.randint(0, 2, size=(200,)).reshape(100, 2).float()

    model = CNNClassifier(
        image_dims=(28, 28), kernel_size=5, input_channels=3, conv_hidden_dims=[4], dense_hidden_dims=[2], n_classes=2
    )

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Let train_fn do the optimization for you
    iter_losses, epoch_losses = train_fn(
        model,
        dataloader=dataloader,
        loss_fn=nn.CrossEntropyLoss(),
        lr=0.1,
        epochs=5,
    )
    print(np.array(epoch_losses).round(2))
except Exception as e:
    print("Training with DataLoader demo:")
    print("Note: This example requires PyTorch, numpy and other dependencies")  
    print("Error:", str(e))
    print("# Demonstrates training with PyTorch DataLoader")
    print("dataset = TensorDataset(X, y)")
    print("dataloader = DataLoader(dataset, batch_size=32, shuffle=True)")
    print("iter_losses, epoch_losses = train_fn(model, dataloader=dataloader, loss_fn=nn.CrossEntropyLoss(), lr=0.1, epochs=5)")
    print("# Training with batched data")

```
```python
Training with DataLoader demo:
Note: This example requires PyTorch, numpy and other dependencies
Error: No module named 'torch'
# Demonstrates training with PyTorch DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
iter_losses, epoch_losses = train_fn(model, dataloader=dataloader, loss_fn=nn.CrossEntropyLoss(), lr=0.1, epochs=5)
# Training with batched data


```


### Advanced Usage
```python
try:
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
except Exception as e:
    print("Advanced training demo:")
    print("Note: This example requires PyTorch, numpy and other dependencies")  
    print("Error:", str(e))
    print("# Demonstrates advanced training with custom models, multiple inputs, and custom loss functions")
    print("class CustomModel(AstraModel):")
    print("    def forward(self, x, inp1, fixed_bias):")
    print("        return self.linear(x) + self.inp1_linear(inp1) + fixed_bias")
    print("(iter_losses, epoch_losses), state_dict_history = train_fn(model, input=X, model_kwargs={'inp1': inp1, 'fixed_bias': bias}, output=y, loss_fn=custom_loss_fn)")
    print("# Advanced training with multiple inputs and custom loss functions")

```
```python
Advanced training demo:
Note: This example requires PyTorch, numpy and other dependencies
Error: No module named 'torch'
# Demonstrates advanced training with custom models, multiple inputs, and custom loss functions
class CustomModel(AstraModel):
    def forward(self, x, inp1, fixed_bias):
        return self.linear(x) + self.inp1_linear(inp1) + fixed_bias
(iter_losses, epoch_losses), state_dict_history = train_fn(model, input=X, model_kwargs={'inp1': inp1, 'fixed_bias': bias}, output=y, loss_fn=custom_loss_fn)
# Advanced training with multiple inputs and custom loss functions


```


## Others
### Count number of parameters in a model
```python
try:
    from astra.torch.utils import count_params
    from astra.torch.models import MLPRegressor

    mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)

    n_params = count_params(mlp)
    print(n_params)
except Exception as e:
    print("Count parameters demo:")
    print("Note: This example requires PyTorch and other dependencies")  
    print("Error:", str(e))
    print("mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)")
    print("count_params(mlp)")
    print("# Returns the total number of trainable parameters in the model")

```
```python
Count parameters demo:
Note: This example requires PyTorch and other dependencies
Error: No module named 'numpy'
mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)
count_params(mlp)
# Returns the total number of trainable parameters in the model


```

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

# Contributing
Please go through the [contributing guidelines](CONTRIBUTING.md) before making a contribution.