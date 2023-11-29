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
{{ load_data }}
```
````python
{{ load_data_output }}
{{ load_data_error }}
````

## Models
### MLPs
```python
{{ mlp }}
```
```python
{{ mlp_output }}
{{ mlp_error }}
```

### CNNs
```python
{{ cnn }}
```
```python
{{ cnn_output }}
{{ cnn_error }}
```

### EfficientNets
```python
{{ efficientnet }}
```
```python
{{ efficientnet_output }}
{{ efficientnet_error }}
```


### ViT
```python
{{ vit }}
```
```python
{{ vit_output }}
{{ vit_error }}
```


## Training
### Train Function Usage
```python
{{ quick_train }}
```
```python
{{ quick_train_output }}
{{ quick_train_error }}
```

### Train with DataLoader
```python
{{ train_with_dataloader }}
```
```python
{{ train_with_dataloader_output }}
{{ train_with_dataloader_error }}
```


### Advanced Usage
```python
{{ advanced_train }}
```
```python
{{ advanced_train_output }}
{{ advanced_train_error }}
```


## Others
### Count number of parameters in a model
```python
{{ count_params }}
```
```python
{{ count_params_output }}
{{ count_params_error }}
```