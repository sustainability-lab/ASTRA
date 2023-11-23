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
