from copy import deepcopy
from tqdm import tqdm
import torch
import optree
from torch.utils.data import TensorDataset, DataLoader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fn(
    model,
    inputs,
    outputs,
    loss_fn,
    lr=None,
    epochs=1,
    optimizer=None,
    batch_size=None,
    shuffle=True,
    verbose=True,
    get_state_dict=False,
):
    if batch_size is None:
        if inputs is not None:
            input_leaf = optree.tree_leaves(inputs)[0]
            if isinstance(input_leaf, torch.Tensor):
                batch_size = len(input_leaf)
                iterable = input_leaf
            else:
                batch_size = 1
                shuffle = False
                iterable = [None]
        else:
            if outputs is not None:
                output_leaf = optree.tree_leaves(outputs)[0]
                if isinstance(output_leaf, torch.Tensor):
                    batch_size = len(output_leaf)
                    iterable = output_leaf
                else:
                    batch_size = 1
                    shuffle = False
                    iterable = [None]
            else:
                batch_size = 1
                shuffle = False
                iterable = [None]

    model.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iter_losses = []
    epoch_losses = []
    state_dict_list = []

    # shuffle
    if shuffle:
        idx = torch.randperm(len(iterable))
    else:
        idx = torch.arange(len(iterable))

    for _ in range(epochs):
        loss_value = 0.0
        pbar = range(0, len(iterable), batch_size)
        if verbose:
            pbar = tqdm(pbar)
        for i in pbar:
            optimizer.zero_grad()

            # Prepare inputs and outputs
            if inputs is not None:
                if isinstance(input_leaf, torch.Tensor):
                    batch_input = optree.tree_map(lambda leaf: leaf[idx[i : i + batch_size]].to(model.device), inputs)
                else:
                    batch_input = inputs
            else:
                batch_input = None

            if isinstance(output_leaf, torch.Tensor):
                batch_output = optree.tree_map(lambda leaf: leaf[idx[i : i + batch_size]].to(model.device), outputs)
            else:
                batch_output = outputs

            pred = model(batch_input)
            loss = loss_fn(pred, batch_output)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            loss_value += loss.item()
            if verbose:
                pbar.set_description(f"Loss: {loss.item():.6f}")

            if get_state_dict:
                state_dict = deepcopy(model.state_dict())
                state_dict_list.append(state_dict)

        # shuffle
        if shuffle:
            idx = torch.randperm(len(iterable))

        epoch_losses.append(loss_value / (len(iterable) / batch_size))
        if verbose:
            print(f"Epoch {len(epoch_losses)}: {epoch_losses[-1]}")

    if get_state_dict:
        return iter_losses, epoch_losses, state_dict_list
    else:
        return iter_losses, epoch_losses


def ravel_pytree(pytree):
    leaves, structure = optree.tree_flatten(pytree)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.numel() for leaf in leaves]
    flat_params = torch.cat([leaf.flatten() for leaf in leaves])

    def unravel_function(flat_params):
        assert flat_params.numel() == sum(sizes), f"Invalid flat_params size {flat_params.numel()} != {sum(sizes)}"
        assert len(flat_params.shape) == 1, f"Invalid flat_params shape {flat_params.shape}"
        flat_leaves = flat_params.split(sizes)
        leaves = [leaf.reshape(shape) for leaf, shape in zip(flat_leaves, shapes)]
        return optree.tree_unflatten(structure, leaves)

    return flat_params, unravel_function
