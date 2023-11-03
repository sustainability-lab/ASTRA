from tqdm import tqdm
import torch
import optree
from torch.utils.data import TensorDataset, DataLoader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fn(model, inputs, outputs, loss_fn, lr, epochs, batch_size=None, shuffle=True, verbose=True):
    if batch_size is None:
        batch_size = len(inputs)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iter_losses = []
    epoch_losses = []

    # shuffle
    if shuffle:
        idx = torch.randperm(len(inputs))
    else:
        idx = torch.arange(len(inputs))

    for _ in range(epochs):
        loss_value = 0.0
        pbar = range(0, len(inputs), batch_size)
        if verbose:
            pbar = tqdm(pbar)
        for i in pbar:
            optimizer.zero_grad()
            pred = model(inputs[idx[i : i + batch_size]].to(model.device))
            loss = loss_fn(pred, outputs[idx[i : i + batch_size]].to(model.device))
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            loss_value += loss.item()
            if verbose:
                pbar.set_description(f"Loss: {loss.item():.6f}")

        # shuffle
        if shuffle:
            idx = torch.randperm(len(inputs))

        epoch_losses.append(loss_value / (len(inputs) / batch_size))
        if verbose:
            print(f"Epoch {len(epoch_losses)}: {epoch_losses[-1]}")

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
