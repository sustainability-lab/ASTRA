from tqdm import tqdm
import torch
import optree
from torch.utils.data import TensorDataset, DataLoader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fn(model, inputs, output, loss_fn, lr, n_epochs, batch_size=None, enable_tqdm=True):
    if batch_size is None:
        batch_size = len(inputs)

    data_loader = DataLoader(TensorDataset(inputs, output), batch_size=batch_size, shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iter_losses = []
    epochs_losses = []
    if enable_tqdm:
        loop = tqdm(range(n_epochs))
    else:
        loop = range(n_epochs)
    for _ in loop:
        loss_value = 0.0
        for x, y in data_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            loss_value += loss.item()
        epochs_losses.append(loss_value / len(data_loader))
        if enable_tqdm:
            loop.set_description(f"Loss: {loss.item():.6f}")

    return {"iter_losses": iter_losses, "epochs_losses": epochs_losses}


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
