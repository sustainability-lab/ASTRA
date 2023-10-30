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
    epoch_losses = []
    outer_loop = range(n_epochs)
    if enable_tqdm:
        pbar = tqdm(total=len(data_loader) * n_epochs)
        n_processed = 0

    for _ in outer_loop:
        loss_value = 0.0
        for x, y in data_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            loss_value += loss.item()
            if enable_tqdm:
                n_processed += len(x)
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.6f}")

        epoch_losses.append(loss_value / len(data_loader))

    return {"iter_losses": iter_losses, "epoch_losses": epoch_losses}


def predict_class(model, inputs, batch_size=None):
    """Generic predict function for classification models.
    Note that we assume that the model predicts the logits of size `n_classes` even for the binary classification case.
    """
    if batch_size is None:
        batch_size = len(inputs)

    data_loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        preds = []
        for x in tqdm(data_loader):
            pred = model(x[0])
            preds.append(pred)
        pred = torch.cat(preds)
    return pred.argmax(dim=1)


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

def torch_set_diff1d(a, b):
    """Return the elements in `a` that are not in `b`."""
    mask = ~a.unsqueeze(1).eq(b).any(-1)
    return torch.masked_select(a, mask)