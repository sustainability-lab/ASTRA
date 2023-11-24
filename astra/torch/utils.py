from copy import deepcopy
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional


def count_params(model):
    params = [(p.numel(), True) if p.requires_grad else (p.numel(), False) for p in model.parameters()]
    sum_fn = lambda params: sum(map(lambda x: x[0], params))
    total_params = sum_fn(params)
    trainable_params = sum_fn(filter(lambda x: x[1] is True, params))
    non_trainable_params = sum_fn(filter(lambda x: x[1] is False, params))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def get_model_device(model):
    return next(model.parameters()).device


def train_fn(
    model: torch.nn.Module,
    loss_fn: callable,
    input: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    lr: Optional[float] = 0.01,
    epochs: int = 1,
    batch_size: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_kwargs: dict = {},
    loss_fn_kwargs: dict = {},
    shuffle: bool = False,
    verbose: bool = True,
    return_state_dict: bool = False,
):
    """Train a model with a loss function.

    Args:
        model: A PyTorch model. It is called with `input` and `**model_kwargs`.
        loss_fn: A callable loss function. First argument is the model output. Second argument is the `output` argument. Third argument is `**loss_fn_kwargs`.
        input: It is passed as a first argument to the `model`.
        model_kwargs: Keyword arguments passed to the `model`.
        output: It is passed as a second argument to the `loss_fn`. First argument to `loss_fn` is the model output.
        loss_fn_kwargs: Keyword arguments passed to the `loss_fn`.
        lr: Learning rate. It is ignored if `optimizer` is explicitly passed.
        epochs: Number of epochs.
        optimizer: A PyTorch optimizer. If None, Adam optimizer with user specified `lr` is used.
        batch_size: Batch size. If None, it is set to the length of `input` or `output` (whichever is not None). If both are None, it is set to 1.
        shuffle: If True, shuffle the data **before** each epoch.
        verbose: If True, print loss values and progress with `tqdm`.
        return_state_dict: If True, store state_dicts of the model at the end of each iteration and return them as a list in addition to the losses.

    Returns:
        If `return_state_dict` is False:
            (iter_losses, epoch_losses)
        Else:
            ((iter_losses, epoch_losses), state_dict_list)
    """

    def get_batch():
        if input is None and output is None:
            yield None, None
            return 0  # execution successfully finished

        in_or_out = output if input is None else input
        if batch_size is None:
            inner_batch_size = len(in_or_out)
        else:
            inner_batch_size = batch_size

        iterable = range(0, len(in_or_out), inner_batch_size)

        if shuffle:
            idx = torch.randperm(len(in_or_out))
        else:
            idx = torch.arange(len(in_or_out))

        for i in iterable:
            if input is not None:
                batch_input = input[idx[i : i + inner_batch_size]]
            else:
                batch_input = None

            if output is not None:
                batch_output = output[idx[i : i + inner_batch_size]]
            else:
                batch_output = None

            yield batch_input, batch_output

    def one_step(batch_input, batch_output):
        optimizer.zero_grad()
        model_output = model(batch_input, **model_kwargs)
        loss = loss_fn(model_output, batch_output, **loss_fn_kwargs)
        loss.backward()
        optimizer.step()
        return loss.item()

    model.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iter_losses = []
    epoch_losses = []
    state_dict_list = []

    pbar = range(epochs)
    if verbose:
        pbar = tqdm(pbar)

    for _ in pbar:
        loss_buffer = []
        for batch_input, batch_output in get_batch():
            loss = one_step(batch_input, batch_output)
            iter_losses.append(loss)
            loss_buffer.append(loss)

            if return_state_dict:
                state_dict_list.append(deepcopy(model.state_dict()))

        epoch_loss = sum(loss_buffer) / len(loss_buffer)
        epoch_losses.append(epoch_loss)

        if verbose:
            pbar.set_description(f"Loss: {epoch_loss:.8f}")

    if return_state_dict:
        return (iter_losses, epoch_losses), state_dict_list
    else:
        return iter_losses, epoch_losses
