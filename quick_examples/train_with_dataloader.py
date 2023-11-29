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
