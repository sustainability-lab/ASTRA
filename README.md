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
from astra.torch.data import load_mnist, load_cifar_10

data = load_cifar_10()
print(data)

```
````python

Traceback (most recent call last):
  File "/usr/lib/python3.12/urllib/request.py", line 1344, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/usr/lib/python3.12/http/client.py", line 1336, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/usr/lib/python3.12/http/client.py", line 1382, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/usr/lib/python3.12/http/client.py", line 1331, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/usr/lib/python3.12/http/client.py", line 1091, in _send_output
    self.send(msg)
  File "/usr/lib/python3.12/http/client.py", line 1035, in send
    self.connect()
  File "/usr/lib/python3.12/http/client.py", line 1470, in connect
    super().connect()
  File "/usr/lib/python3.12/http/client.py", line 1001, in connect
    self.sock = self._create_connection(
                ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 828, in create_connection
    for res in getaddrinfo(host, port, 0, SOCK_STREAM):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 963, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
socket.gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/load_data.py", line 3, in <module>
    data = load_cifar_10()
           ^^^^^^^^^^^^^^^
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/data.py", line 52, in load_cifar_10
    cfar_10_train = datasets.CIFAR10(root=f"{os.environ['TORCH_HOME']}/data", train=True, download=True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/torchvision/datasets/cifar.py", line 66, in __init__
    self.download()
  File "/home/runner/.local/lib/python3.12/site-packages/torchvision/datasets/cifar.py", line 139, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/runner/.local/lib/python3.12/site-packages/torchvision/datasets/utils.py", line 391, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/runner/.local/lib/python3.12/site-packages/torchvision/datasets/utils.py", line 121, in download_url
    url = _get_redirect_url(url, max_hops=max_redirect_hops)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/torchvision/datasets/utils.py", line 66, in _get_redirect_url
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 215, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 515, in open
    response = self._open(req, data)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 532, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 492, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 1392, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/urllib/request.py", line 1347, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>

````

## Models
### MLPs
```python
from astra.torch.models import MLPRegressor

mlp = MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
print(mlp)

```
```python

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/mlp.py", line 1, in <module>
    from astra.torch.models import MLPRegressor
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/models.py", line 30, in <module>
    from astra.torch.utils import get_model_device
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/cnn.py", line 1, in <module>
    from astra.torch.models import CNNClassifier
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/models.py", line 30, in <module>
    from astra.torch.utils import get_model_device
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/efficientnet.py", line 3, in <module>
    from astra.torch.models import EfficientNetClassifier
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/models.py", line 30, in <module>
    from astra.torch.utils import get_model_device
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/vit.py", line 3, in <module>
    from astra.torch.models import ViTClassifier
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/models.py", line 30, in <module>
    from astra.torch.utils import get_model_device
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/quick_train.py", line 4, in <module>
    from astra.torch.utils import train_fn
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

```

### Train with DataLoader
```python
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

```
```python

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/train_with_dataloader.py", line 6, in <module>
    from astra.torch.utils import train_fn
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/advanced_train.py", line 4, in <module>
    from astra.torch.utils import train_fn
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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

Traceback (most recent call last):
  File "/home/runner/work/ASTRA/ASTRA/quick_examples/count_params.py", line 1, in <module>
    from astra.torch.utils import count_params
  File "/home/runner/work/ASTRA/ASTRA/astra/torch/utils.py", line 1, in <module>
    import wandb
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/__init__.py", line 22, in <module>
    from wandb.sdk.lib import wb_logging as _wb_logging
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/__init__.py", line 25, in <module>
    from .artifacts.artifact import Artifact
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/artifacts/artifact.py", line 30, in <module>
    from wandb import data_types, env
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/data_types.py", line 16, in <module>
    from .sdk.data_types.audio import Audio
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/audio.py", line 11, in <module>
    from .base_types.media import BatchableMedia
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/media.py", line 14, in <module>
    from .wb_value import WBValue
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/data_types/base_types/wb_value.py", line 4, in <module>
    from wandb.sdk import wandb_setup
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_setup.py", line 38, in <module>
    from . import wandb_settings
  File "/home/runner/.local/lib/python3.12/site-packages/wandb-0.20.2rc20250616-py3.12.egg/wandb/sdk/wandb_settings.py", line 23, in <module>
    from pydantic import BaseModel, ConfigDict, Field
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/__init__.py", line 5, in <module>
    from ._migration import getattr_migration
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/_migration.py", line 4, in <module>
    from .version import version_short
  File "/home/runner/.local/lib/python3.12/site-packages/pydantic-2.11.7-py3.12.egg/pydantic/version.py", line 5, in <module>
    from pydantic_core import __version__ as __pydantic_core_version__
ModuleNotFoundError: No module named 'pydantic_core'

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