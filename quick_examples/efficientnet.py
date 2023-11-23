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
