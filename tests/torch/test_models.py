import pytest
import torch
import astra.torch.models as models
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    resnet18,
    ResNet18_Weights,
    alexnet,
    AlexNet_Weights,
    densenet121,
    DenseNet121_Weights,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "x, output_dim, out_shape",
    [
        (torch.randn(11, 3), 4, torch.Size([11, 4])),
        (torch.randn(12, 5), 6, torch.Size([12, 6])),
    ],
)
def test_mlp(x, output_dim, out_shape):
    # create a empty class object

    input_dim = x.shape[1]
    model = models.MLPRegressor(input_dim, [7, 9], output_dim).to(device)

    output = model(x.to(device))
    assert output.shape == out_shape

    model = models.MLPClassifier(input_dim, [7, 9], output_dim).to(device)
    output = model(x.to(device))
    assert output.shape == torch.Size([x.shape[0], output_dim])

    pred_logits = model.predict(x.to(device), batch_size=2)
    pred_classes = model.predict_class(x.to(device), batch_size=2)
    assert pred_logits.shape == torch.Size([x.shape[0], output_dim])
    assert pred_classes.shape == torch.Size([x.shape[0]])
    assert torch.all(pred_logits.argmax(dim=1) == pred_classes)
    assert pred_logits.dtype == torch.float32
    assert pred_classes.dtype == torch.int64


@pytest.mark.parametrize(
    "x, output_dim, out_shape",
    [
        (torch.randn(11, 3), 4, torch.Size([11, 4])),
        (torch.randn(12, 5), 6, torch.Size([12, 6])),
    ],
)
def test_siren(x, output_dim, out_shape):
    # create a empty class object

    input_dim = x.shape[1]
    model = models.SIRENRegressor(input_dim, [7, 9], output_dim).to(device)

    output = model(x.to(device))
    assert output.shape == out_shape


@pytest.mark.parametrize("x", [torch.randn(11, 3, 44, 28), torch.randn(11, 5, 224, 180)])
def test_cnn(x):
    batch_dim = x.shape[0]
    image_dim = (x.shape[-2], x.shape[-1])
    n_channels = x.shape[1]
    kernel_size = 3
    conv_hidden_dims = [4, 4, 7]
    dense_hidden_dims = [7, 9]
    output_dim = 12
    dropout = 0.1

    model = models.CNNClassifier(
        image_dim, kernel_size, n_channels, conv_hidden_dims, dense_hidden_dims, output_dim, dropout=dropout
    ).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    # With aggregation
    model = models.CNNClassifier(
        image_dim, kernel_size, n_channels, conv_hidden_dims, dense_hidden_dims, output_dim, adaptive_pooling=True
    ).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    model = models.CNNRegressor(
        image_dim, kernel_size, n_channels, conv_hidden_dims, dense_hidden_dims, output_dim, dropout=dropout
    ).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    # With aggregation
    model = models.CNNRegressor(
        image_dim, kernel_size, n_channels, conv_hidden_dims, dense_hidden_dims, output_dim, adaptive_pooling=True
    ).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 28, 28), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_resnet(x, output_dim):
    batch_dim = x.shape[0]

    model = models.ResNetClassifier(resnet18, ResNet18_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    model = models.ResNetRegressor(resnet18, ResNet18_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 28, 28), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_efficientnet(x, output_dim):
    batch_dim = x.shape[0]

    model = models.EfficientNetClassifier(efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    model = models.EfficientNetRegressor(efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 28, 28), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_vit(x, output_dim):
    batch_dim = x.shape[0]

    if x.shape[-1] != 224 or x.shape[-2] != 224:
        model = models.ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim, transform=False).to(device)

        with pytest.raises(AssertionError):
            out = model(x.to(device))

    model = models.ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim, transform=True).to(device)

    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    if x.shape[-1] != 224 or x.shape[-2] != 224:
        model = models.ViTRegressor(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim, transform=False).to(device)

        with pytest.raises(AssertionError):
            out = model(x.to(device))

    model = models.ViTRegressor(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim, transform=True).to(device)

    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 512, 246), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_alexnet(x, output_dim):
    batch_dim = x.shape[0]

    model = models.AlexNetClassifier(alexnet, AlexNet_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    model = models.AlexNetRegressor(alexnet, AlexNet_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 512, 246), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_densenet(x, output_dim):
    batch_dim = x.shape[0]

    model = models.DenseNetClassifier(densenet121, DenseNet121_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])

    model = models.DenseNetRegressor(densenet121, DenseNet121_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])
