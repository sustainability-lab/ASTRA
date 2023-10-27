import pytest
import torch
import astra.torch.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, vit_b_16, ViT_B_16_Weights

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
    model = models.MLP(input_dim, [7, 9], output_dim).to(device)

    output = model(x.to(device))
    assert output.shape == out_shape


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
    model = models.SIREN(input_dim, [7, 9], output_dim).to(device)

    output = model(x.to(device))
    assert output.shape == out_shape


@pytest.mark.parametrize("x", [torch.randn(11, 3, 28, 28), torch.randn(11, 5, 224, 224)])
def test_cnn(x):
    batch_dim = x.shape[0]
    image_dim = x.shape[-1]
    n_channels = x.shape[1]
    kernel_size = 3
    conv_hidden_dims = [4, 4, 7]
    dense_hidden_dims = [7, 9]
    output_dim = 12

    model = models.CNN(image_dim, kernel_size, n_channels, conv_hidden_dims, dense_hidden_dims, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 28, 28), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_efficientnet(x, output_dim):
    batch_dim = x.shape[0]

    model = models.EfficientNet(efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])


@pytest.mark.parametrize("x, output_dim", [(torch.randn(11, 3, 28, 28), 4), (torch.randn(11, 3, 224, 224), 6)])
def test_vit(x, output_dim):
    batch_dim = x.shape[0]

    model = models.ViT(vit_b_16, ViT_B_16_Weights.DEFAULT, output_dim).to(device)
    out = model(x.to(device))

    assert out.shape == torch.Size([batch_dim, output_dim])
