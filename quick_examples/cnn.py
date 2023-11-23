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
