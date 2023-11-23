from astra.torch.models import MLPRegressor

mlp = MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
print(mlp)
