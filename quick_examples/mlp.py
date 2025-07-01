try:
    from astra.torch.models import MLPRegressor
    
    mlp = MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation="relu", dropout=0.1)
    print(mlp)
except Exception as e:
    print("MLPRegressor demo:")
    print("Note: This example requires PyTorch and other dependencies")
    print("Error:", str(e))
    print("MLPRegressor(input_dim=100, hidden_dims=[128, 64], output_dim=10, activation='relu', dropout=0.1)")
    print("# Creates a multi-layer perceptron with specified architecture")
