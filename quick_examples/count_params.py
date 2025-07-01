try:
    from astra.torch.utils import count_params
    from astra.torch.models import MLPRegressor

    mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)

    n_params = count_params(mlp)
    print(n_params)
except Exception as e:
    print("Count parameters demo:")
    print("Note: This example requires PyTorch and other dependencies")  
    print("Error:", str(e))
    print("mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)")
    print("count_params(mlp)")
    print("# Returns the total number of trainable parameters in the model")
