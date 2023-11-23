from astra.torch.utils import count_params
from astra.torch.models import MLPRegressor

mlp = MLPRegressor(input_dim=2, hidden_dims=[5, 6], output_dim=1)

n_params = count_params(mlp)
print(n_params)
