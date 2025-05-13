from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
import torch
from backend.task import get_weights
from backend.model import vit5m
import flwr as fl
from backend.save_model_strategy import SaveModelStrategy

print("CUDA available: ", torch.cuda.is_available())
num_rounds = 2
fraction_fit = 1

# Initialize model parameters
ndarrays = get_weights(vit5m)
parameters = ndarrays_to_parameters(ndarrays)

# Define strategy
strategy = SaveModelStrategy(
    fraction_fit=fraction_fit,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    min_fit_clients=2,
    min_available_clients=2,
    initial_parameters=parameters,
)
config = ServerConfig(num_rounds=num_rounds)

fl.server.start_server(config=config, strategy=strategy,server_address="[::]:8081")