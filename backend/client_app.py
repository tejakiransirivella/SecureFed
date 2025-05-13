"""vitfed: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from backend.task import get_weights, load_data, set_weights, test, train
from backend.model import vit5m
from time import time

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs,partitionId):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        print(torch.version.cuda)
        print("CUDA available: ", torch.cuda.is_available())
        print("Using device: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        self.device = torch.device(f"cuda:{partitionId}" if torch.cuda.is_available() else "cpu")
        # self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        start=time()
        print("Training model")
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        end=time()
        print("Time taken for training: ",end-start)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        print("Evaluating model")
        set_weights(self.net, parameters)
        start=time()
        loss, accuracy = test(self.net, self.valloader, self.device)
        end=time()
        print("Time taken for evaluation: ",end-start)
        print("Achieved accuracy of ", accuracy)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}




def get_client_fn(partitionId,attack_flag=False):
    def client_fn(context: Context):
        # Load model and data
        print("Partition ID: ", partitionId)
        num_partitions = context.node_config.get("num-partitions") or 10
        trainloader, valloader = load_data(partitionId, num_partitions,attack_flag)
        local_epochs = context.run_config.get("local-epochs") or 5

        # Return Client instance
        return FlowerClient(vit5m, trainloader, valloader, local_epochs,partitionId).to_client()
    return client_fn


# # Flower ClientApp
# app = ClientApp(
#     client_fn,
# )