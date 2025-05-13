"""vitfed: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import medmnist
from torch.utils.data import random_split
from medmnist import INFO
import numpy as np
fds = None  # Cache FederatedDataset

train_data = None  # Cache train_data
test_data = None

def load_data(partition_id: int, num_partitions: int,attack_flag=False):
    """Load partition medmnist data."""
    # Only initialize `FederatedDataset` once
    # global fds
    # if fds is None: 
    #     partitioner = IidPartitioner(num_partitions=num_partitions)
    #     fds = FederatedDataset(
    #         dataset="maxpmx/pathmnist_train",
    #         partitioners={"train": partitioner},
    #     )
    # partition = fds.load_partition(partition_id)
    # # Divide data on each node: 80% train, 20% test
    # partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # pytorch_transforms = Compose(
    #     [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # def apply_transforms(batch):
    #     """Apply transforms to the partition from FederatedDataset."""
    #     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    #     return batch

    # partition_train_test = partition_train_test.with_transform(apply_transforms)
    # trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    # testloader = DataLoader(partition_train_test["test"], batch_size=32)
    # return trainloader, testloader
    global train_data
    global test_data
    if train_data is None:
        DatasetClass = getattr(medmnist, INFO["pathmnist"]["python_class"])

        pytorch_transforms = Compose(
            [ToTensor(), Normalize(mean=[.5], std=[.5])]
        )

        raw_train_data = DatasetClass(split='train', transform=pytorch_transforms, download=True)
        raw_test_data = DatasetClass(split='test', transform=pytorch_transforms, download=True)

        train_size = len(raw_train_data)
        test_size = len(raw_test_data)
        partition_size_train = train_size // num_partitions
        partition_size_test = test_size // num_partitions
        list_of_lengths_train = [partition_size_train] * (num_partitions - 1) + [train_size - partition_size_train * (num_partitions - 1)]
        list_of_lengths_test = [partition_size_test] * (num_partitions - 1) + [test_size - partition_size_test * (num_partitions - 1)]
        train_data = random_split(raw_train_data, list_of_lengths_train)
        test_data=random_split(raw_test_data,list_of_lengths_test)

    partition_train = train_data[partition_id]
    partition_test = test_data[partition_id]
    count=len(partition_train.dataset.labels)
    if attack_flag and partition_id<num_partitions//3:
        #Randomly flip the labels
        partition_train.dataset.labels=(np.random.rand(count,1)*10)//1
        print("Flipped labels for partition ", partition_id)

    trainloader = DataLoader(partition_train, batch_size=32, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch[0].to(device)
            labels = batch[1].squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader)}")

    avg_trainloss = running_loss / len(trainloader)
    print(f"Average training loss: {avg_trainloss}")
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].squeeze().long().to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)