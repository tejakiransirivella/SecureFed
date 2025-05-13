from typing import List, Tuple
import numpy as np
from torchvision.models import ResNet
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes,parameters_to_ndarrays
import torch
def calculate_distances(model:ResNet,results: List[Tuple[ClientProxy, FitRes]]):
    weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
    distances = []
    layers=list(model.state_dict().values())
    distance=0
    for i in range(len(weights_results)):
        weights=weights_results[i][0]
        for i,layer in enumerate(weights):
            distance+=sum((torch.tensor(layer.flatten())-layers[i].flatten())**2)
        distances.append(np.sqrt(distance))
            
    return distances

def calculate_reputation(prevReputation:float,distance:float,alpha:float,iteration:int):
    if distance<alpha:
        return prevReputation+distance-(prevReputation/iteration)
    return prevReputation+distance-np.exp(-(1-distance*(prevReputation/iteration)))

def calculate_trust(reputation:float,distance:float):
    return np.sqrt(reputation**2+distance**2)-np.sqrt((1-reputation)**2+(1-distance)**2)