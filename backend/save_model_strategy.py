from typing import Dict, List, Optional, Tuple,Union,OrderedDict
from backend.model import vit5m
import flwr as fl
import numpy as np
import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from backend.distance import calculate_distances,calculate_reputation,calculate_trust

defense=True
minRoundsBeforeDefense=10
reputations=[]
trusts=[]
ALPHA=0.3
BETA=0.8
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        global reputations,trusts
        client_count=len(results)
        if len(reputations)==0:
            reputations=[1]*client_count
            trusts=[1]*client_count
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(vit5m.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            if defense:
                tmpModel=vit5m 
                tmpModel.load_state_dict(state_dict, strict=True)
                distances=calculate_distances(tmpModel,results)
                mxd=max(distances)
                for i in range(len(distances)):
                    distances[i]=distances[i]/mxd
                for i in range(len(reputations)):
                    reputations[i]=calculate_reputation(reputations[i],distances[i],ALPHA,server_round)
                    trusts[i]=calculate_trust(reputations[i],distances[i])
                if server_round>=minRoundsBeforeDefense:
                    trusted_results=[results[i] for i in range(len(trusts)) if trusts[i]>=BETA]
                else:
                    trusted_results=results
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                    server_round, trusted_results, failures
                )
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )
                params_dict = zip(vit5m.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            vit5m.load_state_dict(state_dict, strict=True)
            if defense:
                distances=[distance.item() for distance in distances]
                reputations=[reputation.item() for reputation in reputations]
                trusts=[trust.item() for trust in trusts]
                print("Round",server_round,"results")
                print("Distances",distances)
                print("Reputations",reputations)
                print("Trusts",trusts)
                print("Dropped clients",[i for i in range(len(trusts)) if trusts[i]<BETA])
            # Save the model
            torch.save(vit5m.state_dict(), f"checkpoints/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics