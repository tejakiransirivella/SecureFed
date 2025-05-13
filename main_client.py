import flwr as fl
from backend.client_app import get_client_fn
import torch
import sys
print("CUDA available: ", torch.cuda.is_available())

partitionID= int(sys.argv[1]) if len(sys.argv) > 1 else 0
attack_flag= int(sys.argv[2]) if len(sys.argv) > 2 else 0

def start_client():
    fl.client.start_client(
        server_address="localhost:8081",
        client_fn=get_client_fn(partitionID,attack_flag),
    )

# threads = []
# for i in range(clients):
#     t = threading.Thread(target=start_client)
#     threads.append(t)
#     t.start()

start_client()