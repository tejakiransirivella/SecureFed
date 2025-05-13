#!/bin/bash

# Check if the number of clients is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_clients>"
    exit 1
fi

# Assign the argument to a variable
NUM_CLIENTS=$1

# Start main server in the background using nohup and redirect output to server.log
nohup python3 -u main_server.py > logs/server.log 2>&1 &

echo "Main server started with PID $!"

# Wait a moment to ensure the server is up before starting clients
sleep 6

# Start the specified number of client instances in the background using nohup
for ((i=0; i<NUM_CLIENTS; i++))
do
    nohup python3 -u main_client.py $i 1 > logs/client_$i.log 2>&1 &
    echo "Client $i started with PID $!"
done

echo "All $NUM_CLIENTS clients started."

