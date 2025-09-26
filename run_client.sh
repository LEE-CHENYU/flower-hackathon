#!/bin/bash
# Start a federated learning client

CLIENT_ID=${1:-0}
echo "Starting Client $CLIENT_ID..."
python run_fl_training.py --mode client --client-id $CLIENT_ID