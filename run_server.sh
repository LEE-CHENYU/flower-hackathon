#!/bin/bash
# Start the federated learning server

echo "Starting Federated Learning Server..."
python run_fl_training.py --mode server --rounds ${1:-3} --clients ${2:-2}