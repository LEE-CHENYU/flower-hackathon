#!/usr/bin/env python3

import argparse
import multiprocessing
import time
import flwr as fl
from pathlib import Path

from data_loader import create_federated_dataloaders
from flower_client import create_client_fn
from flower_server import start_server


def run_server(num_rounds: int, num_clients: int, privacy_budget: float):
    print("\n🚀 Starting Flower Server...")
    start_server(
        num_rounds=num_rounds,
        num_clients=num_clients,
        privacy_budget_threshold=privacy_budget,
        server_address="0.0.0.0:8080"
    )


def run_client(client_id: int, client_dataloaders: dict,
               privacy_budget: float, blur_strength: int):
    time.sleep(5)

    print(f"\n🤖 Starting Client {client_id}...")

    client_fn = create_client_fn(
        client_dataloaders,
        privacy_budget=privacy_budget,
        blur_strength=blur_strength
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client_fn(str(client_id))
    )


def run_simulation(args):
    print("\n" + "="*60)
    print("PRIVACY-PRESERVING FEDERATED LEARNING SYSTEM")
    print("="*60)
    print(f"\n📁 Data directory: {args.data_dir}")
    print(f"👥 Number of clients: {args.num_clients}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"🔒 Privacy budget: {args.privacy_budget}")
    print(f"🌫️  Blur strength: {args.blur_strength}")
    print(f"📦 Batch size: {args.batch_size}")

    print("\n📊 Loading and partitioning data...")
    client_dataloaders = create_federated_dataloaders(
        args.data_dir,
        num_clients=args.num_clients,
        batch_size=args.batch_size
    )

    if args.mode == "simulation":
        print("\n🎮 Running in simulation mode (all clients on same machine)...")

        from flwr.simulation import start_simulation

        client_fn = create_client_fn(
            client_dataloaders,
            privacy_budget=args.privacy_budget,
            blur_strength=args.blur_strength
        )

        def client_fn_wrapper(cid: str):
            return client_fn(cid)

        from flower_server import create_strategy

        strategy = create_strategy(
            num_rounds=args.num_rounds,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            privacy_budget_threshold=args.privacy_budget,
        )

        start_simulation(
            client_fn=client_fn_wrapper,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

    else:
        print("\n🌐 Running in distributed mode...")
        print("Starting server and clients as separate processes...")

        server_process = multiprocessing.Process(
            target=run_server,
            args=(args.num_rounds, args.num_clients, args.privacy_budget)
        )
        server_process.start()

        time.sleep(5)

        client_processes = []
        for client_id in range(args.num_clients):
            client_process = multiprocessing.Process(
                target=run_client,
                args=(client_id, client_dataloaders,
                      args.privacy_budget, args.blur_strength)
            )
            client_process.start()
            client_processes.append(client_process)
            time.sleep(1)

        for process in client_processes:
            process.join()

        server_process.join()

    print("\n✅ Federated learning completed successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Federated Learning for Photo Protection"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/chenyusu/flower-hackathon/omni_coco",
        help="Path to the omni_coco directory"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of federated learning clients"
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Number of federated learning rounds"
    )

    parser.add_argument(
        "--privacy-budget",
        type=float,
        default=1.0,
        help="Privacy budget for differential privacy (lower = more private)"
    )

    parser.add_argument(
        "--blur-strength",
        type=int,
        default=31,
        help="Blur strength for face detection (higher = more blur, must be odd)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "distributed"],
        default="simulation",
        help="Run mode: simulation (single machine) or distributed (multiple processes)"
    )

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"❌ Error: Data directory {args.data_dir} does not exist!")
        return

    run_simulation(args)


if __name__ == "__main__":
    main()