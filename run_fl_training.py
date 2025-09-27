#!/usr/bin/env python3
"""
Main script to run federated learning with LoRA fine-tuning
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
from multiprocessing import Process
import signal
import os

# Silence tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project to path
sys.path.append(str(Path(__file__).parent))

from federated.fl_lora_server import start_server
from federated.fl_lora_client import start_client

def run_server(rounds: int, min_clients: int, local_epochs: int):
    """Run the federated learning server"""
    print("Starting FL Server...")
    start_server(
        num_rounds=rounds,
        min_clients=min_clients,
        local_epochs=local_epochs
    )

def run_client(client_id: int, data_path: str, server_address: str, model_config: str = "tiny-llava"):
    """Run a federated learning client"""
    print(f"Starting Client {client_id} with {model_config} model...")
    from federated.fl_lora_client import DentalFLClient
    import flwr as fl

    client = DentalFLClient(
        client_id=client_id,
        data_path=data_path,
        model_config=model_config
    )
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

def run_simulation(num_clients: int = 3, rounds: int = 10, local_epochs: int = 1, model_config: str = "tiny-llava"):
    """Run federated learning simulation with multiple clients"""

    processes = []

    try:
        # Start server
        server_process = Process(
            target=run_server,
            args=(rounds, num_clients, local_epochs)
        )
        server_process.start()
        processes.append(server_process)

        # Wait longer for server to fully initialize
        print("Waiting for server to initialize...")
        time.sleep(10)  # Increased wait time

        # Start clients with better error handling
        for client_id in range(num_clients):
            client_process = Process(
                target=run_client,
                args=(client_id, "omni_coco", "localhost:8080", model_config)
            )
            client_process.start()
            processes.append(client_process)
            time.sleep(3)  # More stagger between clients

        print(f"\n{'='*60}")
        print(f"Federated Learning Started")
        print(f"Server: 1 instance")
        print(f"Clients: {num_clients} instances")
        print(f"Rounds: {rounds}")
        print(f"Local epochs per round: {local_epochs}")
        print(f"{'='*60}\n")

        # Wait for all processes
        for p in processes:
            p.join()

        print("\n✅ Federated learning completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Shutting down...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        for p in processes:
            if p.is_alive():
                p.kill()
        raise

def evaluate_model(checkpoint_path: str = "checkpoints/lora_adapters/final"):
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)

    try:
        from core.llava_lora_model import LLaVALoRAModel
        from data_loader import FirstPhotoDataset

        # Load model with fine-tuned LoRA
        model = LLaVALoRAModel()

        # Check if checkpoint exists before loading
        if Path(checkpoint_path).exists():
            model.load_lora_adapter(checkpoint_path)
            print(f"Loaded fine-tuned model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, using base model")

        # Load test images
        dataset = FirstPhotoDataset("omni_coco")
        test_images = [item[0] for item in dataset.image_paths[:5]]  # Fixed attribute name

        # Generate diagnoses
        print("\nGenerating diagnoses for test images:")
        for i, image_path in enumerate(test_images, 1):
            print(f"\nImage {i}: {Path(image_path).name}")
            diagnosis = model.generate_diagnosis(image_path)
            print(f"Diagnosis: {diagnosis[:200]}...")

    except Exception as e:
        print(f"Evaluation error: {e}")

def cleanup_old_files():
    """Remove unnecessary files as specified in the plan"""
    files_to_delete = [
        "dental_analysis.py",
        "dental_comparison.py",
        "dental_diagnosis.py",
        "dental_diagnosis_vision.py",
        "dental_simple.py",
        "dental_direct.py",
        "privacy_model.py",
        "visualize_privacy.py",
    ]

    dirs_to_delete = [
        "__pycache__",
        "dental_diagnosis_output",
        "dental_direct_output",
        "dental_output",
        "dental_vision_output",
        "privacy_output",
    ]

    print("\n" + "="*60)
    print("Cleaning up old files")
    print("="*60)

    for file in files_to_delete:
        file_path = Path(file)
        if file_path.exists():
            file_path.unlink()
            print(f"✓ Deleted: {file}")

    for dir_name in dirs_to_delete:
        dir_path = Path(dir_name)
        if dir_path.exists():
            import shutil
            shutil.rmtree(dir_path)
            print(f"✓ Deleted directory: {dir_name}")

    print("\n✅ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning with LoRA Fine-tuning")

    parser.add_argument(
        "--mode",
        choices=["simulate", "server", "client", "evaluate", "cleanup", "list-models"],
        default="simulate",
        help="Execution mode"
    )
    parser.add_argument("--clients", type=int, default=3, help="Number of clients (simulate mode)")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID (client mode)")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--data-path", type=str, default="first_images_dataset", help="Path to data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_adapters/final", help="Checkpoint path for evaluation")
    parser.add_argument(
        "--model-config",
        type=str,
        default="tiny-llava",
        choices=["tiny-llava", "tiny-llava-3b", "llava-7b-qlora", "llava-7b-8bit", "llava-7b-full"],
        help="Model configuration to use"
    )

    args = parser.parse_args()

    if args.mode == "simulate":
        # Run full simulation
        run_simulation(
            num_clients=args.clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            model_config=args.model_config
        )
        # Evaluate after training
        evaluate_model(args.checkpoint)

    elif args.mode == "server":
        # Run server only
        run_server(
            rounds=args.rounds,
            min_clients=args.clients,
            local_epochs=args.local_epochs
        )

    elif args.mode == "client":
        # Run client only
        run_client(
            client_id=args.client_id,
            data_path=args.data_path,
            server_address=args.server_address,
            model_config=args.model_config
        )

    elif args.mode == "evaluate":
        # Evaluate model
        evaluate_model(args.checkpoint)

    elif args.mode == "cleanup":
        # Clean up old files
        cleanup_old_files()

    elif args.mode == "list-models":
        # List available models
        from core.model_configs import print_model_comparison
        print_model_comparison()

if __name__ == "__main__":
    main()