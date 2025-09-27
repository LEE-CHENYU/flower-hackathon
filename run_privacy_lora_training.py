#!/usr/bin/env python3
"""
Privacy-Enabled Federated LoRA Training
Full implementation with actual TinyLLaVA model training and privacy protection
"""

import argparse
import sys
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Process
import torch
import flwr as fl
from typing import Dict, List, Any, Optional

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Import privacy modules
from privacy import (
    PrivacyConfig,
    PrivacyManager,
    ClientPrivacyManager,
    AuthenticationManager,
    SecureAggregationManager
)

# Import federated learning modules
from federated.fl_lora_server import LoRAFedAvg, DentalFLServer
from federated.fl_lora_client import DentalFLClient
from core.llava_lora_model import LLaVALoRAModel
from core.model_configs import get_model_config


class PrivacyEnabledFLClient(DentalFLClient):
    """Privacy-enhanced federated learning client"""

    def __init__(
        self,
        client_id: int,
        data_path: str,
        privacy_manager: Optional[ClientPrivacyManager] = None,
        secure_agg_client = None,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        model_config: str = "tiny-llava"
    ):
        """
        Initialize privacy-enabled FL client

        Args:
            client_id: Client identifier
            data_path: Path to training data
            privacy_manager: Client privacy manager
            secure_agg_client: Secure aggregation client
            batch_size: Training batch size
            learning_rate: Learning rate
            model_config: Model configuration name
        """
        super().__init__(client_id, data_path, batch_size, learning_rate, model_config)

        self.privacy_manager = privacy_manager
        self.secure_agg_client = secure_agg_client

    def _prepare_labels(self):
        """Generate and sanitize labels with privacy protection"""
        print(f"Client {self.client_id}: Generating privacy-protected labels...")

        # Call parent method to generate labels
        super()._prepare_labels()

        # Sanitize labels if privacy manager available
        if self.privacy_manager:
            sanitized_pairs = []
            for img_path, label in self.training_pairs:
                # Sanitize label to remove PII
                sanitized_label = self.privacy_manager._sanitize_client_label(label)
                sanitized_pairs.append((img_path, sanitized_label))

            self.training_pairs = sanitized_pairs
            self.privacy_manager.local_metrics["labels_sanitized"] += len(sanitized_pairs)
            print(f"Client {self.client_id}: Sanitized {len(sanitized_pairs)} labels")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> tuple:
        """Train with privacy protection"""
        # Check if client should participate
        if self.privacy_manager:
            round_num = config.get("server_round", 1)
            if not self.privacy_manager.should_participate(round_num):
                print(f"Client {self.client_id}: Skipping round {round_num} (privacy sampling)")
                # Return empty update
                return self.get_parameters(config), 0, {"skipped": True}

        # Perform training
        result = super().fit(parameters, config)

        # Apply secure aggregation masking if available
        if self.secure_agg_client:
            weights_dict = {}
            weight_names = list(self.model.get_lora_weights().keys())
            for i, name in enumerate(weight_names):
                if i < len(result[0]):
                    weights_dict[name] = result[0][i]

            # Mask weights
            participating = [0, 1, 2]  # Assume 3 clients for demo
            masked = self.secure_agg_client.mask_weights(weights_dict, participating)

            # Replace with masked weights
            masked_arrays = []
            for name in weight_names:
                if name in masked["masked_weights"]:
                    masked_arrays.append(masked["masked_weights"][name]["masked_values"])
                else:
                    masked_arrays.append(result[0][i] if i < len(result[0]) else np.zeros((1,)))

            return masked_arrays, result[1], result[2]

        return result


class PrivacyEnabledFLServer:
    """Privacy-enhanced federated learning server"""

    def __init__(
        self,
        num_rounds: int = 5,
        min_clients: int = 2,
        privacy_config: Optional[PrivacyConfig] = None,
        model_config: str = "tiny-llava",
        output_dir: str = "privacy_training_output"
    ):
        """
        Initialize privacy-enabled FL server

        Args:
            num_rounds: Number of training rounds
            min_clients: Minimum clients before starting
            privacy_config: Privacy configuration
            model_config: Model configuration name
            output_dir: Output directory for results
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize privacy
        self.privacy_config = privacy_config or PrivacyConfig()
        self.shared_auth = AuthenticationManager()
        self.privacy_manager = PrivacyManager(self.privacy_config, shared_auth_manager=self.shared_auth)
        self.secure_agg_manager = SecureAggregationManager(threshold=self.privacy_config.threshold)

        # Initialize FL server
        self.fl_server = DentalFLServer(
            num_rounds=num_rounds,
            min_clients=min_clients,
            min_fit_clients=min_clients,
            min_eval_clients=min_clients,
            local_epochs=1,
            server_address="[::]:8080"
        )

    def start(self):
        """Start privacy-enabled server"""
        print("\n" + "="*70)
        print("PRIVACY-ENABLED FEDERATED LORA TRAINING SERVER")
        print("="*70)
        print(f"Configuration:")
        print(f"  • Model: {self.model_config}")
        print(f"  • Rounds: {self.num_rounds}")
        print(f"  • Min clients: {self.min_clients}")
        print(f"  • Secure aggregation: {'Enabled' if self.privacy_config.enable_secure_aggregation else 'Disabled'}")
        print(f"  • Authentication: {'Enabled' if self.privacy_config.enable_authentication else 'Disabled'}")
        print(f"  • Output directory: {self.output_dir}")
        print("="*70 + "\n")

        # Start FL server
        self.fl_server.start()

        # Save privacy report
        self._save_final_report()

    def _save_final_report(self):
        """Save final privacy report"""
        report = self.privacy_manager.get_privacy_report()
        report["training_config"] = {
            "model": self.model_config,
            "num_rounds": self.num_rounds,
            "min_clients": self.min_clients
        }

        report_path = self.output_dir / "privacy_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✅ Privacy report saved to {report_path}")


def run_privacy_client(
    client_id: int,
    server_address: str,
    data_path: str,
    privacy_config: PrivacyConfig,
    secure_agg_manager: SecureAggregationManager,
    model_config: str = "tiny-llava"
):
    """Run a privacy-enabled client"""
    print(f"\n🔐 Starting Privacy-Enabled Client {client_id}")

    # Create client privacy manager
    client_privacy = ClientPrivacyManager(client_id, privacy_config)

    # Create secure aggregation client
    secure_agg_client = None
    if privacy_config.enable_secure_aggregation:
        secure_agg_client = secure_agg_manager.create_client(client_id, total_clients=3)

    # Create privacy-enabled FL client
    client = PrivacyEnabledFLClient(
        client_id=client_id,
        data_path=data_path,
        privacy_manager=client_privacy,
        secure_agg_client=secure_agg_client,
        model_config=model_config
    )

    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


def run_privacy_fl_training(
    num_clients: int = 3,
    num_rounds: int = 5,
    model_config: str = "tiny-llava",
    output_dir: str = "privacy_training_output"
):
    """
    Run full privacy-enabled federated LoRA training

    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        model_config: Model configuration
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("PRIVACY-ENABLED FEDERATED LORA TRAINING")
    print("="*70)

    # Create privacy configuration
    privacy_config = PrivacyConfig(
        enable_secure_aggregation=True,
        enable_authentication=True,
        enable_tls=True,
        threshold=2,
        track_privacy_metrics=True,
        sanitize_labels=True
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save privacy configuration
    config_path = output_path / "privacy_config.json"
    with open(config_path, 'w') as f:
        json.dump(privacy_config.to_dict(), f, indent=2)
    print(f"✅ Privacy configuration saved to {config_path}")

    processes = []

    try:
        # Start server process
        server_process = Process(
            target=lambda: PrivacyEnabledFLServer(
                num_rounds=num_rounds,
                min_clients=num_clients,
                privacy_config=privacy_config,
                model_config=model_config,
                output_dir=output_dir
            ).start()
        )
        server_process.start()
        processes.append(server_process)

        # Wait for server initialization
        print("\n⏳ Waiting for server to initialize...")
        time.sleep(10)

        # Create shared secure aggregation manager
        secure_agg_manager = SecureAggregationManager(threshold=privacy_config.threshold)
        secure_agg_manager.initialize_round(0, num_clients)

        # Start client processes
        for client_id in range(num_clients):
            client_process = Process(
                target=run_privacy_client,
                args=(
                    client_id,
                    "localhost:8080",
                    "omni_coco",
                    privacy_config,
                    secure_agg_manager,
                    model_config
                )
            )
            client_process.start()
            processes.append(client_process)
            time.sleep(3)  # Stagger client starts

        print(f"\n" + "="*60)
        print(f"Privacy-Enabled Training Started")
        print(f"  • Server: 1 instance")
        print(f"  • Clients: {num_clients} instances")
        print(f"  • Rounds: {num_rounds}")
        print(f"  • Model: {model_config}")
        print(f"  • Output: {output_dir}/")
        print(f"="*60 + "\n")

        # Wait for completion
        for p in processes:
            p.join()

        print("\n✅ Privacy-enabled federated LoRA training completed!")

        # Generate summary
        generate_training_summary(output_path)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
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


def generate_training_summary(output_dir: Path):
    """Generate training summary from privacy outputs"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    # Read privacy report
    report_path = output_dir / "privacy_training_report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)

        print(f"\n📊 Privacy Metrics:")
        print(f"  • Rounds completed: {report['metrics']['rounds_completed']}")
        print(f"  • Secure aggregations: {report['metrics']['secure_aggregations']}")
        print(f"  • Authenticated clients: {report['metrics']['authenticated_clients']}")
        print(f"  • Privacy violations: {report['metrics']['privacy_violations']}")
        print(f"  • Constraints satisfied: {report['constraints_satisfied']}")

    # List checkpoints
    checkpoint_dir = Path("checkpoints/lora_adapters")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("round_*"))
        if checkpoints:
            print(f"\n💾 Model Checkpoints:")
            for cp in sorted(checkpoints)[-3:]:  # Show last 3
                print(f"  • {cp}")

    # List privacy metrics
    metrics_dir = Path("privacy_metrics")
    if metrics_dir.exists():
        metrics_files = list(metrics_dir.glob("round_*.json"))
        if metrics_files:
            print(f"\n📈 Round Metrics:")
            for mf in sorted(metrics_files)[-3:]:  # Show last 3
                print(f"  • {mf}")

    print(f"\n📂 All outputs saved to: {output_dir}/")
    print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Privacy-Enabled Federated LoRA Training")

    parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="Number of federated clients"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="tiny-llava",
        choices=["tiny-llava", "tiny-llava-3b", "llava-7b-qlora"],
        help="Model configuration to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="privacy_training_output",
        help="Output directory for privacy training results"
    )

    args = parser.parse_args()

    # Run privacy-enabled training
    run_privacy_fl_training(
        num_clients=args.clients,
        num_rounds=args.rounds,
        model_config=args.model_config,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()