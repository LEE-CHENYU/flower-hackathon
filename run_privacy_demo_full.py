#!/usr/bin/env python3
"""
Full Privacy-Enabled LoRA Training Demo
Simulates privacy-protected federated learning with LoRA weights
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project to path
import sys
sys.path.append(str(Path(__file__).parent))

from privacy import (
    PrivacyConfig,
    PrivacyManager,
    ClientPrivacyManager,
    AuthenticationManager,
    SecureAggregationManager
)


class PrivacyLoRATraining:
    """Complete privacy-enabled LoRA training simulation"""

    def __init__(self, output_dir: str = "privacy_training_output"):
        """Initialize privacy LoRA training"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Privacy configuration
        self.config = PrivacyConfig(
            enable_secure_aggregation=True,
            enable_authentication=True,
            enable_tls=True,
            threshold=2,
            track_privacy_metrics=True,
            sanitize_labels=True
        )

        # Save configuration
        config_path = self.output_dir / "privacy_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Initialize components
        self.shared_auth = AuthenticationManager()
        self.server_privacy = PrivacyManager(self.config, shared_auth_manager=self.shared_auth)
        self.secure_agg = SecureAggregationManager(threshold=self.config.threshold)

        # Client managers
        self.client_managers = []

        # Training history
        self.training_history = []
        self.model_checkpoints = []

    def simulate_lora_weights(self, client_id: int, round_num: int) -> Dict[str, np.ndarray]:
        """Simulate LoRA weight updates from training"""
        np.random.seed(client_id * 100 + round_num * 10)

        # Simulate TinyLLaVA LoRA dimensions
        weights = {
            "vision_model.lora_A": np.random.randn(16, 768).astype(np.float32) * 0.01,
            "vision_model.lora_B": np.random.randn(768, 16).astype(np.float32) * 0.01,
            "language_model.q_proj.lora_A": np.random.randn(8, 4096).astype(np.float32) * 0.01,
            "language_model.q_proj.lora_B": np.random.randn(4096, 8).astype(np.float32) * 0.01,
            "language_model.v_proj.lora_A": np.random.randn(8, 4096).astype(np.float32) * 0.01,
            "language_model.v_proj.lora_B": np.random.randn(4096, 8).astype(np.float32) * 0.01,
        }

        # Add client-specific patterns
        for key in weights:
            weights[key] += (client_id + 1) * 0.001

        return weights

    def generate_training_data(self, client_id: int) -> Dict[str, List[str]]:
        """Generate training data with PII"""
        images = [f"dental_xray_{client_id}_{i}.jpg" for i in range(10)]

        labels = []
        for i in range(10):
            patient_labels = [
                f"Patient John Doe{i}, SSN: 123-45-{6000+i}, severe cavity in tooth #14",
                f"Mary Smith{i}, email: mary{i}@hospital.com, periodontal disease stage 2",
                f"Treatment for patient {i} at 555-{1234+i}, root canal needed",
                f"Dental exam on {i+1}/20/2024, patient ID: PAT{1000+i}",
                f"Follow-up needed for client at clinic@dental.com, phone: 555-{2000+i}",
                f"Regular cleaning for patient, insurance ID: INS{5000+i}",
                f"Orthodontic consultation, patient age 15, guardian: parent{i}@email.com",
                f"Wisdom tooth extraction scheduled, contact: 555-{3000+i}",
                f"Cavity filling completed for patient {client_id}-{i}",
                f"Dental hygiene assessment, next visit: {(i+2)%12+1}/15/2024"
            ]
            labels.append(patient_labels[i % len(patient_labels)])

        return {"images": images, "labels": labels}

    def run_training_round(self, round_num: int, num_clients: int = 3) -> Dict[str, Any]:
        """Run one round of privacy-enabled training"""
        print(f"\n{'='*70}")
        print(f"TRAINING ROUND {round_num + 1}")
        print(f"{'='*70}")

        # Initialize round
        self.server_privacy.initialize_round(round_num + 1)
        self.secure_agg.initialize_round(round_num + 1, num_clients)

        participating_clients = []
        client_updates = []
        training_metrics = {
            "round": round_num + 1,
            "clients": {},
            "aggregated_weights": None
        }

        # Process each client
        for client_id in range(num_clients):
            client_mgr = self.client_managers[client_id]

            # Check participation
            if client_mgr.should_participate(round_num + 1):
                print(f"\n📱 Client {client_id}:")
                participating_clients.append(client_id)

                # Generate and sanitize data
                data = self.generate_training_data(client_id)
                images, sanitized_labels = client_mgr.prepare_training_data(
                    data["images"], data["labels"]
                )
                print(f"  ✓ Processed {len(images)} images")
                print(f"  ✓ Sanitized {len(sanitized_labels)} labels (removed PII)")

                # Simulate training and get LoRA weights
                weights = self.simulate_lora_weights(client_id, round_num)
                print(f"  ✓ Trained LoRA adapters: {len(weights)} layers")

                # Apply secure aggregation
                secure_agg_client = self.secure_agg.clients.get(client_id)
                if secure_agg_client:
                    masked = secure_agg_client.mask_weights(weights, participating_clients)
                    print(f"  ✓ Applied secure masking to weights")
                else:
                    masked = {"masked_weights": weights}

                # Prepare authenticated update
                update = self.server_privacy.prepare_client_update(client_id, weights)
                update["masked_weights"] = masked.get("masked_weights", weights)
                client_updates.append(update)

                # Track metrics
                training_metrics["clients"][client_id] = {
                    "samples": len(images),
                    "labels_sanitized": client_mgr.local_metrics["labels_sanitized"],
                    "weight_layers": list(weights.keys())
                }

                print(f"  ✓ Prepared secure update with authentication")
            else:
                print(f"\n📱 Client {client_id}: ⏭  Not participating (privacy sampling)")

        # Aggregate weights securely
        print(f"\n🔒 Secure Aggregation:")
        if client_updates:
            aggregated = self.server_privacy.aggregate_secure_updates(client_updates)
            training_metrics["aggregated_weights"] = {
                key: {"shape": val.shape, "mean": float(val.mean()), "std": float(val.std())}
                for key, val in aggregated.items()
            }

            print(f"  ✓ Aggregated {len(aggregated)} weight layers from {len(client_updates)} clients")
            for layer, values in aggregated.items():
                print(f"    • {layer}: shape {values.shape}")

            # Save checkpoint
            checkpoint_path = self.output_dir / f"checkpoint_round_{round_num + 1}"
            checkpoint_path.mkdir(exist_ok=True)
            np.savez(checkpoint_path / "lora_weights.npz", **aggregated)
            self.model_checkpoints.append(str(checkpoint_path))
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")

        # Validate privacy
        print(f"\n🔐 Privacy Validation:")
        if self.server_privacy.validate_privacy_constraints():
            print("  ✅ All privacy constraints satisfied")
        else:
            print("  ⚠️  Privacy constraints need attention")

        # Save round report
        self.server_privacy.finalize_round(round_num + 1)
        training_metrics["privacy_satisfied"] = self.server_privacy.validate_privacy_constraints()
        training_metrics["participating_clients"] = participating_clients

        return training_metrics

    def run_full_training(self, num_rounds: int = 5, num_clients: int = 3):
        """Run complete privacy-enabled training"""
        print("\n" + "="*70)
        print("PRIVACY-ENABLED LORA TRAINING")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"  • Model: TinyLLaVA (simulated)")
        print(f"  • Clients: {num_clients}")
        print(f"  • Rounds: {num_rounds}")
        print(f"  • Output: {self.output_dir}/")

        # Register clients
        print(f"\n🔐 Initializing Privacy System...")
        for client_id in range(num_clients):
            token = self.server_privacy.register_client(client_id)
            client_mgr = ClientPrivacyManager(client_id, self.config)
            self.client_managers.append(client_mgr)
            print(f"  ✓ Client {client_id} registered with secure token")

        # Setup secure aggregation
        self.secure_agg.initialize_round(0, num_clients)
        for client_id in range(num_clients):
            self.secure_agg.create_client(client_id, num_clients)
        self.secure_agg.setup_key_agreement()
        print(f"  ✓ Secure aggregation keys exchanged")

        # Run training rounds
        start_time = time.time()
        for round_num in range(num_rounds):
            round_metrics = self.run_training_round(round_num, num_clients)
            self.training_history.append(round_metrics)

        training_time = time.time() - start_time

        # Generate final report
        self.generate_final_report(num_rounds, training_time)

    def generate_final_report(self, num_rounds: int, training_time: float):
        """Generate comprehensive training report"""
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        # Calculate statistics
        total_samples = sum(
            sum(m["clients"].get(c, {}).get("samples", 0) for c in range(3))
            for m in self.training_history
        )
        total_sanitized = sum(
            mgr.local_metrics["labels_sanitized"]
            for mgr in self.client_managers
        )

        print(f"\n📈 Training Summary:")
        print(f"  • Rounds completed: {num_rounds}")
        print(f"  • Total samples processed: {total_samples}")
        print(f"  • Labels sanitized: {total_sanitized}")
        print(f"  • Training time: {training_time:.2f} seconds")
        print(f"  • Privacy violations: {self.server_privacy.metrics['privacy_violations']}")

        # Final privacy report
        final_report = self.server_privacy.get_privacy_report()
        final_report["training_summary"] = {
            "num_rounds": num_rounds,
            "training_time_seconds": training_time,
            "total_samples": total_samples,
            "total_labels_sanitized": total_sanitized,
            "model_checkpoints": self.model_checkpoints,
            "training_history": self.training_history
        }

        # Save final report
        report_path = self.output_dir / "final_privacy_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        print(f"\n💾 Outputs Saved:")
        print(f"  • Privacy report: {report_path}")
        print(f"  • Training history: {history_path}")
        print(f"  • Model checkpoints: {len(self.model_checkpoints)} saved")
        print(f"  • Privacy metrics: privacy_metrics/")

        print(f"\n🔐 Privacy Status:")
        if final_report["constraints_satisfied"]:
            print("  ✅ All privacy constraints satisfied throughout training")
        else:
            print("  ⚠️  Some privacy constraints were violated")

        print(f"\n✨ Privacy-enabled LoRA training completed successfully!")
        print(f"📂 All outputs saved to: {self.output_dir}/")


def main():
    """Run the privacy LoRA training demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Privacy-Enabled LoRA Training Demo")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--output", type=str, default="privacy_training_output", help="Output directory")

    args = parser.parse_args()

    # Run training
    trainer = PrivacyLoRATraining(output_dir=args.output)
    trainer.run_full_training(num_rounds=args.rounds, num_clients=args.clients)


if __name__ == "__main__":
    main()