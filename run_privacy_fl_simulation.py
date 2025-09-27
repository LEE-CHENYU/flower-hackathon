#!/usr/bin/env python3
"""
Privacy-Enabled Federated Learning Simulation
Runs 3 clients for 5 rounds with full privacy protection
"""

import numpy as np
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Add project to path
sys.path.append(str(Path(__file__).parent))

from privacy import (
    PrivacyConfig,
    PrivacyManager,
    ClientPrivacyManager,
    AuthenticationManager,
    SecureAggregationManager
)


class PrivacyFLSimulation:
    """Simulated privacy-enabled federated learning"""

    def __init__(self, num_clients: int = 3, num_rounds: int = 5):
        """
        Initialize privacy FL simulation

        Args:
            num_clients: Number of federated clients
            num_rounds: Number of training rounds
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds

        # Privacy configuration
        self.config = PrivacyConfig(
            enable_secure_aggregation=True,
            enable_authentication=True,
            enable_tls=True,
            threshold=2,  # Min clients for secret reconstruction
            track_privacy_metrics=True,
            sanitize_labels=True
        )

        # Create shared authentication manager
        self.shared_auth = AuthenticationManager()

        # Initialize server privacy manager
        self.server_privacy = PrivacyManager(self.config, shared_auth_manager=self.shared_auth)

        # Initialize secure aggregation manager
        self.secure_agg_manager = SecureAggregationManager(threshold=self.config.threshold)

        # Client privacy managers
        self.client_managers = []

        # Training metrics
        self.round_metrics = []

    def initialize_clients(self):
        """Initialize and register all clients"""
        print(f"\n📋 Registering {self.num_clients} clients with server...")

        # Register clients with server
        self.client_tokens = {}
        for client_id in range(self.num_clients):
            token = self.server_privacy.register_client(client_id)
            self.client_tokens[client_id] = token

            # Create client privacy manager
            client_manager = ClientPrivacyManager(client_id, self.config)
            self.client_managers.append(client_manager)

            print(f"   ✓ Client {client_id} registered and authenticated")

        # Initialize secure aggregation with server first
        print(f"\n🔐 Setting up secure aggregation...")
        self.secure_agg_manager.initialize_round(0, self.num_clients)  # Initialize server

        # Then create clients
        for client_id in range(self.num_clients):
            self.secure_agg_manager.create_client(client_id, self.num_clients)

        # Perform key agreement
        self.secure_agg_manager.setup_key_agreement()
        print(f"   ✓ Key agreement completed between all clients")

    def generate_client_data(self, client_id: int) -> Dict[str, Any]:
        """
        Generate training data for a client

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with images and labels
        """
        # Simulate images
        num_samples = 5
        images = [f"client_{client_id}_image_{i}.jpg" for i in range(num_samples)]

        # Simulate labels with PII that needs sanitization
        raw_labels = []
        for i in range(num_samples):
            labels_pool = [
                f"Patient {client_id}-{i}, email: patient{i}@hospital.com, phone: 555-{1000+i}",
                f"Dental exam on {i+1}/15/2024 for John Doe{i}",
                f"Follow-up needed, contact: 555-{2000+i}",
                f"Treatment plan for Mary Smith{i} at clinic",
                f"Regular cleaning, patient ID: 123-45-{6000+i}"
            ]
            raw_labels.append(labels_pool[i % len(labels_pool)])

        return {"images": images, "labels": raw_labels}

    def simulate_training(self, client_id: int, round_num: int) -> np.ndarray:
        """
        Simulate local training for a client

        Args:
            client_id: Client identifier
            round_num: Current round number

        Returns:
            LoRA weight updates
        """
        # Simulate training with some variation per client and round
        np.random.seed(client_id * 100 + round_num)

        weights = {
            "lora_A": np.random.randn(8, 768).astype(np.float32) * 0.01,
            "lora_B": np.random.randn(768, 8).astype(np.float32) * 0.01,
        }

        # Add client-specific patterns
        weights["lora_A"] += (client_id + 1) * 0.001
        weights["lora_B"] -= (client_id + 1) * 0.001

        return weights

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Run one round of privacy-enabled federated learning

        Args:
            round_num: Round number (0-indexed)

        Returns:
            Round metrics
        """
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{self.num_rounds}")
        print(f"{'='*70}")

        # Initialize round
        self.server_privacy.initialize_round(round_num + 1)
        self.secure_agg_manager.initialize_round(round_num + 1, self.num_clients)

        # Track participating clients
        participating_clients = []
        client_updates = []

        # Process each client
        for client_id in range(self.num_clients):
            client_manager = self.client_managers[client_id]

            # Check if client participates (privacy-preserving sampling)
            if client_manager.should_participate(round_num + 1):
                print(f"\n👤 Client {client_id}:")
                participating_clients.append(client_id)

                # Generate training data
                data = self.generate_client_data(client_id)

                # Sanitize labels (remove PII)
                images, sanitized_labels = client_manager.prepare_training_data(
                    data["images"], data["labels"]
                )
                print(f"   ✓ Sanitized {len(sanitized_labels)} training labels")

                # Simulate local training
                weights = self.simulate_training(client_id, round_num)
                print(f"   ✓ Completed local training")

                # Apply secure aggregation masking
                masked_update = self.secure_agg_manager.mask_weights(client_id, weights)

                # Prepare update with authentication
                secure_update = self.server_privacy.prepare_client_update(client_id, weights)
                secure_update["masked_weights"] = masked_update.get("masked_weights", weights)

                client_updates.append(secure_update)
                print(f"   ✓ Prepared secure update with masking and authentication")

                # Update client metrics
                metrics = client_manager.get_local_privacy_metrics()
                print(f"   📊 Privacy metrics: {metrics['labels_sanitized']} labels sanitized")
            else:
                print(f"\n👤 Client {client_id}: ⏭  Not participating (privacy sampling)")

        # Aggregate updates securely
        print(f"\n🔄 Secure Aggregation:")
        print(f"   • {len(participating_clients)} clients participating: {participating_clients}")

        if client_updates:
            # Perform secure aggregation
            aggregated_weights = self.server_privacy.aggregate_secure_updates(client_updates)

            print(f"   ✓ Securely aggregated weights from {len(client_updates)} clients")
            for layer, values in aggregated_weights.items():
                print(f"     - {layer}: shape {values.shape}, mean {values.mean():.6f}, std {values.std():.6f}")
        else:
            aggregated_weights = {}
            print(f"   ⚠ No client updates received this round")

        # Validate privacy constraints
        print(f"\n✅ Privacy Validation:")
        if self.server_privacy.validate_privacy_constraints():
            print("   ✓ All privacy constraints satisfied")
        else:
            print("   ⚠ Some privacy constraints need attention")

        # Finalize round
        self.server_privacy.finalize_round(round_num + 1)

        # Collect round metrics
        round_metrics = {
            "round": round_num + 1,
            "participating_clients": participating_clients,
            "num_participants": len(participating_clients),
            "aggregated_layers": list(aggregated_weights.keys()),
            "privacy_violations": self.server_privacy.metrics["privacy_violations"],
            "authenticated_clients": len(self.server_privacy.metrics["authenticated_clients"])
        }

        return round_metrics

    def run_simulation(self):
        """Run the complete FL simulation with privacy"""
        print("\n" + "="*70)
        print("PRIVACY-ENABLED FEDERATED LEARNING SIMULATION")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   • Clients: {self.num_clients}")
        print(f"   • Rounds: {self.num_rounds}")
        print(f"   • Secure Aggregation: Enabled")
        print(f"   • Authentication: Token-based")
        print(f"   • PII Sanitization: Enabled")
        print(f"   • Secret Sharing Threshold: {self.config.threshold}")

        # Initialize clients
        self.initialize_clients()

        # Run training rounds
        start_time = time.time()

        for round_num in range(self.num_rounds):
            round_metrics = self.run_round(round_num)
            self.round_metrics.append(round_metrics)

        training_time = time.time() - start_time

        # Generate final report
        self.generate_final_report(training_time)

    def generate_final_report(self, training_time: float):
        """Generate and save final privacy report"""
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print("="*70)

        # Summary statistics
        total_participants = sum(m["num_participants"] for m in self.round_metrics)
        avg_participants = total_participants / self.num_rounds

        print(f"\n📈 Training Summary:")
        print(f"   • Total rounds completed: {self.num_rounds}")
        print(f"   • Average clients per round: {avg_participants:.1f}")
        print(f"   • Total training time: {training_time:.2f} seconds")
        print(f"   • Privacy violations: {self.server_privacy.metrics['privacy_violations']}")

        # Per-round summary
        print(f"\n📊 Round-by-Round Summary:")
        for metrics in self.round_metrics:
            print(f"   Round {metrics['round']}: {metrics['num_participants']} clients " +
                  f"({metrics['participating_clients']})")

        # Get privacy report
        final_report = self.server_privacy.get_privacy_report()
        final_report["simulation_summary"] = {
            "total_rounds": self.num_rounds,
            "total_clients": self.num_clients,
            "training_time_seconds": training_time,
            "average_participants_per_round": avg_participants,
            "round_metrics": self.round_metrics
        }

        # Save report
        report_path = "privacy_fl_simulation_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print(f"\n💾 Reports Saved:")
        print(f"   • Full privacy report: {report_path}")
        print(f"   • Per-round metrics: privacy_metrics/")

        # Final status
        print(f"\n🔐 Privacy Status:")
        if final_report["constraints_satisfied"]:
            print("   ✅ All privacy constraints satisfied throughout training")
        else:
            print("   ⚠️  Some privacy constraints were violated")

        print(f"\n✨ Privacy-enabled federated learning completed successfully!")


def main():
    """Run privacy-enabled FL simulation"""
    import argparse

    parser = argparse.ArgumentParser(description="Privacy-Enabled FL Simulation")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")

    args = parser.parse_args()

    # Run simulation
    simulation = PrivacyFLSimulation(
        num_clients=args.clients,
        num_rounds=args.rounds
    )

    try:
        simulation.run_simulation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()