#!/usr/bin/env python3
"""
Demo script showing how to run federated learning with privacy features
This demonstrates the integration of privacy protection into the LoRA training flow
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent))

from privacy import PrivacyConfig, PrivacyManager, ClientPrivacyManager


def demo_privacy_features():
    """Demonstrate privacy features for federated learning"""

    print("\n" + "="*70)
    print("FEDERATED LEARNING WITH PRIVACY PROTECTION DEMO")
    print("="*70)

    print("\n📊 Privacy Features Available:")
    print("1. ✅ Secure Aggregation (Barracuda integration)")
    print("2. ✅ Client Authentication (Token-based)")
    print("3. ✅ Secure Communication (TLS/SSL)")
    print("4. ✅ Label Sanitization (PII removal)")
    print("5. ✅ Privacy Metrics Tracking")

    print("\n" + "-"*70)
    print("CONFIGURATION")
    print("-"*70)

    # Configure privacy settings
    config = PrivacyConfig(
        enable_secure_aggregation=True,
        enable_authentication=True,
        enable_tls=True,
        threshold=2,  # Minimum clients for secret reconstruction
        track_privacy_metrics=True,
        sanitize_labels=True
    )

    print("Privacy configuration:")
    print(f"  • Secure Aggregation: {'Enabled' if config.enable_secure_aggregation else 'Disabled'}")
    print(f"  • Authentication: {'Enabled' if config.enable_authentication else 'Disabled'}")
    print(f"  • TLS/SSL: {'Enabled' if config.enable_tls else 'Disabled'}")
    print(f"  • Secret Sharing Threshold: {config.threshold}")
    print(f"  • Privacy Metrics: {'Tracked' if config.track_privacy_metrics else 'Not tracked'}")

    print("\n" + "-"*70)
    print("PRIVACY-ENABLED FEDERATED LEARNING WORKFLOW")
    print("-"*70)

    # Initialize server with privacy (with shared auth manager)
    print("\n1️⃣  Initializing Privacy-Enabled Server")
    from privacy import AuthenticationManager

    # Create shared authentication manager
    shared_auth = AuthenticationManager()

    server_privacy = PrivacyManager(config, shared_auth_manager=shared_auth)
    server_privacy.initialize_round(1)
    print("   ✓ Server privacy manager initialized")
    print("   ✓ Secure aggregation ready")
    print("   ✓ Authentication system active")

    # Simulate 3 clients
    num_clients = 3
    print(f"\n2️⃣  Creating {num_clients} Privacy-Enabled Clients")

    # Register all clients with the server first
    client_tokens = {}
    for client_id in range(num_clients):
        token = server_privacy.register_client(client_id)
        client_tokens[client_id] = token
        print(f"   ✓ Registered client {client_id} with server")

    client_updates = []
    for client_id in range(num_clients):
        print(f"\n   Client {client_id}:")

        # Create client privacy manager
        client_privacy = ClientPrivacyManager(client_id, config)

        # Check if client participates (privacy-preserving sampling)
        if client_privacy.should_participate(round_number=1):
            print(f"   ✓ Participating in this round")

            # Simulate training data with PII
            raw_labels = [
                "Patient John Doe, email: john@example.com, phone: 555-1234",
                "Treatment for patient at clinic on 01/15/2024",
                "Follow-up needed for client Mary Smith"
            ]

            # Sanitize labels
            images = [f"image_{client_id}_{i}.jpg" for i in range(3)]
            _, sanitized_labels = client_privacy.prepare_training_data(images, raw_labels)

            print(f"   ✓ Sanitized {len(sanitized_labels)} labels (removed PII)")

            # Generate sample LoRA weights (simulating training)
            weights = {
                "lora_A": np.random.randn(8, 768).astype(np.float32) * 0.01,
                "lora_B": np.random.randn(768, 8).astype(np.float32) * 0.01
            }

            # Prepare secure update
            secure_update = server_privacy.prepare_client_update(client_id, weights)
            client_updates.append(secure_update)

            print(f"   ✓ Prepared secure update with authentication token")

            # Show metrics
            metrics = client_privacy.get_local_privacy_metrics()
            print(f"   📈 Client metrics: {metrics['labels_sanitized']} labels sanitized")
        else:
            print(f"   ⏭  Skipping this round (privacy-preserving sampling)")

    print(f"\n3️⃣  Secure Aggregation")
    print(f"   • Received updates from {len(client_updates)} clients")

    # Perform secure aggregation
    if client_updates:
        aggregated_weights = server_privacy.aggregate_secure_updates(client_updates)

        print(f"   ✓ Securely aggregated weights")
        for layer, values in aggregated_weights.items():
            print(f"     - {layer}: shape {values.shape}, mean {values.mean():.6f}")

    print(f"\n4️⃣  Privacy Validation")

    # Check privacy constraints
    if server_privacy.validate_privacy_constraints():
        print("   ✅ All privacy constraints satisfied")
    else:
        print("   ⚠️  Some privacy constraints need attention")

    # Generate privacy report
    report = server_privacy.get_privacy_report()
    print(f"\n5️⃣  Privacy Report")
    print(f"   • Rounds completed: {report['metrics']['rounds_completed']}")
    print(f"   • Secure aggregations: {report['metrics']['secure_aggregations']}")
    print(f"   • Authenticated clients: {len(report['metrics']['authenticated_clients'])}")
    print(f"   • Privacy violations: {report['metrics']['privacy_violations']}")

    # Save report
    server_privacy.save_privacy_report("privacy_demo_report.json")
    print(f"   ✓ Full report saved to privacy_demo_report.json")

    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)

    print("\n📝 Next Steps:")
    print("1. Integrate these privacy features into fl_lora_client.py and fl_lora_server.py")
    print("2. Configure privacy settings in model_configs.py")
    print("3. Run full federated learning with: python run_fl_training.py --privacy")
    print("\n💡 For production use:")
    print("• Use proper PKI infrastructure instead of self-signed certificates")
    print("• Configure appropriate privacy budgets for differential privacy")
    print("• Implement secure key distribution for multi-party computation")
    print("• Add audit logging for compliance requirements")


if __name__ == "__main__":
    demo_privacy_features()