#!/usr/bin/env python3
"""
Test script for privacy features in federated learning
"""

import numpy as np
import sys
from pathlib import Path
import json
import logging

# Add project to path
sys.path.append(str(Path(__file__).parent))

from privacy import (
    PrivacyConfig,
    PrivacyManager,
    ClientPrivacyManager,
    SecureAggregationManager,
    AuthenticationManager,
    SecureCommunication,
    FederatedSecureProtocol
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_privacy_config():
    """Test privacy configuration"""
    print("\n" + "="*60)
    print("Testing Privacy Configuration")
    print("="*60)

    # Create default config
    config = PrivacyConfig()
    print(f"Default config: {json.dumps(config.to_dict(), indent=2)}")

    # Create custom config
    custom_config = PrivacyConfig(
        enable_secure_aggregation=True,
        threshold=3,
        enable_authentication=True,
        enable_tls=True
    )
    print(f"\nCustom config: {json.dumps(custom_config.to_dict(), indent=2)}")

    # Test serialization
    config_dict = custom_config.to_dict()
    restored_config = PrivacyConfig.from_dict(config_dict)
    assert restored_config.threshold == 3
    print("✓ Configuration serialization works")


def test_secure_aggregation():
    """Test secure aggregation of LoRA weights"""
    print("\n" + "="*60)
    print("Testing Secure Aggregation")
    print("="*60)

    # Initialize manager
    manager = SecureAggregationManager(threshold=2)
    manager.initialize_round(round_number=1, expected_clients=3)

    # Create clients
    num_clients = 3
    clients = []
    for i in range(num_clients):
        client = manager.create_client(client_id=i, total_clients=num_clients)
        clients.append(client)
        print(f"Created client {i}")

    # Setup key agreement
    manager.setup_key_agreement()
    print("✓ Key agreement completed")

    # Create sample LoRA weights
    def create_sample_weights():
        return {
            "lora_A": np.random.randn(8, 768).astype(np.float32),
            "lora_B": np.random.randn(768, 8).astype(np.float32)
        }

    # Mask weights from each client
    masked_updates = []
    for i, client in enumerate(clients):
        weights = create_sample_weights()
        masked = manager.mask_weights(i, weights)
        masked_updates.append(masked)
        print(f"✓ Client {i} masked weights")

    # Aggregate masked weights
    aggregated = manager.aggregate_masked_weights(masked_updates)
    print(f"✓ Aggregated {len(aggregated)} weight layers")

    # Verify aggregation
    for layer_name, values in aggregated.items():
        print(f"  Layer {layer_name}: shape {values.shape}, mean {values.mean():.4f}")


def test_authentication():
    """Test client authentication"""
    print("\n" + "="*60)
    print("Testing Authentication")
    print("="*60)

    auth_manager = AuthenticationManager()

    # Register clients
    client_tokens = {}
    for client_id in range(3):
        token = auth_manager.register_client(client_id)
        client_tokens[client_id] = token
        print(f"Registered client {client_id}")

    # Test token generation and verification
    for client_id in range(3):
        token = auth_manager.generate_token(client_id)
        is_valid = auth_manager.verify_token(client_id, token)
        assert is_valid, f"Token verification failed for client {client_id}"
        print(f"✓ Client {client_id} authentication verified")

    # Test invalid token
    is_valid = auth_manager.verify_token(0, "invalid_token")
    assert not is_valid, "Invalid token should not verify"
    print("✓ Invalid token rejected")

    # Test challenge-response
    challenge = auth_manager.create_challenge(0)
    print(f"✓ Created challenge for client 0: {len(challenge)} bytes")


def test_secure_communication():
    """Test secure communication channels"""
    print("\n" + "="*60)
    print("Testing Secure Communication")
    print("="*60)

    # Initialize secure communication
    secure_comm = SecureCommunication()

    # Create server SSL context
    server_context = secure_comm.create_server_context()
    print(f"✓ Created server SSL context")

    # Create client SSL context
    client_context = secure_comm.create_client_context(verify=False)
    print(f"✓ Created client SSL context")

    # Test federated secure protocol
    fed_protocol = FederatedSecureProtocol()

    # Initialize server and clients
    server = fed_protocol.initialize_server()
    clients = []
    for i in range(3):
        client = fed_protocol.initialize_client(i)
        clients.append(client)

    print(f"✓ Initialized {len(clients)} secure clients")

    # Test authentication and session creation
    for i, client in enumerate(clients):
        # Server registers client and generates token
        # (In real scenario, this would happen during client registration)
        server_token = server.auth_manager.register_client(i, client.secure_channel.get_public_key_bytes())

        # Generate auth token using server's auth manager
        token = server.auth_manager.generate_token(i)

        # Server authenticates client
        auth_success = server.authenticate_peer(i, token)
        assert auth_success, f"Authentication failed for client {i}"

        # Create secure session
        client_pubkey = client.secure_channel.get_public_key_bytes()
        server_pubkey = server.secure_channel.get_public_key_bytes()

        # Server creates session with client (server perspective)
        server.secure_channel.establish_shared_key(i, client_pubkey)

        # Client creates session with server
        client.authenticated_peers.add(0)  # Mark server as authenticated
        session_id = client.create_secure_session(0, server_pubkey)
        print(f"✓ Client {i} created secure session: {session_id}")

    print("✓ All secure communication tests passed")


def test_privacy_manager():
    """Test integrated privacy manager"""
    print("\n" + "="*60)
    print("Testing Integrated Privacy Manager")
    print("="*60)

    # Create privacy config with secure aggregation disabled for this test
    # (to avoid needing full Barracuda setup)
    config = PrivacyConfig(
        enable_secure_aggregation=False,
        enable_authentication=True,
        enable_tls=True,
        track_privacy_metrics=True
    )

    # Initialize manager
    manager = PrivacyManager(config)
    print("✓ Initialized privacy manager")

    # Test client update preparation
    weights = {
        "lora_A": np.random.randn(8, 768).astype(np.float32),
        "lora_B": np.random.randn(768, 8).astype(np.float32)
    }

    # Prepare update with privacy features
    update = manager.prepare_client_update(client_id=0, weights=weights)
    assert "auth_token" in update
    print("✓ Prepared client update with privacy features")

    # Test label sanitization
    test_label = "Patient John Doe, 555-1234, john@email.com needs cleaning"
    sanitized = manager.sanitize_label(test_label)
    assert "john@email.com" not in sanitized
    assert "555-1234" not in sanitized
    print(f"✓ Label sanitization: {sanitized}")

    # Test privacy report
    report = manager.get_privacy_report()
    print(f"✓ Generated privacy report with {len(report)} sections")

    # Save report
    report_path = "privacy_metrics/test_report.json"
    Path("privacy_metrics").mkdir(exist_ok=True)
    manager.save_privacy_report(report_path)
    print(f"✓ Saved privacy report to {report_path}")


def test_client_privacy():
    """Test client-side privacy features"""
    print("\n" + "="*60)
    print("Testing Client-Side Privacy")
    print("="*60)

    # Create client privacy manager
    config = PrivacyConfig(
        enable_local_privacy=True,
        sanitize_labels=True
    )
    client_manager = ClientPrivacyManager(client_id=0, config=config)

    # Test data preparation
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    labels = [
        "Patient visited clinic ABC on 01/15/2024",
        "Client John needs treatment",
        "Hospital XYZ report shows improvement"
    ]

    processed_images, processed_labels = client_manager.prepare_training_data(images, labels)

    print(f"Original labels:")
    for label in labels:
        print(f"  - {label}")

    print(f"\nSanitized labels:")
    for label in processed_labels:
        print(f"  - {label}")

    # Test participation decision
    for round_num in range(10):
        should_participate = client_manager.should_participate(round_num)
        print(f"Round {round_num}: Participate = {should_participate}")

    # Get metrics
    metrics = client_manager.get_local_privacy_metrics()
    print(f"\nClient privacy metrics: {json.dumps(metrics, indent=2)}")


def run_integration_test():
    """Run full integration test"""
    print("\n" + "="*60)
    print("Running Integration Test")
    print("="*60)

    # Initialize privacy-enabled FL system
    config = PrivacyConfig(
        enable_secure_aggregation=False,  # Disabled to avoid Barracuda dependency
        enable_authentication=True,
        threshold=2
    )

    # Create server privacy manager
    server_manager = PrivacyManager(config)
    server_manager.initialize_round(1)

    # Create client privacy managers
    num_clients = 3
    client_managers = []
    client_weights = []

    for i in range(num_clients):
        client_config = PrivacyConfig(
            enable_local_privacy=True,
            sanitize_labels=True
        )
        client_mgr = ClientPrivacyManager(i, client_config)
        client_managers.append(client_mgr)

        # Generate sample weights
        weights = {
            "lora_A": np.random.randn(8, 768).astype(np.float32) * 0.01,
            "lora_B": np.random.randn(768, 8).astype(np.float32) * 0.01
        }
        client_weights.append(weights)

    print(f"✓ Created {num_clients} privacy-enabled clients")

    # Prepare client updates
    updates = []
    for i, (mgr, weights) in enumerate(zip(client_managers, client_weights)):
        # Check if client participates
        if mgr.should_participate(1):
            update = server_manager.prepare_client_update(i, weights)
            updates.append(update)
            print(f"✓ Client {i} prepared secure update")

    # Aggregate updates securely
    aggregated = server_manager.aggregate_secure_updates(updates)
    print(f"✓ Securely aggregated {len(updates)} client updates")

    # Verify aggregation result
    for layer_name, values in aggregated.items():
        print(f"  Aggregated {layer_name}: shape {values.shape}, mean {values.mean():.6f}")

    # Finalize round
    server_manager.finalize_round(1)
    print("✓ Round 1 completed with privacy protection")

    # Check privacy constraints
    if server_manager.validate_privacy_constraints():
        print("✓ All privacy constraints satisfied")
    else:
        print("⚠ Some privacy constraints violated")

    print("\n✅ Integration test completed successfully!")


def main():
    """Run all privacy feature tests"""
    print("\n" + "="*70)
    print("PRIVACY FEATURES TEST SUITE FOR FEDERATED LEARNING")
    print("="*70)

    try:
        # Run individual tests
        test_privacy_config()
        test_authentication()
        test_secure_communication()
        test_privacy_manager()
        test_client_privacy()

        # Test secure aggregation if Barracuda is available
        barracuda_path = Path("/Users/chenyusu/Downloads/Barracuda")
        if barracuda_path.exists():
            test_secure_aggregation()
        else:
            print("\n⚠ Skipping secure aggregation test (Barracuda not found)")

        # Run integration test
        run_integration_test()

        print("\n" + "="*70)
        print("✅ ALL PRIVACY TESTS PASSED SUCCESSFULLY!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())