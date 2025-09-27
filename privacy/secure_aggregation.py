"""
Secure Aggregation for LoRA Weights using Barracuda Protocol
Integrates secure aggregation from Barracuda into the Flower FL framework
"""

import numpy as np
import hashlib
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# Add Barracuda path for imports
barracuda_path = Path("/Users/chenyusu/Downloads/Barracuda")
if barracuda_path.exists():
    sys.path.insert(0, str(barracuda_path))
    from utils_secureagg import (
        prg_expand, float_to_modint, modint_to_float,
        bytes_to_int_array, shamir_split, shamir_combine,
        encrypt_for_recipient, decrypt_for_recipient
    )
else:
    # Fallback implementations if Barracuda not available
    def prg_expand(seed: bytes, length: int, info: bytes = b"") -> bytes:
        """Pseudo-random generator expansion"""
        import hashlib
        key = hashlib.sha256(seed + info).digest()
        output = b""
        counter = 0
        while len(output) < length:
            output += hashlib.sha256(key + counter.to_bytes(4, 'big')).digest()
            counter += 1
        return output[:length]

    def float_to_modint(vec: np.ndarray, scale: float, M: int, R: int):
        """Convert float array to modular integers"""
        q = np.round(np.clip(vec / scale, -M, M)).astype(np.int64)
        q_shifted = (q + M) % R
        return q_shifted

    def modint_to_float(q_shifted: np.ndarray, scale: float, M: int, R: int):
        """Convert modular integers back to floats"""
        q = (q_shifted.astype(np.int64) - M)
        return q * scale

    def bytes_to_int_array(b: bytes, dtype=np.int64, elems=None):
        """Convert bytes to integer array"""
        if elems is None:
            elems = len(b) // 8
        arr = np.frombuffer(b[: elems * 8], dtype=np.uint64).astype(np.int64)
        return arr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureAggregationClient:
    """Client-side secure aggregation for LoRA weights"""

    def __init__(self, client_id: int, threshold: int, total_clients: int):
        """
        Initialize secure aggregation client

        Args:
            client_id: Unique client identifier
            threshold: Minimum clients for secret reconstruction
            total_clients: Total number of clients in federation
        """
        self.client_id = client_id
        self.threshold = threshold
        self.total_clients = total_clients

        # Quantization parameters for LoRA weights
        self.scale = 1e-3  # Scale factor for quantization
        self.bits = 32
        self.M = 2 ** (self.bits - 1) - 1
        self.R = 2 ** self.bits
        self.bytes_per_elem = 8

        # Generate keys for secure communication
        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

        # Storage for peer information
        self.peer_public_keys = {}  # client_id -> public_key_bytes
        self.pairwise_seeds = {}     # client_id -> seed_bytes
        self.personal_mask_seed = os.urandom(32)  # Random seed for personal mask

        # Shamir shares storage
        self.generated_shares = {}  # recipient_id -> share
        self.received_shares = {}   # sender_id -> encrypted_share

    def get_public_key(self) -> bytes:
        """Get client's public key for key exchange"""
        return self.public_key

    def process_peer_keys(self, peer_keys: Dict[int, bytes]):
        """
        Process public keys from other clients

        Args:
            peer_keys: Mapping of client_id to public_key_bytes
        """
        self.peer_public_keys = peer_keys

        # Generate pairwise seeds with each peer
        for peer_id, peer_pub in peer_keys.items():
            if peer_id == self.client_id:
                continue

            # Derive shared secret via ECDH
            peer_public_key = X25519PublicKey.from_public_bytes(peer_pub)
            shared_secret = self.private_key.exchange(peer_public_key)

            # Derive seed from shared secret
            seed = hashlib.sha256(
                shared_secret + b"pairwise_mask" +
                min(self.client_id, peer_id).to_bytes(4, 'big') +
                max(self.client_id, peer_id).to_bytes(4, 'big')
            ).digest()

            self.pairwise_seeds[peer_id] = seed

    def mask_weights(self, weights: Dict[str, np.ndarray], participating_clients: List[int]) -> Dict[str, Any]:
        """
        Apply secure masking to LoRA weights

        Args:
            weights: Dictionary of LoRA weight arrays
            participating_clients: List of participating client IDs

        Returns:
            Masked weights package with metadata
        """
        masked_weights = {}

        for layer_name, weight_array in weights.items():
            # Flatten the weight array for processing
            original_shape = weight_array.shape
            flat_weights = weight_array.flatten()

            # Quantize weights to integers
            quantized = float_to_modint(flat_weights, self.scale, self.M, self.R)

            # Apply pairwise masks
            masked = quantized.copy().astype(np.int64)
            length_bytes = quantized.size * self.bytes_per_elem

            for peer_id in participating_clients:
                if peer_id == self.client_id:
                    continue

                if peer_id not in self.pairwise_seeds:
                    logger.warning(f"No pairwise seed for peer {peer_id}")
                    continue

                # Generate pairwise mask
                seed = self.pairwise_seeds[peer_id]
                mask_bytes = prg_expand(
                    seed, length_bytes,
                    info=f"lora_layer_{layer_name}".encode()
                )
                mask_array = bytes_to_int_array(mask_bytes, elems=quantized.size)

                # Apply mask based on client ID order
                if self.client_id < peer_id:
                    masked = (masked + mask_array) % self.R
                else:
                    masked = (masked - mask_array) % self.R

            # Apply personal mask
            personal_mask_bytes = prg_expand(
                self.personal_mask_seed, length_bytes,
                info=f"personal_{layer_name}".encode()
            )
            personal_mask = bytes_to_int_array(personal_mask_bytes, elems=quantized.size)
            masked = (masked + personal_mask) % self.R

            # Reshape back to original shape
            masked_weights[layer_name] = {
                "masked_values": masked.reshape(original_shape),
                "shape": original_shape,
                "dtype": str(weight_array.dtype)
            }

        return {
            "client_id": self.client_id,
            "masked_weights": masked_weights,
            "participating_clients": participating_clients
        }

    def create_secret_shares(self) -> Dict[int, bytes]:
        """
        Create Shamir secret shares for personal mask seed

        Returns:
            Dictionary mapping recipient_id to encrypted share
        """
        # Import Barracuda's Shamir implementation if available
        try:
            from sharing_fixed import HexToHexSecretSharer
            hex_secret = self.personal_mask_seed.hex()
            shares = HexToHexSecretSharer.split_secret(
                hex_secret, self.threshold, self.total_clients
            )
        except ImportError:
            # Fallback: simple XOR-based sharing (not cryptographically secure)
            shares = []
            for i in range(self.total_clients):
                if i < self.threshold - 1:
                    share = os.urandom(32)
                else:
                    # Last share is XOR of secret with all previous shares
                    share = self.personal_mask_seed
                    for prev_share in shares:
                        share = bytes(a ^ b for a, b in zip(share, bytes.fromhex(prev_share)))
                shares.append(share.hex())

        # Encrypt shares for each recipient
        encrypted_shares = {}
        client_ids = sorted(self.peer_public_keys.keys())

        for i, recipient_id in enumerate(client_ids):
            if i >= len(shares):
                break

            # Derive encryption key via ECDH
            recipient_pub = X25519PublicKey.from_public_bytes(
                self.peer_public_keys[recipient_id]
            )
            shared_key = self.private_key.exchange(recipient_pub)
            enc_key = hashlib.sha256(shared_key + b"share_encryption").digest()[:32]

            # Encrypt share
            try:
                encrypted_share = encrypt_for_recipient(
                    enc_key, bytes.fromhex(shares[i])
                )
            except:
                # Fallback: simple XOR encryption
                share_bytes = bytes.fromhex(shares[i])
                encrypted_share = bytes(a ^ b for a, b in zip(share_bytes, enc_key))

            encrypted_shares[recipient_id] = encrypted_share
            self.generated_shares[recipient_id] = shares[i]  # Store for potential recovery

        return encrypted_shares

    def store_received_share(self, sender_id: int, encrypted_share: bytes):
        """Store encrypted share received from another client"""
        self.received_shares[sender_id] = encrypted_share


class SecureAggregationServer:
    """Server-side secure aggregation coordinator"""

    def __init__(self, threshold: int, expected_clients: int):
        """
        Initialize secure aggregation server

        Args:
            threshold: Minimum clients for secret reconstruction
            expected_clients: Expected number of clients
        """
        self.threshold = threshold
        self.expected_clients = expected_clients

        # Client tracking
        self.client_keys = {}  # client_id -> public_key_bytes
        self.round_participants = {}  # round -> [client_ids]
        self.received_masked = {}  # client_id -> masked_weights
        self.received_shares = {}  # recipient_id -> {sender_id -> encrypted_share}

        # Quantization parameters (must match clients)
        self.scale = 1e-3
        self.bits = 32
        self.M = 2 ** (self.bits - 1) - 1
        self.R = 2 ** self.bits

    def register_client(self, client_id: int, public_key: bytes):
        """Register client's public key"""
        self.client_keys[client_id] = public_key
        logger.info(f"Registered client {client_id}")

    def broadcast_keys(self) -> Dict[int, bytes]:
        """Broadcast all client public keys"""
        return self.client_keys.copy()

    def collect_masked_weights(self, client_id: int, masked_package: Dict[str, Any]):
        """Collect masked weights from client"""
        self.received_masked[client_id] = masked_package
        logger.debug(f"Received masked weights from client {client_id}")

    def collect_shares(self, recipient_id: int, sender_id: int, encrypted_share: bytes):
        """Collect encrypted secret shares"""
        if recipient_id not in self.received_shares:
            self.received_shares[recipient_id] = {}
        self.received_shares[recipient_id][sender_id] = encrypted_share

    def aggregate_secure(self, round_number: int) -> Dict[str, np.ndarray]:
        """
        Perform secure aggregation of masked weights

        Args:
            round_number: Current round number

        Returns:
            Aggregated LoRA weights
        """
        if not self.received_masked:
            logger.warning("No masked weights received")
            return {}

        # Identify participating and dropped clients
        participating = list(self.received_masked.keys())
        self.round_participants[round_number] = participating

        logger.info(f"Round {round_number}: {len(participating)} clients participating")

        # Aggregate masked weights
        aggregated = {}
        num_clients = len(participating)

        for client_id, package in self.received_masked.items():
            masked_weights = package["masked_weights"]

            for layer_name, layer_data in masked_weights.items():
                masked_values = layer_data["masked_values"]

                if layer_name not in aggregated:
                    aggregated[layer_name] = np.zeros_like(masked_values, dtype=np.int64)

                # Sum masked values (modular arithmetic)
                aggregated[layer_name] = (aggregated[layer_name] + masked_values) % self.R

        # Convert back from modular integers to floats
        final_weights = {}
        for layer_name, summed_masked in aggregated.items():
            # Average the summed values
            averaged = summed_masked // num_clients

            # Convert back to float
            flat_averaged = averaged.flatten()
            float_values = modint_to_float(flat_averaged, self.scale, self.M, self.R)

            # Reshape to original shape
            original_shape = list(self.received_masked.values())[0]["masked_weights"][layer_name]["shape"]
            final_weights[layer_name] = float_values.reshape(original_shape)

        # Clear state for next round
        self.received_masked.clear()

        return final_weights


class SecureAggregationManager:
    """High-level manager for secure aggregation in federated learning"""

    def __init__(self, threshold: int = 2):
        """
        Initialize secure aggregation manager

        Args:
            threshold: Minimum clients for secret reconstruction
        """
        self.threshold = threshold
        self.clients = {}  # client_id -> SecureAggregationClient
        self.server = None
        self.current_round = 0

    def initialize_round(self, round_number: int, expected_clients: int = 3):
        """Initialize secure aggregation for a new round"""
        self.current_round = round_number
        self.server = SecureAggregationServer(self.threshold, expected_clients)
        logger.info(f"Initialized secure aggregation for round {round_number}")

    def create_client(self, client_id: int, total_clients: int) -> SecureAggregationClient:
        """Create secure aggregation client"""
        client = SecureAggregationClient(client_id, self.threshold, total_clients)
        self.clients[client_id] = client

        # Register with server
        if self.server:
            self.server.register_client(client_id, client.get_public_key())

        return client

    def setup_key_agreement(self):
        """Perform key agreement phase"""
        if not self.server:
            raise ValueError("Server not initialized")

        # Broadcast keys to all clients
        all_keys = self.server.broadcast_keys()

        # Each client processes peer keys
        for client_id, client in self.clients.items():
            peer_keys = {cid: key for cid, key in all_keys.items() if cid != client_id}
            client.process_peer_keys(peer_keys)

        logger.info("Completed key agreement phase")

    def mask_weights(self, client_id: int, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Mask weights for secure aggregation

        Args:
            client_id: Client identifier
            weights: LoRA weight updates

        Returns:
            Masked weights package
        """
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not registered for secure aggregation")
            return {"client_id": client_id, "weights": weights}

        client = self.clients[client_id]
        participating = list(self.clients.keys())

        masked_package = client.mask_weights(weights, participating)

        # Submit to server
        if self.server:
            self.server.collect_masked_weights(client_id, masked_package)

        return masked_package

    def aggregate_masked_weights(self, updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Aggregate masked weights from multiple clients

        Args:
            updates: List of masked weight packages

        Returns:
            Aggregated weights
        """
        if not self.server:
            logger.warning("Server not initialized, falling back to simple aggregation")
            return self._simple_aggregate(updates)

        # Process updates if not already collected
        for update in updates:
            if "masked_weights" in update and update["client_id"] not in self.server.received_masked:
                self.server.collect_masked_weights(update["client_id"], update)

        # Perform secure aggregation
        aggregated = self.server.aggregate_secure(self.current_round)

        return aggregated

    def _simple_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Fallback simple aggregation"""
        if not updates:
            return {}

        # Extract weights
        all_weights = []
        for update in updates:
            if "weights" in update:
                all_weights.append(update["weights"])
            elif "masked_weights" in update:
                # Extract from masked format
                weights = {}
                for layer, data in update["masked_weights"].items():
                    weights[layer] = data["masked_values"]
                all_weights.append(weights)

        if not all_weights:
            return {}

        # Average weights
        aggregated = {}
        for key in all_weights[0].keys():
            stacked = np.stack([w[key] for w in all_weights])
            aggregated[key] = np.mean(stacked, axis=0)

        return aggregated