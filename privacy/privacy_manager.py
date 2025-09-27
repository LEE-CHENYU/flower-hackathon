"""
Privacy Manager for Federated Learning with LoRA
Coordinates privacy-preserving features for the FL training flow
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy features"""

    # Secure Aggregation
    enable_secure_aggregation: bool = True
    threshold: int = 2  # Minimum clients for secret reconstruction

    # Authentication
    enable_authentication: bool = True
    require_client_certificates: bool = False

    # Communication Security
    enable_tls: bool = True
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None

    # Privacy Budget Tracking
    track_privacy_metrics: bool = True
    log_privacy_events: bool = True

    # Client-side privacy
    enable_local_privacy: bool = True
    sanitize_labels: bool = True

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "secure_aggregation": {
                "enabled": self.enable_secure_aggregation,
                "threshold": self.threshold
            },
            "authentication": {
                "enabled": self.enable_authentication,
                "require_certificates": self.require_client_certificates
            },
            "communication": {
                "tls_enabled": self.enable_tls,
                "cert_path": self.tls_cert_path,
                "key_path": self.tls_key_path
            },
            "privacy_tracking": {
                "track_metrics": self.track_privacy_metrics,
                "log_events": self.log_privacy_events
            },
            "client_privacy": {
                "local_privacy": self.enable_local_privacy,
                "sanitize_labels": self.sanitize_labels
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PrivacyConfig':
        """Create config from dictionary"""
        return cls(
            enable_secure_aggregation=config_dict.get("secure_aggregation", {}).get("enabled", True),
            threshold=config_dict.get("secure_aggregation", {}).get("threshold", 2),
            enable_authentication=config_dict.get("authentication", {}).get("enabled", True),
            require_client_certificates=config_dict.get("authentication", {}).get("require_certificates", False),
            enable_tls=config_dict.get("communication", {}).get("tls_enabled", True),
            tls_cert_path=config_dict.get("communication", {}).get("cert_path"),
            tls_key_path=config_dict.get("communication", {}).get("key_path"),
            track_privacy_metrics=config_dict.get("privacy_tracking", {}).get("track_metrics", True),
            log_privacy_events=config_dict.get("privacy_tracking", {}).get("log_events", True),
            enable_local_privacy=config_dict.get("client_privacy", {}).get("local_privacy", True),
            sanitize_labels=config_dict.get("client_privacy", {}).get("sanitize_labels", True)
        )


class PrivacyManager:
    """Main privacy manager for federated learning"""

    def __init__(self, config: Optional[PrivacyConfig] = None, shared_auth_manager=None):
        """
        Initialize privacy manager

        Args:
            config: Privacy configuration
            shared_auth_manager: Shared authentication manager for client-server coordination
        """
        self.config = config or PrivacyConfig()
        self.privacy_events = []
        self.metrics = {
            "rounds_completed": 0,
            "secure_aggregations": 0,
            "authenticated_clients": set(),
            "privacy_violations": 0
        }

        # Initialize components based on config
        self.secure_agg = None
        self.auth_manager = None
        self.secure_comm = None

        if self.config.enable_secure_aggregation:
            from .secure_aggregation import SecureAggregationManager
            self.secure_agg = SecureAggregationManager(threshold=self.config.threshold)
            logger.info("Secure aggregation enabled")

        if self.config.enable_authentication:
            from .secure_communication import AuthenticationManager
            # Use shared auth manager if provided, otherwise create new one
            self.auth_manager = shared_auth_manager or AuthenticationManager()
            logger.info("Authentication enabled")

        if self.config.enable_tls:
            from .secure_communication import SecureCommunication
            self.secure_comm = SecureCommunication(
                cert_path=self.config.tls_cert_path,
                key_path=self.config.tls_key_path
            )
            logger.info("TLS communication enabled")

    def register_client(self, client_id: int) -> str:
        """
        Register a client with the privacy manager

        Args:
            client_id: Client identifier

        Returns:
            Authentication token for the client
        """
        if self.auth_manager and self.config.enable_authentication:
            token = self.auth_manager.register_client(client_id)
            self._log_event(f"Registered client {client_id}")
            return token
        return ""

    def prepare_client_update(self, client_id: int, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Prepare client update with privacy features

        Args:
            client_id: Client identifier
            weights: LoRA weight updates

        Returns:
            Privacy-protected update package
        """
        update = {"client_id": client_id, "weights": weights}

        # Apply secure aggregation if enabled
        if self.secure_agg and self.config.enable_secure_aggregation:
            update = self.secure_agg.mask_weights(client_id, weights)
            self._log_event(f"Applied secure aggregation masking for client {client_id}")

        # Add authentication if enabled
        if self.auth_manager and self.config.enable_authentication:
            update["auth_token"] = self.auth_manager.generate_token(client_id)
            self._log_event(f"Generated auth token for client {client_id}")

        return update

    def aggregate_secure_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Securely aggregate client updates

        Args:
            updates: List of client updates

        Returns:
            Aggregated weights
        """
        if self.secure_agg and self.config.enable_secure_aggregation:
            # Verify authentication for all clients
            if self.auth_manager:
                for update in updates:
                    if not self.auth_manager.verify_token(
                        update["client_id"],
                        update.get("auth_token", "")
                    ):
                        logger.warning(f"Authentication failed for client {update['client_id']}")
                        self.metrics["privacy_violations"] += 1
                        continue
                    self.metrics["authenticated_clients"].add(update["client_id"])

            # Perform secure aggregation
            aggregated = self.secure_agg.aggregate_masked_weights(updates)
            self.metrics["secure_aggregations"] += 1
            self._log_event(f"Completed secure aggregation for {len(updates)} clients")
            return aggregated
        else:
            # Fallback to simple averaging
            return self._simple_aggregate(updates)

    def _simple_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Simple averaging aggregation (fallback)"""
        if not updates:
            return {}

        # Extract weights from updates
        all_weights = [u.get("weights", u) for u in updates]

        # Average the weights
        aggregated = {}
        weight_keys = all_weights[0].keys()

        for key in weight_keys:
            stacked = np.stack([w[key] for w in all_weights])
            aggregated[key] = np.mean(stacked, axis=0)

        return aggregated

    def sanitize_label(self, label: str) -> str:
        """
        Sanitize labels to remove PII

        Args:
            label: Original label text

        Returns:
            Sanitized label
        """
        if not self.config.sanitize_labels:
            return label

        # Remove potential PII patterns
        import re

        # Remove email addresses
        label = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', label)

        # Remove phone numbers (various formats)
        label = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', label)
        label = re.sub(r'\b\d{3}-\d{4}\b', '[PHONE]', label)  # Shorter format like 555-1234

        # Remove SSN-like patterns
        label = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[ID]', label)

        # Remove names (simplified - would need NER in production)
        label = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', label)

        return label

    def validate_privacy_constraints(self) -> bool:
        """
        Validate that privacy constraints are met

        Returns:
            True if all constraints are satisfied
        """
        if self.metrics["privacy_violations"] > 0:
            logger.warning(f"Privacy violations detected: {self.metrics['privacy_violations']}")
            return False

        if self.config.enable_authentication:
            if len(self.metrics["authenticated_clients"]) == 0:
                logger.warning("No authenticated clients")
                return False

        return True

    def _log_event(self, event: str):
        """Log privacy event"""
        if self.config.log_privacy_events:
            self.privacy_events.append({
                "timestamp": str(Path(__file__).stat().st_mtime),
                "event": event
            })
            logger.debug(f"Privacy event: {event}")

    def get_privacy_report(self) -> Dict:
        """
        Generate privacy report

        Returns:
            Privacy metrics and events
        """
        return {
            "configuration": self.config.to_dict(),
            "metrics": self.metrics,
            "events": self.privacy_events[-100:],  # Last 100 events
            "constraints_satisfied": self.validate_privacy_constraints()
        }

    def save_privacy_report(self, filepath: str):
        """Save privacy report to file"""
        report = self.get_privacy_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Privacy report saved to {filepath}")

    def initialize_round(self, round_number: int):
        """Initialize privacy tracking for a new round"""
        self.metrics["rounds_completed"] = round_number
        self._log_event(f"Initialized privacy tracking for round {round_number}")

        if self.secure_agg:
            self.secure_agg.initialize_round(round_number)

    def finalize_round(self, round_number: int):
        """Finalize privacy tracking for completed round"""
        self._log_event(f"Finalized privacy tracking for round {round_number}")

        if self.config.track_privacy_metrics:
            metrics_path = Path(f"privacy_metrics/round_{round_number}.json")
            metrics_path.parent.mkdir(exist_ok=True)
            self.save_privacy_report(str(metrics_path))


class ClientPrivacyManager:
    """Client-side privacy manager"""

    def __init__(self, client_id: int, config: Optional[PrivacyConfig] = None):
        """
        Initialize client privacy manager

        Args:
            client_id: Client identifier
            config: Privacy configuration
        """
        self.client_id = client_id
        self.config = config or PrivacyConfig()
        self.local_metrics = {
            "data_samples_used": 0,
            "labels_sanitized": 0,
            "updates_sent": 0
        }

    def prepare_training_data(self, images: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Prepare training data with privacy protections

        Args:
            images: List of image paths
            labels: List of label texts

        Returns:
            Privacy-protected images and labels
        """
        processed_labels = []

        for label in labels:
            if self.config.sanitize_labels:
                # Sanitize labels to remove PII
                processed_label = self._sanitize_client_label(label)
                self.local_metrics["labels_sanitized"] += 1
            else:
                processed_label = label
            processed_labels.append(processed_label)

        self.local_metrics["data_samples_used"] += len(images)

        return images, processed_labels

    def _sanitize_client_label(self, label: str) -> str:
        """Client-side label sanitization"""
        # Hash any potential PII while preserving semantic content
        import re

        # Replace specific medical terms with generic ones
        label = re.sub(r'\b(patient|client)\s+\w+\b', 'individual', label, flags=re.IGNORECASE)

        # Remove dates that might identify specific visits
        label = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', label)

        # Remove specific location references
        label = re.sub(r'\b(clinic|office|hospital)\s+\w+\b', 'facility', label, flags=re.IGNORECASE)

        return label

    def should_participate(self, round_number: int) -> bool:
        """
        Determine if client should participate in round (privacy-preserving sampling)

        Args:
            round_number: Current round number

        Returns:
            True if client should participate
        """
        if not self.config.enable_local_privacy:
            return True

        # Use deterministic sampling based on client ID and round
        participation_hash = hashlib.sha256(
            f"{self.client_id}:{round_number}".encode()
        ).digest()

        # Convert to probability (0-1)
        prob = int.from_bytes(participation_hash[:4], 'big') / (2**32)

        # Participate with 80% probability
        return prob < 0.8

    def get_local_privacy_metrics(self) -> Dict:
        """Get client's local privacy metrics"""
        return self.local_metrics.copy()