"""
Privacy Protection Module for Federated Learning with LoRA
"""

from .privacy_manager import (
    PrivacyConfig,
    PrivacyManager,
    ClientPrivacyManager
)

from .secure_aggregation import (
    SecureAggregationClient,
    SecureAggregationServer,
    SecureAggregationManager
)

from .secure_communication import (
    AuthenticationManager,
    SecureCommunication,
    SecureChannel,
    SecureProtocol,
    FederatedSecureProtocol
)

__all__ = [
    # Privacy Manager
    'PrivacyConfig',
    'PrivacyManager',
    'ClientPrivacyManager',

    # Secure Aggregation
    'SecureAggregationClient',
    'SecureAggregationServer',
    'SecureAggregationManager',

    # Secure Communication
    'AuthenticationManager',
    'SecureCommunication',
    'SecureChannel',
    'SecureProtocol',
    'FederatedSecureProtocol'
]

__version__ = '0.1.0'