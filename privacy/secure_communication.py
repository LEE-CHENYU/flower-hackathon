"""
Secure Communication Module for Federated Learning
Provides authentication and TLS/SSL communication for FL clients and server
"""

import hashlib
import hmac
import json
import ssl
import secrets
import time
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives.asymmetric import rsa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Manages client authentication for federated learning"""

    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize authentication manager

        Args:
            secret_key: Secret key for HMAC (generated if not provided)
        """
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.client_tokens = {}  # client_id -> token
        self.client_keys = {}    # client_id -> public_key
        self.client_challenges = {}  # client_id -> challenge
        self.token_expiry = {}   # token -> expiry_time

    def register_client(self, client_id: int, public_key: Optional[bytes] = None) -> str:
        """
        Register a new client

        Args:
            client_id: Client identifier
            public_key: Optional client public key

        Returns:
            Registration token
        """
        # Generate unique token for client
        token = secrets.token_urlsafe(32)
        self.client_tokens[client_id] = token

        # Store public key if provided
        if public_key:
            self.client_keys[client_id] = public_key

        # Set token expiry (24 hours)
        self.token_expiry[token] = time.time() + 86400

        logger.info(f"Registered client {client_id}")
        return token

    def generate_token(self, client_id: int) -> str:
        """
        Generate authentication token for client

        Args:
            client_id: Client identifier

        Returns:
            Authentication token
        """
        if client_id not in self.client_tokens:
            # Auto-register if not registered
            return self.register_client(client_id)

        # Generate time-based token
        timestamp = str(int(time.time()))
        message = f"{client_id}:{timestamp}".encode()

        # Create HMAC signature
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()

        # Combine into token
        token = f"{client_id}:{timestamp}:{signature}"

        return token

    def verify_token(self, client_id: int, token: str) -> bool:
        """
        Verify client authentication token

        Args:
            client_id: Client identifier
            token: Authentication token

        Returns:
            True if token is valid
        """
        try:
            # Parse token
            parts = token.split(":")
            if len(parts) != 3:
                return False

            token_client_id, timestamp, signature = parts

            # Verify client ID matches
            if int(token_client_id) != client_id:
                logger.warning(f"Client ID mismatch: {token_client_id} != {client_id}")
                return False

            # Verify timestamp (max 1 hour old)
            current_time = int(time.time())
            token_time = int(timestamp)
            if current_time - token_time > 3600:
                logger.warning(f"Token expired for client {client_id}")
                return False

            # Verify HMAC signature
            message = f"{client_id}:{timestamp}".encode()
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Invalid signature for client {client_id}")
                return False

            return True

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False

    def create_challenge(self, client_id: int) -> bytes:
        """
        Create challenge for challenge-response authentication

        Args:
            client_id: Client identifier

        Returns:
            Challenge bytes
        """
        challenge = secrets.token_bytes(32)
        self.client_challenges[client_id] = challenge
        return challenge

    def verify_challenge_response(self, client_id: int, response: bytes, public_key: bytes) -> bool:
        """
        Verify challenge response from client

        Args:
            client_id: Client identifier
            response: Challenge response
            public_key: Client's public key

        Returns:
            True if response is valid
        """
        if client_id not in self.client_challenges:
            logger.warning(f"No challenge found for client {client_id}")
            return False

        challenge = self.client_challenges[client_id]

        try:
            # Verify using client's public key (simplified - would use proper signature in production)
            client_pub = X25519PublicKey.from_public_bytes(public_key)
            expected = hashlib.sha256(challenge + public_key).digest()

            if hmac.compare_digest(response, expected):
                del self.client_challenges[client_id]  # Clear used challenge
                return True

        except Exception as e:
            logger.error(f"Challenge verification error: {e}")

        return False

    def revoke_client(self, client_id: int):
        """Revoke client's authentication"""
        if client_id in self.client_tokens:
            token = self.client_tokens[client_id]
            del self.client_tokens[client_id]
            if token in self.token_expiry:
                del self.token_expiry[token]
            logger.info(f"Revoked authentication for client {client_id}")


class SecureCommunication:
    """Manages secure TLS/SSL communication"""

    def __init__(self, cert_path: Optional[str] = None, key_path: Optional[str] = None):
        """
        Initialize secure communication

        Args:
            cert_path: Path to TLS certificate
            key_path: Path to TLS private key
        """
        self.cert_path = cert_path
        self.key_path = key_path

        # Generate self-signed certificate if not provided
        if not cert_path or not key_path:
            self._generate_self_signed_cert()

    def _generate_self_signed_cert(self):
        """Generate self-signed certificate for testing"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"FL Privacy Demo"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # Save to temporary files
        cert_dir = Path("privacy/certificates")
        cert_dir.mkdir(exist_ok=True)

        self.cert_path = str(cert_dir / "server.crt")
        self.key_path = str(cert_dir / "server.key")

        # Write certificate
        with open(self.cert_path, "wb") as f:
            f.write(cert.public_bytes(Encoding.PEM))

        # Write private key
        with open(self.key_path, "wb") as f:
            f.write(private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.TraditionalOpenSSL,
                NoEncryption()
            ))

        logger.info(f"Generated self-signed certificate at {self.cert_path}")

    def create_server_context(self) -> ssl.SSLContext:
        """
        Create SSL context for server

        Returns:
            Configured SSL context
        """
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Load certificate and key
        if self.cert_path and self.key_path:
            context.load_cert_chain(self.cert_path, self.key_path)

        # Configure security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

        return context

    def create_client_context(self, verify: bool = True) -> ssl.SSLContext:
        """
        Create SSL context for client

        Args:
            verify: Whether to verify server certificate

        Returns:
            Configured SSL context
        """
        if verify:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            if self.cert_path:
                context.load_verify_locations(self.cert_path)
        else:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Configure security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        return context


class SecureChannel:
    """End-to-end secure channel between client and server"""

    def __init__(self, client_id: int):
        """
        Initialize secure channel

        Args:
            client_id: Client identifier
        """
        self.client_id = client_id
        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.shared_keys = {}  # peer_id -> shared_key
        self.session_keys = {}  # session_id -> key

    def get_public_key_bytes(self) -> bytes:
        """Get public key as bytes"""
        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def establish_shared_key(self, peer_id: int, peer_public_key_bytes: bytes) -> bytes:
        """
        Establish shared key with peer using ECDH

        Args:
            peer_id: Peer identifier
            peer_public_key_bytes: Peer's public key

        Returns:
            Derived shared key
        """
        peer_public_key = X25519PublicKey.from_public_bytes(peer_public_key_bytes)
        shared_secret = self.private_key.exchange(peer_public_key)

        # Derive key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=f"fl_session_{self.client_id}_{peer_id}".encode()
        )
        shared_key = hkdf.derive(shared_secret)

        self.shared_keys[peer_id] = shared_key
        return shared_key

    def encrypt_message(self, message: bytes, recipient_id: int) -> Tuple[bytes, bytes]:
        """
        Encrypt message for recipient

        Args:
            message: Message to encrypt
            recipient_id: Recipient identifier

        Returns:
            (nonce, ciphertext) tuple
        """
        if recipient_id not in self.shared_keys:
            raise ValueError(f"No shared key with recipient {recipient_id}")

        # Use AES-GCM encryption (simplified)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = self.shared_keys[recipient_id]
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, message, None)

        return nonce, ciphertext

    def decrypt_message(self, nonce: bytes, ciphertext: bytes, sender_id: int) -> bytes:
        """
        Decrypt message from sender

        Args:
            nonce: Encryption nonce
            ciphertext: Encrypted message
            sender_id: Sender identifier

        Returns:
            Decrypted message
        """
        if sender_id not in self.shared_keys:
            raise ValueError(f"No shared key with sender {sender_id}")

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = self.shared_keys[sender_id]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return plaintext


class SecureProtocol:
    """High-level secure protocol for FL communication"""

    def __init__(self, role: str, identifier: int):
        """
        Initialize secure protocol

        Args:
            role: 'client' or 'server'
            identifier: Unique identifier
        """
        self.role = role
        self.identifier = identifier
        self.auth_manager = AuthenticationManager()
        self.secure_comm = SecureCommunication()
        self.secure_channel = SecureChannel(identifier)

        # Protocol state
        self.authenticated_peers = set()
        self.active_sessions = {}

    def authenticate_peer(self, peer_id: int, auth_token: str) -> bool:
        """
        Authenticate a peer

        Args:
            peer_id: Peer identifier
            auth_token: Authentication token

        Returns:
            True if authentication successful
        """
        if self.auth_manager.verify_token(peer_id, auth_token):
            self.authenticated_peers.add(peer_id)
            logger.info(f"Authenticated peer {peer_id}")
            return True
        return False

    def create_secure_session(self, peer_id: int, peer_public_key: bytes) -> str:
        """
        Create secure session with peer

        Args:
            peer_id: Peer identifier
            peer_public_key: Peer's public key

        Returns:
            Session ID
        """
        if peer_id not in self.authenticated_peers:
            raise ValueError(f"Peer {peer_id} not authenticated")

        # Establish shared key
        session_key = self.secure_channel.establish_shared_key(peer_id, peer_public_key)

        # Create session ID
        session_id = secrets.token_urlsafe(16)
        self.active_sessions[session_id] = {
            "peer_id": peer_id,
            "key": session_key,
            "created": time.time()
        }

        logger.info(f"Created secure session {session_id} with peer {peer_id}")
        return session_id

    def send_secure(self, session_id: str, data: Any) -> Tuple[bytes, bytes]:
        """
        Send data securely through session

        Args:
            session_id: Session identifier
            data: Data to send

        Returns:
            (nonce, ciphertext) tuple
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session {session_id}")

        session = self.active_sessions[session_id]
        peer_id = session["peer_id"]

        # Serialize data
        serialized = json.dumps(data).encode()

        # Encrypt
        nonce, ciphertext = self.secure_channel.encrypt_message(serialized, peer_id)

        return nonce, ciphertext

    def receive_secure(self, session_id: str, nonce: bytes, ciphertext: bytes) -> Any:
        """
        Receive data securely through session

        Args:
            session_id: Session identifier
            nonce: Encryption nonce
            ciphertext: Encrypted data

        Returns:
            Decrypted data
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session {session_id}")

        session = self.active_sessions[session_id]
        peer_id = session["peer_id"]

        # Decrypt
        plaintext = self.secure_channel.decrypt_message(nonce, ciphertext, peer_id)

        # Deserialize
        data = json.loads(plaintext.decode())

        return data

    def close_session(self, session_id: str):
        """Close secure session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Closed session {session_id}")


class FederatedSecureProtocol:
    """Secure protocol specifically for federated learning"""

    def __init__(self):
        """Initialize federated secure protocol"""
        self.server_protocol = None
        self.client_protocols = {}

    def initialize_server(self) -> SecureProtocol:
        """Initialize server-side protocol"""
        self.server_protocol = SecureProtocol("server", 0)
        return self.server_protocol

    def initialize_client(self, client_id: int) -> SecureProtocol:
        """Initialize client-side protocol"""
        protocol = SecureProtocol("client", client_id)
        self.client_protocols[client_id] = protocol
        return protocol

    def secure_weight_transfer(
        self,
        client_id: int,
        weights: Dict[str, Any],
        session_id: str
    ) -> Tuple[bytes, bytes]:
        """
        Securely transfer weights from client to server

        Args:
            client_id: Client identifier
            weights: Weight updates
            session_id: Session identifier

        Returns:
            Encrypted weight package
        """
        if client_id not in self.client_protocols:
            raise ValueError(f"Client {client_id} not initialized")

        client_protocol = self.client_protocols[client_id]

        # Package weights with metadata
        package = {
            "client_id": client_id,
            "timestamp": time.time(),
            "weights": weights
        }

        # Send securely
        return client_protocol.send_secure(session_id, package)

    def verify_and_decrypt_weights(
        self,
        session_id: str,
        nonce: bytes,
        ciphertext: bytes
    ) -> Dict[str, Any]:
        """
        Verify and decrypt received weights

        Args:
            session_id: Session identifier
            nonce: Encryption nonce
            ciphertext: Encrypted weights

        Returns:
            Decrypted weight package
        """
        if not self.server_protocol:
            raise ValueError("Server protocol not initialized")

        # Receive and decrypt
        package = self.server_protocol.receive_secure(session_id, nonce, ciphertext)

        # Verify timestamp (prevent replay attacks)
        current_time = time.time()
        if current_time - package["timestamp"] > 300:  # 5 minute window
            raise ValueError("Timestamp too old - possible replay attack")

        return package