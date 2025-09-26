"""
Federated Learning Client for LoRA Fine-tuning
"""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.llava_lora_model import LLaVALoRAModel
from core.gpt5_label_generator import GPT5LabelGenerator
from core.model_configs import get_model_config
from data_loader import FirstPhotoDataset

class DentalFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        data_path: str,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        model_config: str = "tiny-llava"
    ):
        """
        Initialize Federated Learning Client

        Args:
            client_id: Unique identifier for this client
            data_path: Path to local data
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            model_config: Model configuration name
        """
        self.client_id = client_id
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Get model configuration
        config = get_model_config(model_config)
        print(f"Client {client_id}: Using {model_config} - {config['description']}")

        # Initialize model with configuration
        print(f"Client {client_id}: Initializing model...")
        self.model = LLaVALoRAModel(
            model_name=config["model_name"],
            lora_config=config.get("lora_config"),
            use_quantization=config.get("use_quantization", False),
            quantization_bits=config.get("quantization_bits", 4)
        )

        # Initialize data loader
        self.dataset = FirstPhotoDataset(data_path)
        self.image_paths = [item[1] for item in self.dataset.first_photos]

        # Initialize label generator
        self.label_generator = GPT5LabelGenerator()

        # Generate or load labels
        self._prepare_labels()

        # Initialize optimizer
        if self.model.model is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.model.parameters(),
                lr=self.learning_rate
            )
        else:
            self.optimizer = None

    def _prepare_labels(self):
        """Generate or load training labels"""
        labels_file = f"data/labels_client_{self.client_id}.json"

        if Path(labels_file).exists():
            print(f"Client {self.client_id}: Loading existing labels from {labels_file}")
            import json
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        else:
            print(f"Client {self.client_id}: Generating labels with GPT-5...")
            # Generate labels for subset of images (to save API costs)
            sample_size = min(10, len(self.image_paths))
            sample_images = self.image_paths[:sample_size]

            self.labels = self.label_generator.generate_batch_labels(
                sample_images,
                output_file=labels_file
            )

        # Create training pairs
        self.training_pairs = []
        for label_data in self.labels:
            if label_data["status"] == "success":
                self.training_pairs.append((
                    label_data["image_path"],
                    label_data["diagnosis"]
                ))

        print(f"Client {self.client_id}: Prepared {len(self.training_pairs)} training pairs")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters for server"""
        lora_weights = self.model.get_lora_weights()
        return [weight for weight in lora_weights.values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server"""
        # Reconstruct weight dictionary
        lora_weights = {}
        if self.model.model is not None:
            weight_names = list(self.model.get_lora_weights().keys())
            for i, name in enumerate(weight_names):
                if i < len(parameters):
                    lora_weights[name] = parameters[i]

            self.model.set_lora_weights(lora_weights)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        # Set global model parameters
        self.set_parameters(parameters)

        # Training configuration
        num_epochs = config.get("local_epochs", 1)

        print(f"Client {self.client_id}: Starting local training for {num_epochs} epochs")

        # Perform local training
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            # Create batches
            for i in range(0, len(self.training_pairs), self.batch_size):
                batch = self.training_pairs[i:i + self.batch_size]
                loss = self.model.train_step(batch, self.optimizer)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Client {self.client_id}: Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Get updated parameters
        updated_parameters = self.get_parameters(config)

        # Return metrics
        metrics = {
            "loss": float(avg_loss),
            "num_samples": len(self.training_pairs)
        }

        return updated_parameters, len(self.training_pairs), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local data"""
        self.set_parameters(parameters)

        # Simple evaluation: generate diagnosis for a sample image
        if len(self.image_paths) > 0:
            sample_image = self.image_paths[0]
            diagnosis = self.model.generate_diagnosis(sample_image)

            # Simple quality metric (length of diagnosis)
            quality_score = min(len(diagnosis) / 100.0, 1.0)
        else:
            quality_score = 0.0

        metrics = {
            "quality_score": quality_score,
            "client_id": self.client_id
        }

        return float(quality_score), len(self.training_pairs), metrics

def create_client(client_id: int, data_path: str) -> DentalFLClient:
    """Factory function to create client"""
    return DentalFLClient(client_id, data_path)

def start_client(server_address: str = "localhost:8080", client_id: int = 0, data_path: str = "omni_coco"):
    """Start federated learning client"""
    client = create_client(client_id, data_path)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )