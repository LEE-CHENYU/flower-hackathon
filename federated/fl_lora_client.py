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
from simple_data_loader import SimpleImageDataset

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

        # Initialize data loader - use first_images_dataset for faster training
        first_images_path = Path("/Users/chenyusu/flower-hackathon/first_images_dataset")
        if first_images_path.exists():
            print(f"Client {client_id}: Using extracted first_images_dataset")
            data_path = str(first_images_path)
        else:
            # Fallback to original dataset
            data_path = str(Path(data_path).absolute())

        self.dataset = SimpleImageDataset(data_path)
        self.image_paths = [item[0] for item in self.dataset.image_paths]

        # Limit to subset for each client to reduce costs and training time
        # Use only 3 images per client for testing
        images_per_client = 3
        start_idx = client_id * images_per_client
        end_idx = min(start_idx + images_per_client, len(self.image_paths))

        if start_idx < len(self.image_paths):
            self.image_paths = self.image_paths[start_idx:end_idx]
            print(f"Client {client_id}: Using {len(self.image_paths)} images from index {start_idx} to {end_idx}")
        else:
            # If we run out of images, wrap around
            self.image_paths = self.image_paths[:images_per_client]
            print(f"Client {client_id}: Using first {len(self.image_paths)} images (wrapped)")

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
        """Generate training labels on the fly with GPT-4o"""
        print(f"Client {self.client_id}: Generating labels with GPT-4o for {len(self.image_paths)} images...")

        self.training_pairs = []

        for i, img_path in enumerate(self.image_paths):
            try:
                # Generate label for this specific image
                label_data = self.label_generator.generate_diagnosis_label(img_path)

                if label_data["status"] == "success":
                    self.training_pairs.append((img_path, label_data["diagnosis"]))
                    print(f"  ✓ Generated label {i+1}/{len(self.image_paths)}")
                else:
                    # Fallback to default diagnosis if GPT-4o fails
                    fallback = "Dental image requires professional evaluation. Regular checkup recommended."
                    self.training_pairs.append((img_path, fallback))
                    print(f"  ⚠ Using fallback for image {i+1}")

            except Exception as e:
                # If API fails, use fallback diagnosis
                print(f"  ⚠ API error for image {i+1}: {e}")
                fallback = "Dental image shows teeth structure. Professional evaluation recommended."
                self.training_pairs.append((img_path, fallback))

        print(f"Client {self.client_id}: Generated {len(self.training_pairs)} training pairs on the fly")

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
        num_samples = max(1, len(self.training_pairs))  # Ensure at least 1
        metrics = {
            "loss": float(avg_loss),
            "num_samples": num_samples
        }

        return updated_parameters, num_samples, metrics

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