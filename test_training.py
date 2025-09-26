#!/usr/bin/env python3
"""
Test script for LoRA training with pre-generated labels
"""

import json
from pathlib import Path
import torch
from core.llava_lora_model import LLaVALoRAModel
from core.model_configs import get_model_config

def test_training():
    """Test basic LoRA training with minimal data"""

    print("="*60)
    print("Testing LoRA Training")
    print("="*60)

    # Create dummy labels for testing
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Get first 3 images from dataset
    image_dir = Path("/Users/chenyusu/flower-hackathon/first_images_dataset/train")
    images = list(image_dir.glob("*.JPG"))[:3]

    # Create dummy labels
    labels = []
    for img in images:
        labels.append({
            "image_path": str(img),
            "diagnosis": f"Test diagnosis: The dental image shows healthy teeth with no visible cavities.",
            "status": "success"
        })

    # Save labels
    labels_file = data_dir / "test_labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Created {len(labels)} test labels")

    # Try to load model
    print("\nAttempting to load model...")
    try:
        # Try TinyLLaVA with fallback
        config = get_model_config("tiny-llava")
        model = LLaVALoRAModel(
            model_name=config["model_name"],
            lora_config=config.get("lora_config"),
            use_quantization=config.get("use_quantization", False),
            quantization_bits=config.get("quantization_bits", 4)
        )
        print("✓ Model loaded successfully!")

        # Create training pairs
        training_pairs = [(label["image_path"], label["diagnosis"]) for label in labels]

        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

        # Train one step
        print("\nPerforming training step...")
        loss = model.train_step(training_pairs[:1], optimizer)  # Just one sample
        print(f"Training loss: {loss:.4f}")

        print("\n✓ Training test successful!")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nModel loading failed. This might be due to:")
        print("1. Model download taking too long")
        print("2. Insufficient memory")
        print("3. Missing dependencies")
        print("\nTry using --model-config llava-7b-qlora for 4-bit quantization")

if __name__ == "__main__":
    test_training()