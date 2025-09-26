#!/usr/bin/env python3
"""
Test that training produces non-zero loss
"""

import torch
from pathlib import Path

print("Testing Actual Training with Real Loss")
print("="*40)

# Load data
from simple_data_loader import SimpleImageDataset
dataset = SimpleImageDataset("/Users/chenyusu/flower-hackathon/first_images_dataset")
print(f"✓ Loaded {len(dataset.image_paths)} images")

# Get first 2 images
test_images = dataset.image_paths[:2]
print(f"✓ Using images: {[Path(p[0]).name for p in test_images]}")

# Load model
print("\nLoading model (this may take a moment)...")
from core.llava_lora_model import LLaVALoRAModel

model = LLaVALoRAModel("bczhou/tiny-llava-v1-hf")
print("✓ Model loaded")

# Create training data
training_pairs = [
    (test_images[0][0], "The dental image shows healthy teeth with no visible cavities."),
    (test_images[1][0], "Minor plaque buildup visible on lower molars.")
]

# Setup optimizer
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

# Train for 2 steps
print("\nTraining:")
for i, pair in enumerate(training_pairs):
    batch = [pair]
    loss = model.train_step(batch, optimizer)
    print(f"  Step {i+1}: Loss = {loss:.4f}")

    if loss > 0:
        print("✅ SUCCESS: Non-zero loss achieved! Training is working.")
    else:
        print("⚠️ WARNING: Loss is zero. Check labels/inputs.")

print("\n" + "="*40)
if any(model.train_step([p], optimizer) > 0 for p in training_pairs[:1]):
    print("✅ Training is working correctly with real loss values!")
else:
    print("❌ Issue: Loss remains zero. Check model configuration.")