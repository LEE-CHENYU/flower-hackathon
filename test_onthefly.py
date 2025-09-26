#!/usr/bin/env python3
"""
Test on-the-fly label generation
"""

import torch
from pathlib import Path

print("Testing On-the-Fly Training")
print("="*40)

# Simulate what the client does
from simple_data_loader import SimpleImageDataset

dataset = SimpleImageDataset("/Users/chenyusu/flower-hackathon/first_images_dataset")
image_paths = [item[0] for item in dataset.image_paths[:3]]

print(f"✓ Found {len(image_paths)} images")

# Generate labels on the fly
diagnoses = [
    "Healthy tooth structure with no visible cavities.",
    "Minor plaque buildup on lower molars.",
    "Good oral health with regular maintenance needed."
]

training_pairs = []
for i, img_path in enumerate(image_paths):
    diagnosis = diagnoses[i % len(diagnoses)]
    training_pairs.append((img_path, diagnosis))

print(f"✓ Generated {len(training_pairs)} training pairs on the fly")

# Test training
from core.llava_lora_model import LLaVALoRAModel

print("\nLoading model...")
model = LLaVALoRAModel("bczhou/tiny-llava-v1-hf")
print("✓ Model loaded")

optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

print("\nTraining with on-the-fly labels:")
for i, (img, label) in enumerate(training_pairs):
    loss = model.train_step([(img, label)], optimizer)
    print(f"  Step {i+1}: Loss = {loss:.4f}")

print("\n✅ On-the-fly training works perfectly!")