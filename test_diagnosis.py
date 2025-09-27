#!/usr/bin/env python3
"""
Test the updated diagnosis generation
"""

from core.llava_lora_model import LLaVALoRAModel
from core.model_configs import get_model_config
from pathlib import Path

print("Testing updated model with improved prompt and parameters...")
print("="*60)

# Initialize model
config = get_model_config('tiny-llava')
model = LLaVALoRAModel(
    model_name=config['model_name'],
    lora_config=config.get('lora_config')
)

# Test with several images
test_images = [
    '/Users/chenyusu/flower-hackathon/first_images_dataset/train/2018.01_2018.07.30_first.JPG',
    '/Users/chenyusu/flower-hackathon/first_images_dataset/train/2018.01_2018.08.27_first.JPG',
    '/Users/chenyusu/flower-hackathon/first_images_dataset/train/2018.01_2018.11.05_first.JPG'
]

for i, image_path in enumerate(test_images, 1):
    if Path(image_path).exists():
        print(f"\nImage {i}: {Path(image_path).name}")
        print("-" * 40)
        diagnosis = model.generate_diagnosis(image_path)
        print(f"Diagnosis: {diagnosis[:300]}")

        # Check if toothbrush is mentioned
        if 'toothbrush' in diagnosis.lower():
            print("⚠️  WARNING: Model still mentions toothbrush!")
        else:
            print("✓ No toothbrush hallucination")
    else:
        print(f"Image not found: {image_path}")

print("\n" + "="*60)
print("Test complete!")