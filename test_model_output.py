#!/usr/bin/env python3
"""
Test what the current LLaVA model outputs for dental images
"""
from pathlib import Path
from core.llava_lora_model import LLaVALoRAModel
import torch

def test_model_output():
    # Initialize model
    print("Loading model...")
    model_wrapper = LLaVALoRAModel(model_name="bczhou/tiny-llava-v1-hf")

    # Get sample images
    dataset_dir = Path("first_images_dataset/train")
    sample_images = sorted(list(dataset_dir.glob("*.JPG"))[:3])

    if not sample_images:
        print("No images found")
        return

    print(f"\nTesting with {len(sample_images)} images:")
    print("=" * 60)

    for i, img_path in enumerate(sample_images, 1):
        print(f"\n[Image {i}]: {img_path.name}")

        # Test with different prompts
        prompts = [
            "What do you see in this image?",
            "Describe the dental condition visible in this image.",
            "What daily care would you recommend based on this image?"
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            try:
                # Get model output
                output = model_wrapper.generate(str(img_path), prompt)
                print(f"Response: {output}")
            except Exception as e:
                print(f"Error: {e}")

        print("-" * 60)

if __name__ == "__main__":
    test_model_output()