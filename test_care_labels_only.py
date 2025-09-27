#!/usr/bin/env python3
"""
Test only the care label generation without loading models
"""
import os
from pathlib import Path
from core.gpt5_label_generator import GPT5LabelGenerator

def test_care_labels():
    """Test care label generation"""

    print("=" * 60)
    print("Testing Care Suggestion Generation")
    print("=" * 60)

    # Get sample images from dataset
    dataset_path = Path("/Users/chenyusu/flower-hackathon/first_images_dataset/train")
    images = sorted(list(dataset_path.glob("*.JPG"))[:2])

    if not images:
        print("No images found!")
        return

    # Initialize generator
    print("\nInitializing label generator...")
    generator = GPT5LabelGenerator()
    print(f"Using model: {generator.model}")

    # Test generation for each image
    for i, img_path in enumerate(images, 1):
        print(f"\n[Image {i}]: {img_path.name}")
        print("-" * 40)

        try:
            result = generator.generate_diagnosis_label(str(img_path))

            if result["status"] == "success":
                care_suggestions = result["diagnosis"]
                print(f"✓ Generated care suggestions ({len(care_suggestions)} chars)")
                print(f"\nFirst 400 characters:")
                print(care_suggestions[:400] + "...")
            else:
                print(f"✗ Failed to generate: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_care_labels()