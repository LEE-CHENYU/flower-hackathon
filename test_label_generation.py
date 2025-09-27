#!/usr/bin/env python3
"""
Test label generation with GPT-4o for dental images
"""
import json
import sys
from pathlib import Path
from core.gpt5_label_generator import GPT5LabelGenerator

def test_label_generation():
    # Initialize generator
    print("Initializing GPT label generator...")
    generator = GPT5LabelGenerator()
    print(f"Using model: {generator.model}")

    # Get sample images
    dataset_dir = Path("first_images_dataset/train")
    sample_images = sorted(list(dataset_dir.glob("*.JPG"))[:3])

    if not sample_images:
        print("No images found in first_images_dataset/train/")
        return

    print(f"\nTesting with {len(sample_images)} sample images:")
    print("-" * 80)

    results = []
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n[{i}] Processing: {img_path.name}")

        try:
            # Generate label
            result = generator.generate_diagnosis_label(str(img_path))

            # Display result
            print(f"Status: {result['status']}")
            print(f"Model used: {result['model_used']}")
            print(f"\nDiagnosis:\n{result['diagnosis']}")

            results.append(result)

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "image_path": str(img_path),
                "status": "error",
                "error": str(e)
            })

        print("-" * 80)

    # Save results
    output_file = "test_generated_labels.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Successfully generated {sum(1 for r in results if r.get('status') == 'success')} labels")

if __name__ == "__main__":
    test_label_generation()