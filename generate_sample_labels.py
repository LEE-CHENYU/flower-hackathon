#!/usr/bin/env python3
"""
Generate sample daily care labels for testing
"""
import json
from pathlib import Path
from core.gpt5_label_generator import GPT5LabelGenerator

def generate_sample_labels():
    """Generate labels for a few sample images"""

    # Initialize generator
    print("Initializing GPT label generator...")
    generator = GPT5LabelGenerator()

    dataset_path = Path("/Users/chenyusu/flower-hackathon/first_images_dataset")

    # Get 2 images from each split
    sample_labels = []

    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        images = sorted(list(split_dir.glob("*.JPG"))[:2])

        for img_path in images:
            print(f"\nProcessing: {split}/{img_path.name}")
            print("-" * 60)

            try:
                # Generate label
                result = generator.generate_diagnosis_label(str(img_path))

                label_entry = {
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "split": split,
                    "care_suggestions": result.get("diagnosis", ""),
                    "status": result.get("status", "error"),
                    "model_used": result.get("model_used", "unknown")
                }

                print(f"Model: {label_entry['model_used']}")
                print(f"Status: {label_entry['status']}")
                print(f"\nCare Suggestions:\n{label_entry['care_suggestions'][:300]}...")

                sample_labels.append(label_entry)

            except Exception as e:
                print(f"Error: {e}")
                sample_labels.append({
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "split": split,
                    "care_suggestions": "Error generating suggestions",
                    "status": "error",
                    "error": str(e)
                })

    # Save sample labels
    with open("sample_care_labels.json", "w") as f:
        json.dump(sample_labels, f, indent=2)

    print(f"\n\nSaved {len(sample_labels)} sample labels to sample_care_labels.json")

    # Show summary
    successful = sum(1 for l in sample_labels if l.get("status") == "success")
    print(f"Success rate: {successful}/{len(sample_labels)} ({100*successful/len(sample_labels):.1f}%)")

if __name__ == "__main__":
    generate_sample_labels()