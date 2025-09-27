#!/usr/bin/env python3
"""
Generate daily care suggestion labels for first_images_dataset
"""
import json
import os
from pathlib import Path
from typing import Dict, List
import time
from core.gpt5_label_generator import GPT5LabelGenerator

def generate_labels_for_dataset(dataset_path: str = "/Users/chenyusu/flower-hackathon/first_images_dataset"):
    """Generate labels for all images in the dataset"""

    # Initialize generator
    print("Initializing GPT label generator...")
    generator = GPT5LabelGenerator()

    dataset_path = Path(dataset_path)

    # Process each split
    splits = ["train", "val", "test"]
    all_labels = {}

    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            continue

        # Get all images in this split
        images = list(split_dir.glob("*.JPG")) + list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        print(f"\n{split.upper()}: Found {len(images)} images")

        split_labels = []

        # Generate labels for each image
        for i, img_path in enumerate(images):
            print(f"  [{i+1}/{len(images)}] Processing {img_path.name}...", end="")

            try:
                # Generate label
                result = generator.generate_diagnosis_label(str(img_path))

                # Store result
                label_entry = {
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "split": split,
                    "care_suggestions": result.get("diagnosis", ""),  # Now contains care suggestions
                    "status": result.get("status", "error"),
                    "model_used": result.get("model_used", "unknown")
                }

                split_labels.append(label_entry)
                print(" ✓")

                # Small delay to avoid rate limiting
                if i % 10 == 9:
                    time.sleep(1)

            except Exception as e:
                print(f" ✗ Error: {e}")
                split_labels.append({
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "split": split,
                    "care_suggestions": "Daily dental care: Brush twice daily with fluoride toothpaste, floss once daily, and maintain regular dental checkups.",
                    "status": "error",
                    "error": str(e)
                })

        all_labels[split] = split_labels

        # Save split labels
        output_file = f"labels_{split}.json"
        with open(output_file, "w") as f:
            json.dump(split_labels, f, indent=2)
        print(f"  Saved {len(split_labels)} labels to {output_file}")

    # Save all labels in one file
    with open("all_dataset_labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = 0
    successful = 0

    for split, labels in all_labels.items():
        split_total = len(labels)
        split_success = sum(1 for l in labels if l.get("status") == "success")
        total += split_total
        successful += split_success
        print(f"{split:5}: {split_success}/{split_total} successful")

    print(f"TOTAL: {successful}/{total} successful ({100*successful/total:.1f}%)")

    return all_labels

if __name__ == "__main__":
    labels = generate_labels_for_dataset()