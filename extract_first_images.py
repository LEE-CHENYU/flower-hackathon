#!/usr/bin/env python3
"""
Extract first image from each folder in omni_coco dataset
and organize them into train/test/val structure
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import json

def get_first_image_from_folder(folder_path: Path) -> str:
    """Get the first image file from a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Get all image files and sort them
    images = []
    for ext in image_extensions:
        images.extend(folder_path.glob(f'*{ext}'))
        images.extend(folder_path.glob(f'*{ext.upper()}'))

    if images:
        # Sort by name and return first
        images.sort(key=lambda x: x.name)
        return str(images[0])

    return None

def extract_first_images(source_dir: str, output_dir: str):
    """
    Extract first image from each folder and organize into train/test/val

    Args:
        source_dir: Path to omni_coco directory
        output_dir: Path to output directory for organized first images
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'train': {'folders': 0, 'images': 0, 'files': []},
        'val': {'folders': 0, 'images': 0, 'files': []},
        'test': {'folders': 0, 'images': 0, 'files': []}
    }

    # Process each split
    for split in ['train', 'val', 'test']:
        split_source = source_path / split
        split_output = output_path / split

        if not split_source.exists():
            print(f"Warning: {split_source} does not exist, skipping...")
            continue

        # Create output split directory
        split_output.mkdir(exist_ok=True)
        print(f"\nProcessing {split} split...")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(split_source):
            root_path = Path(root)

            # Skip the split root directory itself
            if root_path == split_source:
                continue

            # Get relative path from split directory
            rel_path = root_path.relative_to(split_source)

            # Look for first image in this folder
            first_image = get_first_image_from_folder(root_path)

            if first_image:
                # Create a meaningful filename
                # Replace path separators with underscores
                new_name = str(rel_path).replace('/', '_').replace('\\', '_')

                # Get original extension
                ext = Path(first_image).suffix

                # Create destination filename
                dest_name = f"{new_name}_first{ext}"
                dest_path = split_output / dest_name

                # Copy the file
                shutil.copy2(first_image, dest_path)

                stats[split]['folders'] += 1
                stats[split]['images'] += 1
                stats[split]['files'].append({
                    'source': str(Path(first_image).relative_to(source_path)),
                    'destination': str(dest_path.relative_to(output_path)),
                    'folder': str(rel_path)
                })

                if stats[split]['images'] % 100 == 0:
                    print(f"  Processed {stats[split]['images']} images...")

        print(f"  Completed {split}: {stats[split]['images']} first images extracted")

    # Save extraction report
    report_path = output_path / 'extraction_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Output directory: {output_path}")
    print("\nSummary:")
    total_images = 0
    for split in ['train', 'val', 'test']:
        count = stats[split]['images']
        total_images += count
        print(f"  {split:5}: {count:4} first images from {stats[split]['folders']} folders")
    print(f"  Total: {total_images:4} images")
    print(f"\nExtraction report saved to: {report_path}")

    return stats

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract first image from each folder in omni_coco dataset'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='/Users/chenyusu/flower-hackathon/omni_coco',
        help='Path to omni_coco directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/chenyusu/flower-hackathon/first_images_dataset',
        help='Path to output directory for first images'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be extracted without copying files'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        source_path = Path(args.source)

        for split in ['train', 'val', 'test']:
            split_path = source_path / split
            if not split_path.exists():
                continue

            count = 0
            for root, dirs, files in os.walk(split_path):
                root_path = Path(root)
                if root_path != split_path:  # Skip root
                    first_img = get_first_image_from_folder(root_path)
                    if first_img:
                        count += 1
                        if count <= 3:  # Show first 3 examples
                            print(f"{split}: {Path(first_img).relative_to(source_path)}")

            print(f"{split}: Would extract {count} first images")
    else:
        extract_first_images(args.source, args.output)

if __name__ == "__main__":
    main()