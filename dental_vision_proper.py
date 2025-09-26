#!/usr/bin/env python3

import os
import subprocess
import json
from pathlib import Path
from data_loader import FirstPhotoDataset


def analyze_dental_image(image_path: str, model: str = "bakllava:7b-v1-q2_K") -> dict:
    """Properly analyze a dental image using BakLLaVA vision model"""

    prompt = """Analyze this dental image. Please identify and describe:

1. TEETH CONDITIONS:
   - Color (white, yellow, brown stains)
   - Alignment issues (crooked, gaps, overlapping)
   - Visible damage (chips, cracks, wear)
   - Dental work (fillings, crowns, bridges)

2. GUM HEALTH:
   - Color (healthy pink vs red/inflamed)
   - Swelling or recession
   - Bleeding signs

3. SPECIFIC PROBLEMS:
   - Cavities or dark spots
   - Plaque or tartar buildup
   - Root exposure
   - Missing teeth

Please be specific about what you actually see in THIS image."""

    try:
        # Run ollama with the image path directly after the model name
        cmd = f'echo "{prompt}" | ollama run {model} "{image_path}"'

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # Extract the actual response (skip the "Added image" line)
            lines = result.stdout.strip().split('\n')
            response = '\n'.join([line for line in lines if not line.startswith('Added image')])
            return {
                "status": "success",
                "analysis": response.strip()
            }
        else:
            return {
                "status": "error",
                "analysis": f"Error: {result.stderr}"
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "analysis": "Analysis timed out"
        }
    except Exception as e:
        return {
            "status": "error",
            "analysis": f"Error: {str(e)}"
        }


def main():
    print("\n" + "="*60)
    print("PROPER DENTAL DIAGNOSIS WITH BAKLLAVA VISION MODEL")
    print("="*60)

    # Get the dental images
    input_dir = "/Users/chenyusu/flower-hackathon/omni_coco"
    dataset = FirstPhotoDataset(input_dir)

    # Output directory
    output_dir = "/Users/chenyusu/flower-hackathon/dental_proper_output"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze first 3 images
    num_samples = 3
    results = []

    print(f"\nAnalyzing {num_samples} dental images...")
    print("-" * 60)

    for idx in range(min(num_samples, len(dataset))):
        _, folder_id, img_path = dataset[idx]

        print(f"\n[Image {idx+1}]")
        print(f"File: {os.path.basename(img_path)}")
        print(f"Path: {folder_id}")
        print("Analyzing...")

        # Analyze the image
        result = analyze_dental_image(img_path)

        if result["status"] == "success":
            print("✓ Analysis complete")
            # Show preview
            analysis = result["analysis"]
            preview = analysis[:500] + "..." if len(analysis) > 500 else analysis
            print(f"\nDiagnosis:\n{preview}")
        else:
            print(f"✗ {result['status']}: {result['analysis']}")

        # Save result
        results.append({
            "image": os.path.basename(img_path),
            "folder": folder_id,
            "status": result["status"],
            "diagnosis": result["analysis"]
        })

        print("-" * 60)

    # Save detailed results
    results_file = os.path.join(output_dir, "dental_diagnoses.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create markdown report
    report_file = os.path.join(output_dir, "diagnosis_report.md")
    with open(report_file, 'w') as f:
        f.write("# Dental Diagnosis Report\n")
        f.write("*Generated using BakLLaVA:7b-v1-q2_K Vision Model*\n\n")

        f.write("## Summary\n")
        successful = [r for r in results if r["status"] == "success"]
        f.write(f"- Total images analyzed: {len(results)}\n")
        f.write(f"- Successful diagnoses: {len(successful)}\n\n")

        f.write("## Detailed Diagnoses\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"### Image {i}: {result['image']}\n")
            f.write(f"**Location**: `{result['folder']}`\n\n")

            if result["status"] == "success":
                f.write("**Diagnosis**:\n\n")
                f.write(f"{result['diagnosis']}\n\n")
            else:
                f.write(f"**Error**: {result['status']}\n\n")

            f.write("---\n\n")

    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print(f"📄 Report: {report_file}")
    print(f"📊 Data: {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()