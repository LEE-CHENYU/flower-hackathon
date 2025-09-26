#!/usr/bin/env python3
"""
Generate test labels for dental images
"""

from pathlib import Path
import json

# Get first 3 images
image_dir = Path("/Users/chenyusu/flower-hackathon/first_images_dataset/train")
images = list(image_dir.glob("*.JPG"))[:3]

print("Generating test labels for quick training...")
print(f"Images: {[img.name for img in images]}")

# Create simple labels without API calls
labels = []
diagnoses = [
    "The dental X-ray shows healthy tooth structure with good enamel density. No visible cavities or decay detected. Gum line appears normal.",
    "Minor tartar buildup observed on lower incisors. Recommend professional cleaning. Otherwise healthy dentition.",
    "Slight gum recession noted around upper molars. Monitor for progression. Good overall oral health maintained."
]

for i, img_path in enumerate(images):
    labels.append({
        "image_path": str(img_path),
        "diagnosis": diagnoses[i % len(diagnoses)],
        "model_used": "manual",
        "status": "success"
    })

# Save labels for each client
Path("data").mkdir(exist_ok=True)

for client_id in range(3):
    # Each client gets the same labels but for their specific images
    start_idx = client_id * 3
    client_images = list(image_dir.glob("*.JPG"))[start_idx:start_idx+3]

    client_labels = []
    for i, img_path in enumerate(client_images):
        client_labels.append({
            "image_path": str(img_path),
            "diagnosis": diagnoses[i % len(diagnoses)],
            "model_used": "manual",
            "status": "success"
        })

    output_file = f"data/labels_client_{client_id}.json"
    with open(output_file, 'w') as f:
        json.dump(client_labels, f, indent=2)

    print(f"✓ Created {output_file} with {len(client_labels)} labels")

print("\n✅ Test labels generated successfully!")