#!/usr/bin/env python3
"""
Minimal test for care suggestions
"""
import json

# Create sample care suggestions for training
sample_care_suggestions = [
    {
        "image": "train_image_001.jpg",
        "care": "Brush twice daily with soft-bristled toothbrush at 45-degree angle. Use fluoride toothpaste and gentle circular motions. Floss daily between all teeth, especially the back molars. Consider an electric toothbrush for more effective plaque removal."
    },
    {
        "image": "train_image_002.jpg",
        "care": "Focus on gum line cleaning with extra attention to lower front teeth where tartar builds up. Use waxed floss for tight spaces. Rinse with antibacterial mouthwash once daily. Limit sugary drinks and snacks between meals."
    },
    {
        "image": "train_image_003.jpg",
        "care": "For sensitive teeth, use desensitizing toothpaste with potassium nitrate. Avoid aggressive brushing that can wear enamel. Wait 30 minutes after eating acidic foods before brushing. Consider using a straw for acidic beverages."
    }
]

print("Sample Care Suggestions for Training:")
print("=" * 60)

for i, item in enumerate(sample_care_suggestions, 1):
    print(f"\n[{i}] {item['image']}")
    print(f"Care: {item['care'][:150]}...")

print("\n" + "=" * 60)
print("These care suggestions will be used for LoRA fine-tuning")
print("The model will learn to generate similar dental care advice")

# Save to file
with open("sample_care_training_data.json", "w") as f:
    json.dump(sample_care_suggestions, f, indent=2)

print("\nSaved to sample_care_training_data.json")