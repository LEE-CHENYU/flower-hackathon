#!/usr/bin/env python3
"""
Simple demo of LoRA training on dental images
"""

import json
from pathlib import Path
import torch
from core.llava_lora_model import LLaVALoRAModel
from core.model_configs import get_model_config

def demo_training():
    """Demo LoRA training with minimal setup"""

    print("="*60)
    print("Dental Diagnosis LoRA Training Demo")
    print("="*60)

    # Setup paths
    image_dir = Path("/Users/chenyusu/flower-hackathon/first_images_dataset/train")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Get first 2 images
    images = list(image_dir.glob("*.JPG"))[:2]

    if not images:
        print("No images found in first_images_dataset/train")
        return

    print(f"\nUsing {len(images)} dental images for demo")

    # Create simple training labels
    training_data = [
        {
            "image": str(images[0]),
            "diagnosis": "The dental X-ray shows healthy tooth structure with no visible cavities. "
                        "Gum line appears normal with no signs of recession or inflammation."
        },
        {
            "image": str(images[1]) if len(images) > 1 else str(images[0]),
            "diagnosis": "Examination reveals minor plaque buildup on lower molars. "
                        "Recommend professional cleaning and improved flossing routine."
        }
    ]

    print("\nLoading TinyLLaVA model with LoRA adapters...")

    try:
        # Load model
        config = get_model_config("tiny-llava")
        model = LLaVALoRAModel(
            model_name=config["model_name"],
            lora_config=config.get("lora_config")
        )

        print("✓ Model loaded successfully!")
        print(f"  - Model: {config['model_name']}")
        print(f"  - LoRA rank: {config['lora_config']['r']}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=1e-4
        )

        # Training loop
        print("\n" + "="*40)
        print("Starting Training")
        print("="*40)

        num_epochs = 2
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            total_loss = 0
            for i, data in enumerate(training_data):
                # Prepare batch
                batch = [(data["image"], data["diagnosis"])]

                # Train step
                loss = model.train_step(batch, optimizer)
                total_loss += loss

                print(f"  Sample {i+1}: Loss = {loss:.4f}")

            avg_loss = total_loss / len(training_data)
            print(f"  Average Loss: {avg_loss:.4f}")

        print("\n" + "="*40)
        print("Training Complete!")
        print("="*40)

        # Save the trained adapter
        save_dir = Path("checkpoints/demo_lora")
        print(f"\nSaving LoRA adapter to {save_dir}...")
        model.save_lora_adapter(str(save_dir))

        # Test generation
        print("\nTesting diagnosis generation on first image...")
        test_image = str(images[0])
        diagnosis = model.generate_diagnosis(test_image, max_length=100)

        print(f"\nGenerated Diagnosis:")
        print("-" * 40)
        print(diagnosis)
        print("-" * 40)

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\nTroubleshooting:")
        print("1. If model download is slow, wait or try again")
        print("2. If memory error, try: --model-config llava-7b-qlora")
        print("3. Check that first_images_dataset exists")

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_training()