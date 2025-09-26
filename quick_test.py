#!/usr/bin/env python3
"""
Quick test to verify LoRA training works
"""

import torch
from pathlib import Path

print("Quick LoRA Training Test")
print("="*40)

# Check if model is cached
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
tiny_llava_cache = list(cache_dir.glob("*tiny-llava*"))

if tiny_llava_cache:
    print("✓ TinyLLaVA model is cached locally")
else:
    print("⚠ TinyLLaVA not cached, will need to download (~3GB)")

# Test with minimal setup
from core.llava_lora_model import LLaVALoRAModel

print("\nLoading model...")
try:
    model = LLaVALoRAModel("bczhou/tiny-llava-v1-hf")
    print("✓ Model loaded successfully")

    # Get a test image
    img = "/Users/chenyusu/flower-hackathon/first_images_dataset/train/2018.01_2018.07.30_first.JPG"

    # Quick training test
    print("\nTesting training step...")
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    batch = [(img, "Test dental diagnosis")]
    loss = model.train_step(batch, optimizer)

    print(f"✓ Training step successful! Loss: {loss:.4f}")

    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        diagnosis = model.generate_diagnosis(img, max_length=50)
        print(f"✓ Generated: {diagnosis[:100]}...")

    print("\n✅ All tests passed! LoRA training is working.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()