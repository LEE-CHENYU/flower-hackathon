#!/usr/bin/env python3
"""
Debug test for label generation with GPT-4o
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()

def test_direct_gpt4o():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return

    client = openai.OpenAI(api_key=api_key)

    # Get a sample image
    dataset_dir = Path("first_images_dataset/train")
    sample_image = sorted(list(dataset_dir.glob("*.JPG")))[0] if list(dataset_dir.glob("*.JPG")) else None

    if not sample_image:
        print("No images found")
        return

    print(f"Testing with: {sample_image}")

    # Try direct GPT-4o call (GPT-5 doesn't exist yet)
    try:
        print("\nAttempting GPT-4o directly...")

        import base64
        with open(sample_image, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this dental image and provide a diagnosis."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_completion_tokens=300
        )

        diagnosis = response.choices[0].message.content
        print(f"\nGPT-4o Response:\n{diagnosis}")

        # Save result
        result = {
            "image": str(sample_image),
            "model": "gpt-4o",
            "diagnosis": diagnosis
        }

        with open("gpt4o_test_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print("\nSaved to gpt4o_test_result.json")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    test_direct_gpt4o()