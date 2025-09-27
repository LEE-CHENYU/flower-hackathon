#!/usr/bin/env python3
"""
Test daily care suggestions generation with GPT-4o
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import base64

load_dotenv()

def test_care_suggestions():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return

    client = openai.OpenAI(api_key=api_key)

    # Get sample images
    dataset_dir = Path("first_images_dataset/train")
    sample_images = sorted(list(dataset_dir.glob("*.JPG"))[:3])

    if not sample_images:
        print("No images found")
        return

    results = []

    for i, img_path in enumerate(sample_images, 1):
        print(f"\n[Image {i}]: {img_path.name}")
        print("-" * 60)

        try:
            with open(img_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a dental hygiene expert. Based on this dental image, provide personalized daily care suggestions:
                            1. Brushing technique recommendations specific to what you observe
                            2. Flossing tips tailored to the visible tooth spacing and gum line
                            3. Recommended oral care products (toothpaste type, mouthwash, etc.)
                            4. Dietary suggestions for better dental health
                            5. Frequency and timing of dental care routines

                            Focus on preventive care and daily habits. Be specific and practical."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_completion_tokens=400
            )

            suggestions = response.choices[0].message.content
            print(f"Daily Care Suggestions:\n{suggestions}")

            results.append({
                "image": str(img_path),
                "suggestions": suggestions,
                "model": "gpt-4o"
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "image": str(img_path),
                "error": str(e)
            })

    # Save results
    with open("care_suggestions_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to care_suggestions_results.json")
    print(f"Successfully generated {len([r for r in results if 'suggestions' in r])} care suggestions")

if __name__ == "__main__":
    test_care_suggestions()