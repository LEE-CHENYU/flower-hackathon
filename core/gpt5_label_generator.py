"""
GPT-5 Label Generator for Dental Images
Generates training labels using OpenAI API
"""

import os
import json
import base64
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()

class GPT5LabelGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_diagnosis_label(self, image_path: str) -> Dict[str, str]:
        """
        Generate dental diagnosis label for an image

        Args:
            image_path: Path to dental image

        Returns:
            Dictionary containing diagnosis and metadata
        """
        try:
            base64_image = self.encode_image(image_path)

            prompt = """You are a dental hygiene expert. Based on this dental image, provide personalized daily care suggestions:
            1. Brushing technique recommendations specific to what you observe
            2. Flossing tips tailored to the visible tooth spacing and gum line
            3. Recommended oral care products (toothpaste type, mouthwash, etc.)
            4. Dietary suggestions for better dental health
            5. Frequency and timing of dental care routines

            Focus on preventive care and daily habits. Be specific and practical."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=400
            )

            diagnosis = response.choices[0].message.content

            return {
                "image_path": image_path,
                "diagnosis": diagnosis,
                "model_used": self.model,
                "status": "success"
            }

        except Exception as e:
            return {
                "image_path": image_path,
                "diagnosis": "",
                "error": str(e),
                "status": "failed"
            }

    def generate_batch_labels(self, image_paths: List[str], output_file: str = "generated_labels.json"):
        """Generate labels for multiple images"""
        labels = []

        for i, image_path in enumerate(image_paths):
            print(f"Generating label {i+1}/{len(image_paths)}: {Path(image_path).name}")
            label = self.generate_diagnosis_label(image_path)
            labels.append(label)

            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(labels, f, indent=2)

        return labels

    def create_training_pairs(self, image_paths: List[str]) -> List[Dict]:
        """Create image-text pairs for training"""
        pairs = []

        for image_path in image_paths:
            label = self.generate_diagnosis_label(image_path)
            if label["status"] == "success":
                pairs.append({
                    "image": image_path,
                    "text": label["diagnosis"]
                })

        return pairs