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
        self.model = "gpt-5"  # As requested, using gpt-5

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

            prompt = """You are an expert dentist. Analyze this dental X-ray or photograph and provide:
            1. Primary diagnosis
            2. Severity assessment (mild/moderate/severe)
            3. Observed conditions (list all visible issues)
            4. Recommended treatment plan
            5. Urgency level (routine/priority/urgent)

            Format your response as a professional dental diagnosis."""

            # Try GPT-5 first, fallback to GPT-4 if needed
            try:
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
                    max_tokens=300
                )
            except Exception as e:
                # Fallback to GPT-4 if GPT-5 not available
                print(f"GPT-5 not available, falling back to GPT-4: {e}")
                self.model = "gpt-4o"
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
                    max_tokens=300
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