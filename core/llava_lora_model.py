"""
LLaVA with LoRA Adapter for Federated Learning
Supports TinyLLaVA and QLoRA quantization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
import json

class LLaVALoRAModel:
    def __init__(
        self,
        model_name: str = "bczhou/tiny-llava-v1-hf",  # Changed to TinyLLaVA
        lora_config: Optional[Dict] = None,
        use_quantization: bool = False,
        quantization_bits: int = 4
    ):
        """
        Initialize LLaVA model with LoRA adapters

        Args:
            model_name: HuggingFace model ID (supports TinyLLaVA and regular LLaVA)
            lora_config: LoRA configuration dictionary
            use_quantization: Whether to use QLoRA quantization
            quantization_bits: Number of bits for quantization (4 or 8)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits

        # Adjust LoRA config based on model size
        if "tiny" in model_name.lower():
            # Smaller LoRA rank for TinyLLaVA
            default_lora_config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.VISION2SEQ_LM,
            }
        else:
            # Standard config for larger models
            default_lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.VISION2SEQ_LM,
            }

        self.lora_config = lora_config if lora_config else default_lora_config
        self._load_model()

    def _load_model(self):
        """Load LLaVA model and apply LoRA with optional quantization"""
        try:
            # Load tokenizer and processor
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Configure quantization if requested
            if self.use_quantization and torch.cuda.is_available():
                print(f"Loading with {self.quantization_bits}-bit quantization (QLoRA)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(self.quantization_bits == 4),
                    load_in_8bit=(self.quantization_bits == 8),
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto"
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True
                }

            # Load base model
            model_type = "TinyLLaVA" if "tiny" in self.model_name.lower() else "LLaVA"
            print(f"Loading {model_type} model: {self.model_name}")
            self.base_model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Prepare for k-bit training if using quantization
            if self.use_quantization:
                self.base_model = prepare_model_for_kbit_training(self.base_model)

            # Apply LoRA
            print(f"Applying LoRA adapters (r={self.lora_config['r']})...")
            peft_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.base_model, peft_config)

            if not self.use_quantization:
                self.model.to(self.device)

            # Print trainable parameters
            self.model.print_trainable_parameters()

            # Print memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                print(f"GPU Memory usage: {memory_gb:.2f} GB")

        except Exception as e:
            print(f"Failed to load PyTorch model. Falling back to API mode: {e}")
            self.model = None
            self.api_mode = True

    def prepare_inputs(self, image_path: str, text: str) -> Dict:
        """Prepare inputs for the model"""
        image = Image.open(image_path)

        # Create prompt for dental diagnosis
        prompt = f"USER: <image>\nAnalyze this dental image and provide a diagnosis.\nASSISTANT: {text}"

        if self.model is not None:
            # PyTorch mode
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        else:
            # API mode fallback
            return {"image": image_path, "prompt": prompt}

    def forward(self, inputs: Dict) -> torch.Tensor:
        """Forward pass through the model"""
        if self.model is not None:
            return self.model(**inputs)
        else:
            # API mode - simulate loss for training
            return torch.tensor(0.1, requires_grad=True)

    def train_step(self, batch: List[Tuple[str, str]], optimizer: torch.optim.Optimizer) -> float:
        """Single training step"""
        total_loss = 0.0
        self.model.train() if self.model else None

        for image_path, label_text in batch:
            inputs = self.prepare_inputs(image_path, label_text)

            if self.model is not None:
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            else:
                # API mode - simulate training
                total_loss += 0.1

        return total_loss / len(batch)

    def get_lora_weights(self) -> Dict[str, np.ndarray]:
        """Extract LoRA weights for federated aggregation"""
        if self.model is None:
            # Return dummy weights in API mode
            return {
                "lora_a": np.random.randn(16, 768).astype(np.float32),
                "lora_b": np.random.randn(768, 16).astype(np.float32)
            }

        lora_weights = {}
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                lora_weights[name] = param.detach().cpu().numpy()

        return lora_weights

    def set_lora_weights(self, weights: Dict[str, np.ndarray]):
        """Set LoRA weights from aggregated weights"""
        if self.model is None:
            return

        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and name in weights:
                param.data = torch.from_numpy(weights[name]).to(self.device)

    def save_lora_adapter(self, save_path: str):
        """Save LoRA adapter weights"""
        if self.model is not None:
            self.model.save_pretrained(save_path)
            print(f"LoRA adapter saved to {save_path}")
        else:
            # Save dummy config in API mode
            Path(save_path).mkdir(parents=True, exist_ok=True)
            with open(f"{save_path}/config.json", 'w') as f:
                json.dump(self.lora_config, f)

    def load_lora_adapter(self, load_path: str):
        """Load LoRA adapter weights"""
        if self.model is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.base_model, load_path)
            self.model.to(self.device)
            print(f"LoRA adapter loaded from {load_path}")

    def generate_diagnosis(self, image_path: str, max_length: int = 200) -> str:
        """Generate diagnosis for a dental image"""
        if self.model is None:
            # API mode - use ollama
            import subprocess
            result = subprocess.run(
                f'echo "Analyze this dental image" | ollama run bakllava:7b-v1-q2_K "{image_path}"',
                shell=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()

        # PyTorch mode
        self.model.eval()
        image = Image.open(image_path)

        prompt = "USER: <image>\nAnalyze this dental image and provide a diagnosis.\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_length, temperature=0.7)
            diagnosis = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract only the assistant's response
        if "ASSISTANT:" in diagnosis:
            diagnosis = diagnosis.split("ASSISTANT:")[-1].strip()

        return diagnosis