"""
Model configurations for different LLaVA variants
"""

MODEL_CONFIGS = {
    "tiny-llava": {
        "model_name": "bczhou/tiny-llava-v1-hf",
        "use_quantization": False,
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "description": "TinyLLaVA 1.5B - Smallest, fastest, good for testing"
    },

    "tiny-llava-3b": {
        "model_name": "bczhou/TinyLLaVA-3.1B",
        "use_quantization": False,
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "description": "TinyLLaVA 3.1B - Best tiny model, comparable to 7B"
    },

    "llava-7b-qlora": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "use_quantization": True,
        "quantization_bits": 4,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "description": "LLaVA 7B with 4-bit QLoRA - Memory efficient (~3GB)"
    },

    "llava-7b-8bit": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "use_quantization": True,
        "quantization_bits": 8,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "description": "LLaVA 7B with 8-bit quantization - Balanced (~6GB)"
    },

    "llava-7b-full": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "use_quantization": False,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        },
        "description": "LLaVA 7B full precision - Best quality (~14GB)"
    },

    "bakllava-api": {
        "model_name": "bakllava:7b-v1-q2_K",
        "use_api": True,
        "description": "BakLLaVA via Ollama API - No fine-tuning, inference only"
    }
}

def get_model_config(config_name: str = "tiny-llava"):
    """
    Get model configuration by name

    Args:
        config_name: Name of the configuration

    Returns:
        Dictionary with model configuration
    """
    if config_name not in MODEL_CONFIGS:
        print(f"Config '{config_name}' not found. Available configs:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  - {name}: {config['description']}")
        print(f"Using default: tiny-llava")
        config_name = "tiny-llava"

    return MODEL_CONFIGS[config_name]

def print_model_comparison():
    """Print comparison of different model configurations"""
    print("\n" + "="*80)
    print("Available Model Configurations")
    print("="*80)
    print(f"{'Config':<20} {'Model':<30} {'Memory':<10} {'Fine-tune':<10}")
    print("-"*80)

    memory_estimates = {
        "tiny-llava": "~4GB",
        "tiny-llava-3b": "~6GB",
        "llava-7b-qlora": "~3GB",
        "llava-7b-8bit": "~6GB",
        "llava-7b-full": "~14GB",
        "bakllava-api": "~2GB"
    }

    for name, config in MODEL_CONFIGS.items():
        model_name = config.get('model_name', 'N/A')
        if len(model_name) > 30:
            model_name = model_name[:27] + "..."
        memory = memory_estimates.get(name, "N/A")
        can_finetune = "✓" if not config.get('use_api', False) else "✗"
        print(f"{name:<20} {model_name:<30} {memory:<10} {can_finetune:<10}")

    print("="*80)
    print("Recommendations:")
    print("- For testing: Use 'tiny-llava' (fastest, smallest)")
    print("- For quality: Use 'tiny-llava-3b' or 'llava-7b-qlora'")
    print("- For limited VRAM: Use QLoRA variants (4-bit or 8-bit)")
    print("- For inference only: Use 'bakllava-api' with Ollama")
    print("="*80 + "\n")

if __name__ == "__main__":
    print_model_comparison()