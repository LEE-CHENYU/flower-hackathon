# Federated Learning with LoRA Fine-tuning for Dental Diagnosis

## Overview

A federated learning system that fine-tunes LLaVA vision-language model using LoRA adapters for dental diagnosis. The system uses GPT-5/GPT-4 API to generate training labels and Flower framework for federated learning.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **Federated Learning**: Privacy-preserving distributed training using Flower framework
- **Vision-Language Model**: LLaVA model for generating diagnostic text from dental images
- **Automated Labeling**: GPT-5/GPT-4 API for generating training labels
- **Minimal Communication**: Only LoRA weights (~20MB) transmitted instead of full model (~7GB)

## Project Structure

```
flower-hackathon/
├── core/                          # Core model components
│   ├── gpt5_label_generator.py   # GPT-5 label generation
│   ├── llava_lora_model.py       # LLaVA with LoRA adapters
│   └── data_loader.py             # Data loading utilities
├── federated/                     # Federated learning components
│   ├── fl_lora_client.py         # FL client implementation
│   └── fl_lora_server.py         # FL server with LoRA aggregation
├── data/                          # Data and labels
│   └── omni_coco/                 # Dental images dataset
├── checkpoints/                   # Model checkpoints
│   └── lora_adapters/             # LoRA adapter weights
├── run_fl_training.py             # Main training script
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo 'OPENAI_API_KEY="your-api-key"' > .env
```

## Usage

### Full Simulation (Recommended)

Run complete federated learning with 3 clients:

```bash
python run_fl_training.py --clients 3 --rounds 10
```

### Separate Server and Clients

#### Start Server
```bash
python run_fl_training.py --mode server --rounds 10 --clients 3
```

#### Start Clients (in separate terminals)
```bash
# Client 0
python run_fl_training.py --mode client --client-id 0

# Client 1
python run_fl_training.py --mode client --client-id 1

# Client 2
python run_fl_training.py --mode client --client-id 2
```

### Evaluate Model

```bash
python run_fl_training.py --mode evaluate --checkpoint checkpoints/lora_adapters/final
```

## Configuration

### LoRA Parameters
```python
{
    "r": 16,                    # Low rank
    "lora_alpha": 32,          # Scaling factor
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none"
}
```

### Federated Learning Parameters
- `--rounds`: Number of federated rounds (default: 10)
- `--clients`: Number of clients (default: 3)
- `--local-epochs`: Local training epochs per round (default: 1)

## How It Works

1. **Label Generation**: GPT-5/GPT-4 analyzes dental images to generate diagnostic labels
2. **Local Training**: Each client trains LoRA adapters on their local data
3. **Weight Aggregation**: Server aggregates LoRA weights using FedAvg
4. **Model Update**: Clients receive updated global LoRA weights
5. **Iteration**: Process repeats for specified rounds

## Advantages

- **Privacy**: Patient data never leaves local clients
- **Efficiency**: Only LoRA weights (20MB) transmitted, not full model (7GB)
- **Scalability**: Can handle many clients with minimal server resources
- **Quality**: Combines local knowledge while maintaining global performance

## Fallback Options

If LLaVA PyTorch loading fails, the system automatically falls back to:
1. API-based training simulation
2. Using pre-trained BakLLaVA for inference
3. GPT-4 if GPT-5 is unavailable

## Requirements

- Python 3.8+
- CUDA GPU (optional but recommended)
- OpenAI API key
- 16GB+ RAM recommended

## License

MIT