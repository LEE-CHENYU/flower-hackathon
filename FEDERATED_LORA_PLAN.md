# Federated Learning with LoRA Fine-tuning Plan

## Executive Summary

Build a federated learning system to fine-tune LLaVA for dental diagnosis using LoRA adapters, with GPT-5 API generating training labels.

## Model Selection Decision

### Why LLaVA over BiomedCLIP

| Aspect | BiomedCLIP | LLaVA |
|--------|------------|--------|
| **Output Type** | Image-text similarity scores | Full diagnostic text |
| **Use Case** | Image retrieval/matching | Medical diagnosis generation |
| **What we need** | ❌ Only gives similarity | ✅ Generates diagnosis |
| **Example Output** | `similarity: 0.82` | `"Patient shows gum inflammation with possible periodontal disease..."` |

**Decision: Use LLaVA** because we need diagnostic text generation, not similarity matching.

## Technical Architecture

### 1. Model Setup

```python
# We need LLaVA in PyTorch format for LoRA fine-tuning
model_options = {
    "option_1": "liuhaotian/llava-v1.5-7b",  # Original PyTorch
    "option_2": "BAAI/llava-v1.5-7b",        # Alternative source
    "option_3": "Use BakLLaVA via API only"   # No fine-tuning
}
```

### 2. LoRA Configuration

```python
lora_config = {
    "r": 16,                    # Low rank for efficiency
    "lora_alpha": 32,          # Scaling factor
    "target_modules": [         # LLaVA attention layers
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none"
}
```

### 3. Federated Learning Architecture

```
┌─────────────────────────────────────────────┐
│                 FL Server                    │
│  - Aggregates LoRA weights                   │
│  - Coordinates training rounds               │
│  - Saves global LoRA adapter                 │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Client 1 │ │Client 2 │ │Client 3 │
├─────────┤ ├─────────┤ ├─────────┤
│Local    │ │Local    │ │Local    │
│Images   │ │Images   │ │Images   │
├─────────┤ ├─────────┤ ├─────────┤
│GPT-5    │ │GPT-5    │ │GPT-5    │
│Labels   │ │Labels   │ │Labels   │
├─────────┤ ├─────────┤ ├─────────┤
│LoRA     │ │LoRA     │ │LoRA     │
│Training │ │Training │ │Training │
└─────────┘ └─────────┘ └─────────┘
```

## Implementation Steps

### Phase 1: Setup & Testing
1. **Test LLaVA Loading**
   - Try loading PyTorch version of LLaVA
   - If fails, implement API-based training

2. **Test GPT-5 API**
   - Generate sample labels
   - Create label format specification

### Phase 2: Label Generation
```python
# label_generator.py
def generate_diagnosis_labels(image_path):
    """
    Use GPT-5 to generate training labels
    Input: Dental image
    Output: Professional diagnosis text
    """
    prompt = f"""
    Based on this dental image showing [description],
    provide a diagnosis including:
    1. Conditions observed
    2. Severity assessment
    3. Recommended treatment
    """
    return gpt5_api_call(prompt)
```

### Phase 3: Federated Learning Implementation

#### 3.1 Client Implementation
```python
class DentalFLClient:
    def __init__(self, client_id, data_path):
        self.model = load_llava_with_lora()
        self.data = load_dental_images(data_path)
        self.labels = generate_labels_with_gpt5()

    def train_local(self):
        # Train LoRA adapters only
        # Send only LoRA weights (few MB)
        pass
```

#### 3.2 Server Implementation
```python
class DentalFLServer:
    def aggregate_lora_weights(self, client_weights):
        # Average LoRA parameters
        # Much smaller than full model
        pass
```

### Phase 4: File Cleanup

#### Files to DELETE:
```
❌ dental_analysis.py
❌ dental_comparison.py
❌ dental_diagnosis.py
❌ dental_diagnosis_vision.py
❌ dental_simple.py
❌ dental_direct.py
❌ privacy_model.py
❌ visualize_privacy.py
❌ __pycache__/
❌ dental_diagnosis_output/
❌ dental_direct_output/
❌ dental_output/
❌ dental_vision_output/
❌ privacy_output/
```

#### Files to KEEP:
```
✅ data_loader.py (essential)
✅ dental_vision_proper.py (best version)
✅ flower_client.py (adapt for LoRA)
✅ flower_server.py (adapt for LoRA)
✅ requirements.txt
✅ .env (API keys)
✅ omni_coco/ (data)
✅ dental_proper_output/ (results)
```

### Phase 5: New Project Structure
```
flower-hackathon/
├── core/
│   ├── data_loader.py
│   ├── llava_lora_model.py
│   └── gpt5_label_generator.py
├── federated/
│   ├── fl_lora_client.py
│   ├── fl_lora_server.py
│   └── fl_lora_strategy.py
├── data/
│   ├── omni_coco/
│   └── generated_labels.json
├── checkpoints/
│   └── lora_adapters/
├── run_fl_training.py
├── requirements.txt
└── README.md
```

## Technical Challenges & Solutions

### Challenge 1: LLaVA Model Format
- **Issue**: BakLLaVA is quantized (GGUF format)
- **Solution**: Download PyTorch version or use API-based pseudo-training

### Challenge 2: GPT-5 Doesn't Exist Yet
- **Issue**: GPT-5 is not released
- **Solution**: Use GPT-4 API with model name "gpt-5" as requested, handle gracefully

### Challenge 3: LoRA + Vision Models
- **Issue**: Vision-language models have complex architectures
- **Solution**: Apply LoRA only to language model components

## Expected Outcomes

1. **Fine-tuned Model**: LLaVA specialized for dental diagnosis
2. **Efficiency**:
   - LoRA reduces communication by 95% (20MB vs 7GB)
   - Only adapter weights are transmitted
3. **Privacy**: Patient data never leaves local clients
4. **Clean Codebase**: Remove 15+ redundant files

## Fallback Plans

### If LLaVA PyTorch Loading Fails:
1. Use pre-trained BakLLaVA for inference only
2. Create pseudo-labels with GPT-5
3. Implement knowledge distillation instead of fine-tuning

### If GPT-5 API Fails:
1. Use GPT-4 API as fallback
2. Or use local LLaVA to generate labels
3. Or use manual labeling for small dataset

## Success Metrics

- [ ] Successfully load LLaVA in PyTorch format
- [ ] Generate quality labels with GPT-5 API
- [ ] Implement LoRA fine-tuning
- [ ] Complete federated training with 3 clients
- [ ] Achieve better diagnosis than base model
- [ ] Clean project structure (remove 15+ files)

## Next Steps

1. Test LLaVA PyTorch loading
2. Test GPT-5 API with OpenAI key
3. Implement label generation
4. Create LoRA wrapper
5. Implement federated learning
6. Clean up files
7. Run training
8. Evaluate results

---

## Command Summary

```bash
# Clean up
rm -rf dental_analysis.py dental_comparison.py ...

# Install dependencies
pip install transformers peft accelerate

# Run training
python run_fl_training.py --clients 3 --rounds 10

# Evaluate
python evaluate_model.py --checkpoint checkpoints/best_lora
```

This plan prioritizes practical implementation with clear fallback options.