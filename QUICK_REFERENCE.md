# Time-MoE Quick Reference

## Quick Commands

### Load Pretrained Model
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="cpu", trust_remote_code=True)
```

### Continue Training (Self-Supervised)
```bash
# Single GPU
python main.py -d <data_path> -m Maple728/TimeMoE-50M

# Multi-GPU
python torch_dist_run.py main.py -d <data_path> -m Maple728/TimeMoE-50M
```

### Fine-tune
```bash
python torch_dist_run.py main.py \
  -d <data_path> \
  -m Maple728/TimeMoE-50M \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --stride 1
```

### Train from Scratch
```bash
python torch_dist_run.py main.py \
  -d <data_path> \
  -m Maple728/TimeMoE-50M \
  --from_scratch
```

### Evaluate
```bash
python run_eval.py -m <model_path> -d dataset/ETT-small/ETTh1.csv -p 96
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-d, --data_path` | Required | Path to training data |
| `-m, --model_path` | `Maple728/TimeMoE-50M` | Pretrained model or checkpoint path |
| `-o, --output_path` | `logs/time_moe` | Output directory |
| `--from_scratch` | False | Train from scratch (random init) |
| `--max_length` | 1024 | Maximum sequence length |
| `--stride` | max_length | Sliding window step |
| `--global_batch_size` | 64 | Global batch size |
| `--micro_batch_size` | 16 | Batch size per device |
| `--num_train_epochs` | 1.0 | Number of epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--min_learning_rate` | 5e-5 | Minimum LR (cosine schedule) |
| `--lr_scheduler_type` | cosine | LR scheduler type |
| `--warmup_ratio` | 0.0 | Warmup ratio |
| `--weight_decay` | 0.1 | Weight decay |
| `--precision` | fp32 | Precision (fp32, fp16, bf16) |
| `--save_strategy` | no | Save strategy (steps, epoch, no) |
| `--save_steps` | None | Save every N steps |
| `--save_total_limit` | None | Max checkpoints to keep |

## Common Use Cases

### 1. Quick Start with Pretrained Model
```bash
python main.py -d ./my_data -m Maple728/TimeMoE-50M --num_train_epochs 1
```

### 2. Production Training
```bash
python torch_dist_run.py main.py \
  -d ./my_data \
  -m Maple728/TimeMoE-50M \
  --num_train_epochs 5 \
  --global_batch_size 128 \
  --precision bf16 \
  --save_strategy epoch \
  --save_total_limit 3
```

### 3. Small Dataset Fine-tuning
```bash
python main.py \
  -d ./small_data \
  -m Maple728/TimeMoE-50M \
  --stride 1 \
  --num_train_epochs 10 \
  --learning_rate 3e-5 \
  --weight_decay 0.2
```

### 4. Resume from Checkpoint
```bash
python main.py -d ./data -m ./logs/time_moe/checkpoint-1000
```

## Data Format

### JSONL Format (Recommended)
```jsonl
{"sequence": [1.0, 2.0, 3.0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...]}
```

### JSON Format
```json
[
  {"sequence": [1.0, 2.0, 3.0, ...]},
  {"sequence": [11.0, 22.0, 33.0, ...]}
]
```

### Pickle Format
Python list or numpy array of sequences.

## Inference Code

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",
    trust_remote_code=True,
)

# Prepare input
context_length = 12
seqs = torch.randn(2, context_length)

# Normalize
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# Predict
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
predictions = output[:, -prediction_length:] * std + mean
```

## Extract Latent Representations

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",
    trust_remote_code=True,
)

# Prepare and normalize input
seqs = torch.randn(2, 12)
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# Extract latent representation
outputs = model.encode(normed_seqs)
latent = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

# Optional: mean pooling for fixed-size representation
pooled = latent.mean(dim=1)  # [batch, hidden_size]
```

See [examples/extract_latent_representation.py](examples/extract_latent_representation.py) for more examples.

## Troubleshooting

### Out of Memory
- Reduce `--micro_batch_size`
- Enable `--gradient_checkpointing`
- Use `--precision bf16` or `--precision fp16`
- Use DeepSpeed with `--deepspeed <config.json>`

### Slow Training
- Install flash-attention: `pip install flash-attn==2.6.3`
- Use `--attn_implementation flash_attention_2`
- Increase `--micro_batch_size`
- Use multiple GPUs: `python torch_dist_run.py ...`

### Small Dataset
- Set `--stride 1` to use all samples
- Increase `--num_train_epochs`
- Lower `--learning_rate`
- Increase `--weight_decay`

## Model Variants

| Model | Parameters | Hugging Face Path |
|-------|-----------|-------------------|
| Base | 50M | `Maple728/TimeMoE-50M` |
| Large | 200M | `Maple728/TimeMoE-200M` |

## Supported Sequence Lengths

- Maximum: 4096 (as trained)
- Recommended: `context_length + prediction_length ≤ 4096`
- Can be extended via fine-tuning

## For More Details

- [Full Training Guide (English)](TRAINING_GUIDE.md)
- [전체 학습 가이드 (한국어)](TRAINING_GUIDE_KR.md)
- [README](README.md)
