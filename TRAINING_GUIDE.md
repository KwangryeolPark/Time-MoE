# Time-MoE Training Guide

This guide explains how to load pretrained checkpoints, perform self-supervised learning, and fine-tune Time-MoE models.

## Table of Contents
1. [Loading Pretrained Checkpoints](#1-loading-pretrained-checkpoints)
2. [Self-Supervised Learning (Continued Pretraining)](#2-self-supervised-learning-continued-pretraining)
3. [Fine-tuning](#3-fine-tuning)
   - [Parameter-Efficient Fine-tuning with LoRA](#33-parameter-efficient-fine-tuning-with-lora)
4. [Checkpoint Management](#4-checkpoint-management)

---

## 1. Loading Pretrained Checkpoints

### 1.1 Basic Usage (Inference)

Load a pretrained model from Hugging Face:

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',  # or 'Maple728/TimeMoE-200M'
    device_map="cpu",  # "cpu" for CPU, "cuda" for GPU
    trust_remote_code=True,
)

# With Flash Attention (if installed)
# model = AutoModelForCausalLM.from_pretrained(
#     'Maple728/TimeMoE-50M',
#     device_map="auto",
#     attn_implementation='flash_attention_2',
#     trust_remote_code=True
# )
```

### 1.2 Loading for Training

Use the `TimeMoeRunner` class to load models for training:

```python
from time_moe.runner import TimeMoeRunner

runner = TimeMoeRunner(
    model_path='Maple728/TimeMoE-50M',  # pretrained model path
    output_path='logs/my_experiment',    # output directory
    seed=9899                            # random seed for reproducibility
)
```

The `load_model()` method works as follows:

```python
# Reference: runner.py load_model method
model = runner.load_model(
    model_path='Maple728/TimeMoE-50M',
    from_scratch=False,  # False: use pretrained weights
    attn_implementation='auto'  # 'auto', 'eager', 'flash_attention_2'
)
```

### 1.3 Loading Local Checkpoints

To load a checkpoint you trained locally:

```python
model = AutoModelForCausalLM.from_pretrained(
    './logs/my_experiment/checkpoint-1000',  # local checkpoint path
    device_map="cpu",
    trust_remote_code=True,
)
```

---

## 2. Self-Supervised Learning (Continued Pretraining)

Continue training a pretrained model on additional data.

### 2.1 Basic Training Commands

**Single GPU or CPU:**
```bash
python main.py -d <data_path> -m Maple728/TimeMoE-50M
```

**Single Node, Multiple GPUs:**
```bash
python torch_dist_run.py main.py -d <data_path> -m Maple728/TimeMoE-50M
```

### 2.2 Key Parameters

```bash
python main.py \
  -d <data_path> \                          # training data path
  -m Maple728/TimeMoE-50M \                 # pretrained model path
  -o logs/continue_training \               # output directory
  --max_length 1024 \                       # maximum sequence length
  --stride 512 \                            # sliding window step (use --stride 1 for small datasets)
  --global_batch_size 64 \                  # global batch size
  --micro_batch_size 16 \                   # micro batch size per device
  --num_train_epochs 3.0 \                  # number of training epochs
  --learning_rate 1e-4 \                    # learning rate
  --min_learning_rate 5e-5 \                # minimum learning rate (for cosine schedule)
  --lr_scheduler_type cosine \              # learning rate scheduler
  --warmup_ratio 0.1 \                      # warmup ratio
  --weight_decay 0.1 \                      # weight decay
  --precision bf16 \                        # precision (fp32, fp16, bf16)
  --normalization_method zero \             # normalization method (none, zero, max)
  --save_strategy steps \                   # save strategy (steps, epoch, no)
  --save_steps 1000 \                       # save every N steps
  --save_total_limit 3 \                    # maximum number of checkpoints to keep
  --logging_steps 10                        # logging frequency
```

### 2.3 Data Preparation

Training data should be in JSONL format:

```jsonl
{"sequence": [1.0, 2.0, 3.0, 4.0, ...]}
{"sequence": [11.0, 22.0, 33.0, 44.0, ...]}
```

JSON and pickle formats are also supported.

### 2.4 Understanding the Training Process

When you run `main.py`, the following happens:

1. **Model Loading**: With `from_scratch=False` (default), pretrained weights are loaded
   ```python
   # From runner.py train_model method
   model = self.load_model(
       model_path=args.model_path,
       from_scratch=from_scratch,  # False loads pretrained weights
       torch_dtype=torch_dtype,
       attn_implementation=train_config.get('attn_implementation', 'eager'),
   )
   ```

2. **Dataset Loading**: Data is loaded and converted to windowed dataset
   ```python
   train_ds = self.get_train_dataset(
       data_path=args.data_path,
       max_length=args.max_length,
       stride=args.stride,
       normalization_method=args.normalization_method,
   )
   ```

3. **Training**: Uses Hugging Face Trainer
   ```python
   trainer = TimeMoeTrainer(
       model=model,
       args=training_args,
       train_dataset=train_ds,
   )
   trainer.train()
   ```

4. **Model Saving**: Saves final model after training
   ```python
   trainer.save_model(self.output_path)
   ```

---

## 3. Fine-tuning

Fine-tuning uses the same process as self-supervised learning, but with different data and hyperparameters.

### 3.1 Fine-tuning Example

**Fine-tune on domain-specific data:**

```bash
python torch_dist_run.py main.py \
  -d ./my_domain_data \                    # domain-specific data
  -m Maple728/TimeMoE-50M \                # pretrained model
  -o logs/finetuned_model \
  --max_length 512 \                       # can use shorter length for fine-tuning
  --stride 1 \                             # recommended stride=1 for small datasets
  --global_batch_size 32 \
  --num_train_epochs 5.0 \
  --learning_rate 5e-5 \                   # lower learning rate for fine-tuning
  --min_learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --precision bf16
```

### 3.2 Fine-tuning vs Self-Supervised Learning

| Aspect | Self-Supervised Learning | Fine-tuning |
|--------|-------------------------|-------------|
| Purpose | Learn general time series patterns | Optimize for specific domain/task |
| Data | Large-scale, diverse domains | Domain-specific, relatively smaller |
| Learning Rate | 1e-4 ~ 1e-3 | 1e-5 ~ 1e-4 (lower) |
| Epochs | 1-3 | 3-10 (more) |
| Stride | 512, 1024 (data efficiency) | 1 (use all samples) |

### 3.3 Parameter-Efficient Fine-tuning with LoRA

**LoRA (Low-Rank Adaptation)** allows fine-tuning Time-MoE with minimal trainable parameters, significantly reducing memory usage and training time while maintaining performance.

#### 3.3.1 Basic LoRA Fine-tuning

**Enable LoRA with default settings:**

```bash
python main.py -d <data_path> -m Maple728/TimeMoE-50M --use_lora
```

This will:
- Apply LoRA to attention layers (q_proj, k_proj, v_proj, o_proj)
- Use rank=8, alpha=16, dropout=0.05 (default values)
- Train only ~0.5-1% of the total parameters

#### 3.3.2 Custom LoRA Configuration

**Adjust LoRA hyperparameters:**

```bash
python main.py -d <data_path> -m Maple728/TimeMoE-50M \
  --use_lora \
  --lora_r 16 \              # LoRA rank (higher = more capacity)
  --lora_alpha 32 \           # LoRA scaling factor
  --lora_dropout 0.05         # Dropout probability
```

**Recommended configurations:**

| Use Case | Rank (r) | Alpha | Target Modules | Description |
|----------|----------|-------|----------------|-------------|
| Quick experiments | 4 | 8 | q_proj,v_proj | Minimal parameters, fast training |
| Balanced (default) | 8 | 16 | q_proj,k_proj,v_proj,o_proj | Good performance/efficiency trade-off |
| High capacity | 16 | 32 | q_proj,k_proj,v_proj,o_proj | Best performance, more parameters |

#### 3.3.3 Custom Target Modules

**Apply LoRA to specific layers:**

```bash
# Attention layers only (default)
python main.py -d <data_path> --use_lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj

# Attention + FFN layers
python main.py -d <data_path> --use_lora \
  --lora_target_modules q_proj,v_proj,gate_proj,down_proj

# Full coverage (attention + all FFN layers)
python main.py -d <data_path> --use_lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

**Available target modules in Time-MoE:**
- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **FFN/MoE Experts**: `gate_proj`, `up_proj`, `down_proj`
- **MoE Router**: `gate`
- **Shared Expert**: `shared_expert`

#### 3.3.4 Complete LoRA Example

```bash
python torch_dist_run.py main.py \
  -d ./my_domain_data \
  -m Maple728/TimeMoE-50M \
  -o logs/lora_finetuned \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 512 \
  --stride 1 \
  --global_batch_size 32 \
  --num_train_epochs 5.0 \
  --learning_rate 1e-4 \
  --min_learning_rate 5e-5 \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --precision bf16
```

#### 3.3.5 LoRA Benefits

**Memory Efficiency:**
- Training with LoRA rank=8: ~0.5-1% trainable parameters
- Significantly reduced GPU memory usage
- Faster training and gradient computation

**Performance:**
- Comparable to full fine-tuning for most tasks
- Better generalization on small datasets
- Prevents catastrophic forgetting

**Example: See [`examples/lora_finetuning_example.py`](examples/lora_finetuning_example.py) for detailed usage patterns.**

### 3.4 Fine-tuning from Local Checkpoint

To start from a local checkpoint:

```bash
python torch_dist_run.py main.py \
  -d ./new_data \
  -m ./logs/my_experiment/checkpoint-5000 \  # local checkpoint path
  -o logs/finetuned_from_checkpoint \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0
```

---

## 4. Checkpoint Management

### 4.1 Checkpoint Saving Options

**Configure save strategy:**

```bash
# Save based on steps
python main.py -d <data> -m <model> \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 5  # keep only last 5 checkpoints

# Save based on epochs
python main.py -d <data> -m <model> \
  --save_strategy epoch \
  --save_total_limit 3  # keep only last 3 epochs

# Don't save checkpoints (only final model)
python main.py -d <data> -m <model> \
  --save_strategy no
```

**Save only model weights (exclude optimizer state):**

```bash
python main.py -d <data> -m <model> \
  --save_only_model
```

### 4.2 Checkpoint Structure

Saved checkpoints have the following structure:

```
logs/my_experiment/
├── checkpoint-1000/
│   ├── config.json              # model configuration
│   ├── model.safetensors        # model weights
│   ├── optimizer.pt             # optimizer state (if save_only_model=False)
│   ├── scheduler.pt             # scheduler state
│   ├── trainer_state.json       # training state
│   └── training_args.bin        # training arguments
├── checkpoint-2000/
│   └── ...
└── tb_logs/                     # TensorBoard logs
```

### 4.3 Resuming Training

To resume interrupted training:

```bash
# Automatically resume from last checkpoint
python main.py -d <data> -m <model> \
  -o logs/my_experiment  # use same output_path

# Trainer automatically finds and resumes from latest checkpoint
```

Manually resume from specific checkpoint:

```bash
python main.py -d <data> \
  -m logs/my_experiment/checkpoint-5000 \  # specify checkpoint
  -o logs/my_experiment_continued
```

---

## 5. Training from Scratch

To train without pretrained weights:

```bash
python torch_dist_run.py main.py \
  -d <data_path> \
  -m Maple728/TimeMoE-50M \  # only architecture (config) is used
  --from_scratch \            # add this flag!
  -o logs/from_scratch \
  --num_train_epochs 10.0 \
  --learning_rate 1e-3
```

With the `--from_scratch` flag:
- Only model configuration is loaded
- Weights are randomly initialized
- Training starts completely fresh

---

## 6. Inference

Making predictions with trained models:

### 6.1 Python Code

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    './logs/my_experiment',  # or 'Maple728/TimeMoE-50M'
    device_map="cpu",
    trust_remote_code=True,
)

# Prepare input
context_length = 12
seqs = torch.randn(2, context_length)  # [batch_size, context_length]

# Normalize
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# Predict
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]

# Denormalize
predictions = normed_predictions * std + mean
```

### 6.2 Benchmark Evaluation

```bash
python run_eval.py \
  -m ./logs/my_experiment \      # trained model path
  -d dataset/ETT-small/ETTh1.csv \
  -p 96 \                        # prediction length
  -c 512 \                       # context length
  -b 32                          # batch size
```

---

## 7. Advanced Settings

### 7.1 Distributed Training (Multi-Node)

```bash
# Run on each node
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_num_gpus>
export RANK=<node_rank>  # 0, 1, 2, ...

python torch_dist_run.py main.py -d <data_path> -m <model_path>
```

### 7.2 Using DeepSpeed

```bash
# Prepare DeepSpeed config file (e.g., ds_config.json)
python torch_dist_run.py main.py \
  -d <data_path> \
  -m <model_path> \
  --deepspeed ./ds_config.json
```

### 7.3 Gradient Checkpointing (Save Memory)

```bash
python main.py -d <data> -m <model> \
  --gradient_checkpointing
```

---

## 8. Code Structure

### 8.1 Key Files

- **`main.py`**: CLI entry point, argument parsing
- **`time_moe/runner.py`**: Training orchestration
  - `TimeMoeRunner.load_model()`: Model loading
  - `TimeMoeRunner.train_model()`: Training execution
- **`time_moe/models/modeling_time_moe.py`**: Model definition
  - `TimeMoeForPrediction`: Prediction model class
- **`time_moe/trainer/hf_trainer.py`**: Custom trainer
  - `TimeMoeTrainer`: Extends Hugging Face Trainer
- **`run_eval.py`**: Evaluation script

### 8.2 Training Flow

```
main.py
  ↓
TimeMoeRunner.train_model()
  ↓
├─ load_model()
│  └─ TimeMoeForPrediction.from_pretrained() (from_scratch=False)
│     or
│  └─ TimeMoeForPrediction(config) (from_scratch=True)
  ↓
├─ get_train_dataset()
│  └─ TimeMoEDataset → TimeMoEWindowDataset
  ↓
├─ TimeMoeTrainer.train()
│  └─ Hugging Face Trainer training loop
  ↓
└─ trainer.save_model()
```

---

## 9. FAQ

**Q1: What's the difference between pretrained and training from scratch?**
- Without `--from_scratch`: Loads pretrained weights → Fine-tuning/continued training
- With `--from_scratch`: Random initialization → Completely new training

**Q2: What's the difference between self-supervised learning and fine-tuning?**
- Technically the same process (both use `main.py`)
- Difference is in data, hyperparameters, and purpose
- Self-supervised: Large unlabeled data for general representation learning
- Fine-tuning: Adapt to specific task/domain

**Q3: What to consider for small datasets?**
- Use `--stride 1` to utilize all samples
- Train for more epochs
- Use lower learning rate
- Increase regularization (higher weight_decay)

**Q4: What's the maximum sequence length?**
- Time-MoE is trained with `max_position_embeddings=4096`
- Recommended: `context_length + prediction_length ≤ 4096`
- Can extend with fine-tuning if longer sequences needed

**Q5: Model sizes?**
- TimeMoE-50M: ~50M parameters
- TimeMoE-200M: ~200M parameters

---

## 10. Example Scenarios

### Scenario 1: Train on Your Data Starting from Pretrained Model

```bash
# Step 1: Train with your data
python torch_dist_run.py main.py \
  -d ./my_data \
  -m Maple728/TimeMoE-50M \
  -o logs/my_trained_model \
  --num_train_epochs 3.0 \
  --learning_rate 1e-4 \
  --save_strategy epoch

# Step 2: Make predictions
python -c "
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    './logs/my_trained_model',
    device_map='cpu',
    trust_remote_code=True
)

seqs = torch.randn(1, 96)
mean, std = seqs.mean(keepdim=True), seqs.std(keepdim=True)
normed_seqs = (seqs - mean) / std
output = model.generate(normed_seqs, max_new_tokens=24)
predictions = output[:, -24:] * std + mean
print('Predictions:', predictions)
"
```

### Scenario 2: Continue Fine-tuning from Intermediate Checkpoint

```bash
# Step 1: Initial training
python main.py -d ./data1 -m Maple728/TimeMoE-50M \
  -o logs/stage1 --num_train_epochs 5 --save_strategy epoch

# Step 2: Fine-tune from specific checkpoint with different data
python main.py -d ./data2 \
  -m ./logs/stage1/checkpoint-3 \  # use 3rd epoch checkpoint
  -o logs/stage2 \
  --num_train_epochs 5 \
  --learning_rate 5e-5  # lower learning rate
```

### Scenario 3: Train from Scratch and Evaluate

```bash
# Training
python torch_dist_run.py main.py \
  -d ./large_dataset \
  -m Maple728/TimeMoE-50M \
  --from_scratch \
  -o logs/scratch_model \
  --num_train_epochs 10.0 \
  --train_steps 100000 \
  --save_strategy steps \
  --save_steps 10000

# Evaluation
python run_eval.py \
  -m ./logs/scratch_model \
  -d dataset/ETT-small/ETTh1.csv \
  -p 96
```

---

## References

- [Time-MoE Paper](https://arxiv.org/abs/2409.16040)
- [Hugging Face Model](https://huggingface.co/Maple728/TimeMoE-50M)
- [Time-300B Dataset](https://huggingface.co/datasets/Maple728/Time-300B)
- [README.md](README.md)
