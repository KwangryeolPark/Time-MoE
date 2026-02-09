# Time-MoE 학습 가이드

이 문서는 Time-MoE 프로젝트에서 pretrained checkpoint를 불러오고, self-supervised learning을 수행하며, fine-tuning하는 방법을 설명합니다.

## 목차
1. [Pretrained Checkpoint 불러오기](#1-pretrained-checkpoint-불러오기)
2. [Self-Supervised Learning (계속 학습)](#2-self-supervised-learning-계속-학습)
3. [Fine-tuning](#3-fine-tuning)
4. [모델 저장 및 체크포인트 관리](#4-모델-저장-및-체크포인트-관리)

---

## 1. Pretrained Checkpoint 불러오기

### 1.1 기본 사용법 (추론용)

Hugging Face에서 제공하는 pretrained 모델을 불러오는 방법:

```python
import torch
from transformers import AutoModelForCausalLM

# 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',  # 또는 'Maple728/TimeMoE-200M'
    device_map="cpu",  # CPU 사용 시 "cpu", GPU 사용 시 "cuda"
    trust_remote_code=True,
)

# Flash Attention이 설치되어 있는 경우
# model = AutoModelForCausalLM.from_pretrained(
#     'Maple728/TimeMoE-50M',
#     device_map="auto",
#     attn_implementation='flash_attention_2',
#     trust_remote_code=True
# )
```

### 1.2 학습을 위한 모델 불러오기

`time_moe.runner.TimeMoeRunner` 클래스를 사용하여 학습용으로 모델을 불러올 수 있습니다:

```python
from time_moe.runner import TimeMoeRunner

runner = TimeMoeRunner(
    model_path='Maple728/TimeMoE-50M',  # pretrained 모델 경로
    output_path='logs/my_experiment',    # 학습 결과 저장 경로
    seed=9899                            # 재현성을 위한 시드
)
```

`TimeMoeRunner.load_model()` 메서드는 다음과 같은 방식으로 작동합니다:

```python
# runner.py의 load_model 메서드 참고
model = runner.load_model(
    model_path='Maple728/TimeMoE-50M',
    from_scratch=False,  # False: pretrained 가중치 사용
    attn_implementation='auto'  # 'auto', 'eager', 'flash_attention_2'
)
```

### 1.3 로컬 체크포인트 불러오기

자신이 학습한 체크포인트를 불러오려면:

```python
# 로컬 경로 지정
model = AutoModelForCausalLM.from_pretrained(
    './logs/my_experiment/checkpoint-1000',  # 로컬 체크포인트 경로
    device_map="cpu",
    trust_remote_code=True,
)
```

---

## 2. Self-Supervised Learning (계속 학습)

Pretrained 모델을 기반으로 추가 데이터로 학습을 계속하는 방법입니다.

### 2.1 기본 학습 명령어

**단일 GPU 또는 CPU:**
```bash
python main.py -d <data_path> -m Maple728/TimeMoE-50M
```

**단일 노드, 다중 GPU:**
```bash
python torch_dist_run.py main.py -d <data_path> -m Maple728/TimeMoE-50M
```

### 2.2 주요 파라미터 설정

```bash
python main.py \
  -d <data_path> \                          # 학습 데이터 경로
  -m Maple728/TimeMoE-50M \                 # pretrained 모델 경로
  -o logs/continue_training \               # 출력 경로
  --max_length 1024 \                       # 최대 시퀀스 길이
  --stride 512 \                            # 슬라이딩 윈도우 스텝 (작은 데이터셋은 --stride 1 권장)
  --global_batch_size 64 \                  # 글로벌 배치 사이즈
  --micro_batch_size 16 \                   # 디바이스당 마이크로 배치 사이즈
  --num_train_epochs 3.0 \                  # 학습 에포크 수
  --learning_rate 1e-4 \                    # 학습률
  --min_learning_rate 5e-5 \                # 최소 학습률 (cosine schedule 사용 시)
  --lr_scheduler_type cosine \              # 학습률 스케줄러 (constant, linear, cosine, constant_with_warmup)
  --warmup_ratio 0.1 \                      # 워밍업 비율
  --weight_decay 0.1 \                      # 가중치 감쇠
  --precision bf16 \                        # 정밀도 (fp32, fp16, bf16)
  --normalization_method zero \             # 정규화 방법 (none, zero, max)
  --save_strategy steps \                   # 저장 전략 (steps, epoch, no)
  --save_steps 1000 \                       # 저장 주기
  --save_total_limit 3 \                    # 최대 체크포인트 개수
  --logging_steps 10                        # 로깅 주기
```

### 2.3 데이터 준비

학습 데이터는 JSONL 형식으로 준비해야 합니다:

```jsonl
{"sequence": [1.0, 2.0, 3.0, 4.0, ...]}
{"sequence": [11.0, 22.0, 33.0, 44.0, ...]}
```

JSON 또는 pickle 형식도 지원됩니다.

### 2.4 학습 프로세스 이해

`main.py`를 실행하면 다음과 같은 과정이 진행됩니다:

1. **모델 로딩**: `from_scratch=False` (기본값)로 설정되어 있어 pretrained 가중치를 불러옵니다
   ```python
   # runner.py의 train_model 메서드에서
   model = self.load_model(
       model_path=args.model_path,
       from_scratch=from_scratch,  # False인 경우 pretrained 사용
       torch_dtype=torch_dtype,
       attn_implementation=train_config.get('attn_implementation', 'eager'),
   )
   ```

2. **데이터셋 로딩**: 지정된 경로의 데이터를 로드하고 윈도우 데이터셋으로 변환
   ```python
   train_ds = self.get_train_dataset(
       data_path=args.data_path,
       max_length=args.max_length,
       stride=args.stride,
       normalization_method=args.normalization_method,
   )
   ```

3. **학습**: Hugging Face Trainer를 사용하여 학습 진행
   ```python
   trainer = TimeMoeTrainer(
       model=model,
       args=training_args,
       train_dataset=train_ds,
   )
   trainer.train()
   ```

4. **모델 저장**: 학습 완료 후 최종 모델 저장
   ```python
   trainer.save_model(self.output_path)
   ```

---

## 3. Fine-tuning

Fine-tuning은 self-supervised learning과 동일한 방식으로 수행됩니다. 차이점은 데이터와 하이퍼파라미터 설정입니다.

### 3.1 Fine-tuning 예시

**특정 도메인 데이터로 fine-tuning:**

```bash
python torch_dist_run.py main.py \
  -d ./my_domain_data \                    # 도메인 특화 데이터
  -m Maple728/TimeMoE-50M \                # pretrained 모델
  -o logs/finetuned_model \
  --max_length 512 \                       # fine-tuning 시 짧은 길이 사용 가능
  --stride 1 \                             # 작은 데이터셋의 경우 stride=1 권장
  --global_batch_size 32 \
  --num_train_epochs 5.0 \
  --learning_rate 5e-5 \                   # fine-tuning 시 더 작은 학습률 사용
  --min_learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --precision bf16
```

### 3.2 Fine-tuning vs Self-Supervised Learning

| 항목 | Self-Supervised Learning | Fine-tuning |
|------|-------------------------|-------------|
| 목적 | 일반적인 시계열 패턴 학습 | 특정 도메인/태스크 최적화 |
| 데이터 | 대규모, 다양한 도메인 | 도메인 특화, 상대적으로 소규모 |
| 학습률 | 1e-4 ~ 1e-3 | 1e-5 ~ 1e-4 (더 작게) |
| 에포크 | 1-3 | 3-10 (더 많이) |
| Stride | 512, 1024 (데이터 효율) | 1 (모든 샘플 활용) |

### 3.3 로컬 체크포인트에서 Fine-tuning

이미 학습한 로컬 체크포인트에서 시작하려면:

```bash
python torch_dist_run.py main.py \
  -d ./new_data \
  -m ./logs/my_experiment/checkpoint-5000 \  # 로컬 체크포인트 경로
  -o logs/finetuned_from_checkpoint \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0
```

---

## 4. 모델 저장 및 체크포인트 관리

### 4.1 체크포인트 저장 옵션

**저장 전략 설정:**

```bash
# Steps 기반 저장
python main.py -d <data> -m <model> \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 5  # 최근 5개 체크포인트만 유지

# Epoch 기반 저장
python main.py -d <data> -m <model> \
  --save_strategy epoch \
  --save_total_limit 3  # 최근 3개 에포크만 유지

# 저장하지 않음 (최종 모델만 저장)
python main.py -d <data> -m <model> \
  --save_strategy no
```

**모델만 저장 (optimizer state 제외):**

```bash
python main.py -d <data> -m <model> \
  --save_only_model
```

### 4.2 체크포인트 구조

저장된 체크포인트는 다음과 같은 구조를 가집니다:

```
logs/my_experiment/
├── checkpoint-1000/
│   ├── config.json              # 모델 설정
│   ├── model.safetensors        # 모델 가중치
│   ├── optimizer.pt             # Optimizer 상태 (save_only_model=False인 경우)
│   ├── scheduler.pt             # Scheduler 상태
│   ├── trainer_state.json       # 학습 상태
│   └── training_args.bin        # 학습 인자
├── checkpoint-2000/
│   └── ...
└── tb_logs/                     # TensorBoard 로그
```

### 4.3 학습 재개

중단된 학습을 재개하려면 Hugging Face Trainer의 기본 기능을 사용:

```bash
# 마지막 체크포인트에서 자동 재개
python main.py -d <data> -m <model> \
  -o logs/my_experiment  # 동일한 output_path 사용

# Trainer가 자동으로 logs/my_experiment에서 최신 체크포인트를 찾아 재개
```

수동으로 특정 체크포인트에서 재개:

```bash
python main.py -d <data> \
  -m logs/my_experiment/checkpoint-5000 \  # 특정 체크포인트 지정
  -o logs/my_experiment_continued
```

---

## 5. 처음부터 학습 (Train from Scratch)

Pretrained 가중치를 사용하지 않고 처음부터 학습하려면:

```bash
python torch_dist_run.py main.py \
  -d <data_path> \
  -m Maple728/TimeMoE-50M \  # 모델 아키텍처만 사용 (config만 로드)
  --from_scratch \            # 이 플래그 추가!
  -o logs/from_scratch \
  --num_train_epochs 10.0 \
  --learning_rate 1e-3
```

`--from_scratch` 플래그를 사용하면:
- 모델 설정(config)만 로드
- 가중치는 랜덤 초기화
- 완전히 새로운 학습 시작

---

## 6. 추론 (Inference)

학습한 모델로 예측하기:

### 6.1 Python 코드

```python
import torch
from transformers import AutoModelForCausalLM

# 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    './logs/my_experiment',  # 또는 'Maple728/TimeMoE-50M'
    device_map="cpu",
    trust_remote_code=True,
)

# 입력 데이터 준비
context_length = 12
seqs = torch.randn(2, context_length)  # [batch_size, context_length]

# 정규화
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# 예측
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]

# 역정규화
predictions = normed_predictions * std + mean
```

### 6.2 벤치마크 평가

```bash
python run_eval.py \
  -m ./logs/my_experiment \      # 학습한 모델 경로
  -d dataset/ETT-small/ETTh1.csv \
  -p 96 \                        # 예측 길이
  -c 512 \                       # 컨텍스트 길이
  -b 32                          # 배치 사이즈
```

---

## 7. 고급 설정

### 7.1 Distributed Training (다중 노드)

```bash
# 각 노드에서 실행
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total_num_gpus>
export RANK=<node_rank>  # 0, 1, 2, ...

python torch_dist_run.py main.py -d <data_path> -m <model_path>
```

### 7.2 DeepSpeed 사용

```bash
# DeepSpeed config 파일 준비 (예: ds_config.json)
python torch_dist_run.py main.py \
  -d <data_path> \
  -m <model_path> \
  --deepspeed ./ds_config.json
```

### 7.3 Gradient Checkpointing (메모리 절약)

```bash
python main.py -d <data> -m <model> \
  --gradient_checkpointing
```

---

## 8. 코드 구조 이해

### 8.1 주요 파일

- **`main.py`**: CLI 진입점, 인자 파싱
- **`time_moe/runner.py`**: 학습 로직 오케스트레이션
  - `TimeMoeRunner.load_model()`: 모델 로딩
  - `TimeMoeRunner.train_model()`: 학습 실행
- **`time_moe/models/modeling_time_moe.py`**: 모델 정의
  - `TimeMoeForPrediction`: 예측용 모델 클래스
- **`time_moe/trainer/hf_trainer.py`**: 커스텀 Trainer
  - `TimeMoeTrainer`: Hugging Face Trainer 확장
- **`run_eval.py`**: 평가 스크립트

### 8.2 학습 플로우

```
main.py
  ↓
TimeMoeRunner.train_model()
  ↓
├─ load_model()
│  └─ TimeMoeForPrediction.from_pretrained() (from_scratch=False)
│     또는
│  └─ TimeMoeForPrediction(config) (from_scratch=True)
  ↓
├─ get_train_dataset()
│  └─ TimeMoEDataset → TimeMoEWindowDataset
  ↓
├─ TimeMoeTrainer.train()
│  └─ Hugging Face Trainer 학습 루프
  ↓
└─ trainer.save_model()
```

---

## 9. 자주 묻는 질문 (FAQ)

**Q1: Pretrained 모델과 처음부터 학습의 차이는?**
- `--from_scratch` 없음: Pretrained 가중치 로드 → Fine-tuning/계속 학습
- `--from_scratch` 있음: 랜덤 초기화 → 완전히 새로운 학습

**Q2: Self-supervised learning과 Fine-tuning의 차이는?**
- 기술적으로 동일한 과정 (모두 `main.py` 사용)
- 차이는 데이터, 하이퍼파라미터, 목적에 있음
- Self-supervised: 대규모 비라벨 데이터로 일반적 표현 학습
- Fine-tuning: 특정 태스크/도메인에 맞게 조정

**Q3: 작은 데이터셋으로 학습할 때 주의사항은?**
- `--stride 1` 사용하여 모든 샘플 활용
- 더 많은 에포크 학습
- 작은 학습률 사용
- 정규화 강화 (weight_decay 증가)

**Q4: 최대 시퀀스 길이는?**
- Time-MoE는 `max_position_embeddings=4096`으로 학습됨
- `context_length + prediction_length ≤ 4096` 권장
- 더 긴 시퀀스가 필요하면 fine-tuning으로 확장 가능

**Q5: 모델 크기는?**
- TimeMoE-50M: 약 50M 파라미터
- TimeMoE-200M: 약 200M 파라미터

---

## 10. 예시 시나리오

### 시나리오 1: Pretrained 모델로 시작하여 자신의 데이터로 학습

```bash
# 1단계: Pretrained 모델로 자신의 데이터 학습
python torch_dist_run.py main.py \
  -d ./my_data \
  -m Maple728/TimeMoE-50M \
  -o logs/my_trained_model \
  --num_train_epochs 3.0 \
  --learning_rate 1e-4 \
  --save_strategy epoch

# 2단계: 학습한 모델로 예측
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

### 시나리오 2: 중간 체크포인트에서 Fine-tuning 계속

```bash
# 1단계: 초기 학습
python main.py -d ./data1 -m Maple728/TimeMoE-50M \
  -o logs/stage1 --num_train_epochs 5 --save_strategy epoch

# 2단계: 특정 체크포인트에서 다른 데이터로 fine-tuning
python main.py -d ./data2 \
  -m ./logs/stage1/checkpoint-3 \  # 3번째 에포크 체크포인트 사용
  -o logs/stage2 \
  --num_train_epochs 5 \
  --learning_rate 5e-5  # 더 작은 학습률
```

### 시나리오 3: 처음부터 학습 후 평가

```bash
# 학습
python torch_dist_run.py main.py \
  -d ./large_dataset \
  -m Maple728/TimeMoE-50M \
  --from_scratch \
  -o logs/scratch_model \
  --num_train_epochs 10.0 \
  --train_steps 100000 \
  --save_strategy steps \
  --save_steps 10000

# 평가
python run_eval.py \
  -m ./logs/scratch_model \
  -d dataset/ETT-small/ETTh1.csv \
  -p 96
```

---

## 참고 자료

- [Time-MoE Paper](https://arxiv.org/abs/2409.16040)
- [Hugging Face 모델](https://huggingface.co/Maple728/TimeMoE-50M)
- [Time-300B 데이터셋](https://huggingface.co/datasets/Maple728/Time-300B)
- [README.md](README.md)
