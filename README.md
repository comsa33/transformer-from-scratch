# Transformer 모델 구현 프로젝트

## 📊 프로젝트 분석 리포트
본 프로젝트의 모든 구현 결과와 논문 재현 검증 내용을 담은 **[최종 분석 리포트](FINAL_REPORT.md)** 를 확인하세요. 리포트에는 30개 이상의 시각화 결과와 함께 각 구성 요소의 상세한 분석이 포함되어 있습니다.

## 개요
이 프로젝트는 Transformer 아키텍처를 처음부터 구현하며 각 구성 요소를 깊이 이해하는 것을 목표로 합니다. Python 3.11과 uv 패키지 매니저를 사용합니다.

## Transformer 아키텍처 상세

### 1. 전체 구조 개요

Transformer는 2017년 ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) 논문에서 처음 소개된 모델로, RNN이나 CNN 없이 오직 Attention 메커니즘만으로 구성된 혁신적인 아키텍처입니다.

```
입력 시퀀스 → Encoder → Context Vectors → Decoder → 출력 시퀀스
```

### 2. 핵심 구성 요소

#### 2.1 Multi-Head Attention

Multi-Head Attention은 Transformer의 핵심 메커니즘입니다. 여러 개의 Attention Head를 병렬로 실행하여 서로 다른 representation subspace에서 정보를 추출합니다.

**구성 요소:**
- **Query (Q)**: 현재 위치에서 어떤 정보를 찾고자 하는지를 나타내는 벡터
- **Key (K)**: 각 위치의 정보가 무엇인지를 나타내는 벡터
- **Value (V)**: 실제로 전달될 정보를 담은 벡터

**수식:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**파라미터:**
- `d_model`: 모델의 차원 (일반적으로 512)
- `num_heads`: Attention Head의 개수 (일반적으로 8)
- `d_k = d_v = d_model / num_heads`: 각 Head의 차원 (일반적으로 64)

#### 2.2 Position-wise Feed-Forward Networks (FFN)

각 위치에 독립적으로 적용되는 완전 연결 신경망입니다.

**구조:**
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**특징:**
- 두 개의 Linear Transformation으로 구성
- 중간에 ReLU 활성화 함수 사용
- Hidden Layer 차원은 일반적으로 `d_model`의 4배 (2048)

#### 2.3 Positional Encoding

Transformer는 순서 정보를 직접적으로 모델링하지 않으므로, 위치 정보를 입력에 추가해야 합니다.

**수식:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

여기서:
- `pos`: 시퀀스 내 위치
- `i`: 차원 인덱스

#### 2.4 Layer Normalization

각 sub-layer 출력에 적용되어 학습을 안정화합니다.

**수식:**
```
LayerNorm(x) = γ * (x - μ) / σ + β
```

여기서:
- `μ`: 평균
- `σ`: 표준편차
- `γ, β`: 학습 가능한 파라미터

### 3. Encoder 구조

Encoder는 N개의 동일한 층으로 구성됩니다 (일반적으로 N=6).

**각 Encoder Layer 구성:**
1. **Multi-Head Self-Attention**
   - 입력 시퀀스의 각 위치가 다른 모든 위치를 참조
   - Residual Connection + Layer Normalization

2. **Position-wise Feed-Forward Network**
   - 각 위치에 독립적으로 적용
   - Residual Connection + Layer Normalization

**Encoder 전체 구조:**
```
Input Embedding → Positional Encoding →
[Encoder Layer 1] → [Encoder Layer 2] → ... → [Encoder Layer N] →
Encoder Output
```

### 4. Decoder 구조

Decoder도 N개의 동일한 층으로 구성됩니다.

**각 Decoder Layer 구성:**
1. **Masked Multi-Head Self-Attention**
   - 미래 위치의 정보를 참조하지 못하도록 마스킹
   - Residual Connection + Layer Normalization

2. **Multi-Head Cross-Attention**
   - Query는 Decoder에서, Key와 Value는 Encoder 출력에서 가져옴
   - Residual Connection + Layer Normalization

3. **Position-wise Feed-Forward Network**
   - Encoder와 동일한 구조
   - Residual Connection + Layer Normalization

**Decoder 전체 구조:**
```
Output Embedding → Positional Encoding →
[Decoder Layer 1] → [Decoder Layer 2] → ... → [Decoder Layer N] →
Linear → Softmax → Output Probabilities
```

### 5. 학습 세부사항

#### 5.1 손실 함수
- **Cross-Entropy Loss**: 다음 토큰 예측을 위한 표준 손실 함수

#### 5.2 최적화
- **Optimizer**: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- **Learning Rate Schedule**: Warmup 기반 스케줄
  ```
  lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
  ```

#### 5.3 정규화 기법
- **Dropout**: 0.1 (각 sub-layer 출력에 적용)
- **Label Smoothing**: ε=0.1

### 6. 구현 모듈 구조

프로젝트의 주요 모듈 구조:

```
test-modeling/
├── transformer/          # Transformer 구현 코드
│   ├── embeddings/      # 임베딩 관련 모듈
│   │   ├── positional.py
│   │   └── token_embedding.py
│   ├── layers/          # 레이어 구현
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── normalization.py
│   │   └── residual.py
│   ├── models/          # 전체 모델 구현
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── transformer.py
│   └── utils/           # 유틸리티
│       └── masking.py
├── tests/               # 테스트 코드
│   ├── test_attention.py
│   ├── test_decoder.py
│   ├── test_encoder.py
│   ├── test_feedforward.py
│   ├── test_layer_normalization.py
│   ├── test_masking.py
│   ├── test_positional_encoding.py
│   ├── test_residual.py
│   ├── test_token_embedding.py
│   └── test_transformer.py
└── outputs/             # 테스트 결과 및 시각화
    ├── attention_patterns.png
    ├── decoder_attention_patterns.png
    ├── encoder_attention_patterns.png
    └── ... (기타 시각화 결과)
```

### 7. 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `d_model` | 512 | 모델의 차원 |
| `num_heads` | 8 | Attention Head 개수 |
| `num_layers` | 6 | Encoder/Decoder 층 수 |
| `d_ff` | 2048 | FFN의 Hidden Layer 차원 |
| `max_seq_length` | 512 | 최대 시퀀스 길이 |
| `vocab_size` | 가변 | 어휘 크기 |
| `dropout_rate` | 0.1 | Dropout 비율 |

## 프로젝트 실행 방법

### 의존성 설치
```bash
# 필요한 패키지 설치 (uv 사용)
uv sync

# 개발 환경 설정 (pre-commit 설치)
uv run pre-commit install
```

### 코드 품질 관리

이 프로젝트는 pre-commit hooks를 사용하여 코드 품질을 관리합니다:

- **Black**: 코드 포맷팅 (line-length: 100)
- **Ruff**: 빠른 Python linter
- **MyPy**: 정적 타입 검사
- **uv-lock**: 의존성 동기화

커밋 전 모든 파일 검사:
```bash
uv run pre-commit run --all-files
```

### Configuration 시스템

프로젝트는 YAML 기반의 설정 시스템을 사용합니다. `configs/` 디렉토리에서 사전 정의된 설정을 확인할 수 있습니다:

- `base.yaml`: 논문 기준 표준 설정 (d_model=512, 6 layers)
- `small.yaml`: 빠른 실험용 작은 모델 (d_model=256, 3 layers)
- `large.yaml`: 고성능 대규모 모델 (d_model=1024, 12 layers)
- `debug.yaml`: 디버깅용 최소 설정 (d_model=128, 2 layers)

### 학습 실행
```bash
# 기본 설정으로 학습
uv run scripts/train_with_config.py

# 특정 설정 파일 사용
uv run scripts/train_with_config.py --config small

# 설정 파일 + 명령줄 옵션
uv run scripts/train_with_config.py --config base --batch-size 64 --learning-rate 0.0005
```

### 테스트 실행
```bash
# 프로젝트 루트 디렉토리에서 실행
# (test-modeling 디렉토리에서)

# 각 모듈별 테스트 실행
uv run tests/test_positional_encoding.py
uv run tests/test_token_embedding.py
uv run tests/test_attention.py
uv run tests/test_transformer.py
# ... 기타 테스트 파일들

# 결과는 outputs/ 디렉토리에 저장됩니다
```

## WMT14 번역 모델 학습 (RTX 3090)

### 1. 데이터 준비

```bash
# 필요한 라이브러리 설치
uv add datasets sentencepiece

# WMT14 데이터 다운로드 및 전처리
uv run scripts/prepare_wmt14_data.py --config configs/rtx3090.yaml

# 디버그용 작은 데이터셋 준비
uv run scripts/prepare_wmt14_data.py --config configs/rtx3090_debug.yaml
```

### 2. 학습 실행

```bash
# RTX 3090에 최적화된 설정으로 학습
uv run train_wmt14.py --config configs/rtx3090.yaml

# 디버그 모드로 빠른 테스트
uv run train_wmt14.py --debug

# 체크포인트에서 재개
uv run train_wmt14.py --config configs/rtx3090.yaml --resume checkpoints/rtx3090/checkpoint-1000
```

### 3. RTX 3090 최적화 설정

**configs/rtx3090.yaml** 주요 설정:
- **배치 크기**: 12 (문장 단위)
- **Gradient Accumulation**: 20 steps (효과적 배치: 240)
- **Mixed Precision (FP16)**: 활성화로 메모리 절약
- **시퀀스 길이 제한**: 100 토큰
- **데이터 서브셋**: 100,000 문장 (전체 4.5M 중)

### 4. 모니터링

학습 중 다음 정보가 표시됩니다:
- 손실값 (Loss)
- 학습률 (Learning Rate)
- GPU 메모리 사용량
- 처리 속도 (samples/sec)

TensorBoard로 상세 모니터링:
```bash
tensorboard --logdir logs/rtx3090
```

### 5. 예상 학습 시간

RTX 3090 기준:
- **디버그 모드**: 약 10-30분
- **100K 서브셋**: 약 12-24시간
- **전체 데이터셋**: 현실적으로 불가능 (메모리 제한)

## 참고 문헌
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), 2017
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (Harvard NLP)
