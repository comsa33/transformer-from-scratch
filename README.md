# Transformer 모델 구현 프로젝트

## 개요
이 프로젝트는 Transformer 아키텍처를 처음부터 구현하며 각 구성 요소를 깊이 이해하는 것을 목표로 합니다. Python 3.11과 uv 패키지 매니저를 사용합니다.

## Transformer 아키텍처 상세

### 1. 전체 구조 개요

Transformer는 2017년 "Attention is All You Need" 논문에서 처음 소개된 모델로, RNN이나 CNN 없이 오직 Attention 메커니즘만으로 구성된 혁신적인 아키텍처입니다.

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
# 필요한 패키지 설치
uv add torch numpy
```

### 테스트 실행
```bash
# 프로젝트 루트 디렉토리에서 실행
# (test-modeling 디렉토리에서)

# 각 모듈별 테스트 실행
uv run python tests/test_positional_encoding.py
uv run python tests/test_token_embedding.py
uv run python tests/test_attention.py
uv run python tests/test_transformer.py
# ... 기타 테스트 파일들

# 결과는 outputs/ 디렉토리에 저장됩니다
```

## 참고 문헌
- Vaswani et al., "Attention Is All You Need", 2017
- The Annotated Transformer (Harvard NLP)