# Transformer 모델 아키텍처 설계

## 프로젝트 구조 개요

이 문서는 Transformer 모델 구현을 위한 상세한 디렉토리 구조와 각 모듈의 역할을 정의합니다.

## 디렉토리 구조

```
test-modeling/
├── README.md                    # 프로젝트 개요 및 사용법
├── ARCHITECTURE.md             # 이 문서 - 아키텍처 설계
├── pyproject.toml              # 프로젝트 설정 및 의존성
├── .gitignore                  # Git 제외 파일 목록
│
├── transformer/                # 핵심 Transformer 구현
│   ├── __init__.py            # 패키지 초기화 및 주요 클래스 export
│   ├── config.py              # 모델 설정 및 하이퍼파라미터
│   │
│   ├── layers/                # 기본 레이어 구현
│   │   ├── __init__.py
│   │   ├── attention.py       # Multi-Head Attention 구현
│   │   ├── feedforward.py     # Position-wise FFN 구현
│   │   ├── normalization.py   # Layer Normalization 구현
│   │   └── residual.py        # Residual Connection 구현
│   │
│   ├── embeddings/            # 임베딩 관련 구현
│   │   ├── __init__.py
│   │   ├── token_embedding.py # 토큰 임베딩
│   │   └── positional.py      # Positional Encoding
│   │
│   ├── models/                # 모델 구성 요소
│   │   ├── __init__.py
│   │   ├── encoder.py         # Encoder 구현
│   │   ├── decoder.py         # Decoder 구현
│   │   └── transformer.py     # 전체 Transformer 모델
│   │
│   └── utils/                 # 유틸리티 함수
│       ├── __init__.py
│       ├── masking.py         # 마스킹 유틸리티
│       └── initialization.py  # 가중치 초기화
│
├── data/                      # 데이터 처리
│   ├── __init__.py
│   ├── dataset.py             # 데이터셋 클래스
│   └── tokenizer.py           # 토크나이저 구현
│
├── training/                  # 학습 관련 코드
│   ├── __init__.py
│   ├── trainer.py             # 학습 루프 구현
│   ├── optimizer.py           # 옵티마이저 및 스케줄러
│   └── loss.py                # 손실 함수 정의
│
├── evaluation/                # 평가 관련 코드
│   ├── __init__.py
│   └── metrics.py             # 평가 메트릭 구현
│
├── scripts/                   # 실행 스크립트
│   └── train_with_config.py  # 설정 기반 학습 스크립트
│
├── tests/                     # 테스트 코드
│   ├── __init__.py
│   ├── test_attention.py      # Attention 레이어 테스트
│   ├── test_encoder.py        # Encoder 테스트
│   ├── test_decoder.py        # Decoder 테스트
│   ├── test_transformer.py    # 전체 모델 테스트
│   └── ... (기타 모듈별 테스트)
│
├── notebooks/                 # 실험 및 시각화 노트북
│   └── (추후 추가 예정)
│
├── outputs/                   # 테스트 결과 및 시각화
│   └── *.png                  # 생성된 시각화 이미지들
│
├── configs/                   # 설정 시스템
│   ├── __init__.py           # 설정 모듈 초기화
│   ├── utils.py              # 설정 로드/저장 유틸리티
│   ├── README.md             # 설정 시스템 문서
│   ├── base.yaml             # 논문 기준 표준 설정
│   ├── small.yaml            # 빠른 실험용 작은 모델
│   ├── large.yaml            # 고성능 대규모 모델
│   └── debug.yaml            # 디버깅용 최소 설정
```

## 모듈별 상세 설계

### 1. Core Layers (`transformer/layers/`)

#### attention.py
```python
class MultiHeadAttention:
    """Multi-Head Attention 메커니즘 구현

    주요 메서드:
    - forward(): 어텐션 계산
    - _scaled_dot_product_attention(): 스케일드 닷 프로덕트 어텐션
    - _split_heads(): 헤드 분할
    - _combine_heads(): 헤드 결합
    """
```

#### feedforward.py
```python
class PositionwiseFeedForward:
    """Position-wise Feed-Forward Network

    구성:
    - 두 개의 Linear 레이어
    - ReLU 활성화 함수
    - Dropout
    """
```

#### normalization.py
```python
class LayerNormalization:
    """Layer Normalization 구현

    특징:
    - 학습 가능한 scale과 shift 파라미터
    - 수치 안정성을 위한 epsilon
    """
```

### 2. Embeddings (`transformer/embeddings/`)

#### token_embedding.py
```python
class TokenEmbedding:
    """토큰을 벡터로 변환

    파라미터:
    - vocab_size: 어휘 크기
    - d_model: 임베딩 차원
    """
```

#### positional.py
```python
class PositionalEncoding:
    """위치 정보 인코딩

    구현 방식:
    - Sinusoidal 인코딩
    - 학습 가능한 인코딩 (선택적)
    """
```

### 3. Models (`transformer/models/`)

#### encoder.py
```python
class EncoderLayer:
    """단일 Encoder 레이어

    구성:
    - Multi-Head Self-Attention
    - Position-wise FFN
    - Residual connections
    - Layer normalization
    """

class TransformerEncoder:
    """전체 Encoder 스택

    구성:
    - N개의 EncoderLayer
    - 입력 임베딩 처리
    """
```

#### decoder.py
```python
class DecoderLayer:
    """단일 Decoder 레이어

    구성:
    - Masked Multi-Head Self-Attention
    - Multi-Head Cross-Attention
    - Position-wise FFN
    - Residual connections
    - Layer normalization
    """

class TransformerDecoder:
    """전체 Decoder 스택

    구성:
    - N개의 DecoderLayer
    - 출력 임베딩 처리
    """
```

### 4. 설정 관리 (`transformer/config.py`)

```python
@dataclass
class TransformerConfig:
    """Transformer 설정 클래스

    주요 파라미터:
    - d_model: 모델 차원 (512)
    - num_heads: 어텐션 헤드 수 (8)
    - num_layers: 레이어 수 (6)
    - d_ff: FFN 히든 차원 (2048)
    - max_seq_length: 최대 시퀀스 길이 (512)
    - vocab_size: 어휘 크기
    - dropout_rate: 드롭아웃 비율 (0.1)
    - attention_dropout: 어텐션 드롭아웃 (0.1)
    - activation: 활성화 함수 ('relu')
    - eps: Layer norm epsilon (1e-6)
    """
```

## 인터페이스 설계

### 1. 기본 인터페이스

```python
# 모델 생성
from transformer import Transformer, TransformerConfig

config = TransformerConfig(
    d_model=512,
    num_heads=8,
    num_layers=6,
    vocab_size=10000
)

model = Transformer(config)

# 학습
outputs = model(
    src_tokens,      # [batch_size, src_len]
    tgt_tokens,      # [batch_size, tgt_len]
    src_mask=None,   # 선택적
    tgt_mask=None    # 선택적
)

# 추론
predictions = model.generate(
    src_tokens,
    max_length=100,
    temperature=1.0
)
```

### 2. 레이어별 인터페이스

```python
# Attention 레이어 단독 사용
from transformer.layers import MultiHeadAttention

attention = MultiHeadAttention(d_model=512, num_heads=8)
output = attention(query, key, value, mask=None)

# Encoder 단독 사용
from transformer.models import TransformerEncoder

encoder = TransformerEncoder(config)
encoder_output = encoder(src_tokens, src_mask)
```

## 확장 가능성

### 1. 변형 모델 지원
- BERT-style (Encoder-only)
- GPT-style (Decoder-only)
- T5-style (Encoder-Decoder)

### 2. 추가 기능
- Relative Position Encoding
- Sparse Attention
- Linear Attention
- Flash Attention

### 3. 최적화
- Mixed Precision Training
- Gradient Checkpointing
- Model Parallelism

## 개발 우선순위

1. **Phase 1**: 핵심 레이어 구현
   - Multi-Head Attention
   - Layer Normalization
   - Position-wise FFN

2. **Phase 2**: 모델 구성
   - Encoder/Decoder 구현
   - 전체 Transformer 조립

3. **Phase 3**: 학습 파이프라인
   - 데이터 로더
   - 학습 루프
   - 평가 메트릭

4. **Phase 4**: 최적화 및 확장
   - 성능 최적화
   - 변형 모델 지원

## 테스트 전략

### 1. 단위 테스트
- 각 레이어의 출력 shape 검증
- Gradient flow 확인
- 수치 안정성 테스트

### 2. 통합 테스트
- End-to-end 모델 테스트
- 학습 수렴 확인
- 추론 품질 검증

### 3. 성능 테스트
- 메모리 사용량 프로파일링
- 연산 속도 벤치마킹
- 배치 크기별 성능 측정
