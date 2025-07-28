# Transformer Configuration Files

이 디렉토리는 다양한 Transformer 모델 설정을 위한 YAML 파일들을 포함합니다.

## 설정 파일 목록

### base.yaml
- **용도**: 논문 "Attention Is All You Need"의 기본 Transformer 설정
- **모델 크기**: d_model=512, 6 layers
- **적합한 경우**: 표준 번역 작업, 중간 규모 데이터셋

### small.yaml
- **용도**: 빠른 실험과 프로토타이핑을 위한 작은 모델
- **모델 크기**: d_model=256, 3 layers
- **적합한 경우**: 빠른 반복 실험, 제한된 컴퓨팅 자원

### large.yaml
- **용도**: 높은 성능을 위한 대규모 모델
- **모델 크기**: d_model=1024, 12 layers
- **적합한 경우**: 대규모 데이터셋, 고성능이 필요한 작업
- **특징**: Mixed precision training, Gradient checkpointing 지원

### debug.yaml
- **용도**: 디버깅과 코드 검증용
- **모델 크기**: d_model=128, 2 layers
- **적합한 경우**: 코드 디버깅, 새로운 기능 테스트
- **특징**: Dropout 비활성화, 자세한 로깅

## 사용 방법

### Python에서 직접 로드
```python
import yaml
from transformer.config import TransformerConfig

# YAML 파일 로드
with open('configs/base.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# TransformerConfig 객체 생성
model_config = TransformerConfig(**config_dict['model'])
```

### Config 유틸리티 사용 (권장)
```python
from configs.utils import load_config

# 전체 설정 로드
config = load_config('base')

# 모델 설정만 가져오기
model_config = config.get_model_config()

# 학습 설정 가져오기
training_config = config.get_training_config()
```

## 커스텀 설정 만들기

새로운 설정 파일을 만들 때는 다음 구조를 따르세요:

```yaml
model:
  d_model: 512
  num_heads: 8
  # ... 기타 모델 설정

training:
  batch_size: 32
  learning_rate: 0.0001
  # ... 기타 학습 설정

optimizer:
  type: "adam"
  # ... 기타 옵티마이저 설정

scheduler:
  type: "transformer"
  # ... 기타 스케줄러 설정

data:
  max_length: 512
  # ... 기타 데이터 설정

logging:
  log_interval: 100
  # ... 기타 로깅 설정

checkpoint:
  save_dir: "checkpoints"
  # ... 기타 체크포인트 설정
```

## 설정 우선순위

1. 명령줄 인자
2. 환경 변수
3. 설정 파일
4. 기본값

## 주의사항

- `vocab_size`는 데이터셋에 따라 동적으로 설정되므로 YAML 파일에는 포함하지 않습니다
- Mixed precision training을 사용할 때는 충분한 GPU 메모리가 필요합니다
- Large 모델을 사용할 때는 gradient accumulation을 고려하세요