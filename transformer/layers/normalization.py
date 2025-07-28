"""
Layer Normalization 구현

Transformer에서는 각 sub-layer의 출력에 Layer Normalization을 적용합니다.
Batch Normalization과 달리 각 샘플 내에서 정규화를 수행하므로
배치 크기에 독립적이고, 추론 시에도 안정적입니다.
"""

import torch
import torch.nn as nn
from typing import Union, List, Tuple


class LayerNormalization(nn.Module):
    """
    Layer Normalization 구현
    
    각 샘플의 특정 차원들에 대해 평균과 분산을 계산하여 정규화합니다.
    학습 가능한 scale(gamma)과 shift(beta) 파라미터를 포함합니다.
    
    Layer Norm 수식:
        y = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    여기서:
        - mean, variance: 지정된 차원에 대한 통계량
        - gamma, beta: 학습 가능한 파라미터
        - eps: 수치 안정성을 위한 작은 값
    
    Args:
        normalized_shape: 정규화할 차원의 shape
        eps: 분산에 더해지는 작은 값 (기본값: 1e-6)
        elementwise_affine: gamma와 beta 사용 여부 (기본값: True)
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        
        # normalized_shape을 튜플로 변환
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # 학습 가능한 파라미터
        if self.elementwise_affine:
            # gamma (scale parameter) - 초기값 1
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            # beta (shift parameter) - 초기값 0
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            # 파라미터를 사용하지 않는 경우
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Layer Normalization 적용
        
        Args:
            x: 입력 텐서 [..., *normalized_shape]
            
        Returns:
            정규화된 텐서 (같은 shape)
        """
        # 정규화할 차원들 계산
        # 예: x.shape = [batch, seq_len, d_model], normalized_shape = (d_model,)
        # -> dims = [-1]
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # 평균과 분산 계산
        # keepdim=True로 브로드캐스팅 가능하게 유지
        mean = x.mean(dim=dims, keepdim=True)
        # unbiased=False: 분산 계산 시 N으로 나눔 (N-1이 아님)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # 정규화
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Affine 변환 적용
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta
        
        return x_normalized
    
    def extra_repr(self) -> str:
        """모듈 정보 문자열"""
        return f'{self.normalized_shape}, eps={self.eps}, ' \
               f'elementwise_affine={self.elementwise_affine}'


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    일부 최신 모델에서 사용하는 변형으로, 평균을 빼지 않고
    RMS(Root Mean Square)로만 정규화합니다.
    
    RMSNorm 수식:
        y = gamma * x / RMS(x) + beta
        where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        normalized_shape: 정규화할 차원
        eps: 수치 안정성을 위한 값
        elementwise_affine: gamma와 beta 사용 여부
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm 적용"""
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # RMS 계산
        ms = (x ** 2).mean(dim=dims, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        
        # 정규화
        x_normalized = x / rms
        
        # Affine 변환
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta
        
        return x_normalized


class PreNormalization(nn.Module):
    """
    Pre-Normalization 래퍼
    
    Transformer의 원 논문은 Post-Normalization을 사용하지만,
    많은 최신 구현은 Pre-Normalization을 사용합니다.
    
    Pre-Norm: LayerNorm(x) -> SubLayer -> Add(x)
    Post-Norm: SubLayer(x) -> Add(x) -> LayerNorm
    
    Args:
        d_model: 모델 차원
        sublayer: 적용할 sub-layer (Attention, FFN 등)
        dropout: dropout 비율
    """
    
    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Pre-normalization 적용
        
        Args:
            x: 입력 텐서
            *args, **kwargs: sublayer에 전달할 추가 인자
            
        Returns:
            Residual connection이 적용된 출력
        """
        # Pre-normalization
        normalized = self.norm(x)
        
        # Sublayer 적용
        output = self.sublayer(normalized, *args, **kwargs)
        
        # Dropout
        output = self.dropout(output)
        
        # Residual connection
        return x + output


class PostNormalization(nn.Module):
    """
    Post-Normalization 래퍼 (원 논문 스타일)
    
    Args:
        d_model: 모델 차원
        sublayer: 적용할 sub-layer
        dropout: dropout 비율
    """
    
    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Post-normalization 적용"""
        # Sublayer 적용
        output = self.sublayer(x, *args, **kwargs)
        
        # Dropout
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        # Post-normalization
        return self.norm(output)


def create_norm_layer(
    norm_type: str,
    normalized_shape: Union[int, List[int], torch.Size],
    eps: float = 1e-6
) -> nn.Module:
    """
    정규화 레이어 생성 팩토리 함수
    
    Args:
        norm_type: 'layer_norm' 또는 'rms_norm'
        normalized_shape: 정규화할 차원
        eps: epsilon 값
        
    Returns:
        정규화 레이어
    """
    if norm_type == 'layer_norm':
        return LayerNormalization(normalized_shape, eps=eps)
    elif norm_type == 'rms_norm':
        return RMSNorm(normalized_shape, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Layer Normalization 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_length = 5
    d_model = 8
    
    # 랜덤 입력 생성
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"입력 shape: {x.shape}")
    print(f"입력 예시 (첫 번째 샘플, 첫 번째 위치):\n{x[0, 0]}\n")
    
    # Layer Normalization 적용
    layer_norm = LayerNormalization(d_model)
    output = layer_norm(x)
    
    print(f"출력 shape: {output.shape}")
    print(f"출력 예시 (첫 번째 샘플, 첫 번째 위치):\n{output[0, 0]}\n")
    
    # 통계 확인
    print("정규화 후 통계 (첫 번째 샘플, 첫 번째 위치):")
    print(f"평균: {output[0, 0].mean():.6f}")
    print(f"표준편차: {output[0, 0].std(unbiased=False):.6f}")
    
    # RMSNorm 비교
    print("\n=== RMSNorm 비교 ===\n")
    rms_norm = RMSNorm(d_model)
    rms_output = rms_norm(x)
    
    print(f"RMSNorm 출력 예시:\n{rms_output[0, 0]}")
    print(f"RMSNorm 평균: {rms_output[0, 0].mean():.6f}")
    print(f"RMSNorm 표준편차: {rms_output[0, 0].std(unbiased=False):.6f}")