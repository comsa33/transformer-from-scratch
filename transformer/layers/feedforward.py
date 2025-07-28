"""
Position-wise Feed-Forward Network 구현

Transformer의 각 위치에 독립적으로 적용되는 Feed-Forward Network입니다.
두 개의 선형 변환과 활성화 함수로 구성됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    각 위치에 독립적으로 적용되는 2층 feed-forward network입니다.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    원논문에서는 hidden dimension을 d_model의 4배로 설정합니다.
    최신 모델들은 다양한 활성화 함수를 사용합니다.
    
    Args:
        d_model: 입력/출력 차원
        d_ff: Feed-forward network의 hidden dimension
        dropout: Dropout 비율
        activation: 활성화 함수 ('relu', 'gelu', 'swish' 등)
        bias: Linear layer에 bias 사용 여부
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        # 첫 번째 선형 변환: d_model -> d_ff
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        
        # 두 번째 선형 변환: d_ff -> d_model
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 활성화 함수 설정
        self.activation = self._get_activation_fn(activation)
        self.activation_name = activation
        
        # 가중치 초기화
        self._init_weights()
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """활성화 함수 반환"""
        activation = activation.lower()
        
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'swish' or activation == 'silu':
            return F.silu
        elif activation == 'mish':
            return F.mish
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _init_weights(self):
        """가중치 초기화"""
        # Xavier uniform 초기화
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        
        if self.w_1.bias is not None:
            nn.init.zeros_(self.w_1.bias)
        if self.w_2.bias is not None:
            nn.init.zeros_(self.w_2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward network 적용
        
        Args:
            x: 입력 텐서 [..., seq_len, d_model]
            
        Returns:
            출력 텐서 [..., seq_len, d_model]
        """
        # 첫 번째 선형 변환 + 활성화
        # [..., seq_len, d_model] -> [..., seq_len, d_ff]
        hidden = self.w_1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # 두 번째 선형 변환
        # [..., seq_len, d_ff] -> [..., seq_len, d_model]
        output = self.w_2(hidden)
        output = self.dropout(output)
        
        return output
    
    def extra_repr(self) -> str:
        """모듈 정보 문자열"""
        return f'd_model={self.d_model}, d_ff={self.d_ff}, ' \
               f'activation={self.activation_name}, dropout={self.dropout_rate}'


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network (GLU 변형)
    
    일부 최신 모델에서 사용하는 gated 구조입니다.
    
    GLU(x) = (xW1 + b1) ⊗ σ(xW2 + b2)
    
    Args:
        d_model: 입력/출력 차원
        d_ff: Hidden dimension
        dropout: Dropout 비율
        activation: Gate에 사용할 활성화 함수
        bias: Bias 사용 여부
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'sigmoid',
        bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate와 value를 위한 projection
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_value = nn.Linear(d_model, d_ff, bias=bias)
        
        # 출력 projection
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Gate 활성화 함수
        self.gate_activation = self._get_activation_fn(activation)
        
        self._init_weights()
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """활성화 함수 반환"""
        if activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'gelu':
            return F.gelu
        else:
            raise ValueError(f"Unknown activation for gate: {activation}")
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.w_gate, self.w_value, self.w_out]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated FFN forward pass"""
        # Gate와 value 계산
        gate = self.gate_activation(self.w_gate(x))
        value = self.w_value(x)
        
        # Gated linear unit
        hidden = gate * value
        hidden = self.dropout(hidden)
        
        # 출력 projection
        output = self.w_out(hidden)
        output = self.dropout(output)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function
    
    LLaMA 등 최신 모델에서 사용하는 활성화 함수입니다.
    
    SwiGLU(x) = (xW1) ⊗ Swish(xW2)
    
    Args:
        d_model: 입력 차원
        d_ff: Hidden dimension (일반적으로 2/3 * 4 * d_model)
        bias: Bias 사용 여부
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        
        # Hidden dimension 설정
        if d_ff is None:
            # LLaMA 스타일: 2/3 * 4 * d_model을 가장 가까운 multiple of 256으로
            d_ff = int(2 * 4 * d_model / 3)
            d_ff = 256 * ((d_ff + 255) // 256)
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # SwiGLU를 위한 3개의 projection
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in [self.w_gate, self.w_up, self.w_down]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass"""
        # SwiGLU activation
        gate = F.silu(self.w_gate(x))  # Swish activation
        up = self.w_up(x)
        
        # Element-wise multiplication
        hidden = gate * up
        
        # Down projection
        output = self.w_down(hidden)
        
        return output


class ExpertFFN(nn.Module):
    """
    Expert Feed-Forward Network (MoE용)
    
    Mixture of Experts 구조에서 사용하는 FFN입니다.
    
    Args:
        d_model: 입력/출력 차원
        d_ff: Hidden dimension
        dropout: Dropout 비율
        activation: 활성화 함수
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Expert별 특화를 위한 추가 파라미터 (선택적)
        self.expert_bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert FFN forward pass"""
        output = self.ffn(x)
        output = output + self.expert_bias
        return output


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Position-wise Feed-Forward Network 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_length = 10
    d_model = 512
    d_ff = 2048
    
    # 기본 FFN 테스트
    ffn = PositionwiseFeedForward(d_model, d_ff, activation='relu')
    x = torch.randn(batch_size, seq_length, d_model)
    output = ffn(x)
    
    print(f"입력 shape: {x.shape}")
    print(f"FFN 출력 shape: {output.shape}")
    print(f"FFN 정보: {ffn}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")
    
    # 다양한 활성화 함수 테스트
    print("\n다양한 활성화 함수:")
    for activation in ['relu', 'gelu', 'swish']:
        ffn = PositionwiseFeedForward(d_model, d_ff, activation=activation)
        output = ffn(x)
        print(f"  {activation}: 출력 평균={output.mean():.4f}, 표준편차={output.std():.4f}")
    
    # Gated FFN 테스트
    print("\n=== Gated FFN 테스트 ===")
    gated_ffn = GatedFeedForward(d_model, d_ff)
    output = gated_ffn(x)
    print(f"Gated FFN 출력 shape: {output.shape}")
    
    # SwiGLU 테스트
    print("\n=== SwiGLU 테스트 ===")
    swiglu = SwiGLU(d_model)
    output = swiglu(x)
    print(f"SwiGLU 출력 shape: {output.shape}")
    print(f"SwiGLU hidden dimension: {swiglu.d_ff}")