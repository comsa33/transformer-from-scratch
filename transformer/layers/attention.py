"""
Attention 메커니즘 구현

Transformer의 핵심인 Scaled Dot-Product Attention과
Multi-Head Attention을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention 계산
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        query: Query 텐서 [..., seq_len_q, d_k]
        key: Key 텐서 [..., seq_len_k, d_k]
        value: Value 텐서 [..., seq_len_k, d_v]
        mask: 마스크 텐서 [..., seq_len_q, seq_len_k] (1: 마스킹)
        dropout: Dropout 레이어
        scale: 스케일 factor (기본값: 1/sqrt(d_k))
        
    Returns:
        (output, attention_weights) 튜플
        - output: [..., seq_len_q, d_v]
        - attention_weights: [..., seq_len_q, seq_len_k]
    """
    # Query의 마지막 차원 크기
    d_k = query.size(-1)
    
    # 스케일 factor 설정
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    
    # Q와 K의 내적 계산: [..., seq_len_q, d_k] x [..., d_k, seq_len_k] = [..., seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 스케일링 적용
    scores = scores * scale
    
    # 마스크 적용 (있는 경우)
    if mask is not None:
        # 마스크가 1인 위치에 매우 작은 값을 더함
        scores = scores.masked_fill(mask == 1, -1e9)
    
    # Softmax를 통해 attention weights 계산
    attention_weights = F.softmax(scores, dim=-1)
    
    # Dropout 적용 (있는 경우)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Value와 가중합 계산: [..., seq_len_q, seq_len_k] x [..., seq_len_k, d_v] = [..., seq_len_q, d_v]
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 구현
    
    여러 개의 attention head를 병렬로 실행하여
    다양한 representation subspace에서 정보를 추출합니다.
    
    Args:
        d_model: 모델의 차원
        num_heads: Attention head의 개수
        dropout: Dropout 비율
        bias: Linear projection에 bias 사용 여부
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 head의 차원
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Query, Key, Value projection layers
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection layer
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform 초기화"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Multi-Head Attention forward pass
        
        Args:
            query: Query 텐서 [batch_size, seq_len_q, d_model]
            key: Key 텐서 [batch_size, seq_len_k, d_model]
            value: Value 텐서 [batch_size, seq_len_k, d_model]
            mask: 마스크 텐서 [batch_size, 1, seq_len_q, seq_len_k] 또는
                              [batch_size, num_heads, seq_len_q, seq_len_k]
            return_attention: Attention weights 반환 여부
            
        Returns:
            return_attention=False: output [batch_size, seq_len_q, d_model]
            return_attention=True: (output, attention_weights)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 1. Linear projections
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        # -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled dot-product attention
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, num_heads, seq_len_q, d_k]
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            dropout=self.attention_dropout,
            scale=self.scale
        )
        
        # 4. Concatenate heads
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, num_heads, d_k]
        # -> [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.d_model)
        
        # 5. Final linear projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            # attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
            return output, attention_weights
        else:
            return output
    
    def get_attention_maps(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention weights만 계산하여 반환 (시각화용)
        
        Returns:
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        with torch.no_grad():
            _, attention_weights = self.forward(
                query, key, value, mask, return_attention=True
            )
        return attention_weights


class SelfAttention(MultiHeadAttention):
    """
    Self-Attention 래퍼 클래스
    
    Query, Key, Value가 모두 같은 입력에서 나오는 경우
    """
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Self-Attention forward pass
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            mask: 마스크 텐서
            return_attention: Attention weights 반환 여부
            
        Returns:
            output 또는 (output, attention_weights)
        """
        return super().forward(x, x, x, mask, return_attention)


class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention 래퍼 클래스
    
    Encoder-Decoder attention에서 사용
    Query는 decoder에서, Key와 Value는 encoder에서 나옵니다.
    """
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cross-Attention forward pass
        
        Args:
            query: Decoder의 입력 [batch_size, target_len, d_model]
            memory: Encoder의 출력 [batch_size, source_len, d_model]
            memory_mask: Encoder 출력에 대한 마스크
            return_attention: Attention weights 반환 여부
            
        Returns:
            output 또는 (output, attention_weights)
        """
        return super().forward(query, memory, memory, memory_mask, return_attention)


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Multi-Head Attention 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_length = 10
    d_model = 512
    num_heads = 8
    
    # Multi-Head Attention 생성
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 더미 입력
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Self-attention (Q=K=V=x)
    output = mha(x, x, x)
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    
    # Attention weights 확인
    output, attention_weights = mha(x, x, x, return_attention=True)
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum (dim=-1): {attention_weights.sum(dim=-1)[0, 0]}")
    
    # 마스크 적용 테스트
    from ..utils.masking import create_look_ahead_mask
    mask = create_look_ahead_mask(seq_length)
    mask = mask.expand(batch_size, num_heads, seq_length, seq_length)
    
    masked_output = mha(x, x, x, mask=mask)
    print(f"\n마스크 적용 출력 shape: {masked_output.shape}")
    
    # Self-Attention 래퍼 테스트
    self_attn = SelfAttention(d_model, num_heads)
    self_output = self_attn(x)
    print(f"\nSelf-Attention 출력 shape: {self_output.shape}")