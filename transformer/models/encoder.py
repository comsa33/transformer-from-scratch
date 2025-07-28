"""
Transformer Encoder 구현

Multi-Head Attention과 Feed-Forward Network를 결합한 Encoder layer와
이를 여러 층으로 쌓은 Encoder stack을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import copy

from ..layers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PreNormResidualConnection,
    PostNormResidualConnection,
    LayerNormalization
)
from ..embeddings import PositionalTokenEmbedding


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    하나의 Encoder layer는 다음 두 sub-layer로 구성됩니다:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    
    각 sub-layer 주위에는 residual connection과 layer normalization이 적용됩니다.
    
    Args:
        d_model: 모델의 dimension
        num_heads: Attention head 수
        d_ff: Feed-forward network의 hidden dimension
        dropout: Dropout 비율
        activation: FFN의 활성화 함수
        norm_first: Pre-LN vs Post-LN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = True
    ):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Residual connections
        if norm_first:
            self.attn_residual = PreNormResidualConnection(d_model, dropout)
            self.ffn_residual = PreNormResidualConnection(d_model, dropout)
        else:
            self.attn_residual = PostNormResidualConnection(d_model, dropout)
            self.ffn_residual = PostNormResidualConnection(d_model, dropout)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_first = norm_first
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Encoder layer forward pass
        
        Args:
            x: 입력 텐서 [batch_size, seq_length, d_model]
            mask: Attention mask [batch_size, 1, 1, seq_length]
            return_attention: Attention weights를 반환할지 여부
            
        Returns:
            출력 텐서 [batch_size, seq_length, d_model]
            (optional) attention weights
        """
        # Self-attention sub-layer
        if return_attention:
            # Attention을 반환할 때는 직접 처리
            normalized = self.attn_residual.norm(x) if hasattr(self.attn_residual, 'norm') else x
            attn_out, attn_weights = self.self_attention(normalized, normalized, normalized, mask, return_attention=True)
            attn_out = x + self.attn_residual.dropout(attn_out)
        else:
            attn_out = self.attn_residual(
                x, 
                lambda x: self.self_attention(x, x, x, mask)
            )
            attn_weights = None
        
        # Feed-forward sub-layer
        output = self.ffn_residual(attn_out, self.feed_forward)
        
        if return_attention:
            return output, attn_weights
        return output


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Stack
    
    여러 개의 Encoder layer를 쌓아서 만든 전체 Encoder입니다.
    
    Args:
        num_layers: Encoder layer 수
        d_model: 모델 dimension
        num_heads: Attention head 수
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary 크기
        max_length: 최대 시퀀스 길이
        dropout: Dropout 비율
        activation: FFN 활성화 함수
        norm_first: Pre-LN vs Post-LN
        share_embeddings: 임베딩 공유 여부
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_length: int = 5000,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = True,
        share_embeddings: bool = False
    ):
        super().__init__()
        
        # 입력 임베딩
        self.embeddings = PositionalTokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_length=max_length,
            dropout=dropout
        )
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        
        # 최종 normalization (Pre-LN 구조에서 필요)
        self.final_norm = LayerNormalization(d_model) if norm_first else None
        
        # 출력 projection (선택적)
        self.output_projection = None
        if share_embeddings:
            # 임베딩 가중치 공유
            self.output_projection = lambda x: torch.matmul(x, self.embeddings.token_embedding.weight.T)
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Encoder forward pass
        
        Args:
            input_ids: 입력 token IDs [batch_size, seq_length]
            mask: Padding mask [batch_size, seq_length]
            return_all_layers: 모든 layer의 출력을 반환할지 여부
            return_attention: Attention weights를 반환할지 여부
            
        Returns:
            출력 텐서 [batch_size, seq_length, d_model]
            (optional) 모든 layer 출력들
            (optional) attention weights
        """
        # Mask 형태 변환 (if needed)
        if mask is not None and mask.dim() == 2:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        # 임베딩
        x = self.embeddings(input_ids)
        
        # 각 layer 통과
        all_layers = []
        all_attentions = []
        
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask, return_attention=True)
                all_attentions.append(attn)
            else:
                x = layer(x, mask)
            
            if return_all_layers:
                all_layers.append(x)
        
        # 최종 normalization (Pre-LN의 경우)
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 출력 projection (if needed)
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        # 반환값 구성
        outputs = (x,)
        if return_all_layers:
            outputs += (all_layers,)
        if return_attention:
            outputs += (all_attentions,)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def get_attention_maps(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        특정 layer(들)의 attention map 반환
        
        Args:
            input_ids: 입력 token IDs
            mask: Padding mask
            layer_idx: 특정 layer index (None이면 모든 layer)
            
        Returns:
            Attention maps
        """
        with torch.no_grad():
            if layer_idx is not None:
                # 특정 layer까지만 실행
                x = self.embeddings(input_ids)
                for i in range(layer_idx + 1):
                    if i == layer_idx:
                        _, attn = self.layers[i](x, mask, return_attention=True)
                        return attn
                    else:
                        x = self.layers[i](x, mask)
            else:
                # 모든 layer의 attention 반환
                _, attentions = self.forward(
                    input_ids, mask, return_attention=True
                )
                return attentions


def create_encoder(
    num_layers: int = 6,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    vocab_size: int = 30000,
    max_length: int = 512,
    dropout: float = 0.1,
    activation: str = 'relu',
    norm_first: bool = True
) -> TransformerEncoder:
    """
    Encoder 생성 헬퍼 함수
    
    기본값은 Attention is All You Need 논문의 base model 설정입니다.
    """
    return TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_length=max_length,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first
    )


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Transformer Encoder 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_length = 10
    vocab_size = 100
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 3
    
    # Encoder 생성
    encoder = create_encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )
    
    # 입력 생성
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    mask = torch.ones(batch_size, seq_length)
    mask[0, 7:] = 0  # 첫 번째 샘플의 마지막 3개 토큰은 padding
    
    # Forward pass
    output = encoder(input_ids, mask)
    
    print(f"입력 shape: {input_ids.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Attention 시각화용
    _, attentions = encoder(input_ids, mask, return_attention=True)
    print(f"\nAttention maps 수: {len(attentions)}")
    print(f"각 attention map shape: {attentions[0].shape}")