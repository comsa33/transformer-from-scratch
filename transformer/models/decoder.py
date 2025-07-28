"""
Transformer Decoder 구현

Self-Attention, Cross-Attention, Feed-Forward Network를 결합한 Decoder layer와
이를 여러 층으로 쌓은 Decoder stack을 구현합니다.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from ..layers import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PreNormResidualConnection,
    PostNormResidualConnection,
    LayerNormalization
)
from ..embeddings import PositionalTokenEmbedding
from ..utils.masking import create_look_ahead_mask, create_padding_mask


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    
    하나의 Decoder layer는 다음 세 sub-layer로 구성됩니다:
    1. Masked Multi-Head Self-Attention
    2. Multi-Head Cross-Attention (Encoder-Decoder Attention)
    3. Position-wise Feed-Forward Network
    
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
        
        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = MultiHeadAttention(
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
            self.self_attn_residual = PreNormResidualConnection(d_model, dropout)
            self.cross_attn_residual = PreNormResidualConnection(d_model, dropout)
            self.ffn_residual = PreNormResidualConnection(d_model, dropout)
        else:
            self.self_attn_residual = PostNormResidualConnection(d_model, dropout)
            self.cross_attn_residual = PostNormResidualConnection(d_model, dropout)
            self.ffn_residual = PostNormResidualConnection(d_model, dropout)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm_first = norm_first
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Decoder layer forward pass
        
        Args:
            x: 입력 텐서 [batch_size, tgt_seq_length, d_model]
            encoder_output: Encoder 출력 [batch_size, src_seq_length, d_model]
            self_attn_mask: Self-attention mask [batch_size, 1, tgt_seq_length, tgt_seq_length]
            cross_attn_mask: Cross-attention mask [batch_size, 1, tgt_seq_length, src_seq_length]
            return_attention: Attention weights를 반환할지 여부
            
        Returns:
            출력 텐서 [batch_size, tgt_seq_length, d_model]
            (optional) self-attention weights, cross-attention weights
        """
        # 1. Masked Self-Attention sub-layer
        if return_attention:
            # Self-attention을 반환할 때는 직접 처리
            normalized = self.self_attn_residual.norm(x) if hasattr(self.self_attn_residual, 'norm') else x
            self_attn_out, self_attn_weights = self.self_attention(
                normalized, normalized, normalized, self_attn_mask, return_attention=True
            )
            self_attn_out = x + self.self_attn_residual.dropout(self_attn_out)
        else:
            self_attn_out = self.self_attn_residual(
                x,
                lambda x: self.self_attention(x, x, x, self_attn_mask)
            )
            self_attn_weights = None
        
        # 2. Cross-Attention sub-layer
        if return_attention:
            # Cross-attention을 반환할 때는 직접 처리
            normalized = self.cross_attn_residual.norm(self_attn_out) if hasattr(self.cross_attn_residual, 'norm') else self_attn_out
            cross_attn_out, cross_attn_weights = self.cross_attention(
                normalized, encoder_output, encoder_output, cross_attn_mask, return_attention=True
            )
            cross_attn_out = self_attn_out + self.cross_attn_residual.dropout(cross_attn_out)
        else:
            cross_attn_out = self.cross_attn_residual(
                self_attn_out,
                lambda x: self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
            )
            cross_attn_weights = None
        
        # 3. Feed-forward sub-layer
        output = self.ffn_residual(cross_attn_out, self.feed_forward)
        
        if return_attention:
            return output, self_attn_weights, cross_attn_weights
        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Stack
    
    여러 개의 Decoder layer를 쌓아서 만든 전체 Decoder입니다.
    
    Args:
        num_layers: Decoder layer 수
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
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
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
        
        # 출력 projection
        if share_embeddings:
            # 임베딩 가중치 공유
            self.output_projection = lambda x: torch.matmul(x, self.embeddings.token_embedding.weight.T)
        else:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Decoder forward pass
        
        Args:
            input_ids: 입력 token IDs [batch_size, tgt_seq_length]
            encoder_output: Encoder 출력 [batch_size, src_seq_length, d_model]
            src_mask: Source padding mask [batch_size, src_seq_length]
            tgt_mask: Target padding mask [batch_size, tgt_seq_length]
            return_all_layers: 모든 layer의 출력을 반환할지 여부
            return_attention: Attention weights를 반환할지 여부
            
        Returns:
            출력 텐서 [batch_size, tgt_seq_length, vocab_size]
            (optional) 모든 layer 출력들
            (optional) self-attention weights, cross-attention weights
        """
        batch_size, tgt_len = input_ids.shape
        device = input_ids.device
        
        # Self-attention mask 생성 (padding + look-ahead)
        # Look-ahead mask
        look_ahead_mask = create_look_ahead_mask(tgt_len).to(device)
        
        # Padding mask for target
        if tgt_mask is not None:
            tgt_padding_mask = create_padding_mask(tgt_mask)
            # Combine look-ahead and padding masks
            self_attn_mask = torch.maximum(look_ahead_mask, tgt_padding_mask)
        else:
            self_attn_mask = look_ahead_mask
        
        # Cross-attention mask (source padding)
        if src_mask is not None:
            # [batch_size, 1, 1, src_seq_length]
            cross_attn_mask = create_padding_mask(src_mask)
            # Expand to [batch_size, 1, tgt_seq_length, src_seq_length]
            cross_attn_mask = cross_attn_mask.expand(-1, -1, tgt_len, -1)
        else:
            cross_attn_mask = None
        
        # 임베딩
        x = self.embeddings(input_ids)
        
        # 각 layer 통과
        all_layers = []
        all_self_attentions = []
        all_cross_attentions = []
        
        for layer in self.layers:
            if return_attention:
                x, self_attn, cross_attn = layer(
                    x, encoder_output, self_attn_mask, cross_attn_mask, return_attention=True
                )
                all_self_attentions.append(self_attn)
                all_cross_attentions.append(cross_attn)
            else:
                x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
            
            if return_all_layers:
                all_layers.append(x)
        
        # 최종 normalization (Pre-LN의 경우)
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 출력 projection
        output = self.output_projection(x)
        
        # 반환값 구성
        outputs = (output,)
        if return_all_layers:
            outputs += (all_layers,)
        if return_attention:
            outputs += (all_self_attentions, all_cross_attentions)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def generate(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive 생성
        
        Args:
            encoder_output: Encoder 출력
            start_token_id: 시작 토큰 ID
            end_token_id: 종료 토큰 ID
            max_length: 최대 생성 길이
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            src_mask: Source padding mask
            
        Returns:
            생성된 시퀀스 [batch_size, seq_length]
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # 시작 토큰으로 초기화
        generated = torch.full((batch_size, 1), start_token_id, device=device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Decoder forward
                logits = self.forward(generated, encoder_output, src_mask)
                
                # 마지막 위치의 logits만 사용
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    min_value = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_value,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Top-p sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # 모든 샘플이 종료 토큰을 생성했는지 확인
                if (next_token == end_token_id).all():
                    break
        
        return generated


def create_decoder(
    num_layers: int = 6,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    vocab_size: int = 30000,
    max_length: int = 512,
    dropout: float = 0.1,
    activation: str = 'relu',
    norm_first: bool = True
) -> TransformerDecoder:
    """
    Decoder 생성 헬퍼 함수
    
    기본값은 Attention is All You Need 논문의 base model 설정입니다.
    """
    return TransformerDecoder(
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
    print("=== Transformer Decoder 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 8
    vocab_size = 100
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 3
    
    # Decoder 생성
    decoder = create_decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )
    
    # 입력 생성
    # Encoder output (실제로는 Encoder에서 나온 출력)
    encoder_output = torch.randn(batch_size, src_seq_length, d_model)
    
    # Decoder input
    target_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_length))
    
    # Masks
    src_mask = torch.ones(batch_size, src_seq_length)
    src_mask[0, 7:] = 0  # 첫 번째 샘플의 source 마지막 3개는 padding
    
    tgt_mask = torch.ones(batch_size, tgt_seq_length)
    tgt_mask[1, 6:] = 0  # 두 번째 샘플의 target 마지막 2개는 padding
    
    # Forward pass
    output = decoder(target_ids, encoder_output, src_mask, tgt_mask)
    
    print(f"Encoder 출력 shape: {encoder_output.shape}")
    print(f"Decoder 입력 shape: {target_ids.shape}")
    print(f"Decoder 출력 shape: {output.shape}")
    print(f"예상 출력 shape: [batch_size={batch_size}, tgt_seq={tgt_seq_length}, vocab_size={vocab_size}]")
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Attention 시각화용
    output, self_attns, cross_attns = decoder(
        target_ids, encoder_output, src_mask, tgt_mask, return_attention=True
    )
    print(f"\nSelf-attention maps 수: {len(self_attns)}")
    print(f"Cross-attention maps 수: {len(cross_attns)}")
    print(f"각 self-attention shape: {self_attns[0].shape}")
    print(f"각 cross-attention shape: {cross_attns[0].shape}")