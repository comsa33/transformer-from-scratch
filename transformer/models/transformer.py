"""
완전한 Transformer 모델 구현

Encoder와 Decoder를 결합한 전체 Transformer 모델입니다.
기계 번역, 텍스트 생성 등 다양한 sequence-to-sequence 작업에 사용됩니다.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
import math

from ..config import TransformerConfig
from .encoder import TransformerEncoder, create_encoder
from .decoder import TransformerDecoder, create_decoder
from ..utils.masking import create_padding_mask


class Transformer(nn.Module):
    """
    완전한 Transformer 모델
    
    Encoder-Decoder 구조로 이루어진 Transformer 모델입니다.
    
    Args:
        num_encoder_layers: Encoder layer 수
        num_decoder_layers: Decoder layer 수
        d_model: 모델 dimension
        num_heads: Attention head 수
        d_ff: Feed-forward dimension
        src_vocab_size: Source vocabulary 크기
        tgt_vocab_size: Target vocabulary 크기
        max_length: 최대 시퀀스 길이
        dropout: Dropout 비율
        activation: FFN 활성화 함수
        norm_first: Pre-LN vs Post-LN
        share_embeddings: 임베딩 공유 여부
        share_encoder_decoder_embeddings: Encoder-Decoder 간 임베딩 공유
    """
    
    def __init__(
        self,
        config: Union[TransformerConfig, None] = None,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        max_length: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        norm_first: Optional[bool] = None,
        share_embeddings: Optional[bool] = None,
        share_encoder_decoder_embeddings: Optional[bool] = None
    ):
        super().__init__()
        
        # Config가 제공된 경우 값 추출
        if config is not None:
            num_encoder_layers = config.num_encoder_layers
            num_decoder_layers = config.num_decoder_layers
            d_model = config.d_model
            num_heads = config.num_heads
            d_ff = config.d_ff
            src_vocab_size = config.vocab_size
            tgt_vocab_size = config.vocab_size
            max_length = config.max_seq_length
            dropout = config.dropout_rate
            activation = config.activation
            norm_first = True  # 기본값
            share_embeddings = config.share_embeddings
            share_encoder_decoder_embeddings = config.share_embeddings
        else:
            # 기본값 설정
            num_encoder_layers = num_encoder_layers or 6
            num_decoder_layers = num_decoder_layers or 6
            d_model = d_model or 512
            num_heads = num_heads or 8
            d_ff = d_ff or 2048
            src_vocab_size = src_vocab_size or 30000
            tgt_vocab_size = tgt_vocab_size or 30000
            max_length = max_length or 512
            dropout = dropout or 0.1
            activation = activation or 'relu'
            norm_first = norm_first if norm_first is not None else True
            share_embeddings = share_embeddings or False
            share_encoder_decoder_embeddings = share_encoder_decoder_embeddings or False
        
        # Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=src_vocab_size,
            max_length=max_length,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            share_embeddings=False  # Encoder는 항상 독립적인 output projection
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=tgt_vocab_size,
            max_length=max_length,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            share_embeddings=share_embeddings
        )
        
        # Encoder-Decoder 간 임베딩 공유 (선택적)
        if share_encoder_decoder_embeddings and src_vocab_size == tgt_vocab_size:
            # Decoder의 임베딩을 Encoder와 공유
            self.decoder.embeddings = self.encoder.embeddings
            
            # Output projection도 공유하는 경우
            if share_embeddings:
                self.decoder.output_projection = lambda x: torch.matmul(
                    x, self.encoder.embeddings.token_embedding.weight.T
                )
        
        # 모델 설정 저장
        self.d_model = d_model
        self.num_heads = num_heads
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 파라미터 초기화
        self._init_parameters()
    
    def _init_parameters(self):
        """
        Xavier uniform 초기화
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_encoder_output: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Transformer forward pass
        
        Args:
            src_ids: Source token IDs [batch_size, src_seq_length]
            tgt_ids: Target token IDs [batch_size, tgt_seq_length]
            src_mask: Source padding mask [batch_size, src_seq_length]
            tgt_mask: Target padding mask [batch_size, tgt_seq_length]
            return_encoder_output: Encoder 출력도 반환할지 여부
            return_attention: Attention weights를 반환할지 여부
            
        Returns:
            출력 logits [batch_size, tgt_seq_length, tgt_vocab_size]
            (optional) encoder output
            (optional) attention weights
        """
        # Encode
        if return_attention:
            encoder_output, encoder_attentions = self.encoder(
                src_ids, src_mask, return_attention=True
            )
        else:
            encoder_output = self.encoder(src_ids, src_mask)
            encoder_attentions = None
        
        # Decode
        if return_attention:
            output, decoder_self_attentions, decoder_cross_attentions = self.decoder(
                tgt_ids, encoder_output, src_mask, tgt_mask, return_attention=True
            )
        else:
            output = self.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
            decoder_self_attentions = None
            decoder_cross_attentions = None
        
        # 반환값 구성
        outputs = (output,)
        if return_encoder_output:
            outputs += (encoder_output,)
        if return_attention:
            outputs += ({
                'encoder': encoder_attentions,
                'decoder_self': decoder_self_attentions,
                'decoder_cross': decoder_cross_attentions
            },)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder만 실행
        
        Args:
            src_ids: Source token IDs
            src_mask: Source padding mask
            
        Returns:
            Encoder 출력
        """
        return self.encoder(src_ids, src_mask)
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder만 실행
        
        Args:
            tgt_ids: Target token IDs
            encoder_output: Encoder 출력
            src_mask: Source padding mask
            tgt_mask: Target padding mask
            
        Returns:
            Decoder 출력 logits
        """
        return self.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
    
    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: int = 1,
        length_penalty: float = 1.0,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        텍스트 생성
        
        Args:
            src_ids: Source token IDs
            start_token_id: 시작 토큰 ID
            end_token_id: 종료 토큰 ID
            max_length: 최대 생성 길이
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            beam_size: Beam search 크기 (1이면 greedy/sampling)
            length_penalty: 길이 패널티
            src_mask: Source padding mask
            
        Returns:
            생성된 시퀀스
        """
        # Encode
        encoder_output = self.encode(src_ids, src_mask)
        
        # Beam search
        if beam_size > 1:
            return self._beam_search(
                encoder_output,
                start_token_id,
                end_token_id,
                max_length,
                beam_size,
                length_penalty,
                src_mask
            )
        
        # Greedy/sampling decoding
        return self.decoder.generate(
            encoder_output,
            start_token_id,
            end_token_id,
            max_length,
            temperature,
            top_k,
            top_p,
            src_mask
        )
    
    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int,
        beam_size: int,
        length_penalty: float,
        src_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Beam search 구현
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # Beam 초기화
        # [batch_size * beam_size, ...]
        encoder_output = encoder_output.repeat_interleave(beam_size, dim=0)
        if src_mask is not None:
            src_mask = src_mask.repeat_interleave(beam_size, dim=0)
        
        # 초기 시퀀스
        sequences = torch.full(
            (batch_size * beam_size, 1),
            start_token_id,
            device=device
        )
        
        # 점수 초기화
        scores = torch.zeros(batch_size * beam_size, device=device)
        scores[beam_size:] = float('-inf')  # 첫 스텝에서는 첫 번째 beam만 사용
        
        # 완료된 시퀀스
        complete_sequences = []
        complete_scores = []
        
        for step in range(max_length - 1):
            # Decode
            logits = self.decode(sequences, encoder_output, src_mask)
            next_token_logits = logits[:, -1, :]  # [batch * beam, vocab]
            
            # Log probabilities
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # 점수 업데이트
            # [batch * beam, vocab]
            next_scores = scores.unsqueeze(1) + log_probs
            
            # Reshape for beam selection
            # [batch, beam * vocab]
            next_scores = next_scores.view(batch_size, -1)
            
            # Top-k beam selection
            # [batch, beam]
            topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)
            
            # 다음 토큰과 beam 인덱스 추출
            next_tokens = topk_indices % self.tgt_vocab_size
            beam_indices = topk_indices // self.tgt_vocab_size
            
            # 시퀀스 업데이트
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            beam_indices = beam_indices + batch_indices * beam_size
            
            sequences = torch.cat([
                sequences[beam_indices.flatten()],
                next_tokens.flatten().unsqueeze(1)
            ], dim=1)
            
            scores = topk_scores.flatten()
            
            # 완료된 시퀀스 확인
            is_complete = (next_tokens.flatten() == end_token_id)
            if is_complete.any():
                # 완료된 시퀀스 저장
                for i, (seq, score) in enumerate(zip(sequences, scores)):
                    if is_complete[i]:
                        # 길이 패널티 적용
                        length = seq.shape[0]
                        normalized_score = score / (length ** length_penalty)
                        complete_sequences.append(seq)
                        complete_scores.append(normalized_score)
                
                # 완료되지 않은 시퀀스만 유지
                incomplete = ~is_complete
                if not incomplete.any():
                    break
                
                sequences = sequences[incomplete]
                scores = scores[incomplete]
                encoder_output = encoder_output[incomplete]
                if src_mask is not None:
                    src_mask = src_mask[incomplete]
        
        # 최고 점수 시퀀스 선택
        if complete_sequences:
            best_idx = torch.tensor(complete_scores).argmax()
            return complete_sequences[best_idx].unsqueeze(0)
        else:
            # 완료된 시퀀스가 없으면 가장 높은 점수의 시퀀스 반환
            return sequences[scores.argmax()].unsqueeze(0)
    
    def get_num_params(self) -> int:
        """모델의 총 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    src_vocab_size: int = 30000,
    tgt_vocab_size: int = 30000,
    max_length: int = 512,
    dropout: float = 0.1,
    activation: str = 'relu',
    norm_first: bool = True,
    share_embeddings: bool = False,
    share_encoder_decoder_embeddings: bool = False
) -> Transformer:
    """
    Transformer 생성 헬퍼 함수
    
    기본값은 Attention is All You Need 논문의 base model 설정입니다.
    """
    return Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_length=max_length,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first,
        share_embeddings=share_embeddings,
        share_encoder_decoder_embeddings=share_encoder_decoder_embeddings
    )


# 사전 정의된 모델 설정
def create_transformer_base(**kwargs) -> Transformer:
    """Base Transformer (논문 기본 설정)"""
    config = {
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1
    }
    config.update(kwargs)
    return create_transformer(**config)


def create_transformer_big(**kwargs) -> Transformer:
    """Big Transformer (논문의 big model)"""
    config = {
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_model': 1024,
        'num_heads': 16,
        'd_ff': 4096,
        'dropout': 0.3
    }
    config.update(kwargs)
    return create_transformer(**config)


def create_transformer_small(**kwargs) -> Transformer:
    """Small Transformer (작은 모델)"""
    config = {
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'd_model': 256,
        'num_heads': 4,
        'd_ff': 1024,
        'dropout': 0.1
    }
    config.update(kwargs)
    return create_transformer(**config)


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Transformer 모델 테스트 ===\n")
    
    # 모델 생성
    model = create_transformer_small(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        max_length=100
    )
    
    # 입력 생성
    batch_size = 2
    src_length = 10
    tgt_length = 8
    
    src_ids = torch.randint(0, 1000, (batch_size, src_length))
    tgt_ids = torch.randint(0, 1000, (batch_size, tgt_length))
    
    # Forward pass
    output = model(src_ids, tgt_ids)
    
    print(f"Source 입력: {src_ids.shape}")
    print(f"Target 입력: {tgt_ids.shape}")
    print(f"출력: {output.shape}")
    print(f"예상 출력: [batch_size={batch_size}, tgt_length={tgt_length}, vocab_size=1000]")
    
    # 파라미터 수
    print(f"\n총 파라미터 수: {model.get_num_params():,}")
    print(f"학습 가능한 파라미터 수: {model.get_num_trainable_params():,}")
    
    # 생성 테스트
    print("\n=== 생성 테스트 ===")
    generated = model.generate(
        src_ids[:1],  # 첫 번째 샘플만
        start_token_id=1,
        end_token_id=2,
        max_length=20,
        temperature=0.8
    )
    print(f"생성된 시퀀스: {generated.shape}")
    print(f"내용: {generated[0].tolist()[:20]}...")