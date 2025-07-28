"""
Token Embedding 구현

토큰(단어, 서브워드 등)을 고차원 벡터 공간으로 매핑하는 임베딩 레이어입니다.
Transformer에서는 이 임베딩에 scaling factor를 적용합니다.
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    토큰을 벡터로 변환하는 임베딩 레이어

    Transformer에서는 임베딩 벡터에 sqrt(d_model)을 곱하는 스케일링을 적용합니다.
    이는 positional encoding과의 크기 균형을 맞추기 위함입니다.

    Args:
        vocab_size: 어휘 크기 (토큰의 총 개수)
        d_model: 임베딩 차원 (모델의 hidden dimension)
        padding_idx: 패딩 토큰의 인덱스 (해당 인덱스의 임베딩은 0으로 유지)
        max_norm: 임베딩 벡터의 최대 norm (None이면 제한 없음)
        scale_embedding: sqrt(d_model)로 스케일링 여부 (기본값: True)
        dropout: Dropout 비율 (기본값: 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        scale_embedding: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.scale_embedding = scale_embedding

        # 임베딩 레이어 생성
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            max_norm=max_norm,
            sparse=False,  # Dense gradient 사용
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # 스케일링 factor
        self.scale = math.sqrt(d_model) if scale_embedding else 1.0

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """
        임베딩 가중치 초기화

        일반적으로 [-0.1, 0.1] 범위의 uniform distribution을 사용하거나
        mean=0, std=1/sqrt(d_model)인 normal distribution을 사용합니다.
        """
        # Normal distribution으로 초기화
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model**-0.5)

        # Padding index가 있으면 해당 임베딩을 0으로 설정
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 ID를 임베딩 벡터로 변환

        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_length]

        Returns:
            임베딩 벡터 [batch_size, seq_length, d_model]
        """
        # 임베딩 lookup
        embeddings = self.embedding(input_ids)

        # 스케일링 적용
        if self.scale_embedding:
            embeddings = embeddings * self.scale

        # Dropout 적용
        embeddings = self.dropout(embeddings)

        return embeddings

    def get_weight(self) -> torch.Tensor:
        """
        임베딩 가중치 행렬 반환

        Returns:
            임베딩 가중치 [vocab_size, d_model]
        """
        return self.embedding.weight

    def set_weight(self, weight: torch.Tensor):
        """
        임베딩 가중치 설정 (사전 학습된 임베딩 로드 시 사용)

        Args:
            weight: 새로운 가중치 텐서 [vocab_size, d_model]
        """
        with torch.no_grad():
            self.embedding.weight.copy_(weight)

            # Padding index가 있으면 해당 임베딩을 0으로 재설정
            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].fill_(0)

    def freeze(self):
        """임베딩 가중치를 고정 (학습하지 않음)"""
        self.embedding.weight.requires_grad = False

    def unfreeze(self):
        """임베딩 가중치를 학습 가능하게 설정"""
        self.embedding.weight.requires_grad = True

    def get_input_embeddings(self):
        """Hugging Face 스타일 호환성을 위한 메서드"""
        return self.embedding

    def set_input_embeddings(self, value):
        """Hugging Face 스타일 호환성을 위한 메서드"""
        self.embedding = value


class PositionalTokenEmbedding(nn.Module):
    """
    Token Embedding과 Positional Encoding을 결합한 클래스

    많은 구현에서 이 두 가지를 함께 처리하므로, 편의를 위해 제공합니다.

    Args:
        vocab_size: 어휘 크기
        d_model: 모델 차원
        max_seq_length: 최대 시퀀스 길이
        padding_idx: 패딩 토큰 인덱스
        dropout: Dropout 비율
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_length: int = 5000,
        padding_idx: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
            scale_embedding=True,
            dropout=0.0,  # Dropout은 나중에 한 번만 적용
        )

        # Positional encoding (이미 구현한 것을 import)
        from .positional import PositionalEncoding

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout,  # 여기서 dropout 적용
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 ID를 임베딩하고 positional encoding을 더함

        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_length]

        Returns:
            Position이 인코딩된 임베딩 [batch_size, seq_length, d_model]
        """
        # Token embedding
        token_embeddings = self.token_embedding(input_ids)

        # Positional encoding 추가
        embeddings = self.positional_encoding(token_embeddings)

        return embeddings


def create_sinusoidal_embeddings(num_embeddings: int, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal 패턴으로 초기화된 임베딩 생성 (특수 토큰용)

    일부 모델에서는 특수 토큰(CLS, SEP 등)에 대해
    sinusoidal 패턴으로 초기화하기도 합니다.

    Args:
        num_embeddings: 임베딩 개수
        embedding_dim: 임베딩 차원

    Returns:
        Sinusoidal 임베딩 텐서 [num_embeddings, embedding_dim]
    """
    embeddings = torch.zeros(num_embeddings, embedding_dim)
    position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)
    )

    embeddings[:, 0::2] = torch.sin(position * div_term)
    if embedding_dim % 2 == 0:
        embeddings[:, 1::2] = torch.cos(position * div_term)
    else:
        embeddings[:, 1::2] = torch.cos(position * div_term[:-1])

    return embeddings


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Token Embedding 테스트 ===\n")

    # 파라미터
    vocab_size = 1000
    d_model = 128
    batch_size = 2
    seq_length = 10

    # Token Embedding 생성
    embedding_layer = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        padding_idx=0,
        dropout=0.0,  # 0을 패딩 토큰으로 사용
    )

    # 랜덤 입력 생성 (1부터 vocab_size-1 사이의 값)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
    # 일부 패딩 추가
    input_ids[0, -3:] = 0  # 첫 번째 샘플의 마지막 3개를 패딩으로

    print(f"입력 토큰 ID shape: {input_ids.shape}")
    print(f"입력 예시:\n{input_ids}\n")

    # Forward pass
    output = embedding_layer(input_ids)
    print(f"출력 shape: {output.shape}")
    print(f"출력 통계 - mean: {output.mean():.4f}, std: {output.std():.4f}")

    # 패딩 위치 확인
    padding_positions = input_ids == 0
    padding_embeddings = output[padding_positions]
    print(f"\n패딩 임베딩의 norm: {padding_embeddings.norm(dim=-1).mean():.4f}")

    # 스케일링 확인
    print(f"\n스케일링 factor: {embedding_layer.scale:.4f}")
    print(f"sqrt(d_model): {math.sqrt(d_model):.4f}")

    # Combined embedding 테스트
    print("\n=== Combined Token + Positional Embedding 테스트 ===\n")

    combined_embedding = PositionalTokenEmbedding(
        vocab_size=vocab_size, d_model=d_model, max_seq_length=100, padding_idx=0
    )

    combined_output = combined_embedding(input_ids)
    print(f"Combined 출력 shape: {combined_output.shape}")
    print(
        f"Combined 출력 통계 - mean: {combined_output.mean():.4f}, std: {combined_output.std():.4f}"
    )
