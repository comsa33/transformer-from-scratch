"""
Masking Utilities for Transformer

Transformer에서 사용하는 다양한 마스킹 기법들을 구현합니다.
- Padding Mask: 패딩 토큰을 무시하기 위한 마스크
- Look-ahead Mask: Decoder에서 미래 정보를 차단하기 위한 마스크
- Combined Mask: 두 마스크를 결합한 형태
"""

import torch


def create_padding_mask(
    seq: torch.Tensor, pad_idx: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    패딩 마스크 생성

    패딩 토큰 위치에 1, 실제 토큰 위치에 0을 가지는 마스크를 생성합니다.
    Attention 계산 시 패딩 위치에 -inf를 더해 softmax 후 0이 되도록 합니다.

    Args:
        seq: 토큰 ID 시퀀스 [batch_size, seq_len]
        pad_idx: 패딩 토큰의 인덱스 (기본값: 0)
        dtype: 출력 마스크의 데이터 타입

    Returns:
        패딩 마스크 [batch_size, 1, 1, seq_len]
        - 1: 패딩 위치 (마스킹됨)
        - 0: 실제 토큰 위치
    """
    # 패딩 토큰 위치 찾기
    mask = (seq == pad_idx).to(dtype)

    # Attention 계산을 위한 shape 변경
    # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
    # 이렇게 하면 [batch_size, num_heads, seq_len, seq_len]과 브로드캐스팅 가능
    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask


def create_look_ahead_mask(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device | None = None
) -> torch.Tensor:
    """
    Look-ahead 마스크 생성 (Causal Mask)

    Decoder self-attention에서 미래 위치의 정보를 보지 못하도록 하는 마스크입니다.
    각 위치에서 자신과 이전 위치들만 볼 수 있도록 제한합니다.

    Args:
        size: 시퀀스 길이
        dtype: 출력 마스크의 데이터 타입
        device: 텐서를 생성할 디바이스

    Returns:
        Look-ahead 마스크 [1, 1, size, size]
        - 상삼각 행렬 (대각선 포함 아래는 0, 위는 1)
    """
    # 상삼각 행렬 생성 (대각선 위가 1)
    mask = torch.triu(torch.ones(size, size, dtype=dtype, device=device), diagonal=1)

    # Broadcasting을 위한 shape 변경
    # [size, size] -> [1, 1, size, size]
    mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


def create_combined_mask(
    seq: torch.Tensor, pad_idx: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    패딩 마스크와 look-ahead 마스크를 결합

    Decoder self-attention에서 사용하는 마스크로,
    패딩 토큰과 미래 위치를 모두 마스킹합니다.

    Args:
        seq: 토큰 ID 시퀀스 [batch_size, seq_len]
        pad_idx: 패딩 토큰의 인덱스
        dtype: 출력 마스크의 데이터 타입

    Returns:
        결합된 마스크 [batch_size, 1, seq_len, seq_len]
    """
    batch_size, seq_len = seq.shape

    # 패딩 마스크 생성
    padding_mask = create_padding_mask(seq, pad_idx, dtype)

    # Look-ahead 마스크 생성
    look_ahead_mask = create_look_ahead_mask(seq_len, dtype, seq.device)

    # 두 마스크 결합 (둘 중 하나라도 1이면 1)
    # padding_mask: [batch_size, 1, 1, seq_len]
    # look_ahead_mask: [1, 1, seq_len, seq_len]
    # 결과: [batch_size, 1, seq_len, seq_len]
    combined_mask = torch.maximum(padding_mask, look_ahead_mask)

    return combined_mask


def create_cross_attention_mask(
    target_seq: torch.Tensor,
    source_seq: torch.Tensor,
    pad_idx: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-attention을 위한 마스크 생성

    Decoder의 encoder-decoder attention에서 사용하는 마스크입니다.
    Target(decoder)과 Source(encoder) 시퀀스의 패딩을 처리합니다.

    Args:
        target_seq: Decoder 입력 시퀀스 [batch_size, target_len]
        source_seq: Encoder 입력 시퀀스 [batch_size, source_len]
        pad_idx: 패딩 토큰의 인덱스
        dtype: 출력 마스크의 데이터 타입

    Returns:
        (target_mask, source_mask) 튜플
        - target_mask: [batch_size, 1, 1, target_len]
        - source_mask: [batch_size, 1, 1, source_len]
    """
    target_mask = create_padding_mask(target_seq, pad_idx, dtype)
    source_mask = create_padding_mask(source_seq, pad_idx, dtype)

    return target_mask, source_mask


def apply_mask(
    scores: torch.Tensor, mask: torch.Tensor | None = None, mask_value: float = -1e9
) -> torch.Tensor:
    """
    Attention scores에 마스크 적용

    마스크가 1인 위치에 매우 작은 값을 더해 softmax 후 0에 가깝게 만듭니다.

    Args:
        scores: Attention scores [..., seq_len, seq_len]
        mask: 마스크 텐서 (1: 마스킹, 0: 유지)
        mask_value: 마스킹할 위치에 더할 값 (기본값: -1e9)

    Returns:
        마스크가 적용된 scores
    """
    if mask is None:
        return scores

    # mask를 scores와 같은 dtype으로 변환
    mask = mask.to(scores.dtype)

    # 마스크가 1인 위치에 mask_value 더하기
    # scores와 mask의 shape이 브로드캐스팅 가능해야 함
    masked_scores = scores + (mask * mask_value)

    return masked_scores


def create_attention_mask(
    query_len: int,
    key_len: int,
    is_causal: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """
    일반적인 attention 마스크 생성

    Args:
        query_len: Query 시퀀스 길이
        key_len: Key 시퀀스 길이
        is_causal: Causal mask 여부
        device: 텐서 디바이스
        dtype: 데이터 타입

    Returns:
        Attention 마스크 또는 None
    """
    if not is_causal:
        return None

    # Causal mask는 query_len x key_len 크기
    # 일반적으로 self-attention에서는 query_len == key_len
    if query_len == key_len:
        return create_look_ahead_mask(query_len, dtype, device)
    else:
        # Cross-attention이나 다른 경우
        mask = torch.triu(
            torch.ones(query_len, key_len, dtype=dtype, device=device),
            diagonal=key_len - query_len + 1,
        )
        return mask.unsqueeze(0).unsqueeze(0)


def expand_mask(
    mask: torch.Tensor, batch_size: int, num_heads: int, tgt_len: int, src_len: int | None = None
) -> torch.Tensor:
    """
    마스크를 multi-head attention에 맞게 확장

    Args:
        mask: 원본 마스크
        batch_size: 배치 크기
        num_heads: Attention head 수
        tgt_len: Target 길이
        src_len: Source 길이 (None이면 tgt_len과 동일)

    Returns:
        확장된 마스크 [batch_size, num_heads, tgt_len, src_len]
    """
    if src_len is None:
        src_len = tgt_len

    # 현재 마스크의 차원 수에 따라 처리
    if mask.dim() == 2:
        # [tgt_len, src_len] -> [batch_size, num_heads, tgt_len, src_len]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, num_heads, tgt_len, src_len)
    elif mask.dim() == 3:
        # [batch_size, tgt_len, src_len] -> [batch_size, num_heads, tgt_len, src_len]
        mask = mask.unsqueeze(1)
        mask = mask.expand(batch_size, num_heads, tgt_len, src_len)
    elif mask.dim() == 4:
        # 이미 올바른 shape
        pass
    else:
        raise ValueError(f"Mask has unexpected number of dimensions: {mask.dim()}")

    return mask


def generate_square_subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """
    PyTorch의 nn.Transformer와 호환되는 정사각형 subsequent 마스크 생성

    Args:
        size: 시퀀스 길이
        device: 텐서 디바이스

    Returns:
        마스크 [size, size] (True: 마스킹, False: 유지)
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask.bool()


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Masking Utilities 테스트 ===\n")

    # 테스트 시퀀스 (0은 패딩)
    batch_size = 2
    seq_length = 8
    pad_idx = 0

    # 패딩이 포함된 시퀀스 생성
    seq = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],  # 5개 토큰 + 3개 패딩
            [1, 2, 3, 0, 0, 0, 0, 0],  # 3개 토큰 + 5개 패딩
        ]
    )

    print(f"입력 시퀀스:\n{seq}\n")

    # 1. 패딩 마스크
    padding_mask = create_padding_mask(seq, pad_idx)
    print(f"패딩 마스크 shape: {padding_mask.shape}")
    print(f"패딩 마스크 (첫 번째 샘플):\n{padding_mask[0, 0, 0]}\n")

    # 2. Look-ahead 마스크
    look_ahead_mask = create_look_ahead_mask(seq_length)
    print(f"Look-ahead 마스크 shape: {look_ahead_mask.shape}")
    print(f"Look-ahead 마스크:\n{look_ahead_mask[0, 0]}\n")

    # 3. 결합된 마스크
    combined_mask = create_combined_mask(seq, pad_idx)
    print(f"결합된 마스크 shape: {combined_mask.shape}")
    print(f"결합된 마스크 (첫 번째 샘플):\n{combined_mask[0, 0]}\n")

    # 4. 마스크 적용 테스트
    scores = torch.randn(batch_size, 1, seq_length, seq_length)
    masked_scores = apply_mask(scores, combined_mask)

    print(f"마스크 적용 전 scores 범위: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"마스크 적용 후 scores 범위: [{masked_scores.min():.2f}, {masked_scores.max():.2f}]")
