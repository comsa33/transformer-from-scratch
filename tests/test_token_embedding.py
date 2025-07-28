"""
Token Embedding 테스트 및 시각화
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.embeddings.token_embedding import TokenEmbedding, PositionalTokenEmbedding, create_sinusoidal_embeddings


def test_basic_embedding():
    """Token Embedding의 기본 기능을 테스트합니다."""
    print("=== Token Embedding 기본 테스트 ===\n")
    
    # 설정
    vocab_size = 100
    d_model = 64
    batch_size = 3
    seq_length = 10
    padding_idx = 0
    
    # Embedding 생성
    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        padding_idx=padding_idx,
        dropout=0.0
    )
    
    # 테스트 입력 생성
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
    input_ids[0, -2:] = padding_idx  # 패딩 추가
    input_ids[1, -4:] = padding_idx
    
    print(f"입력 shape: {input_ids.shape}")
    print(f"첫 번째 시퀀스: {input_ids[0].tolist()}")
    
    # Forward pass
    output = embedding(input_ids)
    
    print(f"\n출력 shape: {output.shape}")
    print(f"출력 dtype: {output.dtype}")
    
    # 스케일링 확인
    print(f"\n스케일링 factor: {embedding.scale:.4f}")
    print(f"sqrt(d_model): {np.sqrt(d_model):.4f}")
    
    # 패딩 확인
    padding_mask = (input_ids == padding_idx)
    padding_embeddings = output[padding_mask]
    non_padding_embeddings = output[~padding_mask]
    
    print(f"\n패딩 임베딩 평균 norm: {padding_embeddings.norm(dim=-1).mean():.6f}")
    print(f"비패딩 임베딩 평균 norm: {non_padding_embeddings.norm(dim=-1).mean():.4f}")
    
    return embedding, output


def test_weight_sharing():
    """임베딩 가중치 공유 테스트 (Encoder-Decoder 간)"""
    print("\n=== 임베딩 가중치 공유 테스트 ===\n")
    
    vocab_size = 50
    d_model = 32
    
    # 두 개의 임베딩 레이어 생성
    encoder_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    decoder_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    
    # 가중치 공유
    decoder_embedding.set_weight(encoder_embedding.get_weight())
    
    # 같은 입력에 대해 테스트
    test_input = torch.tensor([[1, 2, 3, 4, 5]])
    
    encoder_output = encoder_embedding(test_input)
    decoder_output = decoder_embedding(test_input)
    
    # 동일한지 확인
    is_same = torch.allclose(encoder_output, decoder_output)
    print(f"Encoder와 Decoder 임베딩이 동일한가? {is_same}")
    
    # 가중치 업데이트 후 확인
    with torch.no_grad():
        encoder_embedding.get_weight()[1] += 1.0
    
    encoder_output_updated = encoder_embedding(test_input)
    decoder_output_updated = decoder_embedding(test_input)
    
    is_same_after_update = torch.allclose(encoder_output_updated, decoder_output_updated)
    print(f"Encoder 업데이트 후에도 독립적인가? {not is_same_after_update}")


def test_combined_embedding():
    """Token + Positional Embedding 결합 테스트"""
    print("\n=== Combined Embedding 테스트 ===\n")
    
    vocab_size = 100
    d_model = 128
    max_seq_length = 50
    
    # Combined embedding 생성
    combined_embedding = PositionalTokenEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_length=max_seq_length,
        padding_idx=0,
        dropout=0.0
    )
    
    # 다양한 길이의 시퀀스 테스트
    for seq_len in [10, 25, 50]:
        input_ids = torch.randint(1, vocab_size, (2, seq_len))
        output = combined_embedding(input_ids)
        print(f"시퀀스 길이 {seq_len}: 출력 shape {output.shape}")
    
    # Token embedding만 vs Combined 비교
    token_only = combined_embedding.token_embedding(input_ids)
    combined_output = combined_embedding(input_ids)
    
    diff = (combined_output - token_only).abs().mean()
    print(f"\nPositional encoding에 의한 평균 차이: {diff:.4f}")


def visualize_embedding_distribution():
    """임베딩 분포 시각화"""
    print("\n=== 임베딩 분포 시각화 ===\n")
    
    vocab_sizes = [100, 500, 1000]
    d_model = 128
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, vocab_size in enumerate(vocab_sizes):
        embedding = TokenEmbedding(vocab_size, d_model, scale_embedding=True, dropout=0.0)
        weights = embedding.get_weight().detach().numpy()
        
        # 1. 임베딩 norm 분포
        ax = axes[0, 0]
        norms = np.linalg.norm(weights, axis=1)
        ax.hist(norms, bins=50, alpha=0.7, label=f'vocab={vocab_size}')
        ax.set_xlabel('Embedding Norm')
        ax.set_ylabel('Frequency')
        ax.set_title('임베딩 벡터 Norm 분포')
        ax.legend()
        
        # 2. 임베딩 값 분포
        ax = axes[0, 1]
        ax.hist(weights.flatten(), bins=50, alpha=0.7, label=f'vocab={vocab_size}')
        ax.set_xlabel('Embedding Value')
        ax.set_ylabel('Frequency')
        ax.set_title('임베딩 값 분포')
        ax.legend()
    
    # 3. 특정 임베딩의 차원별 값
    ax = axes[1, 0]
    embedding = TokenEmbedding(1000, d_model, scale_embedding=True, dropout=0.0)
    weights = embedding.get_weight().detach().numpy()
    
    # 처음 10개 토큰의 임베딩 표시
    for i in range(10):
        ax.plot(weights[i, :64], alpha=0.7, label=f'token {i}' if i < 5 else '')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title('처음 10개 토큰의 임베딩 값 (처음 64차원)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 코사인 유사도 히트맵
    ax = axes[1, 1]
    # 처음 20개 토큰 간의 코사인 유사도
    weights_subset = weights[:20]
    weights_norm = weights_subset / (np.linalg.norm(weights_subset, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(weights_norm, weights_norm.T)
    
    im = ax.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Token ID')
    ax.set_title('토큰 간 코사인 유사도 (처음 20개)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('outputs/token_embedding_analysis.png', dpi=150)
    print("시각화가 'outputs/token_embedding_analysis.png'에 저장되었습니다.")


def test_gradient_flow():
    """Gradient flow 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")
    
    vocab_size = 50
    d_model = 32
    
    embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    
    # 간단한 손실 계산
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = embedding(input_ids)
    loss = output.mean()
    
    # Backward
    loss.backward()
    
    # Gradient 확인
    grad_norm = embedding.get_weight().grad.norm()
    print(f"임베딩 가중치의 gradient norm: {grad_norm:.4f}")
    
    # 패딩 인덱스의 gradient는 0이어야 함
    padding_grad = embedding.get_weight().grad[0].norm()
    print(f"패딩 토큰의 gradient norm: {padding_grad:.6f}")
    
    # 특정 토큰들의 gradient
    used_tokens = [1, 2, 3, 4, 5]
    for token_id in used_tokens:
        token_grad_norm = embedding.get_weight().grad[token_id].norm()
        print(f"토큰 {token_id}의 gradient norm: {token_grad_norm:.4f}")


def test_special_initialization():
    """특수 초기화 테스트"""
    print("\n=== 특수 초기화 테스트 ===\n")
    
    # Sinusoidal 초기화 테스트
    special_embeddings = create_sinusoidal_embeddings(10, 64)
    
    print(f"Sinusoidal 임베딩 shape: {special_embeddings.shape}")
    print(f"평균: {special_embeddings.mean():.4f}")
    print(f"표준편차: {special_embeddings.std():.4f}")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.imshow(special_embeddings.numpy().T, aspect='auto', cmap='RdBu')
    plt.xlabel('Special Token ID')
    plt.ylabel('Dimension')
    plt.title('Sinusoidal Pattern으로 초기화된 특수 토큰 임베딩')
    plt.colorbar()
    plt.savefig('outputs/special_token_embeddings.png')
    print("특수 토큰 임베딩이 'outputs/special_token_embeddings.png'에 저장되었습니다.")


if __name__ == "__main__":
    # 1. 기본 테스트
    embedding, output = test_basic_embedding()
    
    # 2. 가중치 공유 테스트
    test_weight_sharing()
    
    # 3. Combined embedding 테스트
    test_combined_embedding()
    
    # 4. 분포 시각화
    visualize_embedding_distribution()
    
    # 5. Gradient flow 테스트
    test_gradient_flow()
    
    # 6. 특수 초기화 테스트
    test_special_initialization()
    
    print("\n모든 테스트가 완료되었습니다!")