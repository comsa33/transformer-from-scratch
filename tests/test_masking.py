"""
Masking Utilities 테스트 및 시각화
"""

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.utils.masking import (
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask,
    create_cross_attention_mask,
    apply_mask,
    create_attention_mask,
    expand_mask
)


def test_padding_mask():
    """패딩 마스크 테스트"""
    print("=== 패딩 마스크 테스트 ===\n")
    
    # 테스트 시퀀스
    seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],  # 5개 토큰 + 3개 패딩
        [1, 2, 0, 0, 0, 0, 0, 0],  # 2개 토큰 + 6개 패딩
        [1, 2, 3, 4, 5, 6, 7, 8]   # 패딩 없음
    ])
    
    print(f"입력 시퀀스:\n{seq}\n")
    
    # 패딩 마스크 생성
    mask = create_padding_mask(seq, pad_idx=0)
    
    print(f"마스크 shape: {mask.shape}")
    print(f"마스크 (batch_size x 1 x 1 x seq_len):")
    
    for i in range(seq.shape[0]):
        print(f"  샘플 {i}: {mask[i, 0, 0].tolist()}")
    
    # 마스크 적용 효과 확인
    print("\n마스크 적용 효과:")
    scores = torch.ones(3, 1, 8, 8)
    masked_scores = apply_mask(scores, mask)
    
    # Softmax 적용 후 확인
    attention_weights = F.softmax(masked_scores, dim=-1)
    
    print(f"샘플 0의 첫 번째 행 attention weights:")
    print(f"  {attention_weights[0, 0, 0].tolist()}")
    print(f"  (패딩 위치의 가중치가 0에 가까움)")
    
    return seq, mask


def test_look_ahead_mask():
    """Look-ahead 마스크 테스트"""
    print("\n=== Look-ahead 마스크 테스트 ===\n")
    
    size = 6
    mask = create_look_ahead_mask(size)
    
    print(f"마스크 shape: {mask.shape}")
    print(f"Look-ahead 마스크 ({size}x{size}):")
    print(mask[0, 0].int())
    
    # 마스크 의미 설명
    print("\n각 위치에서 볼 수 있는 토큰:")
    for i in range(size):
        visible = [j for j in range(size) if mask[0, 0, i, j] == 0]
        print(f"  위치 {i}: {visible}")
    
    return mask


def test_combined_mask():
    """결합된 마스크 테스트"""
    print("\n=== 결합된 마스크 테스트 ===\n")
    
    # 패딩이 있는 시퀀스
    seq = torch.tensor([
        [1, 2, 3, 4, 0, 0],
        [1, 2, 3, 4, 5, 6]  # 패딩 없음
    ])
    
    combined_mask = create_combined_mask(seq, pad_idx=0)
    
    print(f"입력 시퀀스:\n{seq}\n")
    print(f"결합된 마스크 shape: {combined_mask.shape}")
    
    # 각 샘플의 마스크 표시
    for i in range(seq.shape[0]):
        print(f"\n샘플 {i} 마스크:")
        print(combined_mask[i, 0].int())
        
        # 마스크 해석
        if i == 0:
            print("  - 대각선 위: Look-ahead 마스크")
            print("  - 마지막 2열: 패딩 마스크")
    
    return seq, combined_mask


def test_cross_attention_mask():
    """Cross-attention 마스크 테스트"""
    print("\n=== Cross-attention 마스크 테스트 ===\n")
    
    # Encoder와 Decoder 시퀀스
    source_seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],  # Encoder: 5개 토큰
        [1, 2, 3, 0, 0, 0, 0, 0]   # Encoder: 3개 토큰
    ])
    
    target_seq = torch.tensor([
        [1, 2, 3, 0, 0, 0],  # Decoder: 3개 토큰
        [1, 2, 3, 4, 5, 6]   # Decoder: 6개 토큰
    ])
    
    target_mask, source_mask = create_cross_attention_mask(target_seq, source_seq)
    
    print(f"Source 시퀀스 (Encoder):\n{source_seq}")
    print(f"Target 시퀀스 (Decoder):\n{target_seq}\n")
    
    print(f"Source 마스크: {source_mask[0, 0, 0].tolist()}")
    print(f"Target 마스크: {target_mask[0, 0, 0].tolist()}")
    
    return source_seq, target_seq, source_mask, target_mask


def visualize_masks():
    """마스크 시각화"""
    print("\n=== 마스크 시각화 ===\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 패딩 마스크
    seq = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    padding_mask = create_padding_mask(seq)
    
    ax = axes[0, 0]
    im = ax.imshow(padding_mask[0, 0, 0:1], cmap='RdBu', vmin=0, vmax=1)
    ax.set_title('패딩 마스크')
    ax.set_xlabel('시퀀스 위치')
    ax.set_yticks([])
    ax.set_xticks(range(8))
    ax.set_xticklabels(['1', '2', '3', '4', '5', 'PAD', 'PAD', 'PAD'])
    
    # 2. Look-ahead 마스크
    look_ahead = create_look_ahead_mask(8)
    
    ax = axes[0, 1]
    im = ax.imshow(look_ahead[0, 0], cmap='RdBu', vmin=0, vmax=1)
    ax.set_title('Look-ahead 마스크')
    ax.set_xlabel('Key 위치')
    ax.set_ylabel('Query 위치')
    
    # 3. 결합된 마스크
    combined = create_combined_mask(seq)
    
    ax = axes[0, 2]
    im = ax.imshow(combined[0, 0], cmap='RdBu', vmin=0, vmax=1)
    ax.set_title('결합된 마스크 (Decoder Self-Attention)')
    ax.set_xlabel('Key 위치')
    ax.set_ylabel('Query 위치')
    
    # 4. Attention scores (마스크 적용 전)
    scores = torch.randn(1, 1, 8, 8) * 2
    
    ax = axes[1, 0]
    im = ax.imshow(scores[0, 0], cmap='coolwarm', vmin=-3, vmax=3)
    ax.set_title('Attention Scores (마스크 적용 전)')
    ax.set_xlabel('Key 위치')
    ax.set_ylabel('Query 위치')
    plt.colorbar(im, ax=ax)
    
    # 5. Attention scores (마스크 적용 후)
    masked_scores = apply_mask(scores, combined)
    
    ax = axes[1, 1]
    im = ax.imshow(masked_scores[0, 0], cmap='coolwarm', vmin=-10, vmax=3)
    ax.set_title('Attention Scores (마스크 적용 후)')
    ax.set_xlabel('Key 위치')
    ax.set_ylabel('Query 위치')
    plt.colorbar(im, ax=ax)
    
    # 6. Attention weights (Softmax 후)
    attention_weights = F.softmax(masked_scores, dim=-1)
    
    ax = axes[1, 2]
    im = ax.imshow(attention_weights[0, 0], cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Attention Weights (Softmax 후)')
    ax.set_xlabel('Key 위치')
    ax.set_ylabel('Query 위치')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('outputs/masking_visualization.png', dpi=150)
    print("시각화가 'outputs/masking_visualization.png'에 저장되었습니다.")


def test_mask_expansion():
    """마스크 확장 테스트"""
    print("\n=== 마스크 확장 테스트 ===\n")
    
    # 2D 마스크
    mask_2d = torch.ones(4, 4)
    expanded = expand_mask(mask_2d, batch_size=2, num_heads=8, tgt_len=4, src_len=4)
    print(f"2D 마스크 {mask_2d.shape} -> {expanded.shape}")
    
    # 3D 마스크
    mask_3d = torch.ones(2, 4, 4)
    expanded = expand_mask(mask_3d, batch_size=2, num_heads=8, tgt_len=4, src_len=4)
    print(f"3D 마스크 {mask_3d.shape} -> {expanded.shape}")
    
    # 4D 마스크 (이미 올바른 shape)
    mask_4d = torch.ones(2, 8, 4, 4)
    expanded = expand_mask(mask_4d, batch_size=2, num_heads=8, tgt_len=4, src_len=4)
    print(f"4D 마스크 {mask_4d.shape} -> {expanded.shape}")


def test_mask_effectiveness():
    """마스크 효과성 테스트"""
    print("\n=== 마스크 효과성 테스트 ===\n")
    
    # 시퀀스와 마스크 생성
    seq = torch.tensor([[1, 2, 3, 0, 0, 0]])
    mask = create_combined_mask(seq)
    
    # 랜덤 attention scores
    scores = torch.randn(1, 1, 6, 6)
    
    # 마스크 적용
    masked_scores = apply_mask(scores, mask)
    
    # Softmax
    weights = F.softmax(masked_scores, dim=-1)
    
    print("각 위치의 attention 분포:")
    for i in range(6):
        weight_dist = weights[0, 0, i].tolist()
        print(f"  위치 {i}: {[f'{w:.3f}' for w in weight_dist]}")
        
        # 검증
        if i < 3:  # 실제 토큰
            # Look-ahead: i+1 위치까지만 0이 아님
            assert all(w < 0.001 for w in weight_dist[i+1:]), f"Look-ahead 실패 at {i}"
            # 패딩: 위치 3,4,5는 0
            assert all(weight_dist[j] < 0.001 for j in [3, 4, 5]), f"패딩 마스크 실패 at {i}"
    
    print("\n✅ 모든 마스크가 올바르게 작동합니다!")


def analyze_computational_impact():
    """마스크의 계산 영향 분석"""
    print("\n=== 계산 영향 분석 ===\n")
    
    import time
    
    sizes = [64, 128, 256, 512]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}\n")
    
    for size in sizes:
        # 마스크 생성 시간
        start = time.time()
        for _ in range(100):
            mask = create_look_ahead_mask(size, device=device)
        mask_time = (time.time() - start) / 100 * 1000
        
        # 마스크 적용 시간
        scores = torch.randn(32, 8, size, size, device=device)
        mask = create_look_ahead_mask(size, device=device)
        mask = mask.expand(32, 8, size, size)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            masked = apply_mask(scores, mask)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        apply_time = (time.time() - start) / 100 * 1000
        
        print(f"Size {size}x{size}:")
        print(f"  마스크 생성: {mask_time:.3f} ms")
        print(f"  마스크 적용: {apply_time:.3f} ms")


if __name__ == "__main__":
    # 1. 패딩 마스크 테스트
    seq, padding_mask = test_padding_mask()
    
    # 2. Look-ahead 마스크 테스트
    look_ahead_mask = test_look_ahead_mask()
    
    # 3. 결합된 마스크 테스트
    seq, combined_mask = test_combined_mask()
    
    # 4. Cross-attention 마스크 테스트
    source_seq, target_seq, source_mask, target_mask = test_cross_attention_mask()
    
    # 5. 마스크 시각화
    visualize_masks()
    
    # 6. 마스크 확장 테스트
    test_mask_expansion()
    
    # 7. 마스크 효과성 테스트
    test_mask_effectiveness()
    
    # 8. 계산 영향 분석
    analyze_computational_impact()
    
    print("\n모든 테스트가 완료되었습니다!")