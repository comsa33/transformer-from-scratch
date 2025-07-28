"""
Attention 메커니즘 테스트 및 시각화
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.layers.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention
)
from transformer.utils.masking import create_look_ahead_mask, create_padding_mask


def test_scaled_dot_product_attention():
    """Scaled Dot-Product Attention 테스트"""
    print("=== Scaled Dot-Product Attention 테스트 ===\n")
    
    # 간단한 예제
    batch_size = 1
    seq_len = 4
    d_k = 8
    
    # Q, K, V 생성
    torch.manual_seed(42)
    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_k)
    
    # Attention 계산
    output, weights = scaled_dot_product_attention(query, key, value)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Attention weights 확인
    print(f"\nAttention weights (첫 번째 행):")
    print(weights[0, 0].tolist())
    print(f"합계: {weights[0, 0].sum().item():.6f}")
    
    # 스케일링 효과 확인
    scores_unscaled = torch.matmul(query, key.transpose(-2, -1))
    scale = 1.0 / np.sqrt(d_k)
    scores_scaled = scores_unscaled * scale
    
    print(f"\n스케일링 전 scores 범위: [{scores_unscaled.min():.2f}, {scores_unscaled.max():.2f}]")
    print(f"스케일링 후 scores 범위: [{scores_scaled.min():.2f}, {scores_scaled.max():.2f}]")
    
    return output, weights


def test_multi_head_attention():
    """Multi-Head Attention 테스트"""
    print("\n=== Multi-Head Attention 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 4
    
    # MHA 생성
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    
    # 입력 생성
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = mha(x, x, x, return_attention=True)
    
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # 각 head의 차원 확인
    print(f"\n설정:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k (per head): {mha.d_k}")
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")
    
    # 각 레이어별 파라미터
    for name, param in mha.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,}")
    
    return mha, x, output, attention_weights


def test_attention_with_mask():
    """마스크가 적용된 Attention 테스트"""
    print("\n=== 마스크가 적용된 Attention 테스트 ===\n")
    
    batch_size = 1
    seq_len = 6
    d_model = 32
    num_heads = 2
    
    # Self-Attention 생성
    self_attn = SelfAttention(d_model, num_heads, dropout=0.0)
    
    # 입력 생성
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 마스크 없이
    output_no_mask, weights_no_mask = self_attn(x, return_attention=True)
    
    # 2. Look-ahead 마스크 적용
    look_ahead_mask = create_look_ahead_mask(seq_len)
    look_ahead_mask = look_ahead_mask.expand(batch_size, num_heads, seq_len, seq_len)
    output_causal, weights_causal = self_attn(x, mask=look_ahead_mask, return_attention=True)
    
    print("마스크 없는 경우 - 첫 번째 헤드의 attention weights:")
    print(weights_no_mask[0, 0].detach().numpy().round(3))
    
    print("\nLook-ahead 마스크 적용 - 첫 번째 헤드의 attention weights:")
    print(weights_causal[0, 0].detach().numpy().round(3))
    
    # 차이 확인
    diff = (output_no_mask - output_causal).abs().mean()
    print(f"\n출력 차이 (평균): {diff:.6f}")
    
    return weights_no_mask, weights_causal


def test_cross_attention():
    """Cross-Attention 테스트 (Encoder-Decoder)"""
    print("\n=== Cross-Attention 테스트 ===\n")
    
    batch_size = 2
    src_len = 8  # Encoder 시퀀스 길이
    tgt_len = 6  # Decoder 시퀀스 길이
    d_model = 64
    num_heads = 4
    
    # Cross-Attention 생성
    cross_attn = CrossAttention(d_model, num_heads, dropout=0.0)
    
    # Encoder 출력 (memory)와 Decoder 입력 생성
    memory = torch.randn(batch_size, src_len, d_model)  # Encoder 출력
    query = torch.randn(batch_size, tgt_len, d_model)   # Decoder 입력
    
    # Forward pass
    output, attention_weights = cross_attn(query, memory, return_attention=True)
    
    print(f"Query (Decoder) shape: {query.shape}")
    print(f"Memory (Encoder) shape: {memory.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Attention pattern 확인
    print(f"\nCross-attention pattern:")
    print(f"  각 decoder 위치는 모든 encoder 위치를 참조 가능")
    print(f"  Attention matrix: [{tgt_len} x {src_len}]")
    
    return cross_attn, attention_weights


def visualize_attention_patterns():
    """Attention 패턴 시각화"""
    print("\n=== Attention 패턴 시각화 ===\n")
    
    # 간단한 시퀀스로 테스트
    seq_len = 8
    d_model = 32
    num_heads = 4
    
    # 특별한 패턴을 가진 입력 생성
    x = torch.zeros(1, seq_len, d_model)
    # 각 위치에 고유한 패턴 부여
    for i in range(seq_len):
        x[0, i, i*4:(i+1)*4] = 1.0
    
    # Self-Attention
    self_attn = SelfAttention(d_model, num_heads, dropout=0.0)
    
    # 다양한 마스크 조건에서 attention 계산
    _, weights_no_mask = self_attn(x, return_attention=True)
    
    mask = create_look_ahead_mask(seq_len)
    mask = mask.expand(1, num_heads, seq_len, seq_len)
    _, weights_causal = self_attn(x, mask=mask, return_attention=True)
    
    # 시각화
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 각 head의 attention 패턴 표시
    for head in range(num_heads):
        # 마스크 없는 경우
        ax = axes[0, head]
        sns.heatmap(weights_no_mask[0, head].detach().numpy(), 
                    ax=ax, cmap='Blues', vmin=0, vmax=1,
                    cbar_kws={'label': 'Attention Weight'})
        ax.set_title(f'Head {head+1} (마스크 없음)')
        ax.set_xlabel('Key 위치')
        ax.set_ylabel('Query 위치')
        
        # Causal 마스크 적용
        ax = axes[1, head]
        sns.heatmap(weights_causal[0, head].detach().numpy(), 
                    ax=ax, cmap='Blues', vmin=0, vmax=1,
                    cbar_kws={'label': 'Attention Weight'})
        ax.set_title(f'Head {head+1} (Causal 마스크)')
        ax.set_xlabel('Key 위치')
        ax.set_ylabel('Query 위치')
    
    plt.tight_layout()
    plt.savefig('outputs/attention_patterns.png', dpi=150)
    print("시각화가 'outputs/attention_patterns.png'에 저장되었습니다.")


def test_attention_properties():
    """Attention의 수학적 특성 테스트"""
    print("\n=== Attention 수학적 특성 테스트 ===\n")
    
    d_k = 16
    seq_len = 5
    
    # 1. Permutation equivariance 테스트
    print("1. Permutation Equivariance 테스트:")
    query = torch.randn(1, seq_len, d_k)
    key = torch.randn(1, seq_len, d_k)
    value = torch.randn(1, seq_len, d_k)
    
    # 원본 attention
    out1, _ = scaled_dot_product_attention(query, key, value)
    
    # 순서를 바꾼 후 attention
    perm = torch.tensor([4, 2, 0, 3, 1])
    query_perm = query[:, perm]
    key_perm = key[:, perm]
    value_perm = value[:, perm]
    
    out2, _ = scaled_dot_product_attention(query_perm, key_perm, value_perm)
    
    # 결과를 원래 순서로 복원
    inv_perm = torch.argsort(perm)
    out2_restored = out2[:, inv_perm]
    
    # 차이 확인
    diff = (out1 - out2_restored).abs().max()
    print(f"  Permutation 전후 차이: {diff:.6f}")
    print(f"  {'✅ 통과' if diff < 1e-5 else '❌ 실패'}")
    
    # 2. Attention weights 합이 1인지 확인
    print("\n2. Attention Weights 정규화 테스트:")
    _, weights = scaled_dot_product_attention(query, key, value)
    weight_sums = weights.sum(dim=-1)
    
    print(f"  Weights 합의 평균: {weight_sums.mean():.6f}")
    print(f"  Weights 합의 표준편차: {weight_sums.std():.6f}")
    print(f"  {'✅ 통과' if weight_sums.std() < 1e-5 else '❌ 실패'}")
    
    # 3. Scale factor의 영향
    print("\n3. Scale Factor 영향 분석:")
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    entropies = []
    
    for scale in scales:
        _, weights = scaled_dot_product_attention(query, key, value, scale=scale)
        # Entropy 계산 (분포의 집중도 측정)
        entropy = -(weights * weights.log()).sum(dim=-1).mean()
        entropies.append(entropy.item())
    
    print("  Scale | Entropy")
    print("  ------|--------")
    for scale, entropy in zip(scales, entropies):
        print(f"  {scale:5.1f} | {entropy:7.4f}")
    
    print("\n  (낮은 entropy = 더 집중된 attention)")


def test_gradient_flow():
    """Gradient flow 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")
    
    d_model = 64
    num_heads = 4
    seq_len = 10
    
    # MHA 생성
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 입력
    x = torch.randn(2, seq_len, d_model, requires_grad=True)
    
    # Forward & backward
    output = mha(x, x, x)
    loss = output.mean()
    loss.backward()
    
    # Gradient 통계
    print("입력 gradient:")
    print(f"  Norm: {x.grad.norm():.4f}")
    print(f"  Mean: {x.grad.mean():.6f}")
    print(f"  Std: {x.grad.std():.6f}")
    
    print("\n파라미터 gradients:")
    for name, param in mha.named_parameters():
        if param.grad is not None:
            print(f"  {name}: norm={param.grad.norm():.4f}")


def benchmark_attention():
    """성능 벤치마크"""
    print("\n=== 성능 벤치마크 ===\n")
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 다양한 설정으로 테스트
    configs = [
        (32, 128, 512, 8),   # (batch, seq_len, d_model, heads)
        (32, 256, 512, 8),
        (16, 512, 512, 8),
        (32, 128, 1024, 16),
        (8, 1024, 512, 8),
    ]
    
    for batch_size, seq_len, d_model, num_heads in configs:
        mha = MultiHeadAttention(d_model, num_heads).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Warmup
        for _ in range(10):
            _ = mha(x, x, x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 벤치마크
        start = time.time()
        iterations = 100
        
        for _ in range(iterations):
            output = mha(x, x, x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations * 1000
        
        print(f"Config (B={batch_size}, L={seq_len}, D={d_model}, H={num_heads}):")
        print(f"  평균 시간: {avg_time:.2f} ms/iter")
        print(f"  처리량: {batch_size * seq_len / avg_time:.1f} K tokens/s")


if __name__ == "__main__":
    # 1. Scaled Dot-Product Attention 테스트
    output, weights = test_scaled_dot_product_attention()
    
    # 2. Multi-Head Attention 테스트
    mha, x, output, attention_weights = test_multi_head_attention()
    
    # 3. 마스크 적용 테스트
    weights_no_mask, weights_causal = test_attention_with_mask()
    
    # 4. Cross-Attention 테스트
    cross_attn, cross_weights = test_cross_attention()
    
    # 5. Attention 패턴 시각화
    visualize_attention_patterns()
    
    # 6. 수학적 특성 테스트
    test_attention_properties()
    
    # 7. Gradient flow 테스트
    test_gradient_flow()
    
    # 8. 성능 벤치마크
    benchmark_attention()
    
    print("\n모든 테스트가 완료되었습니다!")