"""
Positional Encoding 테스트 및 시각화
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 직접 import하여 순환 참조 회피
from transformer.embeddings.positional import PositionalEncoding, create_sinusoidal_positions


def test_positional_encoding():
    """Positional Encoding의 기본 동작을 테스트합니다."""
    print("=== Positional Encoding 테스트 ===\n")
    
    # 파라미터 설정
    d_model = 128  # 작은 차원으로 테스트
    seq_length = 50
    batch_size = 2
    
    # Positional Encoding 생성
    pe_layer = PositionalEncoding(d_model=d_model, max_seq_length=100, dropout=0.0)
    
    # 더미 입력 생성 (모두 0인 텐서)
    dummy_input = torch.zeros(batch_size, seq_length, d_model)
    
    # Positional encoding 적용
    output = pe_layer(dummy_input)
    
    print(f"입력 shape: {dummy_input.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"출력이 0이 아닌지 확인: {not torch.allclose(output, torch.zeros_like(output))}")
    
    # 첫 번째 배치의 첫 5개 위치, 첫 8개 차원 출력
    print("\n첫 5개 위치의 positional encoding (첫 8차원):")
    print(output[0, :5, :8])
    
    # 각 위치의 norm 계산
    position_norms = torch.norm(output[0], dim=1)
    print(f"\n각 위치의 L2 norm 평균: {position_norms.mean():.4f}")
    print(f"각 위치의 L2 norm 표준편차: {position_norms.std():.4f}")
    
    return output[0].numpy()  # 첫 번째 배치 반환


def visualize_positional_encoding():
    """Positional Encoding의 패턴을 시각화합니다."""
    print("\n=== Positional Encoding 시각화 ===\n")
    
    d_model = 256
    seq_length = 100
    
    # Sinusoidal positions 생성
    pe_values = create_sinusoidal_positions(seq_length, d_model).numpy()
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 처음 몇 개 차원의 sinusoidal 패턴
    ax = axes[0, 0]
    for i in range(0, 8, 2):  # 짝수 차원 (sin)
        ax.plot(pe_values[:50, i], label=f'dim {i} (sin)', alpha=0.8)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Sinusoidal Patterns (Even Dimensions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 홀수 차원 (cosine)
    ax = axes[0, 1]
    for i in range(1, 8, 2):  # 홀수 차원 (cos)
        ax.plot(pe_values[:50, i], label=f'dim {i} (cos)', alpha=0.8)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Cosine Patterns (Odd Dimensions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 전체 패턴 히트맵
    ax = axes[1, 0]
    im = ax.imshow(pe_values[:50, :64].T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Positional Encoding Heatmap (First 64 dims)')
    plt.colorbar(im, ax=ax)
    
    # 4. 다른 주파수 비교
    ax = axes[1, 1]
    positions = [10, 20, 30, 40]
    x = np.arange(d_model)
    for pos in positions:
        ax.plot(x[:64], pe_values[pos, :64], label=f'pos={pos}', alpha=0.8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Encoding Values at Different Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/positional_encoding_analysis.png', dpi=150)
    print("시각화가 'outputs/positional_encoding_analysis.png'에 저장되었습니다.")
    
    return pe_values


def analyze_properties():
    """Positional Encoding의 수학적 특성을 분석합니다."""
    print("\n=== Positional Encoding 특성 분석 ===\n")
    
    d_model = 512
    seq_length = 100
    
    pe_values = create_sinusoidal_positions(seq_length, d_model)
    
    # 1. 직교성 확인 (서로 다른 위치의 내적)
    dot_products = []
    for i in range(10):
        for j in range(i+1, 10):
            dot_product = torch.dot(pe_values[i], pe_values[j])
            dot_products.append(dot_product.item())
    
    print(f"서로 다른 위치 간 내적 평균: {np.mean(dot_products):.6f}")
    print(f"서로 다른 위치 간 내적 표준편차: {np.std(dot_products):.6f}")
    
    # 2. 주기성 확인
    print("\n차원별 주기:")
    for dim in [0, 2, 4, 8, 16, 32]:
        if dim < d_model:
            # 주기 계산 (대략적)
            period = 2 * np.pi * (10000 ** (dim / d_model))
            print(f"  차원 {dim}: 약 {period:.1f} positions")
    
    # 3. 상대적 위치 정보
    print("\n상대적 위치 정보 보존 확인:")
    pos1, pos2 = 20, 25
    pos3, pos4 = 40, 45  # 같은 거리 차이
    
    diff1 = pe_values[pos2] - pe_values[pos1]
    diff2 = pe_values[pos4] - pe_values[pos3]
    
    similarity = torch.cosine_similarity(diff1.unsqueeze(0), diff2.unsqueeze(0))
    print(f"  위치 차이 {pos2-pos1}의 유사도: {similarity.item():.4f}")


if __name__ == "__main__":
    # 1. 기본 테스트
    encoding_output = test_positional_encoding()
    
    # 2. 시각화
    pe_values = visualize_positional_encoding()
    
    # 3. 특성 분석
    analyze_properties()
    
    print("\n모든 테스트가 완료되었습니다!")