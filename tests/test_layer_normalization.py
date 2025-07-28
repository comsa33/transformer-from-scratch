"""
Layer Normalization 테스트 및 시각화
"""

import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from transformer.layers.normalization import (
    LayerNormalization,
    PostNormalization,
    PreNormalization,
    RMSNorm,
)


def test_basic_layer_norm():
    """Layer Normalization의 기본 동작을 테스트합니다."""
    print("=== Layer Normalization 기본 테스트 ===\n")

    # 설정
    batch_size = 3
    seq_length = 4
    d_model = 64

    # 테스트 입력 생성
    x = torch.randn(batch_size, seq_length, d_model)

    # Layer Norm 생성
    layer_norm = LayerNormalization(d_model)

    # Forward pass
    output = layer_norm(x)

    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")

    # 각 위치에서의 통계 확인
    for b in range(batch_size):
        for s in range(seq_length):
            mean = output[b, s].mean().item()
            std = output[b, s].std(unbiased=False).item()
            print(f"배치 {b}, 위치 {s}: mean={mean:.6f}, std={std:.6f}")

    # 평균적인 통계
    overall_mean = output.view(-1, d_model).mean(dim=1).mean().item()
    overall_std = output.view(-1, d_model).std(dim=1, unbiased=False).mean().item()

    print(f"\n전체 평균의 평균: {overall_mean:.6f}")
    print(f"전체 표준편차의 평균: {overall_std:.6f}")

    return x, output


def test_gradient_flow():
    """Gradient flow 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")

    d_model = 32
    x = torch.randn(2, 5, d_model, requires_grad=True)

    # Layer Norm
    layer_norm = LayerNormalization(d_model)
    output = layer_norm(x)

    # 간단한 손실
    loss = output.mean()
    loss.backward()

    print(f"입력 gradient norm: {x.grad.norm():.4f}")
    print(f"Gamma gradient norm: {layer_norm.gamma.grad.norm():.4f}")
    print(f"Beta gradient norm: {layer_norm.beta.grad.norm():.4f}")

    # Gradient의 분포
    print("\n입력 gradient 통계:")
    print(f"  평균: {x.grad.mean():.6f}")
    print(f"  표준편차: {x.grad.std():.6f}")
    print(f"  최대값: {x.grad.max():.6f}")
    print(f"  최소값: {x.grad.min():.6f}")


def compare_normalization_methods():
    """Layer Norm vs RMS Norm 비교"""
    print("\n=== Normalization 방법 비교 ===\n")

    d_model = 128
    x = torch.randn(1, 10, d_model)

    # 각 정규화 방법 적용
    layer_norm = LayerNormalization(d_model)
    rms_norm = RMSNorm(d_model)

    ln_output = layer_norm(x)
    rms_output = rms_norm(x)

    # 통계 비교
    print("Layer Normalization:")
    print(f"  출력 평균: {ln_output.mean(dim=-1).mean():.6f}")
    print(f"  출력 표준편차: {ln_output.std(dim=-1, unbiased=False).mean():.6f}")

    print("\nRMS Normalization:")
    print(f"  출력 평균: {rms_output.mean(dim=-1).mean():.6f}")
    print(f"  출력 표준편차: {rms_output.std(dim=-1, unbiased=False).mean():.6f}")

    # 차이 분석
    diff = (ln_output - rms_output).abs()
    print(f"\n두 방법의 평균 차이: {diff.mean():.6f}")
    print(f"최대 차이: {diff.max():.6f}")

    return x, ln_output, rms_output


def test_pre_post_normalization():
    """Pre-Norm vs Post-Norm 비교"""
    print("\n=== Pre-Norm vs Post-Norm 테스트 ===\n")

    d_model = 64

    # 간단한 sub-layer (선형 변환)
    class SimpleSubLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)

        def forward(self, x):
            return self.linear(x)

    # Pre-Norm과 Post-Norm 생성
    sublayer1 = SimpleSubLayer(d_model)
    sublayer2 = SimpleSubLayer(d_model)

    # 같은 가중치로 초기화
    sublayer2.load_state_dict(sublayer1.state_dict())

    pre_norm = PreNormalization(d_model, sublayer1, dropout=0.0)
    post_norm = PostNormalization(d_model, sublayer2, dropout=0.0)

    # 테스트 입력
    x = torch.randn(2, 5, d_model)

    # Forward pass
    pre_output = pre_norm(x)
    post_output = post_norm(x)

    print(f"입력 norm: {x.norm():.4f}")
    print(f"Pre-Norm 출력 norm: {pre_output.norm():.4f}")
    print(f"Post-Norm 출력 norm: {post_output.norm():.4f}")

    # Residual connection 강도
    pre_residual = (pre_output - x).norm() / x.norm()
    post_residual = (post_output - x).norm() / x.norm()

    print(f"\nPre-Norm residual 상대 강도: {pre_residual:.4f}")
    print(f"Post-Norm residual 상대 강도: {post_residual:.4f}")


def visualize_normalization_effects():
    """정규화 효과 시각화"""
    print("\n=== 정규화 효과 시각화 ===\n")

    # 다양한 분포의 입력 생성
    d_model = 64

    # 1. 정규 분포
    x_normal = torch.randn(1, 100, d_model)

    # 2. 균등 분포
    x_uniform = torch.rand(1, 100, d_model) * 2 - 1

    # 3. 편향된 분포
    x_skewed = torch.randn(1, 100, d_model) ** 3

    # 4. 큰 분산
    x_large_var = torch.randn(1, 100, d_model) * 10

    inputs = [
        ("정규 분포", x_normal),
        ("균등 분포", x_uniform),
        ("편향 분포", x_skewed),
        ("큰 분산", x_large_var),
    ]

    layer_norm = LayerNormalization(d_model)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, x) in enumerate(inputs):
        # 원본 분포
        ax = axes[0, idx]
        ax.hist(x.flatten().numpy(), bins=50, alpha=0.7, color="blue")
        ax.set_title(f"{name} - 원본")
        ax.set_xlabel("값")
        ax.set_ylabel("빈도")
        ax.set_ylim(0, 1500)

        # 정규화 후
        output = layer_norm(x)
        ax = axes[1, idx]
        ax.hist(output.flatten().detach().numpy(), bins=50, alpha=0.7, color="green")
        ax.set_title(f"{name} - Layer Norm 후")
        ax.set_xlabel("값")
        ax.set_ylabel("빈도")
        ax.set_ylim(0, 1500)

        # 통계 표시
        mean = output.mean(dim=-1).mean().item()
        std = output.std(dim=-1, unbiased=False).mean().item()
        ax.text(
            0.02,
            0.95,
            f"평균: {mean:.3f}\n표준편차: {std:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    plt.tight_layout()
    plt.savefig("outputs/layer_norm_effects.png", dpi=150)
    print("시각화가 'outputs/layer_norm_effects.png'에 저장되었습니다.")


def test_numerical_stability():
    """수치 안정성 테스트"""
    print("\n=== 수치 안정성 테스트 ===\n")

    d_model = 32

    # 극단적인 값들
    test_cases = [
        ("매우 작은 값", torch.ones(1, 5, d_model) * 1e-8),
        ("매우 큰 값", torch.ones(1, 5, d_model) * 1e8),
        ("0에 가까운 분산", torch.ones(1, 5, d_model) + torch.randn(1, 5, d_model) * 1e-8),
        ("NaN 포함", torch.randn(1, 5, d_model)),
    ]

    # NaN 추가
    test_cases[3][1][0, 0, 0] = float("nan")

    layer_norm = LayerNormalization(d_model)

    for name, x in test_cases:
        try:
            output = layer_norm(x)
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            print(f"{name}:")
            print(f"  입력 범위: [{x.min():.2e}, {x.max():.2e}]")
            print(f"  출력 범위: [{output.min():.2e}, {output.max():.2e}]")
            print(f"  NaN 포함: {has_nan}")
            print(f"  Inf 포함: {has_inf}")

        except Exception as e:
            print(f"{name}: 오류 발생 - {e}")


def benchmark_performance():
    """성능 벤치마크"""
    print("\n=== 성능 벤치마크 ===\n")

    import time

    # 다양한 크기로 테스트
    sizes = [
        (32, 128, 512),  # 작은 배치
        (128, 128, 512),  # 중간 배치
        (256, 128, 512),  # 큰 배치
        (32, 512, 512),  # 긴 시퀀스
        (32, 128, 2048),  # 큰 차원
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for batch_size, seq_length, d_model in sizes:
        x = torch.randn(batch_size, seq_length, d_model, device=device)
        layer_norm = LayerNormalization(d_model).to(device)

        # Warmup
        for _ in range(10):
            _ = layer_norm(x)

        # 벤치마크
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        iterations = 100

        for _ in range(iterations):
            layer_norm(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations * 1000  # ms

        print(f"Shape {x.shape}: {avg_time:.3f} ms/iter")


if __name__ == "__main__":
    # 1. 기본 테스트
    x, output = test_basic_layer_norm()

    # 2. Gradient flow 테스트
    test_gradient_flow()

    # 3. 정규화 방법 비교
    x, ln_out, rms_out = compare_normalization_methods()

    # 4. Pre/Post Norm 비교
    test_pre_post_normalization()

    # 5. 효과 시각화
    visualize_normalization_effects()

    # 6. 수치 안정성
    test_numerical_stability()

    # 7. 성능 벤치마크
    benchmark_performance()

    print("\n모든 테스트가 완료되었습니다!")
