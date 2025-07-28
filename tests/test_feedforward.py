"""
Feed-Forward Network 테스트 및 분석
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.layers.feedforward import (
    PositionwiseFeedForward,
    GatedFeedForward,
    SwiGLU,
    ExpertFFN
)


def test_basic_ffn():
    """기본 FFN 테스트"""
    print("=== 기본 Feed-Forward Network 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    seq_length = 10
    d_model = 128
    d_ff = 512
    
    # FFN 생성
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    
    # 입력 생성
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Forward pass
    output = ffn(x)
    
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Hidden dimension: {d_ff} (확장 비율: {d_ff/d_model:.1f}x)")
    
    # 파라미터 수 계산
    w1_params = ffn.w_1.weight.numel() + (ffn.w_1.bias.numel() if ffn.w_1.bias is not None else 0)
    w2_params = ffn.w_2.weight.numel() + (ffn.w_2.bias.numel() if ffn.w_2.bias is not None else 0)
    total_params = w1_params + w2_params
    
    print(f"\n파라미터 수:")
    print(f"  W1: {d_model} x {d_ff} + {d_ff} = {w1_params:,}")
    print(f"  W2: {d_ff} x {d_model} + {d_model} = {w2_params:,}")
    print(f"  총합: {total_params:,}")
    
    return ffn, x, output


def test_activation_functions():
    """다양한 활성화 함수 비교"""
    print("\n=== 활성화 함수 비교 ===\n")
    
    d_model = 64
    d_ff = 256
    x = torch.randn(1, 100, d_model)
    
    activations = ['relu', 'gelu', 'swish', 'mish', 'tanh']
    results = {}
    
    for activation in activations:
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0, activation=activation)
        output = ffn(x)
        
        # 통계 수집
        results[activation] = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'zeros': (output == 0).float().mean().item(),  # ReLU의 경우 0인 비율
            'output': output
        }
        
        print(f"{activation:8s}: mean={results[activation]['mean']:7.4f}, "
              f"std={results[activation]['std']:7.4f}, "
              f"zeros={results[activation]['zeros']:5.1%}")
    
    return results


def test_gradient_flow():
    """Gradient flow 분석"""
    print("\n=== Gradient Flow 분석 ===\n")
    
    d_model = 128
    d_ff = 512
    seq_length = 50
    
    # 다양한 활성화 함수로 테스트
    activations = ['relu', 'gelu', 'swish']
    
    for activation in activations:
        print(f"\n{activation.upper()} 활성화:")
        
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0, activation=activation)
        x = torch.randn(2, seq_length, d_model, requires_grad=True)
        
        # Forward pass
        output = ffn(x)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient 통계
        print(f"  입력 gradient norm: {x.grad.norm():.4f}")
        print(f"  W1 gradient norm: {ffn.w_1.weight.grad.norm():.4f}")
        print(f"  W2 gradient norm: {ffn.w_2.weight.grad.norm():.4f}")
        
        # Gradient가 0인 비율 (vanishing gradient 확인)
        grad_zeros = (x.grad == 0).float().mean()
        print(f"  입력 gradient zeros: {grad_zeros:.1%}")


def test_gated_variants():
    """Gated FFN 변형들 테스트"""
    print("\n=== Gated FFN 변형 테스트 ===\n")
    
    d_model = 128
    d_ff = 512
    batch_size = 2
    seq_length = 10
    
    x = torch.randn(batch_size, seq_length, d_model)
    
    # 1. 기본 FFN
    basic_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    basic_output = basic_ffn(x)
    
    # 2. Gated FFN
    gated_ffn = GatedFeedForward(d_model, d_ff, dropout=0.0)
    gated_output = gated_ffn(x)
    
    # 3. SwiGLU
    swiglu = SwiGLU(d_model)
    swiglu_output = swiglu(x)
    
    print("출력 통계:")
    print(f"  Basic FFN: mean={basic_output.mean():.4f}, std={basic_output.std():.4f}")
    print(f"  Gated FFN: mean={gated_output.mean():.4f}, std={gated_output.std():.4f}")
    print(f"  SwiGLU:    mean={swiglu_output.mean():.4f}, std={swiglu_output.std():.4f}")
    
    # 파라미터 수 비교
    print("\n파라미터 수:")
    print(f"  Basic FFN: {sum(p.numel() for p in basic_ffn.parameters()):,}")
    print(f"  Gated FFN: {sum(p.numel() for p in gated_ffn.parameters()):,}")
    print(f"  SwiGLU:    {sum(p.numel() for p in swiglu.parameters()):,}")
    
    return basic_output, gated_output, swiglu_output


def visualize_ffn_behavior():
    """FFN 동작 시각화"""
    print("\n=== FFN 동작 시각화 ===\n")
    
    # 간단한 입력으로 테스트
    d_model = 64
    d_ff = 256
    seq_length = 20
    
    # 특정 패턴을 가진 입력 생성
    x = torch.zeros(1, seq_length, d_model)
    # 각 위치에 다른 강도의 신호
    for i in range(seq_length):
        x[0, i, :] = torch.randn(d_model) * (i / seq_length)
    
    # 다양한 FFN 적용
    ffn_relu = PositionwiseFeedForward(d_model, d_ff, activation='relu', dropout=0.0)
    ffn_gelu = PositionwiseFeedForward(d_model, d_ff, activation='gelu', dropout=0.0)
    gated_ffn = GatedFeedForward(d_model, d_ff, dropout=0.0)
    
    out_relu = ffn_relu(x)
    out_gelu = ffn_gelu(x)
    out_gated = gated_ffn(x)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 입력 패턴
    ax = axes[0, 0]
    im = ax.imshow(x[0].T.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax.set_title('입력 패턴')
    ax.set_xlabel('시퀀스 위치')
    ax.set_ylabel('특징 차원')
    plt.colorbar(im, ax=ax)
    
    # 2. ReLU FFN 출력
    ax = axes[0, 1]
    im = ax.imshow(out_relu[0].T.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax.set_title('ReLU FFN 출력')
    ax.set_xlabel('시퀀스 위치')
    ax.set_ylabel('특징 차원')
    plt.colorbar(im, ax=ax)
    
    # 3. GELU FFN 출력
    ax = axes[1, 0]
    im = ax.imshow(out_gelu[0].T.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax.set_title('GELU FFN 출력')
    ax.set_xlabel('시퀀스 위치')
    ax.set_ylabel('특징 차원')
    plt.colorbar(im, ax=ax)
    
    # 4. Gated FFN 출력
    ax = axes[1, 1]
    im = ax.imshow(out_gated[0].T.detach().numpy(), cmap='coolwarm', aspect='auto')
    ax.set_title('Gated FFN 출력')
    ax.set_xlabel('시퀀스 위치')
    ax.set_ylabel('특징 차원')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('outputs/ffn_behavior.png', dpi=150)
    print("시각화가 'outputs/ffn_behavior.png'에 저장되었습니다.")


def test_position_independence():
    """위치 독립성 테스트"""
    print("\n=== 위치 독립성 테스트 ===\n")
    
    d_model = 64
    d_ff = 256
    
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    
    # 같은 벡터를 다른 위치에 배치
    test_vector = torch.randn(1, d_model)
    x = torch.zeros(1, 5, d_model)
    x[0, 0] = test_vector
    x[0, 2] = test_vector
    x[0, 4] = test_vector
    
    # FFN 적용
    output = ffn(x)
    
    # 같은 입력에 대해 같은 출력인지 확인
    out0 = output[0, 0]
    out2 = output[0, 2]
    out4 = output[0, 4]
    
    diff_02 = (out0 - out2).abs().max()
    diff_04 = (out0 - out4).abs().max()
    
    print(f"위치 0과 2의 출력 차이: {diff_02:.6f}")
    print(f"위치 0과 4의 출력 차이: {diff_04:.6f}")
    print(f"{'✅ 위치 독립적' if diff_02 < 1e-5 and diff_04 < 1e-5 else '❌ 위치 의존적'}")


def benchmark_ffn_variants():
    """FFN 변형들의 성능 벤치마크"""
    print("\n=== FFN 성능 벤치마크 ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 테스트 설정
    batch_size = 32
    seq_length = 128
    d_model = 512
    d_ff = 2048
    iterations = 100
    
    x = torch.randn(batch_size, seq_length, d_model, device=device)
    
    variants = {
        'Basic FFN': PositionwiseFeedForward(d_model, d_ff, dropout=0.0),
        'Gated FFN': GatedFeedForward(d_model, d_ff, dropout=0.0),
        'SwiGLU': SwiGLU(d_model)
    }
    
    for name, model in variants.items():
        model = model.to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 벤치마크
        start = time.time()
        for _ in range(iterations):
            output = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations * 1000
        throughput = batch_size * seq_length / avg_time
        
        print(f"{name:12s}: {avg_time:6.2f} ms/iter, {throughput:7.1f} K tokens/s")


def analyze_activation_patterns():
    """활성화 패턴 분석"""
    print("\n=== 활성화 패턴 분석 ===\n")
    
    d_model = 128
    d_ff = 512
    
    # ReLU FFN의 활성화 패턴 분석
    ffn = PositionwiseFeedForward(d_model, d_ff, activation='relu', dropout=0.0)
    
    # 다양한 입력 크기로 테스트
    input_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    activation_rates = []
    
    for scale in input_scales:
        x = torch.randn(100, 20, d_model) * scale
        
        # 중간 활성화 값 추출을 위한 hook
        activations = []
        def hook(module, input, output):
            activations.append(output)
        
        handle = ffn.w_1.register_forward_hook(hook)
        _ = ffn(x)
        handle.remove()
        
        # ReLU 후 활성화된 뉴런 비율
        hidden = activations[0]
        active_ratio = (hidden > 0).float().mean()
        activation_rates.append(active_ratio.item())
        
        print(f"입력 scale {scale:3.1f}: 활성화 비율 {active_ratio:.1%}")
    
    return input_scales, activation_rates


if __name__ == "__main__":
    # 1. 기본 FFN 테스트
    ffn, x, output = test_basic_ffn()
    
    # 2. 활성화 함수 비교
    activation_results = test_activation_functions()
    
    # 3. Gradient flow 분석
    test_gradient_flow()
    
    # 4. Gated 변형 테스트
    basic_out, gated_out, swiglu_out = test_gated_variants()
    
    # 5. FFN 동작 시각화
    visualize_ffn_behavior()
    
    # 6. 위치 독립성 테스트
    test_position_independence()
    
    # 7. 성능 벤치마크
    benchmark_ffn_variants()
    
    # 8. 활성화 패턴 분석
    scales, rates = analyze_activation_patterns()
    
    print("\n모든 테스트가 완료되었습니다!")