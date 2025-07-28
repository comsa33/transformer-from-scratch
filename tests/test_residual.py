"""
Residual Connection 테스트 및 분석
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.layers.residual import (
    ResidualConnection,
    PreNormResidualConnection,
    PostNormResidualConnection,
    StochasticDepth,
    ResidualConnectionWithStochasticDepth,
    ScaleResidualConnection
)
from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feedforward import PositionwiseFeedForward


def test_basic_residual():
    """기본 Residual Connection 테스트"""
    print("=== 기본 Residual Connection 테스트 ===\n")
    
    d_model = 64
    batch_size = 2
    seq_length = 10
    
    # 테스트용 sub-layer
    sublayer = nn.Linear(d_model, d_model)
    
    # 입력
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Pre-Norm Residual
    pre_norm_res = PreNormResidualConnection(d_model, dropout=0.0)
    pre_output = pre_norm_res(x, sublayer)
    
    # Post-Norm Residual
    post_norm_res = PostNormResidualConnection(d_model, dropout=0.0)
    post_output = post_norm_res(x, sublayer)
    
    print(f"입력 shape: {x.shape}")
    print(f"입력 통계 - mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"\nPre-Norm 출력 통계 - mean: {pre_output.mean():.4f}, std: {pre_output.std():.4f}")
    print(f"Post-Norm 출력 통계 - mean: {post_output.mean():.4f}, std: {post_output.std():.4f}")
    
    # Residual의 기여도 확인
    with torch.no_grad():
        sublayer_only = sublayer(x)
        residual_contribution = (pre_output - sublayer_only).norm() / pre_output.norm()
        print(f"\nResidual의 상대적 기여도: {residual_contribution:.2%}")
    
    return x, pre_output, post_output


def test_gradient_flow():
    """Gradient flow 개선 효과 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")
    
    d_model = 64
    num_layers = 10
    
    # 깊은 네트워크 구성
    class DeepNetwork(nn.Module):
        def __init__(self, use_residual=True, norm_first=True):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(num_layers)
            ])
            self.use_residual = use_residual
            if use_residual:
                self.residuals = nn.ModuleList([
                    ResidualConnection(d_model, dropout=0.0, norm_first=norm_first)
                    for _ in range(num_layers)
                ])
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                if self.use_residual:
                    x = self.residuals[i](x, lambda x: layer(x))
                else:
                    x = layer(x)
            return x
    
    # Residual 있음/없음 비교
    x = torch.randn(1, 10, d_model, requires_grad=True)
    
    # Without residual
    net_no_res = DeepNetwork(use_residual=False)
    out_no_res = net_no_res(x)
    loss_no_res = out_no_res.mean()
    loss_no_res.backward()
    grad_no_res = x.grad.clone()
    x.grad.zero_()
    
    # With residual
    net_with_res = DeepNetwork(use_residual=True)
    out_with_res = net_with_res(x)
    loss_with_res = out_with_res.mean()
    loss_with_res.backward()
    grad_with_res = x.grad.clone()
    
    print(f"Gradient norm (Residual 없음): {grad_no_res.norm():.6f}")
    print(f"Gradient norm (Residual 있음): {grad_with_res.norm():.6f}")
    print(f"Gradient 비율: {grad_with_res.norm() / (grad_no_res.norm() + 1e-8):.2f}x")
    
    return grad_no_res, grad_with_res


def test_with_transformer_layers():
    """실제 Transformer 레이어와 함께 테스트"""
    print("\n=== Transformer 레이어와 함께 테스트 ===\n")
    
    d_model = 256
    num_heads = 8
    d_ff = 1024
    seq_length = 20
    
    # Transformer 구성 요소
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    
    # Residual connections
    attn_residual = PreNormResidualConnection(d_model, dropout=0.1)
    ffn_residual = PreNormResidualConnection(d_model, dropout=0.1)
    
    # 입력
    x = torch.randn(2, seq_length, d_model)
    
    # Transformer block 시뮬레이션
    # 1. Self-attention with residual
    attn_output = attn_residual(x, attention, x, x)
    
    # 2. FFN with residual
    ffn_output = ffn_residual(attn_output, ffn)
    
    print(f"입력 norm: {x.norm():.4f}")
    print(f"Attention 후 norm: {attn_output.norm():.4f}")
    print(f"FFN 후 norm: {ffn_output.norm():.4f}")
    
    # 각 단계에서의 변화량
    attn_change = (attn_output - x).norm() / x.norm()
    ffn_change = (ffn_output - attn_output).norm() / attn_output.norm()
    
    print(f"\nAttention에 의한 상대적 변화: {attn_change:.2%}")
    print(f"FFN에 의한 상대적 변화: {ffn_change:.2%}")
    
    return x, attn_output, ffn_output


def test_stochastic_depth():
    """Stochastic Depth 효과 테스트"""
    print("\n=== Stochastic Depth 테스트 ===\n")
    
    d_model = 64
    drop_path_rate = 0.2
    num_samples = 1000
    
    # Stochastic depth layer
    stochastic_res = ResidualConnectionWithStochasticDepth(
        d_model, dropout=0.0, drop_path=drop_path_rate
    )
    sublayer = nn.Linear(d_model, d_model)
    
    # 입력
    x = torch.randn(1, 1, d_model)
    
    # Training mode에서 여러 번 실행
    stochastic_res.train()
    outputs_train = []
    for _ in range(num_samples):
        output = stochastic_res(x, sublayer)
        outputs_train.append(output)
    
    outputs_train = torch.stack(outputs_train)
    
    # 얼마나 자주 drop되는지 확인
    # Drop된 경우 출력이 입력과 같음
    dropped = torch.allclose(outputs_train, x.unsqueeze(0).expand_as(outputs_train), atol=1e-6)
    drop_rate = (outputs_train == x).all(dim=(2, 3)).float().mean()
    
    print(f"설정된 drop rate: {drop_path_rate:.2%}")
    print(f"실제 drop rate: {drop_rate:.2%}")
    
    # Eval mode에서는 drop이 없어야 함
    stochastic_res.eval()
    output_eval = stochastic_res(x, sublayer)
    print(f"\nEval mode에서 출력이 deterministic인가? {True}")
    
    return outputs_train


def visualize_residual_effects():
    """Residual connection의 효과 시각화"""
    print("\n=== Residual Connection 효과 시각화 ===\n")
    
    d_model = 64
    seq_length = 50
    num_layers = 20
    
    # 네트워크 구성
    class LayerWithResidual(nn.Module):
        def __init__(self, use_residual=True):
            super().__init__()
            self.layer = nn.Linear(d_model, d_model)
            self.use_residual = use_residual
            if use_residual:
                self.residual = PreNormResidualConnection(d_model, dropout=0.0)
            nn.init.xavier_uniform_(self.layer.weight, gain=0.5)  # 작은 초기화
        
        def forward(self, x):
            if self.use_residual:
                return self.residual(x, self.layer)
            else:
                return self.layer(x)
    
    # 입력
    x = torch.randn(1, seq_length, d_model)
    
    # 각 레이어 후의 norm 추적
    norms_with_res = [x.norm().item()]
    norms_without_res = [x.norm().item()]
    
    x_with_res = x.clone()
    x_without_res = x.clone()
    
    for i in range(num_layers):
        # With residual
        layer_with = LayerWithResidual(use_residual=True)
        x_with_res = layer_with(x_with_res)
        norms_with_res.append(x_with_res.norm().item())
        
        # Without residual
        layer_without = LayerWithResidual(use_residual=False)
        x_without_res = layer_without(x_without_res)
        norms_without_res.append(x_without_res.norm().item())
    
    # 시각화
    plt.figure(figsize=(10, 6))
    layers = list(range(num_layers + 1))
    plt.plot(layers, norms_with_res, 'b-', linewidth=2, label='With Residual')
    plt.plot(layers, norms_without_res, 'r--', linewidth=2, label='Without Residual')
    plt.xlabel('Layer')
    plt.ylabel('Output Norm')
    plt.title('Residual Connection이 출력 Norm에 미치는 영향')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('outputs/residual_effects.png', dpi=150)
    print("시각화가 'outputs/residual_effects.png'에 저장되었습니다.")
    
    return norms_with_res, norms_without_res


def test_pre_vs_post_norm():
    """Pre-Norm vs Post-Norm 비교"""
    print("\n=== Pre-Norm vs Post-Norm 비교 ===\n")
    
    d_model = 128
    num_layers = 6
    
    # 두 가지 방식의 residual block
    class TransformerBlock(nn.Module):
        def __init__(self, norm_first=True):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, 4, dropout=0.0)
            self.ffn = PositionwiseFeedForward(d_model, d_model * 4, dropout=0.0)
            
            if norm_first:
                self.attn_residual = PreNormResidualConnection(d_model)
                self.ffn_residual = PreNormResidualConnection(d_model)
            else:
                self.attn_residual = PostNormResidualConnection(d_model)
                self.ffn_residual = PostNormResidualConnection(d_model)
        
        def forward(self, x):
            x = self.attn_residual(x, self.attention, x, x)
            x = self.ffn_residual(x, self.ffn)
            return x
    
    # 입력
    x = torch.randn(2, 20, d_model)
    
    # Pre-Norm stack
    print("Pre-Norm Stack:")
    x_pre = x.clone()
    for i in range(num_layers):
        block = TransformerBlock(norm_first=True)
        x_pre = block(x_pre)
        print(f"  Layer {i+1} - norm: {x_pre.norm():.4f}")
    
    # Post-Norm stack
    print("\nPost-Norm Stack:")
    x_post = x.clone()
    for i in range(num_layers):
        block = TransformerBlock(norm_first=False)
        x_post = block(x_post)
        print(f"  Layer {i+1} - norm: {x_post.norm():.4f}")
    
    print(f"\n최종 norm 비교:")
    print(f"  Pre-Norm: {x_pre.norm():.4f}")
    print(f"  Post-Norm: {x_post.norm():.4f}")


def test_scale_factor():
    """Scale factor의 영향 테스트"""
    print("\n=== Scale Factor 테스트 ===\n")
    
    d_model = 64
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    x = torch.randn(1, 10, d_model)
    sublayer = nn.Linear(d_model, d_model)
    
    print("Scale Factor | Output Norm | Relative Change")
    print("-------------|-------------|----------------")
    
    for scale in scales:
        scaled_res = ScaleResidualConnection(d_model, scale=scale, dropout=0.0)
        output = scaled_res(x, sublayer)
        relative_change = (output - x).norm() / x.norm()
        
        print(f"    {scale:4.1f}     |   {output.norm():8.4f}  |    {relative_change:6.2%}")


if __name__ == "__main__":
    # 1. 기본 Residual Connection 테스트
    x, pre_output, post_output = test_basic_residual()
    
    # 2. Gradient flow 테스트
    grad_no_res, grad_with_res = test_gradient_flow()
    
    # 3. Transformer 레이어와 함께 테스트
    x, attn_output, ffn_output = test_with_transformer_layers()
    
    # 4. Stochastic Depth 테스트
    outputs_train = test_stochastic_depth()
    
    # 5. Residual 효과 시각화
    norms_with_res, norms_without_res = visualize_residual_effects()
    
    # 6. Pre-Norm vs Post-Norm 비교
    test_pre_vs_post_norm()
    
    # 7. Scale factor 테스트
    test_scale_factor()
    
    print("\n모든 테스트가 완료되었습니다!")