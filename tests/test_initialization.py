"""
가중치 초기화 테스트 및 분석
"""

import sys

sys.path.append(".")


import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from transformer.utils.initialization import (
    TransformerInitializer,
    he_normal_,
    he_uniform_,
    init_bert_params,
    init_gpt2_params,
    init_transformer_params,
    normal_,
    truncated_normal_,
    uniform_,
    xavier_normal_,
    xavier_uniform_,
)


def test_basic_initializations():
    """기본 초기화 함수들 테스트"""
    print("=== 기본 초기화 함수 테스트 ===\n")

    # 테스트용 텐서
    shape = (256, 512)  # (out_features, in_features)

    # 1. Xavier 초기화
    tensor_xavier_u = torch.empty(shape)
    xavier_uniform_(tensor_xavier_u)
    print(f"Xavier Uniform - mean: {tensor_xavier_u.mean():.4f}, std: {tensor_xavier_u.std():.4f}")

    tensor_xavier_n = torch.empty(shape)
    xavier_normal_(tensor_xavier_n)
    print(f"Xavier Normal - mean: {tensor_xavier_n.mean():.4f}, std: {tensor_xavier_n.std():.4f}")

    # 2. He 초기화
    tensor_he_u = torch.empty(shape)
    he_uniform_(tensor_he_u)
    print(f"He Uniform - mean: {tensor_he_u.mean():.4f}, std: {tensor_he_u.std():.4f}")

    tensor_he_n = torch.empty(shape)
    he_normal_(tensor_he_n)
    print(f"He Normal - mean: {tensor_he_n.mean():.4f}, std: {tensor_he_n.std():.4f}")

    # 3. Truncated Normal
    tensor_trunc = torch.empty(shape)
    truncated_normal_(tensor_trunc, mean=0, std=0.02, a=-2.0, b=2.0)
    print(f"Truncated Normal - mean: {tensor_trunc.mean():.4f}, std: {tensor_trunc.std():.4f}")
    print(f"  min: {tensor_trunc.min():.4f}, max: {tensor_trunc.max():.4f}")

    return {
        "xavier_uniform": tensor_xavier_u,
        "xavier_normal": tensor_xavier_n,
        "he_uniform": tensor_he_u,
        "he_normal": tensor_he_n,
        "truncated_normal": tensor_trunc,
    }


def visualize_initialization_distributions(tensors: dict):
    """초기화 분포 시각화"""
    print("\n=== 초기화 분포 시각화 ===\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, tensor) in enumerate(tensors.items()):
        ax = axes[idx]
        values = tensor.flatten().numpy()

        # 히스토그램
        n, bins, patches = ax.hist(
            values, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black"
        )

        # 통계 정보
        mean = values.mean()
        std = values.std()
        ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.4f}")
        ax.axvline(
            mean + std, color="green", linestyle="--", linewidth=1, label=f"±1 STD: {std:.4f}"
        )
        ax.axvline(mean - std, color="green", linestyle="--", linewidth=1)

        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 마지막 subplot은 비워둠
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/initialization_distributions.png", dpi=150)
    print("분포 시각화가 'outputs/initialization_distributions.png'에 저장되었습니다.")


def test_model_initialization():
    """실제 모델에 초기화 적용 테스트"""
    print("\n=== 모델 초기화 테스트 ===\n")

    # 간단한 Transformer 블록 생성
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, d_model=512, n_heads=8, d_ff=2048):
            super().__init__()
            self.attention = nn.Linear(d_model, d_model * 3)  # Q, K, V
            self.output_proj = nn.Linear(d_model, d_model)
            self.ffn1 = nn.Linear(d_model, d_ff)
            self.ffn2 = nn.Linear(d_ff, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.embedding = nn.Embedding(1000, d_model)

    # 다양한 초기화 방법 테스트
    init_methods = ["xavier_uniform", "xavier_normal", "he_uniform", "bert", "gpt2"]

    print("초기화 방법별 가중치 통계:")
    print("-" * 70)
    print(f"{'Method':<15} | {'Linear Mean':<12} | {'Linear Std':<12} | {'Embed Std':<12}")
    print("-" * 70)

    for method in init_methods:
        model = SimpleTransformerBlock()

        if method == "bert":
            model.apply(init_bert_params)
        elif method == "gpt2":
            model.apply(lambda m: init_gpt2_params(m, n_layers=6))
        else:
            model.apply(lambda m: init_transformer_params(m, d_model=512, init_type=method))

        # 통계 수집
        linear_weights = []
        for name, param in model.named_parameters():
            if "weight" in name and "norm" not in name and "embedding" not in name:
                linear_weights.append(param.data.flatten())

        linear_weights = torch.cat(linear_weights)
        embed_std = model.embedding.weight.data.std().item()

        print(
            f"{method:<15} | {linear_weights.mean():.6f}    | "
            f"{linear_weights.std():.6f}    | {embed_std:.6f}"
        )


def test_gradient_flow_with_initialization():
    """초기화 방법에 따른 gradient flow 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")

    class DeepLinear(nn.Module):
        def __init__(self, d_model=256, n_layers=10):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.activation = nn.ReLU()

        def forward(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

    # 다양한 초기화로 테스트
    init_types = ["xavier_uniform", "he_uniform", "normal_0.02"]
    batch_size = 32
    d_model = 256

    gradient_norms = {}

    for init_type in init_types:
        model = DeepLinear(d_model=d_model, n_layers=10)

        # 초기화
        if init_type == "xavier_uniform":
            for layer in model.layers:
                xavier_uniform_(layer.weight)
                layer.bias.data.zero_()
        elif init_type == "he_uniform":
            for layer in model.layers:
                he_uniform_(layer.weight)
                layer.bias.data.zero_()
        elif init_type == "normal_0.02":
            for layer in model.layers:
                normal_(layer.weight, std=0.02)
                layer.bias.data.zero_()

        # Forward & backward
        x = torch.randn(batch_size, d_model)
        output = model(x)
        loss = output.mean()
        loss.backward()

        # Gradient norm 수집
        grad_norms = []
        for i, layer in enumerate(model.layers):
            grad_norm = layer.weight.grad.norm().item()
            grad_norms.append(grad_norm)

        gradient_norms[init_type] = grad_norms

    # 시각화
    plt.figure(figsize=(10, 6))
    for init_type, norms in gradient_norms.items():
        plt.plot(range(1, len(norms) + 1), norms, marker="o", label=init_type)

    plt.xlabel("Layer Index")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Flow with Different Initializations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.savefig("outputs/gradient_flow_initialization.png", dpi=150)
    print("Gradient flow 분석이 'outputs/gradient_flow_initialization.png'에 저장되었습니다.")

    # 결과 출력
    print("\nGradient norm by layer:")
    for init_type, norms in gradient_norms.items():
        print(f"\n{init_type}:")
        print(f"  First layer: {norms[0]:.6f}")
        print(f"  Last layer: {norms[-1]:.6f}")
        print(f"  Ratio (last/first): {norms[-1] / norms[0]:.6f}")


def test_transformer_initializer():
    """TransformerInitializer 클래스 테스트"""
    print("\n=== TransformerInitializer 테스트 ===\n")

    # 간단한 Transformer 모델
    class MiniTransformer(nn.Module):
        def __init__(self, d_model=512, n_heads=8, n_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(10000, d_model)
            self.pos_embedding = nn.Embedding(512, d_model)

            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attention": nn.Linear(d_model, d_model * 3),
                            "output": nn.Linear(d_model, d_model),
                            "ffn1": nn.Linear(d_model, d_model * 4),
                            "ffn2": nn.Linear(d_model * 4, d_model),
                            "norm1": nn.LayerNorm(d_model),
                            "norm2": nn.LayerNorm(d_model),
                        }
                    )
                    for _ in range(n_layers)
                ]
            )

            self.output_projection = nn.Linear(d_model, 10000)
            self.output_projection.is_output_projection = True  # GPT-2 스타일용 마커

    # 다양한 초기화 전략 테스트
    strategies = ["xavier_uniform", "bert", "gpt2"]

    for strategy in strategies:
        print(f"\n{strategy} 초기화:")
        model = MiniTransformer()

        initializer = TransformerInitializer(d_model=512, n_layers=6, init_type=strategy)
        initializer.initialize(model)

        # 가중치 통계 확인
        print(f"  Embedding std: {model.embedding.weight.std():.4f}")
        print(f"  First attention std: {model.layers[0]['attention'].weight.std():.4f}")
        print(f"  First FFN std: {model.layers[0]['ffn1'].weight.std():.4f}")
        print(f"  Output projection std: {model.output_projection.weight.std():.4f}")

        # Norm 레이어 확인
        print(f"  LayerNorm weight mean: {model.layers[0]['norm1'].weight.mean():.4f}")
        print(f"  LayerNorm bias mean: {model.layers[0]['norm1'].bias.mean():.4f}")


def compare_initialization_effects():
    """초기화 방법이 학습 초기에 미치는 영향 비교"""
    print("\n=== 초기화 방법의 학습 영향 비교 ===\n")

    # 간단한 classification 태스크
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=256, output_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 데이터 생성
    torch.manual_seed(42)
    X = torch.randn(1000, 100)
    torch.randint(0, 10, (1000,))

    # 다양한 초기화로 학습
    init_methods = {
        "Xavier": lambda m: xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None,
        "He": lambda m: he_uniform_(m.weight) if isinstance(m, nn.Linear) else None,
        "Normal(0.02)": lambda m: normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else None,
        "Uniform": lambda m: uniform_(m.weight, -0.1, 0.1) if isinstance(m, nn.Linear) else None,
    }

    results = {}

    for name, init_fn in init_methods.items():
        model = SimpleClassifier()
        model.apply(init_fn)

        # 첫 번째 forward pass의 출력 분포
        with torch.no_grad():
            output = model(X[:100])
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_max = output.abs().max().item()

        results[name] = {"mean": output_mean, "std": output_std, "max": output_max}

    # 결과 출력
    print("초기화 방법별 첫 forward pass 출력 통계:")
    print("-" * 60)
    print(f"{'Method':<15} | {'Mean':<10} | {'Std':<10} | {'Max Abs':<10}")
    print("-" * 60)

    for name, stats in results.items():
        print(
            f"{name:<15} | {stats['mean']:>10.4f} | {stats['std']:>10.4f} | {stats['max']:>10.4f}"
        )


if __name__ == "__main__":
    # 1. 기본 초기화 함수 테스트
    tensors = test_basic_initializations()

    # 2. 초기화 분포 시각화
    visualize_initialization_distributions(tensors)

    # 3. 모델 초기화 테스트
    test_model_initialization()

    # 4. Gradient flow 테스트
    test_gradient_flow_with_initialization()

    # 5. TransformerInitializer 테스트
    test_transformer_initializer()

    # 6. 초기화 효과 비교
    compare_initialization_effects()

    print("\n모든 테스트가 완료되었습니다!")
