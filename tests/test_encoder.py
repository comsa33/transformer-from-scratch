"""
Transformer Encoder 테스트 및 분석
"""

import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import torch

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from transformer.models.encoder import EncoderLayer, create_encoder
from transformer.utils.masking import create_padding_mask


def test_encoder_layer():
    """단일 Encoder Layer 테스트"""
    print("=== Encoder Layer 테스트 ===\n")

    # 파라미터
    batch_size = 2
    seq_length = 10
    d_model = 128
    num_heads = 8
    d_ff = 512

    # Encoder layer 생성
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)

    # 입력 생성
    x = torch.randn(batch_size, seq_length, d_model)

    # Mask 생성 (첫 번째 샘플의 마지막 3개는 padding)
    padding = torch.ones(batch_size, seq_length)
    padding[0, 7:] = 0
    mask = create_padding_mask(padding)

    # Forward pass
    output = encoder_layer(x, mask)

    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"입력과 출력의 shape이 같은가? {x.shape == output.shape}")

    # Attention weights 확인
    output_with_attn, attn_weights = encoder_layer(x, mask, return_attention=True)
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(
        f"Expected shape: [batch_size={batch_size}, num_heads={num_heads}, "
        f"seq_length={seq_length}, seq_length={seq_length}]"
    )

    return encoder_layer, x, output, attn_weights


def test_full_encoder():
    """전체 Encoder Stack 테스트"""
    print("\n=== Full Encoder Stack 테스트 ===\n")

    # 파라미터
    batch_size = 2
    seq_length = 15
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4

    # Encoder 생성
    encoder = create_encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_length=100,
        dropout=0.1,
    )

    # 입력 생성
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Padding mask 생성
    mask = torch.ones(batch_size, seq_length)
    mask[0, 10:] = 0  # 첫 번째 샘플의 마지막 5개는 padding
    mask[1, 12:] = 0  # 두 번째 샘플의 마지막 3개는 padding

    # Forward pass
    output = encoder(input_ids, mask)

    print(f"입력 shape: {input_ids.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Encoder layers: {num_layers}")
    print(f"총 파라미터 수: {sum(p.numel() for p in encoder.parameters()):,}")

    # 각 layer의 출력 확인
    output, all_layers = encoder(input_ids, mask, return_all_layers=True)
    print(f"\n각 layer 출력 수: {len(all_layers)}")

    # 각 layer 후의 norm 변화
    print("\n각 layer 후의 output norm:")
    x = encoder.embeddings(input_ids)
    print(f"  Embedding: {x.norm():.4f}")
    for i, layer_out in enumerate(all_layers):
        print(f"  Layer {i + 1}: {layer_out.norm():.4f}")

    return encoder, input_ids, mask, output


def visualize_attention_patterns(encoder, input_ids, mask):
    """Attention 패턴 시각화"""
    print("\n=== Attention 패턴 시각화 ===\n")

    # 모든 layer의 attention 가져오기
    _, attentions = encoder(input_ids, mask, return_attention=True)

    # 첫 번째 샘플의 attention 시각화
    sample_idx = 0
    num_layers = len(attentions)
    attentions[0].shape[1]

    # Layer별 평균 attention
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for layer_idx in range(min(4, num_layers)):
        # 모든 head의 평균
        avg_attention = attentions[layer_idx][sample_idx].mean(dim=0)

        ax = axes[layer_idx]
        im = ax.imshow(avg_attention.detach().numpy(), cmap="Blues", aspect="auto")
        ax.set_title(f"Layer {layer_idx + 1} - 평균 Attention")
        ax.set_xlabel("Key 위치")
        ax.set_ylabel("Query 위치")

        # Padding 영역 표시
        seq_len = mask[sample_idx].sum().int()
        ax.axvline(x=seq_len - 0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=seq_len - 0.5, color="red", linestyle="--", alpha=0.5)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("outputs/encoder_attention_patterns.png", dpi=150)
    print("Attention 패턴이 'outputs/encoder_attention_patterns.png'에 저장되었습니다.")

    return attentions


def analyze_layer_contributions():
    """각 Layer의 기여도 분석"""
    print("\n=== Layer별 기여도 분석 ===\n")

    # 작은 모델로 테스트
    encoder = create_encoder(
        num_layers=6, d_model=128, num_heads=4, d_ff=512, vocab_size=100, dropout=0.0
    )

    # 입력
    input_ids = torch.randint(0, 100, (1, 20))

    # 각 layer 후의 변화량 측정
    x = encoder.embeddings(input_ids)
    prev_output = x
    changes = []

    for i, layer in enumerate(encoder.layers):
        curr_output = layer(prev_output)

        # 변화량 계산 (L2 norm)
        change = (curr_output - prev_output).norm() / prev_output.norm()
        changes.append(change.item())

        print(f"Layer {i + 1} 상대적 변화량: {change:.4f}")
        prev_output = curr_output

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(changes) + 1), changes)
    plt.xlabel("Layer")
    plt.ylabel("상대적 변화량")
    plt.title("각 Encoder Layer의 기여도")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/encoder_layer_contributions.png", dpi=150)
    print("\n기여도 그래프가 'outputs/encoder_layer_contributions.png'에 저장되었습니다.")

    return changes


def test_positional_encoding_effect():
    """Positional Encoding의 효과 테스트"""
    print("\n=== Positional Encoding 효과 테스트 ===\n")

    # Encoder 생성
    encoder = create_encoder(num_layers=2, d_model=256, num_heads=8, vocab_size=100)

    # 같은 토큰이 다른 위치에 있을 때
    batch_size = 3
    seq_length = 10
    token_id = 42

    # 세 가지 입력:
    # 1. 토큰이 처음에
    # 2. 토큰이 중간에
    # 3. 토큰이 끝에
    input_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    input_ids[0, 0] = token_id
    input_ids[1, 5] = token_id
    input_ids[2, 9] = token_id

    # Encoder 통과
    with torch.no_grad():
        output = encoder(input_ids)

    # 해당 위치의 출력 비교
    output_pos_0 = output[0, 0]
    output_pos_5 = output[1, 5]
    output_pos_9 = output[2, 9]

    # 차이 계산
    diff_0_5 = (output_pos_0 - output_pos_5).norm()
    diff_0_9 = (output_pos_0 - output_pos_9).norm()
    diff_5_9 = (output_pos_5 - output_pos_9).norm()

    print(f"같은 토큰 {token_id}의 위치별 출력 차이:")
    print(f"  위치 0 vs 5: {diff_0_5:.4f}")
    print(f"  위치 0 vs 9: {diff_0_9:.4f}")
    print(f"  위치 5 vs 9: {diff_5_9:.4f}")
    print(
        f"\n{'✅ Positional encoding이 작동합니다!' if diff_0_5 > 0.1 else '❌ 위치 정보가 반영되지 않습니다!'}"
    )


def test_gradient_flow():
    """Encoder의 gradient flow 테스트"""
    print("\n=== Gradient Flow 테스트 ===\n")

    # Pre-LN과 Post-LN 비교
    configs = [("Pre-LN", True), ("Post-LN", False)]

    for name, norm_first in configs:
        print(f"\n{name} 구조:")

        encoder = create_encoder(
            num_layers=6,
            d_model=128,
            num_heads=4,
            d_ff=512,
            vocab_size=100,
            norm_first=norm_first,
            dropout=0.0,
        )

        # 입력
        input_ids = torch.randint(0, 100, (2, 30))
        input_ids.requires_grad = False

        # Forward & backward
        output = encoder(input_ids)
        loss = output.mean()
        loss.backward()

        # 각 layer의 gradient norm 확인
        for i, layer in enumerate(encoder.layers):
            attn_grad = layer.self_attention.w_q.weight.grad.norm()
            ffn_grad = layer.feed_forward.w_1.weight.grad.norm()
            print(f"  Layer {i + 1} - Attention grad: {attn_grad:.4f}, FFN grad: {ffn_grad:.4f}")


def compare_architectures():
    """다양한 Encoder 아키텍처 비교"""
    print("\n=== 아키텍처 비교 ===\n")

    configs = [
        ("Base", {"num_layers": 6, "d_model": 512, "num_heads": 8, "d_ff": 2048}),
        ("Small", {"num_layers": 4, "d_model": 256, "num_heads": 4, "d_ff": 1024}),
        ("Large", {"num_layers": 12, "d_model": 768, "num_heads": 12, "d_ff": 3072}),
        ("Deep-Narrow", {"num_layers": 12, "d_model": 256, "num_heads": 8, "d_ff": 1024}),
        ("Wide-Shallow", {"num_layers": 3, "d_model": 1024, "num_heads": 16, "d_ff": 4096}),
    ]

    print("아키텍처 | Layers | d_model | Heads | d_ff | Parameters")
    print("-" * 60)

    for name, config in configs:
        encoder = create_encoder(vocab_size=30000, **config)
        params = sum(p.numel() for p in encoder.parameters())

        print(
            f"{name:12} | {config['num_layers']:6} | {config['d_model']:7} | "
            f"{config['num_heads']:5} | {config['d_ff']:4} | {params:,}"
        )


if __name__ == "__main__":
    # 1. Encoder Layer 테스트
    encoder_layer, x, output, attn_weights = test_encoder_layer()

    # 2. Full Encoder 테스트
    encoder, input_ids, mask, output = test_full_encoder()

    # 3. Attention 패턴 시각화
    attentions = visualize_attention_patterns(encoder, input_ids, mask)

    # 4. Layer별 기여도 분석
    changes = analyze_layer_contributions()

    # 5. Positional Encoding 효과
    test_positional_encoding_effect()

    # 6. Gradient Flow 테스트
    test_gradient_flow()

    # 7. 아키텍처 비교
    compare_architectures()

    print("\n모든 테스트가 완료되었습니다!")
