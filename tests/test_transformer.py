"""
전체 Transformer 모델 테스트 및 분석
"""

import sys

sys.path.append(".")

import time

import matplotlib.pyplot as plt
import torch

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from transformer.models.transformer import (
    create_transformer,
    create_transformer_base,
    create_transformer_big,
    create_transformer_small,
)


def test_basic_transformer():
    """기본 Transformer 동작 테스트"""
    print("=== 기본 Transformer 테스트 ===\n")

    # Small 모델로 테스트
    model = create_transformer_small(src_vocab_size=1000, tgt_vocab_size=1000, max_length=100)

    # 입력 생성
    batch_size = 2
    src_length = 15
    tgt_length = 12

    src_ids = torch.randint(0, 1000, (batch_size, src_length))
    tgt_ids = torch.randint(0, 1000, (batch_size, tgt_length))

    # Masks
    src_mask = torch.ones(batch_size, src_length)
    src_mask[0, 10:] = 0  # 첫 번째 source의 마지막 5개는 padding

    tgt_mask = torch.ones(batch_size, tgt_length)
    tgt_mask[1, 8:] = 0  # 두 번째 target의 마지막 4개는 padding

    # Forward pass
    output = model(src_ids, tgt_ids, src_mask, tgt_mask)

    print("모델 구성:")
    print(f"  Encoder layers: {model.encoder.num_layers}")
    print(f"  Decoder layers: {model.decoder.num_layers}")
    print(f"  Model dimension: {model.d_model}")
    print(f"  Attention heads: {model.num_heads}")

    print("\n입출력:")
    print(f"  Source shape: {src_ids.shape}")
    print(f"  Target shape: {tgt_ids.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  예상 shape: [{batch_size}, {tgt_length}, 1000]")

    print("\n파라미터:")
    print(f"  총 파라미터: {model.get_num_params():,}")
    print(f"  Encoder 파라미터: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Decoder 파라미터: {sum(p.numel() for p in model.decoder.parameters()):,}")

    return model, src_ids, tgt_ids, output


def test_attention_visualization():
    """Attention 시각화 테스트"""
    print("\n=== Attention 시각화 ===\n")

    # 작은 모델과 짧은 시퀀스로 테스트
    model = create_transformer(
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=128,
        num_heads=4,
        src_vocab_size=100,
        tgt_vocab_size=100,
    )

    # 입력
    src_ids = torch.randint(0, 100, (1, 8))
    tgt_ids = torch.randint(0, 100, (1, 6))

    # Forward with attention
    output, encoder_output, attentions = model(
        src_ids, tgt_ids, return_encoder_output=True, return_attention=True
    )

    # Attention 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Encoder self-attention (첫 번째 layer)
    enc_attn = attentions["encoder"][0][0].mean(dim=0)  # 평균 over heads
    ax = axes[0, 0]
    im = ax.imshow(enc_attn.detach().numpy(), cmap="Blues", aspect="auto")
    ax.set_title("Encoder Self-Attention (Layer 1)")
    ax.set_xlabel("Key 위치")
    ax.set_ylabel("Query 위치")
    plt.colorbar(im, ax=ax)

    # Decoder self-attention (첫 번째 layer)
    dec_self_attn = attentions["decoder_self"][0][0].mean(dim=0)
    ax = axes[0, 1]
    im = ax.imshow(dec_self_attn.detach().numpy(), cmap="Greens", aspect="auto")
    ax.set_title("Decoder Self-Attention (Layer 1)")
    ax.set_xlabel("Key 위치")
    ax.set_ylabel("Query 위치")
    plt.colorbar(im, ax=ax)

    # Decoder cross-attention (첫 번째 layer)
    dec_cross_attn = attentions["decoder_cross"][0][0].mean(dim=0)
    ax = axes[0, 2]
    im = ax.imshow(dec_cross_attn.detach().numpy(), cmap="Reds", aspect="auto")
    ax.set_title("Decoder Cross-Attention (Layer 1)")
    ax.set_xlabel("Encoder 위치")
    ax.set_ylabel("Decoder 위치")
    plt.colorbar(im, ax=ax)

    # 두 번째 layer attentions
    if len(attentions["encoder"]) > 1:
        # Encoder self-attention (두 번째 layer)
        enc_attn = attentions["encoder"][1][0].mean(dim=0)
        ax = axes[1, 0]
        im = ax.imshow(enc_attn.detach().numpy(), cmap="Blues", aspect="auto")
        ax.set_title("Encoder Self-Attention (Layer 2)")
        ax.set_xlabel("Key 위치")
        ax.set_ylabel("Query 위치")
        plt.colorbar(im, ax=ax)

        # Decoder self-attention (두 번째 layer)
        dec_self_attn = attentions["decoder_self"][1][0].mean(dim=0)
        ax = axes[1, 1]
        im = ax.imshow(dec_self_attn.detach().numpy(), cmap="Greens", aspect="auto")
        ax.set_title("Decoder Self-Attention (Layer 2)")
        ax.set_xlabel("Key 위치")
        ax.set_ylabel("Query 위치")
        plt.colorbar(im, ax=ax)

        # Decoder cross-attention (두 번째 layer)
        dec_cross_attn = attentions["decoder_cross"][1][0].mean(dim=0)
        ax = axes[1, 2]
        im = ax.imshow(dec_cross_attn.detach().numpy(), cmap="Reds", aspect="auto")
        ax.set_title("Decoder Cross-Attention (Layer 2)")
        ax.set_xlabel("Encoder 위치")
        ax.set_ylabel("Decoder 위치")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("outputs/transformer_attention_all.png", dpi=150)
    print("전체 attention 패턴이 'outputs/transformer_attention_all.png'에 저장되었습니다.")

    return attentions


def test_generation():
    """텍스트 생성 테스트"""
    print("\n=== 텍스트 생성 테스트 ===\n")

    model = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100)

    # Source 입력
    src_ids = torch.randint(5, 50, (1, 10))  # 5-50 사이의 토큰

    print("1. Greedy Decoding:")
    generated_greedy = model.generate(
        src_ids, start_token_id=1, end_token_id=2, max_length=20, temperature=1.0
    )
    print(f"   생성 길이: {generated_greedy.shape[1]}")
    print(f"   생성 시퀀스: {generated_greedy[0].tolist()}")

    print("\n2. Sampling (high temperature):")
    generated_sample = model.generate(
        src_ids, start_token_id=1, end_token_id=2, max_length=20, temperature=1.5
    )
    print(f"   생성 길이: {generated_sample.shape[1]}")
    print(f"   생성 시퀀스: {generated_sample[0].tolist()}")

    print("\n3. Top-k Sampling:")
    generated_topk = model.generate(
        src_ids, start_token_id=1, end_token_id=2, max_length=20, temperature=0.8, top_k=10
    )
    print(f"   생성 길이: {generated_topk.shape[1]}")
    print(f"   생성 시퀀스: {generated_topk[0].tolist()}")

    print("\n4. Beam Search:")
    generated_beam = model.generate(
        src_ids, start_token_id=1, end_token_id=2, max_length=20, beam_size=3, length_penalty=1.0
    )
    print(f"   생성 길이: {generated_beam.shape[1]}")
    print(f"   생성 시퀀스: {generated_beam[0].tolist()}")

    return model


def compare_model_variants():
    """다양한 모델 변형 비교"""
    print("\n=== 모델 변형 비교 ===\n")

    vocab_size = 30000

    variants = [
        ("Small", create_transformer_small),
        ("Base", create_transformer_base),
        ("Big", create_transformer_big),
    ]

    print("모델    | Enc/Dec Layers | d_model | Heads | d_ff  | Parameters")
    print("-" * 65)

    for name, create_fn in variants:
        model = create_fn(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size)

        enc_layers = model.encoder.num_layers
        dec_layers = model.decoder.num_layers
        d_model = model.d_model
        num_heads = model.num_heads
        d_ff = model.encoder.layers[0].feed_forward.d_ff
        params = model.get_num_params()

        print(
            f"{name:7} | {enc_layers:3}/{dec_layers:3}        | {d_model:7} | {num_heads:5} | "
            f"{d_ff:5} | {params:,}"
        )


def test_embedding_sharing():
    """임베딩 공유 테스트"""
    print("\n=== 임베딩 공유 테스트 ===\n")

    # 1. 기본 모델 (공유 없음)
    model_no_share = create_transformer_small(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        share_embeddings=False,
        share_encoder_decoder_embeddings=False,
    )

    # 2. Decoder input/output 임베딩 공유
    model_share_dec = create_transformer_small(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        share_embeddings=True,
        share_encoder_decoder_embeddings=False,
    )

    # 3. Encoder-Decoder 임베딩 공유
    model_share_all = create_transformer_small(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        share_embeddings=True,
        share_encoder_decoder_embeddings=True,
    )

    print("임베딩 공유 설정별 파라미터 수:")
    print(f"  공유 없음: {model_no_share.get_num_params():,}")
    print(f"  Decoder 임베딩 공유: {model_share_dec.get_num_params():,}")
    print(f"  전체 임베딩 공유: {model_share_all.get_num_params():,}")

    # 공유 확인
    if model_share_all.encoder.embeddings is model_share_all.decoder.embeddings:
        print("\n✅ Encoder-Decoder 임베딩이 올바르게 공유됨")
    else:
        print("\n❌ Encoder-Decoder 임베딩 공유 실패")


def benchmark_inference_speed():
    """추론 속도 벤치마크"""
    print("\n=== 추론 속도 벤치마크 ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = create_transformer_small(src_vocab_size=10000, tgt_vocab_size=10000).to(device)

    model.eval()

    # 테스트 설정
    batch_sizes = [1, 4, 8, 16]
    src_length = 50
    tgt_length = 50
    iterations = 20

    print("Batch Size | Inference Time (ms) | Throughput (samples/sec)")
    print("-" * 60)

    with torch.no_grad():
        for batch_size in batch_sizes:
            src_ids = torch.randint(0, 10000, (batch_size, src_length), device=device)
            tgt_ids = torch.randint(0, 10000, (batch_size, tgt_length), device=device)

            # Warmup
            for _ in range(5):
                _ = model(src_ids, tgt_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # 벤치마크
            start = time.time()
            for _ in range(iterations):
                model(src_ids, tgt_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.time() - start
            avg_time = (elapsed / iterations) * 1000  # ms
            throughput = batch_size / (avg_time / 1000)  # samples/sec

            print(f"{batch_size:10} | {avg_time:18.2f} | {throughput:20.1f}")


def analyze_gradient_flow():
    """Gradient flow 분석"""
    print("\n=== Gradient Flow 분석 ===\n")

    model = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100)

    # 입력
    src_ids = torch.randint(0, 100, (2, 10))
    tgt_ids = torch.randint(0, 100, (2, 8))

    # Forward & backward
    output = model(src_ids, tgt_ids)
    loss = output.mean()  # 더미 loss
    loss.backward()

    # Gradient norms 수집
    print("Layer별 Gradient Norms:")
    print("\nEncoder:")
    for i, layer in enumerate(model.encoder.layers):
        attn_grad = layer.self_attention.w_q.weight.grad.norm()
        ffn_grad = layer.feed_forward.w_1.weight.grad.norm()
        print(f"  Layer {i + 1} - Attention: {attn_grad:.4f}, FFN: {ffn_grad:.4f}")

    print("\nDecoder:")
    for i, layer in enumerate(model.decoder.layers):
        self_attn_grad = layer.self_attention.w_q.weight.grad.norm()
        cross_attn_grad = layer.cross_attention.w_q.weight.grad.norm()
        ffn_grad = layer.feed_forward.w_1.weight.grad.norm()
        print(
            f"  Layer {i + 1} - Self-Attn: {self_attn_grad:.4f}, "
            f"Cross-Attn: {cross_attn_grad:.4f}, FFN: {ffn_grad:.4f}"
        )


if __name__ == "__main__":
    # 1. 기본 동작 테스트
    model, src_ids, tgt_ids, output = test_basic_transformer()

    # 2. Attention 시각화
    attentions = test_attention_visualization()

    # 3. 생성 테스트
    test_generation()

    # 4. 모델 변형 비교
    compare_model_variants()

    # 5. 임베딩 공유 테스트
    test_embedding_sharing()

    # 6. 추론 속도 벤치마크
    benchmark_inference_speed()

    # 7. Gradient flow 분석
    analyze_gradient_flow()

    print("\n모든 테스트가 완료되었습니다!")
