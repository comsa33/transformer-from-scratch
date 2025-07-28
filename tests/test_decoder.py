"""
Transformer Decoder 테스트 및 분석
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from transformer.models.decoder import DecoderLayer, TransformerDecoder, create_decoder
from transformer.models.encoder import create_encoder
from transformer.utils.masking import create_padding_mask, create_look_ahead_mask


def test_decoder_layer():
    """단일 Decoder Layer 테스트"""
    print("=== Decoder Layer 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 8
    d_model = 128
    num_heads = 8
    d_ff = 512
    
    # Decoder layer 생성
    decoder_layer = DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0
    )
    
    # 입력 생성
    decoder_input = torch.randn(batch_size, tgt_seq_length, d_model)
    encoder_output = torch.randn(batch_size, src_seq_length, d_model)
    
    # Mask 생성
    # Self-attention mask (look-ahead + padding)
    look_ahead_mask = create_look_ahead_mask(tgt_seq_length)
    
    # Forward pass
    output = decoder_layer(decoder_input, encoder_output, look_ahead_mask)
    
    print(f"Decoder 입력 shape: {decoder_input.shape}")
    print(f"Encoder 출력 shape: {encoder_output.shape}")
    print(f"Decoder 출력 shape: {output.shape}")
    print(f"출력이 입력과 같은 shape인가? {output.shape == decoder_input.shape}")
    
    # Attention weights 확인
    output, self_attn, cross_attn = decoder_layer(
        decoder_input, encoder_output, look_ahead_mask, return_attention=True
    )
    
    print(f"\nSelf-attention shape: {self_attn.shape}")
    print(f"Cross-attention shape: {cross_attn.shape}")
    
    return decoder_layer, decoder_input, encoder_output, output


def test_full_decoder():
    """전체 Decoder Stack 테스트"""
    print("\n=== Full Decoder Stack 테스트 ===\n")
    
    # 파라미터
    batch_size = 2
    src_seq_length = 15
    tgt_seq_length = 12
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4
    
    # Encoder와 Decoder 생성
    encoder = create_encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )
    
    decoder = create_decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size
    )
    
    # 입력 생성
    src_ids = torch.randint(0, vocab_size, (batch_size, src_seq_length))
    tgt_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_length))
    
    # Masks
    src_mask = torch.ones(batch_size, src_seq_length)
    src_mask[0, 10:] = 0  # 첫 번째 source의 마지막 5개는 padding
    
    tgt_mask = torch.ones(batch_size, tgt_seq_length)
    tgt_mask[1, 9:] = 0  # 두 번째 target의 마지막 3개는 padding
    
    # Encoder forward
    with torch.no_grad():
        encoder_output = encoder(src_ids, src_mask)
    
    # Decoder forward
    output = decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
    
    print(f"Source 입력 shape: {src_ids.shape}")
    print(f"Target 입력 shape: {tgt_ids.shape}")
    print(f"Encoder 출력 shape: {encoder_output.shape}")
    print(f"Decoder 출력 shape: {output.shape}")
    print(f"예상 출력 shape: [batch_size={batch_size}, tgt_seq={tgt_seq_length}, vocab_size={vocab_size}]")
    print(f"\nDecoder 파라미터 수: {sum(p.numel() for p in decoder.parameters()):,}")
    
    return encoder, decoder, src_ids, tgt_ids, src_mask, tgt_mask, encoder_output, output


def visualize_attention_patterns(decoder, tgt_ids, encoder_output, src_mask, tgt_mask):
    """Decoder의 Attention 패턴 시각화"""
    print("\n=== Attention 패턴 시각화 ===\n")
    
    # Attention weights 가져오기
    _, self_attentions, cross_attentions = decoder(
        tgt_ids, encoder_output, src_mask, tgt_mask, return_attention=True
    )
    
    # 첫 번째 샘플, 첫 번째 layer의 attention 시각화
    sample_idx = 0
    layer_idx = 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Self-attention (평균)
    self_attn = self_attentions[layer_idx][sample_idx].mean(dim=0)
    im1 = ax1.imshow(self_attn.detach().numpy(), cmap='Blues', aspect='auto')
    ax1.set_title(f'Layer {layer_idx + 1} - Self Attention (Masked)')
    ax1.set_xlabel('Key/Value 위치')
    ax1.set_ylabel('Query 위치')
    
    # Look-ahead mask 영역 표시
    for i in range(self_attn.shape[0]):
        for j in range(i + 1, self_attn.shape[1]):
            ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=True, color='gray', alpha=0.3))
    
    plt.colorbar(im1, ax=ax1)
    
    # Cross-attention (평균)
    cross_attn = cross_attentions[layer_idx][sample_idx].mean(dim=0)
    im2 = ax2.imshow(cross_attn.detach().numpy(), cmap='Reds', aspect='auto')
    ax2.set_title(f'Layer {layer_idx + 1} - Cross Attention')
    ax2.set_xlabel('Encoder 위치')
    ax2.set_ylabel('Decoder 위치')
    
    # Source padding 영역 표시
    if src_mask is not None:
        src_len = src_mask[sample_idx].sum().int()
        ax2.axvline(x=src_len-0.5, color='blue', linestyle='--', alpha=0.5)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('outputs/decoder_attention_patterns.png', dpi=150)
    print("Attention 패턴이 'outputs/decoder_attention_patterns.png'에 저장되었습니다.")
    
    return self_attentions, cross_attentions


def test_look_ahead_mask():
    """Look-ahead mask 동작 확인"""
    print("\n=== Look-ahead Mask 테스트 ===\n")
    
    seq_length = 8
    mask = create_look_ahead_mask(seq_length)
    
    # 시각화
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[0, 0].numpy(), cmap='binary', aspect='auto')
    plt.title('Look-ahead Mask')
    plt.xlabel('Key/Value 위치')
    plt.ylabel('Query 위치')
    
    # 격자 추가
    for i in range(seq_length):
        plt.axhline(y=i+0.5, color='gray', linestyle='-', linewidth=0.5)
        plt.axvline(x=i+0.5, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/look_ahead_mask.png', dpi=150)
    print("Look-ahead mask가 'outputs/look_ahead_mask.png'에 저장되었습니다.")
    
    # 마스크 값 확인
    print("\nMask 값 (0=attend, 1=mask):")
    print(mask[0, 0].numpy().astype(int))


def test_generation():
    """Autoregressive 생성 테스트"""
    print("\n=== Autoregressive 생성 테스트 ===\n")
    
    # 작은 모델로 테스트
    vocab_size = 100
    d_model = 128
    num_heads = 4
    
    encoder = create_encoder(
        num_layers=2,
        d_model=d_model,
        num_heads=num_heads,
        vocab_size=vocab_size
    )
    
    decoder = create_decoder(
        num_layers=2,
        d_model=d_model,
        num_heads=num_heads,
        vocab_size=vocab_size
    )
    
    # Source 시퀀스
    src_ids = torch.randint(0, vocab_size, (1, 10))
    
    # Encoder forward
    with torch.no_grad():
        encoder_output = encoder(src_ids)
    
    # 생성
    start_token = 1  # <start>
    end_token = 2    # <end>
    
    generated = decoder.generate(
        encoder_output,
        start_token_id=start_token,
        end_token_id=end_token,
        max_length=20,
        temperature=0.8,
        top_k=10
    )
    
    print(f"Source shape: {src_ids.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated sequence: {generated[0].tolist()}")
    
    # 생성 과정 시각화
    visualize_generation_process(decoder, encoder_output, generated)


def visualize_generation_process(decoder, encoder_output, generated):
    """생성 과정의 attention 시각화"""
    print("\n생성 과정 attention 시각화...")
    
    # 마지막 생성 스텝의 attention
    with torch.no_grad():
        _, self_attns, cross_attns = decoder(
            generated, encoder_output, return_attention=True
        )
    
    # 마지막 layer의 cross-attention
    last_cross_attn = cross_attns[-1][0].mean(dim=0)  # 평균 over heads
    
    plt.figure(figsize=(10, 6))
    plt.imshow(last_cross_attn.detach().numpy(), cmap='hot', aspect='auto')
    plt.title('생성 과정의 Cross-Attention (마지막 layer)')
    plt.xlabel('Source 위치')
    plt.ylabel('Generated 위치')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('outputs/generation_attention.png', dpi=150)
    print("생성 attention이 'outputs/generation_attention.png'에 저장되었습니다.")


def analyze_layer_contributions():
    """Decoder 각 Layer의 기여도 분석"""
    print("\n=== Decoder Layer별 기여도 분석 ===\n")
    
    # 작은 모델로 테스트
    decoder = create_decoder(
        num_layers=6,
        d_model=128,
        num_heads=4,
        d_ff=512,
        vocab_size=100,
        dropout=0.0
    )
    
    # 더미 입력
    batch_size = 1
    tgt_seq = 20
    src_seq = 15
    d_model = 128
    
    tgt_ids = torch.randint(0, 100, (batch_size, tgt_seq))
    encoder_output = torch.randn(batch_size, src_seq, d_model)
    
    # 각 layer 후의 변화량 측정
    x = decoder.embeddings(tgt_ids)
    prev_output = x
    
    look_ahead_mask = create_look_ahead_mask(tgt_seq)
    
    self_changes = []
    cross_changes = []
    ffn_changes = []
    
    for i, layer in enumerate(decoder.layers):
        # Self-attention 후
        normalized = layer.self_attn_residual.norm(prev_output)
        self_attn_out = layer.self_attention(normalized, normalized, normalized, look_ahead_mask)
        self_attn_out = prev_output + layer.self_attn_residual.dropout(self_attn_out)
        self_change = (self_attn_out - prev_output).norm() / prev_output.norm()
        self_changes.append(self_change.item())
        
        # Cross-attention 후
        normalized = layer.cross_attn_residual.norm(self_attn_out)
        cross_attn_out = layer.cross_attention(normalized, encoder_output, encoder_output)
        cross_attn_out = self_attn_out + layer.cross_attn_residual.dropout(cross_attn_out)
        cross_change = (cross_attn_out - self_attn_out).norm() / self_attn_out.norm()
        cross_changes.append(cross_change.item())
        
        # FFN 후
        curr_output = layer.ffn_residual(cross_attn_out, layer.feed_forward)
        ffn_change = (curr_output - cross_attn_out).norm() / cross_attn_out.norm()
        ffn_changes.append(ffn_change.item())
        
        print(f"Layer {i+1} - Self: {self_change:.4f}, Cross: {cross_change:.4f}, FFN: {ffn_change:.4f}")
        prev_output = curr_output
    
    # 시각화
    layers = list(range(1, len(self_changes) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, self_changes, 'b-o', label='Self-Attention')
    plt.plot(layers, cross_changes, 'r-s', label='Cross-Attention')
    plt.plot(layers, ffn_changes, 'g-^', label='Feed-Forward')
    
    plt.xlabel('Layer')
    plt.ylabel('상대적 변화량')
    plt.title('Decoder 각 Sub-layer의 기여도')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/decoder_layer_contributions.png', dpi=150)
    print("\n기여도 그래프가 'outputs/decoder_layer_contributions.png'에 저장되었습니다.")


def compare_decoder_architectures():
    """다양한 Decoder 아키텍처 비교"""
    print("\n=== Decoder 아키텍처 비교 ===\n")
    
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
        decoder = create_decoder(vocab_size=30000, **config)
        params = sum(p.numel() for p in decoder.parameters())
        
        print(f"{name:12} | {config['num_layers']:6} | {config['d_model']:7} | "
              f"{config['num_heads']:5} | {config['d_ff']:4} | {params:,}")


if __name__ == "__main__":
    # 1. Decoder Layer 테스트
    decoder_layer, decoder_input, encoder_output, output = test_decoder_layer()
    
    # 2. Full Decoder 테스트
    encoder, decoder, src_ids, tgt_ids, src_mask, tgt_mask, encoder_output, output = test_full_decoder()
    
    # 3. Look-ahead mask 테스트
    test_look_ahead_mask()
    
    # 4. Attention 패턴 시각화
    self_attentions, cross_attentions = visualize_attention_patterns(
        decoder, tgt_ids, encoder_output, src_mask, tgt_mask
    )
    
    # 5. Autoregressive 생성 테스트
    test_generation()
    
    # 6. Layer별 기여도 분석
    analyze_layer_contributions()
    
    # 7. 아키텍처 비교
    compare_decoder_architectures()
    
    print("\n모든 테스트가 완료되었습니다!")