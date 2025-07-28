"""
Loss 함수 테스트 및 분석
"""

import sys

sys.path.append(".")


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from training.loss import (
    ContrastiveLoss,
    CrossEntropyLoss,
    FocalLoss,
    MaskedLanguageModelingLoss,
    SequenceGenerationLoss,
)


def test_cross_entropy_loss():
    """Cross Entropy Loss 테스트"""
    print("=== Cross Entropy Loss 테스트 ===\n")

    batch_size = 4
    seq_len = 10
    vocab_size = 100

    # 랜덤 logits과 targets 생성
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 일부를 padding으로 설정
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[0, 7:] = 0  # 첫 번째 샘플의 마지막 3개
    padding_mask[1, 8:] = 0  # 두 번째 샘플의 마지막 2개

    # 1. 기본 Cross Entropy
    print("1. 기본 Cross Entropy Loss:")
    criterion = CrossEntropyLoss(ignore_index=-100)
    loss = criterion(logits, targets, padding_mask)
    print(f"   Loss: {loss.item():.4f}")

    # 2. Label Smoothing 적용
    print("\n2. Label Smoothing (0.1) 적용:")
    criterion_smooth = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    loss_smooth = criterion_smooth(logits, targets, padding_mask)
    print(f"   Loss without smoothing: {loss.item():.4f}")
    print(f"   Loss with smoothing: {loss_smooth.item():.4f}")
    print(f"   Difference: {(loss_smooth - loss).item():.4f}")

    # 3. Reduction 옵션 테스트
    print("\n3. Reduction 옵션:")
    for reduction in ["mean", "sum", "none"]:
        criterion_red = CrossEntropyLoss(reduction=reduction)
        loss_red = criterion_red(logits, targets, padding_mask)
        if reduction == "none":
            print(f"   {reduction}: shape {loss_red.shape}")
        else:
            print(f"   {reduction}: {loss_red.item():.4f}")

    return logits, targets, padding_mask


def test_sequence_generation_loss():
    """Sequence Generation Loss 테스트"""
    print("\n=== Sequence Generation Loss 테스트 ===\n")

    batch_size = 4
    seq_len = 20
    vocab_size = 1000

    # 모델 출력과 타겟 생성
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len))

    # 패딩 추가
    pad_token_id = 0
    targets[0, 15:] = pad_token_id
    targets[1, 17:] = pad_token_id

    # Loss 계산
    criterion = SequenceGenerationLoss(
        vocab_size=vocab_size, pad_token_id=pad_token_id, label_smoothing=0.1
    )

    loss, metrics = criterion(logits, targets)

    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Correct tokens: {metrics['correct_tokens']}")

    return loss, metrics


def test_masked_lm_loss():
    """Masked Language Modeling Loss 테스트"""
    print("\n=== Masked LM Loss 테스트 ===\n")

    batch_size = 4
    seq_len = 128
    vocab_size = 30000

    # 모델 출력 시뮬레이션
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 15%를 마스킹
    masked_positions = torch.rand(batch_size, seq_len) < 0.15

    # MLM Loss
    criterion = MaskedLanguageModelingLoss(vocab_size=vocab_size, label_smoothing=0.0)

    loss, accuracy = criterion(predictions, labels, masked_positions)

    num_masked = masked_positions.sum().item()
    print(f"마스킹된 토큰 수: {num_masked}")
    print(f"전체 토큰 수: {batch_size * seq_len}")
    print(f"마스킹 비율: {num_masked / (batch_size * seq_len):.1%}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item():.2%}")

    return loss, accuracy


def test_contrastive_loss():
    """Contrastive Loss 테스트"""
    print("\n=== Contrastive Loss 테스트 ===\n")

    batch_size = 32
    embedding_dim = 256

    # 두 개의 뷰에서 임베딩 생성
    embeddings1 = torch.randn(batch_size, embedding_dim)
    embeddings2 = embeddings1 + torch.randn(batch_size, embedding_dim) * 0.1  # 약간의 노이즈 추가

    # Negative samples 추가
    embeddings2[batch_size // 2 :] = torch.randn(batch_size // 2, embedding_dim)

    # Loss 계산
    criterion = ContrastiveLoss(temperature=0.07)
    loss = criterion(embeddings1, embeddings2)

    print(f"Contrastive Loss: {loss.item():.4f}")

    # Temperature 영향 분석
    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
    losses = []

    for temp in temperatures:
        criterion_temp = ContrastiveLoss(temperature=temp)
        loss_temp = criterion_temp(embeddings1, embeddings2)
        losses.append(loss_temp.item())

    print("\nTemperature별 Loss:")
    for temp, loss_val in zip(temperatures, losses):
        print(f"  τ={temp}: {loss_val:.4f}")

    return temperatures, losses


def test_focal_loss():
    """Focal Loss 테스트"""
    print("\n=== Focal Loss 테스트 ===\n")

    num_classes = 10
    batch_size = 100

    # 불균형 데이터 시뮬레이션 (클래스 0이 90%)
    targets = torch.zeros(batch_size, dtype=torch.long)
    targets[90:] = torch.randint(1, num_classes, (10,))

    # Logits 생성 (클래스 0에 편향된 예측)
    logits = torch.randn(batch_size, num_classes)
    logits[:, 0] += 2.0  # 클래스 0에 높은 점수

    # 다양한 gamma 값으로 테스트
    gammas = [0.0, 0.5, 1.0, 2.0, 5.0]
    losses = []

    for gamma in gammas:
        criterion = FocalLoss(gamma=gamma)
        loss = criterion(logits, targets)
        losses.append(loss.item())

    print("Gamma별 Focal Loss:")
    for gamma, loss_val in zip(gammas, losses):
        print(f"  γ={gamma}: {loss_val:.4f}")

    # Alpha 가중치 추가 테스트
    alpha = torch.ones(num_classes)
    alpha[0] = 0.25  # 다수 클래스의 가중치 감소

    criterion_alpha = FocalLoss(alpha=alpha, gamma=2.0)
    loss_alpha = criterion_alpha(logits, targets)
    print(f"\nAlpha 가중치 적용 후 Loss: {loss_alpha.item():.4f}")

    return gammas, losses


def visualize_label_smoothing_effect():
    """Label Smoothing 효과 시각화"""
    print("\n=== Label Smoothing 효과 시각화 ===\n")

    vocab_size = 10
    smoothing_values = [0.0, 0.1, 0.2, 0.3]

    fig, axes = plt.subplots(1, len(smoothing_values), figsize=(15, 4))

    for idx, smoothing in enumerate(smoothing_values):
        # 원-핫 분포
        target_dist = torch.zeros(vocab_size)
        target_dist[3] = 1.0  # 정답은 인덱스 3

        # Label smoothing 적용
        if smoothing > 0:
            target_dist = (1.0 - smoothing) * target_dist + smoothing / vocab_size

        # 시각화
        ax = axes[idx]
        ax.bar(range(vocab_size), target_dist.numpy())
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Token ID")
        ax.set_ylabel("Probability")
        ax.set_title(f"Label Smoothing = {smoothing}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/label_smoothing_effect.png", dpi=150)
    print("Label smoothing 효과가 'outputs/label_smoothing_effect.png'에 저장되었습니다.")


def visualize_focal_loss_weights():
    """Focal Loss 가중치 시각화"""
    print("\n=== Focal Loss 가중치 시각화 ===\n")

    # 확률 범위
    p = torch.linspace(0.01, 0.99, 100)

    # 다양한 gamma 값
    gammas = [0, 0.5, 1, 2, 5]

    plt.figure(figsize=(10, 6))

    for gamma in gammas:
        focal_weight = (1 - p) ** gamma
        plt.plot(p.numpy(), focal_weight.numpy(), label=f"γ={gamma}")

    plt.xlabel("예측 확률 (p)")
    plt.ylabel("Focal Weight: (1-p)^γ")
    plt.title("Focal Loss 가중치 함수")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/focal_loss_weights.png", dpi=150)
    print("Focal loss 가중치가 'outputs/focal_loss_weights.png'에 저장되었습니다.")


def compare_loss_functions():
    """다양한 Loss 함수 비교"""
    print("\n=== Loss 함수 비교 ===\n")

    batch_size = 32
    seq_len = 50
    vocab_size = 1000

    # 공통 데이터 생성
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    mask[:, 40:] = 0  # 마지막 10개는 패딩

    # 다양한 loss 함수
    losses = {
        "Standard CE": CrossEntropyLoss(),
        "CE + Smoothing(0.1)": CrossEntropyLoss(label_smoothing=0.1),
        "CE + Smoothing(0.2)": CrossEntropyLoss(label_smoothing=0.2),
    }

    results = {}
    for name, criterion in losses.items():
        loss = criterion(logits, targets, mask)
        results[name] = loss.item()

    # 결과 출력
    print("Loss 함수별 결과:")
    print("-" * 40)
    for name, loss_val in results.items():
        print(f"{name:20s}: {loss_val:.4f}")

    # Perplexity 계산
    print("\nPerplexity:")
    print("-" * 40)
    for name, loss_val in results.items():
        perplexity = np.exp(min(loss_val, 100))  # Overflow 방지
        print(f"{name:20s}: {perplexity:.2f}")


def test_gradient_behavior():
    """Loss 함수들의 gradient 동작 테스트"""
    print("\n=== Gradient 동작 테스트 ===\n")

    # 간단한 모델
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            return self.output(self.embedding(x))

    vocab_size = 100
    hidden_size = 64
    model = SimpleModel(vocab_size, hidden_size)

    # 데이터
    inputs = torch.randint(0, vocab_size, (4, 10))
    targets = torch.randint(0, vocab_size, (4, 10))

    # 다양한 loss로 gradient 계산
    loss_fns = {
        "Standard": CrossEntropyLoss(),
        "Smoothed": CrossEntropyLoss(label_smoothing=0.1),
        "Focal": FocalLoss(gamma=2.0),
    }

    gradient_norms = {}

    for name, criterion in loss_fns.items():
        model.zero_grad()

        outputs = model(inputs)
        if name == "Focal":
            # Focal loss는 2D input 필요
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        else:
            loss = criterion(outputs, targets)

        loss.backward()

        # Gradient norm 계산
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm**0.5

        gradient_norms[name] = total_norm

    print("Loss 함수별 Gradient Norm:")
    for name, norm in gradient_norms.items():
        print(f"  {name}: {norm:.4f}")


if __name__ == "__main__":
    # 1. Cross Entropy Loss 테스트
    logits, targets, mask = test_cross_entropy_loss()

    # 2. Sequence Generation Loss 테스트
    loss, metrics = test_sequence_generation_loss()

    # 3. Masked LM Loss 테스트
    mlm_loss, mlm_acc = test_masked_lm_loss()

    # 4. Contrastive Loss 테스트
    temps, contrastive_losses = test_contrastive_loss()

    # 5. Focal Loss 테스트
    gammas, focal_losses = test_focal_loss()

    # 6. Label Smoothing 효과 시각화
    visualize_label_smoothing_effect()

    # 7. Focal Loss 가중치 시각화
    visualize_focal_loss_weights()

    # 8. Loss 함수 비교
    compare_loss_functions()

    # 9. Gradient 동작 테스트
    test_gradient_behavior()

    print("\n모든 테스트가 완료되었습니다!")
