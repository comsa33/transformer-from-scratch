"""
Evaluation Metrics 테스트
"""

import sys

sys.path.append(".")


import matplotlib.pyplot as plt
import numpy as np
import torch

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from evaluation.metrics import (
    AccuracyMetric,
    BLEUMetric,
    EarlyStopping,
    F1Metric,
    PerplexityMetric,
    ROUGEMetric,
    TokenAccuracyMetric,
    get_metrics_for_task,
)


def test_accuracy_metric():
    """Accuracy 메트릭 테스트"""
    print("=== Accuracy Metric 테스트 ===\n")

    try:
        metric = AccuracyMetric()

        # 배치 1
        predictions = torch.tensor([0, 1, 2, 1, 0])
        references = torch.tensor([0, 1, 2, 0, 0])
        metric.update(predictions, references)

        # 배치 2 (logits 형태)
        logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
        references2 = torch.tensor([0, 1, 2])
        metric.update(logits, references2)

        result = metric.compute()
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Expected: {7 / 8:.4f} (7 correct out of 8)")
    except ImportError as e:
        print(f"AccuracyMetric 테스트 생략: {e}")


def test_f1_metric():
    """F1 메트릭 테스트"""
    print("\n=== F1 Metric 테스트 ===\n")

    try:
        # Macro F1
        metric_macro = F1Metric(average="macro")

        predictions = torch.tensor([0, 1, 2, 1, 0, 2, 2, 0])
        references = torch.tensor([0, 1, 2, 0, 0, 2, 1, 0])

        metric_macro.update(predictions, references)
        result_macro = metric_macro.compute()

        print("Macro Average:")
        print(f"  F1 Score: {result_macro['f1_macro']:.4f}")
        print(f"  Precision: {result_macro['precision_macro']:.4f}")
        print(f"  Recall: {result_macro['recall_macro']:.4f}")

        # Weighted F1
        metric_weighted = F1Metric(average="weighted")
        metric_weighted.update(predictions, references)
        result_weighted = metric_weighted.compute()

        print("\nWeighted Average:")
        print(f"  F1 Score: {result_weighted['f1_weighted']:.4f}")
    except ImportError as e:
        print(f"F1Metric 테스트 생략: {e}")


def test_perplexity_metric():
    """Perplexity 메트릭 테스트"""
    print("\n=== Perplexity Metric 테스트 ===\n")

    metric = PerplexityMetric()

    # 여러 배치 시뮬레이션
    losses = [2.5, 2.3, 2.1, 1.9, 1.7]
    tokens_per_batch = 100

    for loss in losses:
        metric.update(torch.tensor(loss), tokens_per_batch)

    result = metric.compute()
    print(f"Average Loss: {result['loss']:.4f}")
    print(f"Perplexity: {result['perplexity']:.4f}")


def test_bleu_metric():
    """BLEU 메트릭 테스트"""
    print("\n=== BLEU Metric 테스트 ===\n")

    try:
        metric = BLEUMetric()

        # 예제 번역
        predictions = [
            "The cat is on the mat",
            "I love machine learning",
            "This is a test sentence",
        ]

        references = ["The cat is on the mat", "I love deep learning", "This is a sample sentence"]

        metric.update(predictions, references)
        result = metric.compute()

        print(f"BLEU Score: {result['bleu']:.2f}")
        print(f"Brevity Penalty: {result['bleu_bp']:.4f}")
        print(f"N-gram Precisions: {result['bleu_precisions']}")
    except ImportError as e:
        print(f"BLEUMetric 테스트 생략: {e}")


def test_rouge_metric():
    """ROUGE 메트릭 테스트"""
    print("\n=== ROUGE Metric 테스트 ===\n")

    try:
        metric = ROUGEMetric()

        predictions = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
        ]

        references = [
            "A quick brown fox jumped over a lazy dog",
            "Machine learning is part of artificial intelligence",
        ]

        metric.update(predictions, references)
        result = metric.compute()

        for rouge_type, score in result.items():
            print(f"{rouge_type}: {score:.4f}")
    except ImportError as e:
        print(f"ROUGEMetric 테스트 생략: {e}")


def test_token_accuracy_metric():
    """Token Accuracy 메트릭 테스트"""
    print("\n=== Token Accuracy Metric 테스트 ===\n")

    metric = TokenAccuracyMetric(ignore_index=-100)

    # 배치 예제
    predictions = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    references = torch.tensor(
        [[1, 2, 3, 4, -100], [6, 7, 9, 9, 10]]  # 마지막 토큰은 무시  # 3번째 토큰 틀림
    )

    metric.update(predictions, references)
    result = metric.compute()

    print(f"Token Accuracy: {result['token_accuracy']:.4f}")
    print(f"Expected: {8 / 9:.4f} (8 correct out of 9 valid tokens)")


def test_metric_collection():
    """MetricCollection 테스트"""
    print("\n=== MetricCollection 테스트 ===\n")

    # 분류 태스크용 메트릭 컬렉션
    metrics = get_metrics_for_task("classification")

    # sklearn이 없으면 다른 태스크로 테스트
    if len(metrics.metrics) == 0:
        print("Classification metrics not available, testing language modeling metrics instead")
        metrics = get_metrics_for_task("language_modeling")

        # 언어 모델링 메트릭 테스트
        loss = torch.tensor(2.0)
        num_tokens = 100
        predictions = torch.randint(0, 100, (2, 10))
        references = torch.randint(0, 100, (2, 10))

        metrics.update(
            loss=loss, num_tokens=num_tokens, predictions=predictions, references=references
        )
    else:
        # 분류 메트릭 테스트
        predictions = torch.tensor(
            [[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0], [2.0, 0.1, 0.1]]
        )
        references = torch.tensor([0, 1, 2, 1])

        metrics.update(predictions=predictions, references=references)

    # 계산
    results = metrics.compute()

    print("Metrics:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")


def test_early_stopping():
    """Early Stopping 테스트"""
    print("\n=== Early Stopping 테스트 ===\n")

    # Early stopping 초기화 (낮을수록 좋은 메트릭)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode="min")

    # 손실값 시뮬레이션
    losses = [1.0, 0.9, 0.85, 0.84, 0.835, 0.834, 0.833, 0.832]

    print("Training Progress:")
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss)
        print(
            f"  Epoch {epoch + 1}: Loss={loss:.3f}, "
            f"Best={early_stopping.best_score:.3f}, "
            f"Counter={early_stopping.counter}, "
            f"Stop={should_stop}"
        )

        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break


def visualize_metric_comparison():
    """여러 메트릭 비교 시각화"""
    print("\n=== 메트릭 비교 시각화 ===\n")

    # 에폭별 메트릭 시뮬레이션
    epochs = np.arange(1, 21)

    # 메트릭 값 시뮬레이션
    train_loss = 2.5 * np.exp(-0.15 * epochs) + 0.3 + 0.05 * np.random.randn(20)
    val_loss = 2.5 * np.exp(-0.12 * epochs) + 0.35 + 0.08 * np.random.randn(20)

    train_acc = 1 - np.exp(-0.2 * epochs) + 0.05 * np.random.randn(20)
    val_acc = 1 - np.exp(-0.18 * epochs) + 0.08 * np.random.randn(20)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss 플롯
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, "r--", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("학습 및 검증 손실")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy 플롯
    ax2.plot(epochs, train_acc, "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_acc, "r--", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("학습 및 검증 정확도")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig("outputs/metric_comparison.png", dpi=150)
    print("메트릭 비교 그래프가 'outputs/metric_comparison.png'에 저장되었습니다.")


def visualize_confusion_matrix():
    """혼동 행렬 시각화"""
    print("\n=== 혼동 행렬 시각화 ===\n")

    # 예제 데이터 (3개 클래스)
    n_classes = 3
    class_names = ["Class A", "Class B", "Class C"]

    # 혼동 행렬 생성 (실제보다는 시뮬레이션)
    confusion_matrix = np.array([[85, 10, 5], [15, 75, 10], [8, 12, 80]])

    # 정규화
    confusion_matrix_normalized = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 원본 혼동 행렬
    ax1.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("혼동 행렬 (Count)")

    # 텍스트 주석
    for i in range(n_classes):
        for j in range(n_classes):
            ax1.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

    ax1.set_xticks(np.arange(n_classes))
    ax1.set_yticks(np.arange(n_classes))
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel("예측 레이블")
    ax1.set_ylabel("실제 레이블")

    # 정규화된 혼동 행렬
    im2 = ax2.imshow(confusion_matrix_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax2.set_title("정규화된 혼동 행렬 (%)")

    # 텍스트 주석
    for i in range(n_classes):
        for j in range(n_classes):
            ax2.text(
                j,
                i,
                f"{confusion_matrix_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    ax2.set_xticks(np.arange(n_classes))
    ax2.set_yticks(np.arange(n_classes))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel("예측 레이블")
    ax2.set_ylabel("실제 레이블")

    # 컬러바 추가
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("혼동 행렬이 'outputs/confusion_matrix.png'에 저장되었습니다.")


if __name__ == "__main__":
    # 1. Accuracy 메트릭 테스트
    test_accuracy_metric()

    # 2. F1 메트릭 테스트
    test_f1_metric()

    # 3. Perplexity 메트릭 테스트
    test_perplexity_metric()

    # 4. BLEU 메트릭 테스트
    test_bleu_metric()

    # 5. ROUGE 메트릭 테스트
    test_rouge_metric()

    # 6. Token Accuracy 메트릭 테스트
    test_token_accuracy_metric()

    # 7. MetricCollection 테스트
    test_metric_collection()

    # 8. Early Stopping 테스트
    test_early_stopping()

    # 9. 메트릭 비교 시각화
    visualize_metric_comparison()

    # 10. 혼동 행렬 시각화
    visualize_confusion_matrix()

    print("\n모든 테스트가 완료되었습니다!")
