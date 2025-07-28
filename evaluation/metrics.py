"""
Evaluation Metrics 구현

다양한 NLP 태스크에 대한 평가 메트릭을 구현합니다.
"""

import warnings

import numpy as np
import torch

# Optional dependencies를 try-except로 처리
try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
except ImportError:
    warnings.warn("scikit-learn not installed. Some metrics may not be available.", stacklevel=2)
    accuracy_score = None
    f1_score = None
    precision_recall_fscore_support = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    warnings.warn("rouge-score not installed. ROUGE metrics will not be available.", stacklevel=2)
    rouge_scorer = None

try:
    import sacrebleu
except ImportError:
    warnings.warn("sacrebleu not installed. BLEU metrics will not be available.", stacklevel=2)
    sacrebleu = None


class Metric:
    """평가 메트릭 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        """메트릭 초기화"""
        raise NotImplementedError

    def update(self, predictions, references):
        """예측값과 정답으로 메트릭 업데이트"""
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        """최종 메트릭 계산"""
        raise NotImplementedError


class AccuracyMetric(Metric):
    """정확도 메트릭"""

    def __init__(self):
        super().__init__("accuracy")
        if accuracy_score is None:
            raise ImportError("scikit-learn is required for AccuracyMetric")

    def reset(self):
        self.predictions = []
        self.references = []

    def update(self, predictions: torch.Tensor, references: torch.Tensor):
        """배치 업데이트"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)

        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.references.extend(references.cpu().numpy().tolist())

    def compute(self) -> dict[str, float]:
        """정확도 계산"""
        acc = accuracy_score(self.references, self.predictions)
        return {self.name: acc}


class F1Metric(Metric):
    """F1 Score 메트릭"""

    def __init__(self, average: str = "macro"):
        super().__init__(f"f1_{average}")
        if f1_score is None:
            raise ImportError("scikit-learn is required for F1Metric")
        self.average = average

    def reset(self):
        self.predictions = []
        self.references = []

    def update(self, predictions: torch.Tensor, references: torch.Tensor):
        """배치 업데이트"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)

        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.references.extend(references.cpu().numpy().tolist())

    def compute(self) -> dict[str, float]:
        """F1 score 계산"""
        f1 = f1_score(self.references, self.predictions, average=self.average)
        precision, recall, _, _ = precision_recall_fscore_support(
            self.references, self.predictions, average=self.average
        )

        return {
            self.name: f1,
            f"precision_{self.average}": precision,
            f"recall_{self.average}": recall,
        }


class PerplexityMetric(Metric):
    """Perplexity 메트릭 (언어 모델링)"""

    def __init__(self):
        super().__init__("perplexity")

    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, loss: torch.Tensor, num_tokens: int):
        """배치 업데이트"""
        self.total_loss += loss.item() * num_tokens
        self.total_tokens += num_tokens

    def compute(self) -> dict[str, float]:
        """Perplexity 계산"""
        avg_loss = self.total_loss / self.total_tokens
        perplexity = np.exp(avg_loss)

        return {self.name: perplexity, "loss": avg_loss}


class BLEUMetric(Metric):
    """BLEU Score 메트릭 (번역)"""

    def __init__(self, tokenizer=None):
        super().__init__("bleu")
        if sacrebleu is None:
            raise ImportError("sacrebleu is required for BLEUMetric")
        self.tokenizer = tokenizer

    def reset(self):
        self.predictions = []
        self.references = []

    def update(self, predictions: list[str], references: list[str]):
        """배치 업데이트"""
        self.predictions.extend(predictions)
        self.references.extend(references)

    def compute(self) -> dict[str, float]:
        """BLEU score 계산"""
        # SacreBLEU 사용
        bleu = sacrebleu.corpus_bleu(self.predictions, [self.references])

        return {
            self.name: bleu.score,
            "bleu_bp": bleu.bp,  # Brevity penalty
            "bleu_precisions": bleu.precisions,  # n-gram precisions
        }


class ROUGEMetric(Metric):
    """ROUGE Score 메트릭 (요약)"""

    def __init__(self, rouge_types: list[str] | None = None):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]
        if rouge_scorer is None:
            raise ImportError("rouge-score is required for ROUGEMetric")
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        super().__init__("rouge")

    def reset(self):
        self.scores = {rouge_type: [] for rouge_type in self.rouge_types}

    def update(self, predictions: list[str], references: list[str]):
        """배치 업데이트"""
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                self.scores[rouge_type].append(scores[rouge_type].fmeasure)

    def compute(self) -> dict[str, float]:
        """ROUGE scores 계산"""
        results = {}
        for rouge_type in self.rouge_types:
            results[rouge_type] = np.mean(self.scores[rouge_type])

        return results


class ExactMatchMetric(Metric):
    """Exact Match 메트릭 (QA)"""

    def __init__(self):
        super().__init__("exact_match")

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: list[str], references: list[str]):
        """배치 업데이트"""
        for pred, ref in zip(predictions, references):
            # 정규화: 소문자 변환, 공백 정리
            pred_normalized = pred.lower().strip()
            ref_normalized = ref.lower().strip()

            if pred_normalized == ref_normalized:
                self.correct += 1
            self.total += 1

    def compute(self) -> dict[str, float]:
        """Exact match 계산"""
        em = self.correct / self.total if self.total > 0 else 0.0
        return {self.name: em}


class TokenAccuracyMetric(Metric):
    """토큰 수준 정확도 메트릭"""

    def __init__(self, ignore_index: int = -100):
        super().__init__("token_accuracy")
        self.ignore_index = ignore_index

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, references: torch.Tensor):
        """배치 업데이트"""
        if predictions.dim() == 3:  # (batch_size, seq_len, vocab_size)
            predictions = predictions.argmax(dim=-1)

        # ignore_index 제외
        mask = references != self.ignore_index

        self.correct += ((predictions == references) & mask).sum().item()
        self.total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """토큰 정확도 계산"""
        acc = self.correct / self.total if self.total > 0 else 0.0
        return {self.name: acc}


class MetricCollection:
    """여러 메트릭을 관리하는 컬렉션"""

    def __init__(self, metrics: list[Metric]):
        self.metrics = {metric.name: metric for metric in metrics}

    def reset(self):
        """모든 메트릭 초기화"""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, **kwargs):
        """메트릭 업데이트"""
        for name, metric in self.metrics.items():
            # 메트릭에 필요한 인자가 있는지 확인
            if name == "accuracy" or name.startswith("f1"):
                if "predictions" in kwargs and "references" in kwargs:
                    metric.update(kwargs["predictions"], kwargs["references"])
            elif name == "perplexity":
                if "loss" in kwargs and "num_tokens" in kwargs:
                    metric.update(kwargs["loss"], kwargs["num_tokens"])
            elif name == "bleu" or name == "rouge" or name == "exact_match":
                if "pred_texts" in kwargs and "ref_texts" in kwargs:
                    metric.update(kwargs["pred_texts"], kwargs["ref_texts"])
            elif name == "token_accuracy" and "predictions" in kwargs and "references" in kwargs:
                metric.update(kwargs["predictions"], kwargs["references"])

    def compute(self) -> dict[str, float]:
        """모든 메트릭 계산"""
        results = {}
        for metric in self.metrics.values():
            results.update(metric.compute())
        return results


def get_metrics_for_task(task: str) -> MetricCollection:
    """태스크에 맞는 메트릭 컬렉션 반환"""

    metrics = []

    if task == "classification":
        if accuracy_score is not None:
            metrics.extend(
                [AccuracyMetric(), F1Metric(average="macro"), F1Metric(average="weighted")]
            )

    elif task == "language_modeling":
        metrics.extend([PerplexityMetric(), TokenAccuracyMetric()])

    elif task == "translation":
        if sacrebleu is not None:
            metrics.append(BLEUMetric())
        metrics.append(TokenAccuracyMetric())

    elif task == "summarization":
        if rouge_scorer is not None:
            metrics.append(ROUGEMetric())
        metrics.append(TokenAccuracyMetric())

    elif task == "question_answering":
        metrics.append(ExactMatchMetric())
        if f1_score is not None:
            metrics.append(F1Metric(average="macro"))

    else:
        warnings.warn(f"Unknown task: {task}. Using default metrics.", stacklevel=2)
        if accuracy_score is not None:
            metrics.append(AccuracyMetric())
        metrics.append(TokenAccuracyMetric())

    return MetricCollection(metrics)


# Early Stopping 클래스
class EarlyStopping:
    """Early Stopping 기능"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: 성능 개선이 없을 때 기다리는 epoch 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'min' 또는 'max' (메트릭이 작을수록 좋은지, 클수록 좋은지)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

        if mode == "min":
            self.is_better = lambda a, b: a < b - min_delta
        else:
            self.is_better = lambda a, b: a > b + min_delta

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: 현재 epoch의 점수

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
