"""
Transformer 학습을 위한 Loss 함수들

Cross-entropy loss with label smoothing, masked loss 등을 구현합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossEntropyLoss(nn.Module):
    """패딩을 고려한 Cross Entropy Loss
    
    Args:
        ignore_index: 무시할 토큰 인덱스 (보통 padding token)
        reduction: 'mean', 'sum', 'none' 중 선택
        label_smoothing: Label smoothing 값 (0.0 ~ 1.0)
    """
    
    def __init__(self, 
                 ignore_index: int = -100,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] 또는 [batch_size * seq_len, vocab_size]
            targets: [batch_size, seq_len] 또는 [batch_size * seq_len]
            mask: [batch_size, seq_len] 또는 [batch_size * seq_len] - 유효한 위치는 1
            
        Returns:
            loss: 스칼라 또는 reduction='none'인 경우 텐서
        """
        # Reshape if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
        
        # Label smoothing이 있는 경우
        if self.label_smoothing > 0:
            loss = self._label_smoothed_nll_loss(logits, targets)
        else:
            loss = F.cross_entropy(
                logits, 
                targets, 
                ignore_index=self.ignore_index,
                reduction='none'
            )
        
        # Mask 적용
        if mask is not None:
            loss = loss * mask
            
        # Reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1.0)
            else:
                # ignore_index가 있는 경우를 위해 유효한 토큰 수 계산
                valid_tokens = (targets != self.ignore_index).sum()
                return loss.sum() / valid_tokens.clamp(min=1.0)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
    
    def _label_smoothed_nll_loss(self, 
                                 logits: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """Label smoothing을 적용한 negative log likelihood loss
        
        Label smoothing은 target distribution을 다음과 같이 변경:
        - 정답 레이블: 1 - smoothing
        - 나머지 레이블: smoothing / (vocab_size - 1)
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 정답에 대한 loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Label smoothing: 전체 분포에 대한 KL divergence
        smooth_loss = -log_probs.mean(dim=-1)
        
        # 조합
        loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        
        # ignore_index 처리
        if self.ignore_index >= 0:
            pad_mask = targets == self.ignore_index
            loss = loss.masked_fill(pad_mask, 0.0)
        
        return loss


class MaskedLanguageModelingLoss(nn.Module):
    """BERT 스타일 Masked Language Modeling Loss
    
    마스킹된 토큰들에 대해서만 loss를 계산합니다.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.criterion = CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='none',
            label_smoothing=label_smoothing
        )
    
    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                masked_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions: [batch_size, seq_len, hidden_size] - 모델 출력
            labels: [batch_size, seq_len] - 원본 토큰 ID
            masked_positions: [batch_size, seq_len] - 마스킹된 위치는 1
            
        Returns:
            loss: 평균 loss
            accuracy: 마스킹된 토큰에 대한 정확도
        """
        # 마스킹된 위치의 예측값만 추출
        batch_size, seq_len = labels.shape
        masked_indices = masked_positions.bool()
        
        if not masked_indices.any():
            # 마스킹된 토큰이 없는 경우
            return torch.tensor(0.0, device=predictions.device), torch.tensor(0.0)
        
        masked_predictions = predictions[masked_indices]  # [num_masked, hidden_size]
        masked_labels = labels[masked_indices]  # [num_masked]
        
        # Linear projection to vocab size (보통 모델에 포함되지만 여기서는 별도 처리)
        # 실제로는 모델의 output layer를 사용해야 함
        
        # Loss 계산
        loss = self.criterion(masked_predictions, masked_labels)
        loss = loss.mean()
        
        # Accuracy 계산
        with torch.no_grad():
            predictions_argmax = masked_predictions.argmax(dim=-1)
            accuracy = (predictions_argmax == masked_labels).float().mean()
        
        return loss, accuracy


class SequenceGenerationLoss(nn.Module):
    """시퀀스 생성 태스크를 위한 Loss (번역, 요약 등)
    
    Teacher forcing을 사용하며, 다음 토큰을 예측하는 loss입니다.
    """
    
    def __init__(self,
                 vocab_size: int,
                 pad_token_id: int = 0,
                 label_smoothing: float = 0.0,
                 ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.criterion = CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=label_smoothing
        )
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] - 모델 출력
            targets: [batch_size, seq_len] - 정답 시퀀스
            target_mask: [batch_size, seq_len] - 유효한 위치는 1
            
        Returns:
            loss: 평균 loss
            metrics: 추가 메트릭 (perplexity, accuracy 등)
        """
        # Shift: 이전 토큰들로 다음 토큰 예측
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        
        if target_mask is not None:
            shift_mask = target_mask[:, 1:].contiguous()
        else:
            # 패딩 토큰 자동 감지
            shift_mask = (shift_labels != self.pad_token_id).float()
        
        # Loss 계산
        loss = self.criterion(shift_logits, shift_labels, shift_mask)
        
        # 추가 메트릭 계산
        with torch.no_grad():
            # Perplexity
            perplexity = torch.exp(loss.clamp(max=100))  # Overflow 방지
            
            # Accuracy
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels) * shift_mask
            accuracy = correct.sum() / shift_mask.sum().clamp(min=1.0)
            
            # Token-level metrics
            total_tokens = shift_mask.sum().item()
            correct_tokens = correct.sum().item()
        
        metrics = {
            'perplexity': perplexity.item(),
            'accuracy': accuracy.item(),
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens
        }
        
        return loss, metrics


class ContrastiveLoss(nn.Module):
    """대조 학습을 위한 Loss (SimCLR, CLIP 스타일)
    
    Positive pairs는 가깝게, negative pairs는 멀게 만듭니다.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self,
                embeddings1: torch.Tensor,
                embeddings2: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embedding_dim] - 첫 번째 뷰
            embeddings2: [batch_size, embedding_dim] - 두 번째 뷰
            labels: [batch_size] - 같은 클래스면 1, 다르면 0 (없으면 diagonal이 positive)
            
        Returns:
            loss: InfoNCE loss
        """
        batch_size = embeddings1.size(0)
        
        # L2 정규화
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # 전체 임베딩 결합
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # [2*batch_size, dim]
        
        # 유사도 행렬 계산
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [2*batch_size, 2*batch_size]
        
        # 자기 자신과의 유사도는 제외
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Positive pairs 인덱스
        if labels is None:
            # 기본값: (i, i+batch_size)가 positive pair
            labels = torch.arange(batch_size, device=embeddings.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
        else:
            # 주어진 labels 사용
            labels = torch.cat([labels, labels], dim=0)
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class FocalLoss(nn.Module):
    """클래스 불균형 문제를 위한 Focal Loss
    
    쉬운 예제의 가중치를 줄이고 어려운 예제에 집중합니다.
    """
    
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 ignore_index: int = -100,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] 또는 [batch_size, seq_len, num_classes]
            targets: [batch_size] 또는 [batch_size, seq_len]
            
        Returns:
            loss: Focal loss
        """
        # Reshape if needed
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # Softmax probabilities
        p = F.softmax(logits, dim=-1)
        
        # Get class probabilities
        class_mask = F.one_hot(targets, num_classes=logits.size(-1))
        probs = (p * class_mask).sum(dim=-1)
        
        # Focal weight
        focal_weight = (1.0 - probs) ** self.gamma
        
        # Cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weight if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Handle ignore index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask
        
        # Reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            if self.ignore_index >= 0:
                return focal_loss.sum() / mask.sum().clamp(min=1.0)
            else:
                return focal_loss.mean()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def create_transformer_loss(task_type: str = 'generation', **kwargs):
    """태스크에 맞는 loss 함수 생성
    
    Args:
        task_type: 'generation', 'mlm', 'classification', 'contrastive' 중 선택
        **kwargs: Loss 함수별 추가 인자
        
    Returns:
        Loss 함수 인스턴스
    """
    if task_type == 'generation':
        return SequenceGenerationLoss(**kwargs)
    elif task_type == 'mlm':
        return MaskedLanguageModelingLoss(**kwargs)
    elif task_type == 'classification':
        return CrossEntropyLoss(**kwargs)
    elif task_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif task_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")