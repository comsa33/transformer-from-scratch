"""
Transformer 학습을 위한 Optimizer와 Learning Rate Scheduler

Adam optimizer with warmup, cosine annealing 등을 구현합니다.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, Union, List, Dict
import warnings


class WarmupScheduler(_LRScheduler):
    """Warmup을 포함한 Learning Rate Scheduler 기본 클래스"""
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_warmup_factor(self) -> float:
        """현재 step에서의 warmup factor 계산"""
        if self.last_epoch < self.warmup_steps:
            return float(self.last_epoch) / float(max(1, self.warmup_steps))
        return 1.0


class LinearWarmupScheduler(WarmupScheduler):
    """선형 Warmup Scheduler
    
    warmup_steps 동안 0에서 base_lr까지 선형적으로 증가
    """
    
    def get_lr(self) -> List[float]:
        warmup_factor = self.get_warmup_factor()
        return [base_lr * warmup_factor for base_lr in self.base_lrs]


class TransformerScheduler(WarmupScheduler):
    """Transformer 논문의 Learning Rate Schedule
    
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 d_model: int,
                 warmup_steps: int = 4000,
                 last_epoch: int = -1):
        self.d_model = d_model
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5)
        
        if step < self.warmup_steps:
            # Warmup phase
            return [base_lr * scale * step * (self.warmup_steps ** (-1.5)) 
                    for base_lr in self.base_lrs]
        else:
            # Decay phase
            return [base_lr * scale * (step ** (-0.5)) 
                    for base_lr in self.base_lrs]


class CosineAnnealingWarmupScheduler(WarmupScheduler):
    """Cosine Annealing with Linear Warmup
    
    Warmup 후 cosine annealing으로 learning rate 감소
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 eta_min: float = 0,
                 last_epoch: int = -1):
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.get_warmup_factor()
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / \
                      (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class PolynomialDecayScheduler(WarmupScheduler):
    """Polynomial (Linear) Decay with Warmup
    
    Warmup 후 polynomial decay로 learning rate 감소
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 end_lr: float = 0.0,
                 power: float = 1.0,
                 last_epoch: int = -1):
        self.max_steps = max_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.get_warmup_factor()
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_steps) / \
                      (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            decay_factor = (1 - progress) ** self.power
            return [self.end_lr + (base_lr - self.end_lr) * decay_factor
                    for base_lr in self.base_lrs]


class ExponentialWarmupScheduler(WarmupScheduler):
    """Exponential Decay with Warmup"""
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 gamma: float = 0.95,
                 last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, warmup_steps, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.get_warmup_factor()
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Exponential decay
            decay_steps = self.last_epoch - self.warmup_steps
            return [base_lr * (self.gamma ** decay_steps) 
                    for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """One Cycle Learning Rate Schedule
    
    Leslie Smith의 1cycle policy 구현
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 max_lr: Union[float, List[float]],
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.0,
                 final_div_factor: float = 10000.0,
                 last_epoch: int = -1):
        
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # max_lr을 리스트로 변환
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = max_lr
        
        # 초기 learning rate 설정
        for idx, group in enumerate(optimizer.param_groups):
            group['initial_lr'] = self.max_lrs[idx] / div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step_num = self.last_epoch
        
        if step_num > self.total_steps:
            warnings.warn(f"Epoch {step_num} exceeds total_steps {self.total_steps}")
            step_num = self.total_steps
        
        # Phase 1: 증가 구간
        if step_num <= self.total_steps * self.pct_start:
            pct = step_num / (self.total_steps * self.pct_start)
            return [base_lr + pct * (max_lr - base_lr) 
                    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)]
        
        # Phase 2: 감소 구간
        else:
            step_num_decrease = step_num - self.total_steps * self.pct_start
            total_decrease_steps = self.total_steps * (1 - self.pct_start)
            
            if self.anneal_strategy == 'cos':
                pct = (1 + math.cos(math.pi * step_num_decrease / total_decrease_steps)) / 2
            else:  # linear
                pct = 1 - step_num_decrease / total_decrease_steps
            
            return [max_lr * pct + (max_lr / self.final_div_factor) * (1 - pct)
                    for max_lr in self.max_lrs]


class CyclicLRWithWarmup(_LRScheduler):
    """Cyclic Learning Rate with Warmup
    
    Warmup 후 cyclic learning rate 적용
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 base_lr: Union[float, List[float]],
                 max_lr: Union[float, List[float]],
                 warmup_steps: int,
                 step_size_up: int = 2000,
                 step_size_down: Optional[int] = None,
                 mode: str = 'triangular',
                 gamma: float = 1.0,
                 last_epoch: int = -1):
        
        self.warmup_steps = warmup_steps
        
        # base_lr과 max_lr을 리스트로 변환
        if isinstance(base_lr, float):
            self.base_lrs_cycle = [base_lr] * len(optimizer.param_groups)
        else:
            self.base_lrs_cycle = base_lr
            
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = max_lr
        
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Cyclic phase
        cycle_epoch = self.last_epoch - self.warmup_steps
        cycle = math.floor(1 + cycle_epoch / (self.step_size_up + self.step_size_down))
        x = abs(cycle_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = self.gamma ** cycle_epoch
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        
        return [base_lr + (max_lr - base_lr) * max(0, (1 - x)) * scale
                for base_lr, max_lr in zip(self.base_lrs_cycle, self.max_lrs)]


def create_optimizer(model: torch.nn.Module,
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas: tuple = (0.9, 0.999),
                    eps: float = 1e-8,
                    **kwargs) -> optim.Optimizer:
    """Optimizer 생성 헬퍼 함수
    
    Args:
        model: 학습할 모델
        optimizer_type: 'adam', 'adamw', 'sgd', 'adagrad' 중 선택
        learning_rate: 학습률
        weight_decay: Weight decay (L2 regularization)
        betas: Adam의 beta 파라미터
        eps: Adam의 epsilon
        **kwargs: 추가 optimizer 인자
        
    Returns:
        Optimizer 인스턴스
    """
    # Weight decay를 적용하지 않을 파라미터들
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
    
    # 파라미터 그룹 분리
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        }
    ]
    
    # Optimizer 생성
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    elif optimizer_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=eps,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = 'transformer',
                    warmup_steps: int = 4000,
                    max_steps: Optional[int] = None,
                    **kwargs) -> _LRScheduler:
    """Learning Rate Scheduler 생성 헬퍼 함수
    
    Args:
        optimizer: Optimizer 인스턴스
        scheduler_type: scheduler 종류
        warmup_steps: Warmup 단계 수
        max_steps: 최대 학습 단계 수 (일부 scheduler에 필요)
        **kwargs: 추가 scheduler 인자
        
    Returns:
        Scheduler 인스턴스
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'transformer':
        return TransformerScheduler(
            optimizer,
            d_model=kwargs.get('d_model', 512),
            warmup_steps=warmup_steps
        )
    elif scheduler_type == 'linear':
        return LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
    elif scheduler_type == 'cosine':
        if max_steps is None:
            raise ValueError("max_steps is required for cosine scheduler")
        return CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'polynomial':
        if max_steps is None:
            raise ValueError("max_steps is required for polynomial scheduler")
        return PolynomialDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            end_lr=kwargs.get('end_lr', 0.0),
            power=kwargs.get('power', 1.0)
        )
    elif scheduler_type == 'exponential':
        return ExponentialWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'onecycle':
        if max_steps is None:
            raise ValueError("max_steps is required for onecycle scheduler")
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=max_steps,
            pct_start=kwargs.get('pct_start', 0.3)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")