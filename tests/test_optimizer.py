"""
Optimizer와 Learning Rate Scheduler 테스트 및 시각화
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from training.optimizer import (
    LinearWarmupScheduler,
    TransformerScheduler,
    CosineAnnealingWarmupScheduler,
    PolynomialDecayScheduler,
    ExponentialWarmupScheduler,
    OneCycleLR,
    CyclicLRWithWarmup,
    create_optimizer,
    create_scheduler
)


def create_dummy_model(input_dim=100, hidden_dim=256, output_dim=10):
    """테스트용 더미 모델 생성"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def test_optimizer_creation():
    """Optimizer 생성 테스트"""
    print("=== Optimizer 생성 테스트 ===\n")
    
    model = create_dummy_model()
    
    # 다양한 optimizer 테스트
    optimizers = {
        'Adam': create_optimizer(model, 'adam', learning_rate=1e-3),
        'AdamW': create_optimizer(model, 'adamw', learning_rate=1e-3, weight_decay=0.1),
        'SGD': create_optimizer(model, 'sgd', learning_rate=0.1, momentum=0.9),
        'Adagrad': create_optimizer(model, 'adagrad', learning_rate=0.01)
    }
    
    for name, opt in optimizers.items():
        print(f"{name}:")
        print(f"  파라미터 그룹 수: {len(opt.param_groups)}")
        for i, group in enumerate(opt.param_groups):
            print(f"  그룹 {i}: lr={group['lr']:.4f}, weight_decay={group['weight_decay']}")
        print()
    
    # Weight decay 적용 확인
    adamw = optimizers['AdamW']
    for i, group in enumerate(adamw.param_groups):
        param_names = []
        for p in group['params']:
            for n, param in model.named_parameters():
                if param is p:
                    param_names.append(n)
                    break
        print(f"그룹 {i} 파라미터: {param_names[:2]}...")  # 처음 2개만 표시


def test_warmup_schedulers():
    """Warmup Scheduler 테스트"""
    print("\n=== Warmup Scheduler 테스트 ===\n")
    
    model = create_dummy_model()
    base_lr = 1e-3
    warmup_steps = 1000
    
    # Linear Warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = LinearWarmupScheduler(optimizer, warmup_steps=warmup_steps)
    
    # 처음 몇 스텝의 learning rate
    lrs = []
    for step in range(10):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print("Linear Warmup - 처음 10 스텝의 LR:")
    for i, lr in enumerate(lrs):
        print(f"  Step {i}: {lr:.6f}")
    
    # Warmup 완료 시점
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = LinearWarmupScheduler(optimizer, warmup_steps=warmup_steps)
    
    for _ in range(warmup_steps):
        scheduler.step()
    
    print(f"\nWarmup 완료 시점 (step {warmup_steps}):")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"  목표 LR: {base_lr:.6f}")


def visualize_scheduler_curves():
    """다양한 Scheduler의 Learning Rate 곡선 시각화"""
    print("\n=== Learning Rate 곡선 시각화 ===\n")
    
    model = create_dummy_model()
    base_lr = 1e-3
    warmup_steps = 500
    max_steps = 5000
    
    schedulers = {
        'Transformer': TransformerScheduler(
            torch.optim.Adam(model.parameters(), lr=base_lr),
            d_model=512,
            warmup_steps=warmup_steps
        ),
        'Cosine': CosineAnnealingWarmupScheduler(
            torch.optim.Adam(model.parameters(), lr=base_lr),
            warmup_steps=warmup_steps,
            max_steps=max_steps
        ),
        'Polynomial': PolynomialDecayScheduler(
            torch.optim.Adam(model.parameters(), lr=base_lr),
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            power=1.0
        ),
        'Exponential': ExponentialWarmupScheduler(
            torch.optim.Adam(model.parameters(), lr=base_lr),
            warmup_steps=warmup_steps,
            gamma=0.995
        )
    }
    
    # 각 scheduler의 LR 곡선 수집
    lr_curves = {}
    steps = list(range(max_steps))
    
    for name, scheduler in schedulers.items():
        lrs = []
        for _ in steps:
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        lr_curves[name] = lrs
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, lrs) in enumerate(lr_curves.items()):
        ax = axes[idx]
        ax.plot(steps, lrs, linewidth=2)
        ax.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup End')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{name} Scheduler')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Y축 스케일 조정
        if name == 'Transformer':
            ax.set_ylim(0, max(lrs) * 1.1)
    
    plt.tight_layout()
    plt.savefig('outputs/lr_scheduler_curves.png', dpi=150)
    print("Learning rate 곡선이 'outputs/lr_scheduler_curves.png'에 저장되었습니다.")


def test_onecycle_lr():
    """OneCycleLR 테스트 및 시각화"""
    print("\n=== OneCycleLR 테스트 ===\n")
    
    model = create_dummy_model()
    total_steps = 1000
    max_lr = 1e-2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # LR 곡선 수집
    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(total_steps), lrs, linewidth=2)
    
    # 주요 지점 표시
    pct_start = 0.3
    peak_step = int(total_steps * pct_start)
    plt.axvline(x=peak_step, color='red', linestyle='--', alpha=0.5, label=f'Peak (step {peak_step})')
    plt.axhline(y=max_lr, color='green', linestyle='--', alpha=0.5, label=f'Max LR ({max_lr})')
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/onecycle_lr.png', dpi=150)
    print("OneCycleLR 곡선이 'outputs/onecycle_lr.png'에 저장되었습니다.")
    
    # 주요 통계
    print(f"초기 LR: {lrs[0]:.6f}")
    print(f"최대 LR: {max(lrs):.6f} (at step {lrs.index(max(lrs))})")
    print(f"최종 LR: {lrs[-1]:.6f}")


def test_cyclic_lr():
    """CyclicLR with Warmup 테스트"""
    print("\n=== CyclicLR with Warmup 테스트 ===\n")
    
    model = create_dummy_model()
    base_lr = 1e-4
    max_lr = 1e-2
    warmup_steps = 200
    step_size = 300
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CyclicLRWithWarmup(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        step_size_up=step_size,
        mode='triangular'
    )
    
    # LR 곡선 수집
    lrs = []
    total_steps = 2000
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(total_steps), lrs, linewidth=2)
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cyclic LR with Warmup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/cyclic_lr_warmup.png', dpi=150)
    print("Cyclic LR 곡선이 'outputs/cyclic_lr_warmup.png'에 저장되었습니다.")


def compare_warmup_strategies():
    """다양한 Warmup 전략 비교"""
    print("\n=== Warmup 전략 비교 ===\n")
    
    model = create_dummy_model()
    base_lr = 1e-3
    warmup_steps = 500
    
    # 다양한 warmup 길이
    warmup_lengths = [100, 500, 1000, 2000]
    
    plt.figure(figsize=(10, 6))
    
    for warmup in warmup_lengths:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_steps=warmup,
            max_steps=5000,
            eta_min=1e-5
        )
        
        lrs = []
        for _ in range(1500):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        plt.plot(range(1500), lrs, label=f'Warmup {warmup} steps', linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup 길이에 따른 Learning Rate 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/warmup_comparison.png', dpi=150)
    print("Warmup 비교가 'outputs/warmup_comparison.png'에 저장되었습니다.")


def test_scheduler_with_training():
    """실제 학습 시뮬레이션과 함께 테스트"""
    print("\n=== 학습 시뮬레이션 테스트 ===\n")
    
    # 간단한 학습 태스크
    torch.manual_seed(42)
    model = create_dummy_model()
    
    # 데이터 생성
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    
    # Optimizer와 Scheduler
    optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3, weight_decay=0.01)
    scheduler = create_scheduler(
        optimizer, 
        'cosine',
        warmup_steps=100,
        max_steps=500
    )
    
    # 학습 시뮬레이션
    losses = []
    lrs = []
    
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    
    for epoch in range(10):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Scheduler step (per iteration)
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        
        avg_loss = epoch_loss / (len(X) // batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {lrs[-1]:.6f}")
    
    # 결과 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss 곡선
    ax1.plot(losses, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Learning Rate 곡선
    ax2.plot(lrs, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/training_simulation.png', dpi=150)
    print("\n학습 시뮬레이션이 'outputs/training_simulation.png'에 저장되었습니다.")


def analyze_transformer_schedule():
    """Transformer 논문의 스케줄 상세 분석"""
    print("\n=== Transformer Schedule 분석 ===\n")
    
    d_models = [128, 256, 512, 1024]
    warmup_steps = 4000
    
    plt.figure(figsize=(10, 6))
    
    for d_model in d_models:
        model = create_dummy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # base_lr = 1 for visualization
        scheduler = TransformerScheduler(
            optimizer,
            d_model=d_model,
            warmup_steps=warmup_steps
        )
        
        lrs = []
        steps = list(range(20000))
        for _ in steps:
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        plt.plot(steps, lrs, label=f'd_model={d_model}', linewidth=2)
    
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate (relative)')
    plt.title('Transformer Learning Rate Schedule for Different Model Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20000)
    plt.savefig('outputs/transformer_schedule_analysis.png', dpi=150)
    print("Transformer schedule 분석이 'outputs/transformer_schedule_analysis.png'에 저장되었습니다.")
    
    # 주요 지점에서의 learning rate
    print("\n주요 지점에서의 Learning Rate (d_model=512):")
    model = create_dummy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = TransformerScheduler(optimizer, d_model=512, warmup_steps=4000)
    
    checkpoints = [1, 1000, 4000, 8000, 16000, 32000]
    for step in checkpoints:
        for _ in range(step - scheduler.last_epoch - 1):
            scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"  Step {step:6d}: {lr:.6f}")


if __name__ == "__main__":
    # 1. Optimizer 생성 테스트
    test_optimizer_creation()
    
    # 2. Warmup Scheduler 테스트
    test_warmup_schedulers()
    
    # 3. 다양한 Scheduler 곡선 시각화
    visualize_scheduler_curves()
    
    # 4. OneCycleLR 테스트
    test_onecycle_lr()
    
    # 5. CyclicLR 테스트
    test_cyclic_lr()
    
    # 6. Warmup 전략 비교
    compare_warmup_strategies()
    
    # 7. 학습 시뮬레이션
    test_scheduler_with_training()
    
    # 8. Transformer schedule 분석
    analyze_transformer_schedule()
    
    print("\n모든 테스트가 완료되었습니다!")