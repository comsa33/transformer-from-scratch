"""
Simple Training Script 예제

실제 전체 training script는 다른 모든 모듈들이 필요하므로,
여기서는 간단한 학습 루프의 예제를 보여줍니다.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path

from transformer.models.transformer import create_transformer_small
from evaluation.metrics import get_metrics_for_task, EarlyStopping


def create_dummy_data(n_samples=100, seq_length=20, vocab_size=100):
    """더미 데이터 생성"""
    # 랜덤 시퀀스 생성
    src_ids = torch.randint(1, vocab_size, (n_samples, seq_length))
    tgt_ids = torch.randint(1, vocab_size, (n_samples, seq_length))
    
    # 패딩 추가
    src_ids[:, -5:] = 0
    tgt_ids[:, -5:] = 0
    
    return src_ids, tgt_ids


def train_step(model, batch, criterion, optimizer, device):
    """단일 학습 스텝"""
    src_ids, tgt_ids = batch
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    
    # Teacher forcing을 위해 target을 shift
    tgt_input = tgt_ids[:, :-1]
    tgt_output = tgt_ids[:, 1:]
    
    # Forward pass
    output = model(src_ids, tgt_input)
    
    # Loss 계산
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, dataloader, criterion, device, metrics=None):
    """모델 평가"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src_ids, tgt_ids = batch
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            
            output = model(src_ids, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # 토큰 수 계산 (패딩 제외)
            mask = tgt_output != 0
            num_tokens = mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # 메트릭 업데이트
            if metrics:
                metrics.update(
                    predictions=output,
                    references=tgt_output,
                    loss=loss,
                    num_tokens=num_tokens
                )
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Simple Transformer Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("=== Simple Transformer Training Example ===\n")
    
    # 설정
    vocab_size = 100
    device = torch.device(args.device)
    
    # 모델 생성
    model = create_transformer_small(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_length=100
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Using device: {device}\n")
    
    # 더미 데이터 생성
    train_src, train_tgt = create_dummy_data(n_samples=200)
    val_src, val_tgt = create_dummy_data(n_samples=50)
    
    train_dataset = TensorDataset(train_src, train_tgt)
    val_dataset = TensorDataset(val_src, val_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Loss와 Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 메트릭과 Early Stopping
    metrics = get_metrics_for_task("language_modeling")
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    # 학습 루프
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            loss = train_step(model, batch, criterion, optimizer, device)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 검증
        metrics.reset()
        val_loss = evaluate(model, val_loader, criterion, device, metrics)
        val_metrics = metrics.compute()
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_metrics.get('perplexity', 0):.2f}")
        
        # Early stopping 체크
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()