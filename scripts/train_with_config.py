"""
Configuration 파일을 사용한 Transformer 학습 스크립트
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.utils import load_config, save_config
from transformer.models.transformer import Transformer
from data.dataset import TranslationDataset
from data.tokenizer import SimpleTokenizer
from training.trainer import TransformerTrainer
from training.optimizer import get_optimizer_and_scheduler
from evaluation.metrics import MetricsTracker


def setup_logging(config):
    """로깅 설정"""
    log_dir = Path(config.checkpoint.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_dataloaders(config, tokenizer):
    """데이터로더 생성"""
    # 더미 데이터 생성
    src_texts = [
        "Hello world",
        "How are you?",
        "I love transformers",
        "Machine learning is amazing",
        "Natural language processing"
    ] * 20
    
    tgt_texts = [
        "안녕 세계",
        "어떻게 지내세요?",
        "나는 트랜스포머를 좋아해",
        "기계 학습은 놀라워",
        "자연어 처리"
    ] * 20
    
    # 데이터셋 생성
    train_dataset = TranslationDataset(
        src_texts[:80], tgt_texts[:80], 
        tokenizer, tokenizer,
        max_length=config.data.max_length
    )
    
    val_dataset = TranslationDataset(
        src_texts[80:], tgt_texts[80:],
        tokenizer, tokenizer,
        max_length=config.data.max_length
    )
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Transformer 학습 with Config")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='base',
        help='설정 파일 이름 또는 경로 (기본값: base)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='학습 디바이스'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )
    
    # 명령줄 인자로 설정 덮어쓰기
    parser.add_argument('--batch-size', type=int, help='배치 크기')
    parser.add_argument('--learning-rate', type=float, help='학습률')
    parser.add_argument('--num-epochs', type=int, help='에폭 수')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 설정 덮어쓰기
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.max_steps = args.num_epochs * 100  # 임시 변환
    
    # 로깅 설정
    logger = setup_logging(config)
    logger.info(f"설정 파일 로드: {args.config}")
    logger.info(f"디바이스: {args.device}")
    
    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 설정 저장
    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")
    logger.info(f"설정 저장: {save_dir / 'config.yaml'}")
    
    # 토크나이저 생성
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + \
            [f'token_{i}' for i in range(1000)]
    tokenizer = SimpleTokenizer(vocab)
    
    # vocab_size 업데이트
    config.model['vocab_size'] = len(tokenizer)
    model_config = config.get_model_config()
    
    # 모델 생성
    model = Transformer(model_config)
    model = model.to(args.device)
    
    logger.info(f"모델 생성 완료:")
    logger.info(f"  - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - d_model: {model_config.d_model}")
    logger.info(f"  - num_heads: {model_config.num_heads}")
    logger.info(f"  - 인코더 레이어: {model_config.num_encoder_layers}")
    logger.info(f"  - 디코더 레이어: {model_config.num_decoder_layers}")
    
    # 데이터로더 생성
    train_dataloader, val_dataloader = create_dataloaders(config, tokenizer)
    
    # 옵티마이저와 스케줄러 생성
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        d_model=model_config.d_model,
        optimizer_type=config.optimizer.type,
        scheduler_type=config.scheduler.type,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay
    )
    
    # 메트릭 트래커 생성
    metrics_tracker = MetricsTracker()
    
    # 트레이너 생성
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        metrics_tracker=metrics_tracker,
        checkpoint_dir=config.checkpoint.save_dir,
        log_interval=config.logging.log_interval,
        eval_interval=config.logging.eval_interval,
        save_interval=config.logging.save_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        label_smoothing=config.training.label_smoothing
    )
    
    # 학습 시작
    logger.info("학습 시작...")
    logger.info(f"  - 배치 크기: {config.training.batch_size}")
    logger.info(f"  - 학습률: {config.training.learning_rate}")
    logger.info(f"  - Warmup 스텝: {config.training.warmup_steps}")
    logger.info(f"  - 최대 스텝: {config.training.max_steps}")
    
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=config.training.max_steps // len(train_dataloader)  # 스텝을 에폭으로 변환
    )
    
    logger.info("학습 완료!")
    
    # 최종 메트릭 저장
    final_metrics = metrics_tracker.get_average_metrics()
    logger.info("최종 메트릭:")
    for key, value in final_metrics.items():
        logger.info(f"  - {key}: {value:.4f}")


if __name__ == "__main__":
    main()