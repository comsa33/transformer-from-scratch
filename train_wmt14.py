#!/usr/bin/env python3
"""
WMT14 번역 모델 학습 스크립트

RTX 3090에 최적화된 설정으로 Transformer 모델을 학습합니다.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.utils import load_config
from data.wmt_dataloader import create_dataloaders
from training.resource_monitor import ResourceMonitor, ResourceMonitorCallback
from training.tensorboard_callback import TensorBoardCallback
from training.trainer import Trainer, TrainingConfig
from training.visualization_callback import TrainingVisualizationCallback
from transformer.models.transformer import Transformer


def create_model(config: dict, vocab_size: int, device: torch.device) -> Transformer:
    """모델 생성"""
    model_config = config["model"]

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        d_ff=model_config["d_ff"],
        max_length=model_config["max_sequence_length"],
        dropout=model_config["dropout"],
        activation=model_config.get("activation", "relu"),  # 기본값은 relu
    )

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n모델 파라미터:")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  학습 가능한 파라미터: {trainable_params:,}")
    print(f"  모델 크기: {total_params * 4 / 1024**2:.2f} MB (fp32)")

    return model.to(device)


def collate_fn(batch, pad_id=0):
    """Trainer와 호환되는 collate 함수"""
    # 소스와 타겟 분리
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    # 패딩 적용
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # 타겟을 입력과 레이블로 분리
    tgt_input = tgt_padded[:, :-1]
    tgt_labels = tgt_padded[:, 1:]

    # 마스크 생성
    src_mask = (src_padded != pad_id).float()
    tgt_mask = (tgt_input != pad_id).float()

    return {
        "src_ids": src_padded,
        "tgt_ids": tgt_input,
        "labels": tgt_labels,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def compute_metrics(eval_preds):
    """평가 메트릭 계산"""
    predictions, labels = eval_preds

    # 패딩 토큰 제외하고 정확도 계산
    mask = labels != 0  # 패딩이 아닌 위치
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return {"accuracy": accuracy.item()}


def setup_logging(output_dir: str):
    """로깅 디렉토리 설정"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="WMT14 번역 모델 학습")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtx3090.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트에서 재개",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (작은 데이터셋 사용)",
    )
    args = parser.parse_args()

    # 디버그 모드면 디버그 설정 사용
    if args.debug:
        args.config = "configs/rtx3090_debug.yaml"

    # 설정 로드
    config = load_config(args.config)

    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("\nGPU 정보:")
        print(f"  디바이스: {gpu_name}")
        print(f"  메모리: {gpu_memory:.1f} GB")
    else:
        print("\n경고: GPU를 사용할 수 없습니다. CPU로 학습합니다.")

    # 데이터 준비 확인
    data_dir = Path(config["paths"]["data_dir"])
    data_stats_path = data_dir / "data_stats.json"

    if not data_stats_path.exists():
        print("\n데이터가 준비되지 않았습니다.")
        print("먼저 다음 명령을 실행하세요:")
        print(f"  python scripts/prepare_wmt14_data.py --config {args.config}")
        return

    with open(data_stats_path) as f:
        data_stats = json.load(f)

    print("\n데이터 정보:")
    print(f"  학습 샘플: {data_stats['train_size']:,}")
    print(f"  검증 샘플: {data_stats['val_size']:,}")
    print(f"  테스트 샘플: {data_stats['test_size']:,}")
    print(f"  어휘 크기: {data_stats['vocab_size']:,}")

    # 로깅 설정
    setup_logging(config["paths"]["checkpoint_dir"])

    # 데이터로더 생성
    print("\n데이터로더 생성 중...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        prefetch_factor=config["data"]["prefetch_factor"],
    )

    # collate 함수 설정
    def collate_with_pad(batch):
        return collate_fn(batch, pad_id=0)

    # 모델 생성
    print("\n모델 생성 중...")
    model = create_model(config, data_stats["vocab_size"], device)

    # Training 설정
    training_args = TrainingConfig(
        output_dir=config["paths"]["checkpoint_dir"],
        num_train_epochs=config["training"]["max_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"] * 2,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        adam_beta1=config["training"]["betas"][0],
        adam_beta2=config["training"]["betas"][1],
        adam_epsilon=config["training"]["eps"],
        max_grad_norm=config["training"]["gradient_clip_val"],
        lr_scheduler_type="linear",  # 원본 논문의 스케줄러와 유사
        warmup_steps=config["training"]["warmup_steps"],
        fp16=config["training"]["use_fp16"],
        logging_steps=config["logging"]["log_every_n_steps"],
        eval_steps=config["training"]["validate_every_n_steps"],
        save_steps=config["training"]["save_every_n_steps"],
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        seed=42,
        dataloader_num_workers=config["data"]["num_workers"],
        label_names=["labels"],
        resume_from_checkpoint=args.resume,
    )

    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['paths']['checkpoint_dir'].split('/')[-1]}_{timestamp}"

    # 콜백들 생성
    callbacks = []

    # 1. 시각화 콜백
    viz_callback = TrainingVisualizationCallback(
        output_dir=f"outputs/training/{experiment_name}",
        update_frequency=10,  # 10스텝마다 업데이트
        save_plots=True,
        update_report=True,
    )
    callbacks.append(viz_callback)

    # 2. TensorBoard 콜백
    tb_callback = TensorBoardCallback(log_dir=f"logs/tensorboard/{experiment_name}")
    callbacks.append(tb_callback)

    # 3. 리소스 모니터링 콜백
    resource_monitor = ResourceMonitor(output_dir=f"outputs/resources/{experiment_name}")
    resource_callback = ResourceMonitorCallback(
        monitor=resource_monitor,
        frequency=10,  # 10스텝마다 기록
    )
    callbacks.append(resource_callback)

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=collate_with_pad,
        compute_metrics=None,  # 일단 비활성화
        callbacks=callbacks,  # 모든 콜백 추가
    )

    # 학습 시작
    print("\n학습을 시작합니다...")
    print(f"설정 파일: {args.config}")
    print(f"Mixed Precision (FP16): {'활성화' if config['training']['use_fp16'] else '비활성화'}")
    print(f"Gradient Accumulation Steps: {config['training']['gradient_accumulation_steps']}")
    print(
        f"효과적인 배치 크기: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}"
    )

    try:
        # 학습 실행
        trainer.train()

        # 최종 평가
        print("\n최종 평가 중...")
        eval_results = trainer.evaluate()
        print(f"평가 결과: {eval_results}")

        # 테스트 세트 평가
        # if test_loader is not None:
        #     print("\n테스트 세트 평가 중...")
        #     test_results = trainer.evaluate(eval_dataset=test_loader.dataset)
        #     print(f"테스트 결과: {test_results}")

        print("\n학습이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
        print(f"체크포인트가 {config['paths']['checkpoint_dir']}에 저장되었습니다.")
        print(
            f"재개하려면: python train_wmt14.py --config {args.config} --resume <checkpoint_path>"
        )

    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
