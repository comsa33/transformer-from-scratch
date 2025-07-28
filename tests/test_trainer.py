"""
Trainer 테스트 및 학습 시뮬레이션
"""

import sys

sys.path.append(".")

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from training.trainer import Trainer, TrainingConfig
from transformer.models.transformer import create_transformer_small


class DummyTranslationDataset(Dataset):
    """테스트용 더미 번역 데이터셋"""

    def __init__(
        self, size: int = 1000, src_len: int = 20, tgt_len: int = 25, vocab_size: int = 1000
    ):
        self.size = size
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.vocab_size = vocab_size

        # 더미 데이터 생성
        torch.manual_seed(42)
        self.src_ids = torch.randint(1, vocab_size, (size, src_len))
        self.tgt_ids = torch.randint(1, vocab_size, (size, tgt_len))

        # 일부 패딩 추가
        for i in range(size):
            pad_len = np.random.randint(0, 5)
            if pad_len > 0:
                self.src_ids[i, -pad_len:] = 0
                self.tgt_ids[i, -pad_len:] = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "src_ids": self.src_ids[idx],
            "tgt_ids": self.tgt_ids[idx],
            "labels": self.tgt_ids[idx],
        }


def custom_data_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    """배치 데이터 준비"""
    batch = {}

    # 각 키별로 텐서 스택
    for key in features[0]:
        batch[key] = torch.stack([f[key] for f in features])

    return batch


def compute_metrics(eval_pred: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """평가 메트릭 계산"""
    predictions, labels = eval_pred

    # 정확도 계산
    mask = labels != -100  # padding 제외
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    return {"accuracy": accuracy.item()}


class LoggingCallback:
    """로깅 콜백 예제"""

    def on_train_begin(self, trainer, state, model):
        print("🚀 Training started!")

    def on_epoch_begin(self, trainer, state, model):
        print(f"\n📅 Epoch {state.epoch + 1} started")

    def on_epoch_end(self, trainer, state, model):
        print(f"✅ Epoch {state.epoch + 1} completed")

    def on_train_end(self, trainer, state, model):
        print("🎉 Training completed!")

    def on_evaluate(self, trainer, state, model):
        print("📊 Running evaluation...")


def test_basic_training():
    """기본 학습 테스트"""
    print("=== 기본 학습 테스트 ===\n")

    # 작은 모델과 데이터셋
    model = create_transformer_small(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        max_length=100,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=128,
        num_heads=4,
    )

    # 데이터셋
    train_dataset = DummyTranslationDataset(size=200)
    eval_dataset = DummyTranslationDataset(size=50)

    # 학습 설정
    training_args = TrainingConfig(
        output_dir="./test_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-3,
        warmup_steps=10,
        logging_steps=5,
        eval_steps=15,
        save_steps=20,
        fp16=False,  # CPU에서 테스트
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
        compute_metrics=None,  # 일단 메트릭 계산 제외
        callbacks=[LoggingCallback()],
    )

    # 학습
    print("Starting training...")
    history = trainer.train()

    # 결과 출력
    print(f"\n학습 완료! 총 스텝: {trainer.state.global_step}")

    # 정리
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")

    return history


def test_gradient_accumulation():
    """Gradient Accumulation 테스트"""
    print("\n=== Gradient Accumulation 테스트 ===\n")

    model = create_transformer_small(
        src_vocab_size=100,
        tgt_vocab_size=100,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_model=64,
    )

    train_dataset = DummyTranslationDataset(size=64, vocab_size=100)

    # 다양한 accumulation steps 테스트
    accumulation_steps = [1, 2, 4, 8]
    results = {}

    for acc_steps in accumulation_steps:
        training_args = TrainingConfig(
            output_dir=f"./test_acc_{acc_steps}",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=acc_steps,
            learning_rate=1e-3,
            logging_steps=5,
            eval_steps=-1,  # 평가 안함
            save_steps=-1,  # 저장 안함
        )

        trainer = Trainer(
            model=model.to("cpu"),  # 모델 초기화
            args=training_args,
            train_dataset=train_dataset,
            data_collator=custom_data_collator,
        )

        history = trainer.train()

        # 실제 배치 크기
        effective_batch_size = 8 * acc_steps
        results[acc_steps] = {
            "effective_batch_size": effective_batch_size,
            "final_loss": history[-1]["loss"] if history else 0,
        }

        # 정리
        if os.path.exists(f"./test_acc_{acc_steps}"):
            shutil.rmtree(f"./test_acc_{acc_steps}")

    print("Gradient Accumulation 결과:")
    print("-" * 50)
    print("Acc Steps | Effective Batch | Final Loss")
    print("-" * 50)
    for acc_steps, res in results.items():
        print(f"{acc_steps:9d} | {res['effective_batch_size']:15d} | {res['final_loss']:.4f}")

    return results


def test_scheduler_integration():
    """Scheduler 통합 테스트"""
    print("\n=== Scheduler 통합 테스트 ===\n")

    model = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100, d_model=128)

    train_dataset = DummyTranslationDataset(size=100, vocab_size=100)

    # 다양한 scheduler 테스트
    schedulers = ["linear", "cosine", "transformer"]
    lr_histories = {}

    for scheduler_type in schedulers:
        training_args = TrainingConfig(
            output_dir=f"./test_scheduler_{scheduler_type}",
            num_train_epochs=2,
            per_device_train_batch_size=10,
            learning_rate=1e-3,
            lr_scheduler_type=scheduler_type,
            warmup_steps=20,
            logging_steps=5,
        )

        trainer = Trainer(
            model=model.to("cpu"),
            args=training_args,
            train_dataset=train_dataset,
            data_collator=custom_data_collator,
        )

        # LR 추적을 위한 수정
        lrs = []
        original_train = trainer.train

        def track_lr():
            history = original_train()
            # 로그에서 LR 추출
            for log in history:
                if "learning_rate" in log:
                    lrs.append(log["learning_rate"])
            return history

        trainer.train = track_lr
        trainer.train()

        lr_histories[scheduler_type] = lrs

        # 정리
        if os.path.exists(f"./test_scheduler_{scheduler_type}"):
            shutil.rmtree(f"./test_scheduler_{scheduler_type}")

    # LR 곡선 시각화
    plt.figure(figsize=(10, 6))
    for scheduler_type, lrs in lr_histories.items():
        plt.plot(lrs, label=scheduler_type, linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules in Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/trainer_lr_schedules.png", dpi=150)
    print("LR 스케줄이 'outputs/trainer_lr_schedules.png'에 저장되었습니다.")

    return lr_histories


def test_checkpoint_save_load():
    """체크포인트 저장/로드 테스트"""
    print("\n=== 체크포인트 저장/로드 테스트 ===\n")

    # 모델과 데이터
    model = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100, d_model=64)
    train_dataset = DummyTranslationDataset(size=100, vocab_size=100)

    # 첫 번째 학습 (체크포인트 저장)
    training_args = TrainingConfig(
        output_dir="./checkpoint_test",
        num_train_epochs=2,
        per_device_train_batch_size=10,
        save_steps=5,
        logging_steps=5,
        save_total_limit=2,
    )

    trainer1 = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )

    trainer1.train()
    final_step1 = trainer1.state.global_step

    print(f"첫 번째 학습 완료. 최종 스텝: {final_step1}")

    # 체크포인트 확인
    checkpoints = [d for d in os.listdir("./checkpoint_test") if d.startswith("checkpoint-")]
    print(f"저장된 체크포인트: {checkpoints}")

    # 체크포인트에서 재개
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint_path = os.path.join("./checkpoint_test", latest_checkpoint)

        # 새 모델과 trainer
        model2 = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100, d_model=64)

        training_args2 = TrainingConfig(
            output_dir="./checkpoint_test",
            num_train_epochs=3,  # 추가 epoch
            per_device_train_batch_size=10,
            resume_from_checkpoint=checkpoint_path,
            logging_steps=5,
        )

        trainer2 = Trainer(
            model=model2,
            args=training_args2,
            train_dataset=train_dataset,
            data_collator=custom_data_collator,
        )

        # 체크포인트 로드 및 학습 재개
        trainer2._load_checkpoint(checkpoint_path)
        print(f"\n체크포인트 '{latest_checkpoint}'에서 재개")
        print(f"재개 시점 스텝: {trainer2.state.global_step}")

        # 모델 가중치 비교
        state1 = model.state_dict()
        state2 = model2.state_dict()

        weight_diff = 0
        for key in state1:
            if key in state2:
                diff = (state1[key] - state2[key]).abs().mean().item()
                weight_diff += diff

        print(f"체크포인트 로드 후 가중치 차이: {weight_diff:.6f}")

    # 정리
    if os.path.exists("./checkpoint_test"):
        shutil.rmtree("./checkpoint_test")

    return checkpoints


def test_mixed_precision():
    """Mixed Precision 학습 테스트"""
    print("\n=== Mixed Precision 테스트 ===\n")

    if not torch.cuda.is_available():
        print("CUDA가 없어서 Mixed Precision 테스트를 건너뜁니다.")
        return

    model = create_transformer_small(src_vocab_size=100, tgt_vocab_size=100, d_model=128)

    train_dataset = DummyTranslationDataset(size=100, vocab_size=100)

    # FP16 vs FP32 비교
    results = {}

    for use_fp16 in [False, True]:
        training_args = TrainingConfig(
            output_dir=f"./fp16_test_{use_fp16}",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            fp16=use_fp16,
            logging_steps=10,
        )

        trainer = Trainer(
            model=model.cuda(),
            args=training_args,
            train_dataset=train_dataset,
            data_collator=custom_data_collator,
        )

        import time

        start_time = time.time()
        history = trainer.train()
        train_time = time.time() - start_time

        results[f"FP{16 if use_fp16 else 32}"] = {
            "time": train_time,
            "final_loss": history[-1]["loss"] if history else 0,
        }

        # 정리
        if os.path.exists(f"./fp16_test_{use_fp16}"):
            shutil.rmtree(f"./fp16_test_{use_fp16}")

    print("Mixed Precision 결과:")
    for precision, res in results.items():
        print(f"{precision}: 시간={res['time']:.2f}s, Loss={res['final_loss']:.4f}")


def test_training_config():
    """TrainingConfig 저장/로드 테스트"""
    print("\n=== TrainingConfig 테스트 ===\n")

    # 설정 생성
    config = TrainingConfig(
        output_dir="./config_test",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        fp16=True,
        gradient_checkpointing=True,
        custom_param="test_value",  # 추가 파라미터
    )

    # JSON으로 저장
    os.makedirs("./config_test", exist_ok=True)
    config.save_to_json("./config_test/config.json")

    # 로드
    loaded_config = TrainingConfig.from_json("./config_test/config.json")

    # 비교
    print("원본 설정:")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  warmup_ratio: {config.warmup_ratio}")
    print(f"  custom_param: {config.custom_param}")

    print("\n로드된 설정:")
    print(f"  learning_rate: {loaded_config.learning_rate}")
    print(f"  warmup_ratio: {loaded_config.warmup_ratio}")
    print(f"  custom_param: {loaded_config.custom_param}")

    # 정리
    if os.path.exists("./config_test"):
        shutil.rmtree("./config_test")


def visualize_training_history(history: list[dict]):
    """학습 이력 시각화"""
    if not history:
        return

    # 데이터 추출
    steps = [log["step"] for log in history if "loss" in log]
    losses = [log["loss"] for log in history if "loss" in log]
    lrs = [log["learning_rate"] for log in history if "learning_rate" in log]

    eval_steps = [log["step"] for log in history if "eval_loss" in log]
    eval_losses = [log["eval_loss"] for log in history if "eval_loss" in log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Loss 그래프
    ax1.plot(steps, losses, "b-", label="Train Loss", linewidth=2)
    if eval_steps:
        ax1.plot(eval_steps, eval_losses, "r--", label="Eval Loss", linewidth=2, marker="o")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Evaluation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning Rate 그래프
    ax2.plot(steps[: len(lrs)], lrs, "g-", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=150)
    print("\n학습 이력이 'outputs/training_history.png'에 저장되었습니다.")


if __name__ == "__main__":
    # 1. 기본 학습 테스트
    history = test_basic_training()

    # 2. Gradient Accumulation 테스트
    grad_acc_results = test_gradient_accumulation()

    # 3. Scheduler 통합 테스트
    lr_histories = test_scheduler_integration()

    # 4. 체크포인트 저장/로드 테스트
    checkpoints = test_checkpoint_save_load()

    # 5. Mixed Precision 테스트
    test_mixed_precision()

    # 6. TrainingConfig 테스트
    test_training_config()

    # 7. 학습 이력 시각화
    if history:
        visualize_training_history(history)

    print("\n모든 테스트가 완료되었습니다!")
