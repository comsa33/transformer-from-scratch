"""
TensorBoard 실시간 로깅 콜백

학습 중 메트릭을 TensorBoard에 기록합니다.
"""

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardCallback:
    """TensorBoard 로깅 콜백"""

    def __init__(self, log_dir: str = "logs/tensorboard"):
        if not TENSORBOARD_AVAILABLE:
            print("경고: TensorBoard가 설치되지 않았습니다. 로깅이 비활성화됩니다.")
            self.writer = None
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

        print(f"\nTensorBoard 로그 디렉토리: {self.log_dir}")
        print(f"TensorBoard 실행: tensorboard --logdir {self.log_dir}")

    def on_train_begin(self, trainer, state, model):
        """학습 시작 시 호출"""
        if self.writer is None:
            return

        # 모델 구조 기록 (옵션)
        # dummy_input = torch.randn(1, 10, model.d_model)
        # self.writer.add_graph(model, dummy_input)

    def on_log(self, trainer, state, model, logs: dict[str, Any]):
        """로그 이벤트 시 호출"""
        if self.writer is None:
            return

        step = logs.get("step", 0)

        # 학습 메트릭
        if "loss" in logs:
            self.writer.add_scalar("train/loss", logs["loss"], step)

        if "learning_rate" in logs:
            self.writer.add_scalar("train/learning_rate", logs["learning_rate"], step)

        # 검증 메트릭
        if "eval_loss" in logs:
            self.writer.add_scalar("eval/loss", logs["eval_loss"], step)

        # 추가 메트릭들
        for key, value in logs.items():
            if key not in ["loss", "learning_rate", "eval_loss", "step", "epoch"]:
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"metrics/{key}", value, step)

    def on_epoch_end(self, trainer, state, model):
        """에폭 종료 시 호출"""
        if self.writer is None:
            return

        # 에폭별 평균 메트릭 기록 가능
        epoch = state.epoch
        if hasattr(state, "log_history") and state.log_history:
            # 에폭 평균 계산
            epoch_losses = [
                log["loss"]
                for log in state.log_history
                if log.get("epoch") == epoch and "loss" in log
            ]
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)

    def on_train_end(self, trainer, state, model):
        """학습 종료 시 호출"""
        if self.writer is not None:
            self.writer.close()

    def on_save(self, trainer, state, model):
        """체크포인트 저장 시 호출"""
        if self.writer is None:
            return

        # 체크포인트 저장 이벤트 기록
        step = state.global_step
        self.writer.add_text("checkpoints", f"Saved at step {step}", step)
