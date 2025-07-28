"""
학습 과정 시각화 및 로깅 콜백

학습 중 실시간으로 결과를 시각화하고 저장합니다.
"""

import json
import os

# 한글 폰트 설정 - 더 안정적인 방법
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()

    # 사용 가능한 한글 폰트 찾기
    font_list = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic", "UnDotum"]

    font_name = None
    for font in korean_fonts:
        if font in font_list:
            font_name = font
            break

    if font_name:
        plt.rcParams["font.family"] = font_name
    else:
        # 폰트를 찾지 못한 경우 직접 경로 지정
        if system == "Linux":
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                fm.fontManager.addfont(font_path)
                plt.rcParams["font.family"] = "NanumGothic"
        elif system == "Windows":
            plt.rcParams["font.family"] = "Malgun Gothic"
        elif system == "Darwin":  # macOS
            plt.rcParams["font.family"] = "AppleGothic"

    plt.rcParams["axes.unicode_minus"] = False

    # 폰트 캐시 재생성
    import contextlib

    with contextlib.suppress(Exception):
        fm._rebuild()


# 한글 폰트 설정 적용
setup_korean_font()


class TrainingVisualizationCallback:
    """학습 과정을 시각화하고 기록하는 콜백"""

    def __init__(
        self,
        output_dir: str = "outputs/training",
        update_frequency: int = 10,
        save_plots: bool = True,
        update_report: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.update_frequency = update_frequency
        self.save_plots = save_plots
        self.update_report = update_report

        # 메트릭 저장
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": [],
            "timestamp": [],
        }

        # 학습 시작 시간
        self.start_time = None
        self.config = None

    def on_train_begin(self, trainer, state, model):
        """학습 시작 시 호출"""
        self.start_time = datetime.now()
        self.config = trainer.args.to_dict() if hasattr(trainer.args, "to_dict") else {}

        # 초기 정보 저장
        info = {
            "start_time": self.start_time.isoformat(),
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "config": self.config,
        }

        with open(self.output_dir / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n시각화 출력 디렉토리: {self.output_dir}")

    def on_log(self, trainer, state, model, logs: dict[str, Any]):
        """로그 이벤트 시 호출"""
        # 메트릭 수집
        if "loss" in logs:
            self.history["train_loss"].append(logs["loss"])
            self.history["step"].append(logs.get("step", len(self.history["train_loss"])))
            self.history["epoch"].append(logs.get("epoch", 0))
            self.history["learning_rate"].append(logs.get("learning_rate", 0))
            self.history["timestamp"].append(datetime.now().isoformat())

        if "eval_loss" in logs:
            self.history["eval_loss"].append(
                {
                    "step": logs.get("step", len(self.history["train_loss"])),
                    "value": logs["eval_loss"],
                }
            )

        # 주기적으로 시각화 업데이트
        if len(self.history["train_loss"]) % self.update_frequency == 0:
            self._update_visualizations()
            self._save_metrics()

    def on_evaluate(self, trainer, state, model):
        """평가 시작 시 호출"""
        pass

    def on_save(self, trainer, state, model):
        """체크포인트 저장 시 호출"""
        self._update_visualizations()
        self._save_metrics()
        if self.update_report:
            self._update_final_report()

    def on_train_end(self, trainer, state, model):
        """학습 종료 시 호출"""
        # 최종 시각화
        self._update_visualizations(final=True)
        self._save_metrics()

        # 학습 요약 생성
        summary = self._generate_summary()
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self.update_report:
            self._update_final_report(final=True)

        print(f"\n학습 완료! 결과가 {self.output_dir}에 저장되었습니다.")

    def _update_visualizations(self, final: bool = False):
        """시각화 업데이트"""
        if not self.save_plots or len(self.history["train_loss"]) == 0:
            return

        # 1. 손실 그래프
        self._plot_loss_curves(final)

        # 2. 학습률 그래프
        self._plot_learning_rate(final)

        # 3. 학습 진행 상황 대시보드
        self._plot_training_dashboard(final)

    def _plot_loss_curves(self, final: bool = False):
        """손실 곡선 플롯"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 학습 손실
        steps = self.history["step"]
        train_loss = self.history["train_loss"]
        ax.plot(steps, train_loss, label="Train Loss", alpha=0.8)

        # 검증 손실
        if self.history["eval_loss"]:
            eval_steps = [e["step"] for e in self.history["eval_loss"]]
            eval_values = [e["value"] for e in self.history["eval_loss"]]
            ax.plot(eval_steps, eval_values, "o-", label="Eval Loss", markersize=8)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("학습 손실 곡선")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 로그 스케일 옵션
        if train_loss and max(train_loss) / min(train_loss) > 10:
            ax.set_yscale("log")

        plt.tight_layout()
        filename = "loss_curves_final.png" if final else "loss_curves.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_learning_rate(self, final: bool = False):
        """학습률 변화 플롯"""
        if not self.history["learning_rate"]:
            return

        fig, ax = plt.subplots(figsize=(10, 4))

        steps = self.history["step"]
        lr = self.history["learning_rate"]

        ax.plot(steps, lr, color="orange", linewidth=2)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning Rate")
        ax.set_title("학습률 스케줄")
        ax.grid(True, alpha=0.3)

        # 과학적 표기법 사용
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        filename = "learning_rate_final.png" if final else "learning_rate.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_training_dashboard(self, final: bool = False):
        """종합 대시보드"""
        fig = plt.figure(figsize=(15, 10))

        # 2x2 그리드
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. 손실 곡선
        ax1 = fig.add_subplot(gs[0, 0])
        if self.history["train_loss"]:
            ax1.plot(self.history["step"], self.history["train_loss"], label="Train")
            if self.history["eval_loss"]:
                eval_steps = [e["step"] for e in self.history["eval_loss"]]
                eval_values = [e["value"] for e in self.history["eval_loss"]]
                ax1.plot(eval_steps, eval_values, "o-", label="Eval")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("손실 변화")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 학습률
        ax2 = fig.add_subplot(gs[0, 1])
        if self.history["learning_rate"]:
            ax2.plot(self.history["step"], self.history["learning_rate"], "orange")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("학습률 변화")
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 3. 에폭별 평균 손실
        ax3 = fig.add_subplot(gs[1, 0])
        if self.history["epoch"] and self.history["train_loss"]:
            # 에폭별 평균 계산
            epoch_losses = {}
            for epoch, loss in zip(self.history["epoch"], self.history["train_loss"]):
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(loss)

            epochs = sorted(epoch_losses.keys())
            avg_losses = [np.mean(epoch_losses[e]) for e in epochs]

            ax3.bar(epochs, avg_losses, alpha=0.7)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Average Loss")
            ax3.set_title("에폭별 평균 손실")
            ax3.grid(True, alpha=0.3, axis="y")

        # 4. 학습 통계
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        # 통계 텍스트
        stats_text = self._get_current_stats()
        # 현재 설정된 한글 폰트 사용
        font_family = plt.rcParams["font.family"]
        if isinstance(font_family, list):
            font_family = font_family[0]
        ax4.text(
            0.1,
            0.9,
            stats_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily=font_family,
        )

        plt.suptitle(
            f"학습 진행 상황 대시보드 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=16
        )

        filename = "training_dashboard_final.png" if final else "training_dashboard.png"
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def _get_current_stats(self) -> str:
        """현재 학습 통계 텍스트 생성"""
        if not self.history["train_loss"]:
            return "학습 데이터 없음"

        stats = []
        stats.append("=== 학습 통계 ===\n")

        # 현재 스텝/에폭
        current_step = self.history["step"][-1] if self.history["step"] else 0
        current_epoch = self.history["epoch"][-1] if self.history["epoch"] else 0
        stats.append(f"현재 스텝: {current_step}")
        stats.append(f"현재 에폭: {current_epoch}")

        # 손실 통계
        recent_loss = self.history["train_loss"][-1] if self.history["train_loss"] else 0
        min_loss = min(self.history["train_loss"]) if self.history["train_loss"] else 0
        stats.append(f"\n최근 손실: {recent_loss:.4f}")
        stats.append(f"최소 손실: {min_loss:.4f}")

        if self.history["eval_loss"]:
            recent_eval = self.history["eval_loss"][-1]["value"]
            min_eval = min(e["value"] for e in self.history["eval_loss"])
            stats.append(f"\n최근 검증 손실: {recent_eval:.4f}")
            stats.append(f"최소 검증 손실: {min_eval:.4f}")

        # 학습 시간
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            stats.append(f"\n경과 시간: {str(elapsed).split('.')[0]}")

        # 학습률
        if self.history["learning_rate"]:
            current_lr = self.history["learning_rate"][-1]
            stats.append(f"\n현재 학습률: {current_lr:.6f}")

        return "\n".join(stats)

    def _save_metrics(self):
        """메트릭을 JSON 파일로 저장"""
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def _generate_summary(self) -> dict[str, Any]:
        """학습 요약 생성"""
        if not self.history["train_loss"]:
            return {}

        summary = {
            "total_steps": len(self.history["train_loss"]),
            "total_epochs": max(self.history["epoch"]) if self.history["epoch"] else 0,
            "final_train_loss": self.history["train_loss"][-1],
            "min_train_loss": min(self.history["train_loss"]),
            "training_time": (
                str(datetime.now() - self.start_time).split(".")[0]
                if self.start_time
                else "Unknown"
            ),
        }

        if self.history["eval_loss"]:
            summary["final_eval_loss"] = self.history["eval_loss"][-1]["value"]
            summary["min_eval_loss"] = min(e["value"] for e in self.history["eval_loss"])

        return summary

    def _update_final_report(self, final: bool = False):
        """FINAL_REPORT.md 업데이트"""
        report_path = Path("FINAL_REPORT.md")
        if not report_path.exists():
            return

        # 현재 리포트 읽기
        with open(report_path, encoding="utf-8") as f:
            content = f.read()

        # WMT14 학습 섹션 찾기 또는 추가
        section_marker = "## WMT14 번역 모델 학습 결과"

        if section_marker not in content:
            # 섹션 추가
            content += f"\n\n{section_marker}\n\n"

        # 섹션 내용 생성
        section_content = self._generate_report_section(final)

        # 섹션 업데이트
        if section_marker in content:
            # 기존 섹션 찾기
            start_idx = content.find(section_marker)

            # 다음 섹션 찾기 (## 로 시작하는 다음 라인)
            next_section_idx = content.find("\n## ", start_idx + len(section_marker))

            if next_section_idx == -1:
                # 마지막 섹션인 경우
                content = content[:start_idx] + section_marker + "\n\n" + section_content
            else:
                # 중간 섹션인 경우
                content = (
                    content[:start_idx]
                    + section_marker
                    + "\n\n"
                    + section_content
                    + "\n"
                    + content[next_section_idx:]
                )

        # 리포트 저장
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _generate_report_section(self, final: bool = False) -> str:
        """리포트 섹션 내용 생성"""
        lines = []

        # 학습 상태
        status = "완료" if final else "진행 중"
        lines.append(f"**학습 상태**: {status}")
        lines.append(f"**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 학습 설정
        if self.config:
            lines.append("### 학습 설정")
            lines.append(
                f"- **배치 크기**: {self.config.get('per_device_train_batch_size', 'N/A')}"
            )
            lines.append(f"- **학습률**: {self.config.get('learning_rate', 'N/A')}")
            lines.append(f"- **에폭 수**: {self.config.get('num_train_epochs', 'N/A')}")
            lines.append(
                f"- **Mixed Precision**: {'활성화' if self.config.get('fp16', False) else '비활성화'}"
            )
            lines.append("")

        # 현재 진행 상황
        if self.history["train_loss"]:
            current_step = self.history["step"][-1] if self.history["step"] else 0
            current_epoch = self.history["epoch"][-1] if self.history["epoch"] else 0
            recent_loss = self.history["train_loss"][-1]
            min_loss = min(self.history["train_loss"])

            lines.append("### 학습 진행 상황")
            lines.append(f"- **현재 스텝**: {current_step}")
            lines.append(f"- **현재 에폭**: {current_epoch}")
            lines.append(f"- **최근 학습 손실**: {recent_loss:.4f}")
            lines.append(f"- **최소 학습 손실**: {min_loss:.4f}")

            if self.history["eval_loss"]:
                recent_eval = self.history["eval_loss"][-1]["value"]
                min_eval = min(e["value"] for e in self.history["eval_loss"])
                lines.append(f"- **최근 검증 손실**: {recent_eval:.4f}")
                lines.append(f"- **최소 검증 손실**: {min_eval:.4f}")

            if self.start_time:
                elapsed = datetime.now() - self.start_time
                lines.append(f"- **경과 시간**: {str(elapsed).split('.')[0]}")

            lines.append("")

        # 시각화 결과
        lines.append("### 학습 곡선")
        lines.append("")
        lines.append("#### 손실 변화")
        lines.append(f"![Loss Curves]({self.output_dir}/loss_curves.png)")
        lines.append("")
        lines.append("#### 학습률 변화")
        lines.append(f"![Learning Rate]({self.output_dir}/learning_rate.png)")
        lines.append("")
        lines.append("#### 종합 대시보드")
        lines.append(f"![Training Dashboard]({self.output_dir}/training_dashboard.png)")
        lines.append("")

        # 체크포인트 정보
        if self.config.get("output_dir"):
            lines.append("### 체크포인트")
            lines.append(f"체크포인트는 `{self.config['output_dir']}`에 저장됩니다.")
            lines.append("")

        return "\n".join(lines)
