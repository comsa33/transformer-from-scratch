"""
시스템 리소스 모니터링 유틸리티

GPU 메모리, CPU 사용률 등을 추적합니다.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch


class ResourceMonitor:
    """시스템 리소스 모니터링"""

    def __init__(self, output_dir: str = "outputs/resources"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "timestamp": [],
            "step": [],
            "gpu_memory_used": [],
            "gpu_memory_total": [],
            "gpu_utilization": [],
            "cpu_percent": [],
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()

        # GPU 정보 출력
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nGPU 모니터링 시작: {gpu_name} ({gpu_memory:.1f} GB)")

    def record(self, step: int = 0):
        """현재 리소스 상태 기록"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "elapsed_time": time.time() - self.start_time,
        }

        # GPU 메트릭
        if self.device.type == "cuda":
            # 메모리 사용량
            memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = (memory_used / memory_total) * 100

            record["gpu_memory_used_gb"] = round(memory_used, 2)
            record["gpu_memory_total_gb"] = round(memory_total, 2)
            record["gpu_memory_percent"] = round(memory_percent, 1)

            # GPU 사용률 (nvidia-ml-py 사용 시)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                record["gpu_utilization"] = util.gpu
                record["gpu_memory_util"] = util.memory
                pynvml.nvmlShutdown()
            except Exception:
                pass

        # CPU 메트릭
        try:
            import psutil

            record["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            record["memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        # 기록 저장
        for key, value in record.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

        return record

    def save(self):
        """모니터링 데이터 저장"""
        output_file = (
            self.output_dir / f"resource_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_summary(self) -> dict:
        """리소스 사용 요약"""
        if not self.history.get("gpu_memory_used_gb"):
            return {}

        summary = {
            "max_gpu_memory_gb": max(self.history.get("gpu_memory_used_gb", [0])),
            "avg_gpu_memory_gb": sum(self.history.get("gpu_memory_used_gb", []))
            / len(self.history["gpu_memory_used_gb"]),
            "max_gpu_memory_percent": max(self.history.get("gpu_memory_percent", [0])),
        }

        if "gpu_utilization" in self.history and self.history["gpu_utilization"]:
            summary["avg_gpu_utilization"] = sum(self.history["gpu_utilization"]) / len(
                self.history["gpu_utilization"]
            )

        if "cpu_percent" in self.history and self.history["cpu_percent"]:
            summary["avg_cpu_percent"] = sum(self.history["cpu_percent"]) / len(
                self.history["cpu_percent"]
            )

        return summary

    def print_current_usage(self):
        """현재 리소스 사용량 출력"""
        if self.device.type == "cuda":
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = (memory_used / memory_total) * 100

            print(
                f"\rGPU 메모리: {memory_used:.1f}/{memory_total:.1f} GB ({memory_percent:.1f}%)",
                end="",
            )


class ResourceMonitorCallback:
    """리소스 모니터링 콜백"""

    def __init__(self, monitor: ResourceMonitor, frequency: int = 10):
        self.monitor = monitor
        self.frequency = frequency
        self.step_count = 0

    def on_train_begin(self, trainer, state, model):
        """학습 시작 시 호출"""
        self.monitor.record(0)

    def on_log(self, trainer, state, model, logs: dict):
        """로그 이벤트 시 호출"""
        self.step_count += 1

        if self.step_count % self.frequency == 0:
            step = logs.get("step", self.step_count)
            self.monitor.record(step)

    def on_train_end(self, trainer, state, model):
        """학습 종료 시 호출"""
        self.monitor.save()

        # 요약 출력
        summary = self.monitor.get_summary()
        if summary:
            print("\n\n=== 리소스 사용 요약 ===")
            print(f"최대 GPU 메모리: {summary.get('max_gpu_memory_gb', 0):.2f} GB")
            print(f"평균 GPU 메모리: {summary.get('avg_gpu_memory_gb', 0):.2f} GB")
            if "avg_gpu_utilization" in summary:
                print(f"평균 GPU 사용률: {summary['avg_gpu_utilization']:.1f}%")
            if "avg_cpu_percent" in summary:
                print(f"평균 CPU 사용률: {summary['avg_cpu_percent']:.1f}%")
