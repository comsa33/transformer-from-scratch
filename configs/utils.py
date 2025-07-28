"""
Configuration 로드 및 관리 유틸리티
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from transformer.config import TransformerConfig


class ConfigDict(dict):
    """점 표기법을 지원하는 설정 딕셔너리"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def get_model_config(self) -> TransformerConfig:
        """TransformerConfig 객체 반환"""
        if "model" not in self:
            raise ValueError("설정에 'model' 섹션이 없습니다.")

        model_dict = (
            self["model"].copy() if isinstance(self["model"], dict) else asdict(self["model"])
        )

        # vocab_size가 없으면 임시값 설정 (나중에 데이터셋에서 설정됨)
        if "vocab_size" not in model_dict:
            model_dict["vocab_size"] = 30000  # 임시 기본값

        return TransformerConfig(**model_dict)

    def get_training_config(self) -> "ConfigDict":
        """학습 설정 반환"""
        if "training" not in self:
            raise ValueError("설정에 'training' 섹션이 없습니다.")
        return ConfigDict(self["training"])

    def get_optimizer_config(self) -> "ConfigDict":
        """옵티마이저 설정 반환"""
        if "optimizer" not in self:
            raise ValueError("설정에 'optimizer' 섹션이 없습니다.")
        return ConfigDict(self["optimizer"])

    def get_scheduler_config(self) -> "ConfigDict":
        """스케줄러 설정 반환"""
        if "scheduler" not in self:
            raise ValueError("설정에 'scheduler' 섹션이 없습니다.")
        return ConfigDict(self["scheduler"])

    def merge(self, other: dict[str, Any]) -> "ConfigDict":
        """다른 설정과 병합 (deep merge)"""

        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        merged = deep_merge(self.copy(), other)
        return ConfigDict(merged)


def load_config(config_name_or_path: str | Path) -> ConfigDict:
    """설정 파일 로드

    Args:
        config_name_or_path: 설정 이름 (예: 'base', 'small') 또는 파일 경로

    Returns:
        ConfigDict: 로드된 설정
    """
    # 설정 이름으로 전달된 경우
    if isinstance(config_name_or_path, str) and not config_name_or_path.endswith(
        (".yaml", ".yml", ".json")
    ):
        config_dir = Path(__file__).parent
        config_path = config_dir / f"{config_name_or_path}.yaml"
    else:
        config_path = Path(config_name_or_path)

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    # 파일 확장자에 따라 로드
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {config_path.suffix}")

    return ConfigDict(config_dict)


def save_config(config: ConfigDict | dict, save_path: str | Path) -> None:
    """설정을 파일로 저장

    Args:
        config: 저장할 설정
        save_path: 저장 경로
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ConfigDict를 일반 dict로 변환
    def to_dict(obj):
        if isinstance(obj, ConfigDict | dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        else:
            return obj

    config_dict = to_dict(config)

    # 파일 확장자에 따라 저장
    if save_path.suffix in [".yaml", ".yml"]:
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    elif save_path.suffix == ".json":
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {save_path.suffix}")


def merge_configs(*configs: ConfigDict | dict | str | Path) -> ConfigDict:
    """여러 설정을 병합

    Args:
        *configs: 병합할 설정들 (뒤의 설정이 우선순위가 높음)

    Returns:
        ConfigDict: 병합된 설정
    """
    merged = ConfigDict()

    for config in configs:
        if isinstance(config, str | Path):
            config = load_config(config)
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            config = ConfigDict(config)

        merged = merged.merge(config)

    return merged


def create_config_from_args(args: Any) -> ConfigDict:
    """명령줄 인자에서 설정 생성

    Args:
        args: argparse 또는 다른 인자 객체

    Returns:
        ConfigDict: 생성된 설정
    """
    # 기본 설정 로드
    config = load_config(args.config) if hasattr(args, "config") and args.config else ConfigDict()

    # 명령줄 인자로 덮어쓰기
    args_dict = vars(args) if hasattr(args, "__dict__") else args

    for key, value in args_dict.items():
        if value is not None and key != "config":
            # 중첩된 키 처리 (예: model.d_model)
            keys = key.split(".")
            current = config

            for k in keys[:-1]:
                if k not in current:
                    current[k] = ConfigDict()
                current = current[k]

            current[keys[-1]] = value

    return config
