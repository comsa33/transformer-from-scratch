#!/usr/bin/env python3
"""
Hugging Face datasets를 사용한 WMT14 데이터 준비 스크립트

RTX 3090용 서브셋 생성 및 BPE 토크나이저 학습
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from tqdm import tqdm


def check_dependencies():
    """필요한 라이브러리가 설치되어 있는지 확인합니다."""
    import importlib.util

    required_modules = ["datasets", "sentencepiece", "torch"]
    missing = []

    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)

    if missing:
        print("필요한 라이브러리가 없습니다. 다음 명령어로 설치하세요:")
        print("uv pip install datasets sentencepiece")
        return False

    return True


def prepare_wmt14_data(config_path: str, use_cache: bool = True):
    """WMT14 데이터를 준비하고 전처리합니다."""
    if not check_dependencies():
        return

    import sentencepiece as spm
    from datasets import load_dataset

    # 설정 로드
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== WMT14 En-De 데이터 준비 ===\n")

    # 1. 데이터셋 로드
    print("Hugging Face에서 WMT14 데이터셋 로드 중...")
    dataset = load_dataset(
        "wmt14",
        "de-en",
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )

    # 2. 서브셋 생성
    train_size = config["data"]["train_size"]
    val_size = config["data"]["val_size"]
    test_size = config["data"]["test_size"]

    print("\n서브셋 생성:")
    print(f"  학습: {train_size} 문장")
    print(f"  검증: {val_size} 문장")
    print(f"  테스트: {test_size} 문장")

    # 학습 데이터 서브셋
    train_subset = dataset["train"].select(range(min(train_size, len(dataset["train"]))))
    val_subset = dataset["validation"].select(range(min(val_size, len(dataset["validation"]))))
    test_subset = dataset["test"].select(range(min(test_size, len(dataset["test"]))))

    # 3. 원시 텍스트 파일로 저장 (토크나이저 학습용)
    print("\n원시 텍스트 파일 생성 중...")

    def save_raw_text(subset, prefix: str):
        en_path = data_dir / f"{prefix}.en"
        de_path = data_dir / f"{prefix}.de"

        with (
            open(en_path, "w", encoding="utf-8") as en_f,
            open(de_path, "w", encoding="utf-8") as de_f,
        ):
            for item in tqdm(subset, desc=f"{prefix} 저장"):
                # 'translation' 키 안에 언어별 텍스트가 있음
                en_text = item["translation"]["en"].strip()
                de_text = item["translation"]["de"].strip()
                en_f.write(en_text + "\n")
                de_f.write(de_text + "\n")

        return en_path, de_path

    train_en, train_de = save_raw_text(train_subset, "train")
    val_en, val_de = save_raw_text(val_subset, "val")
    test_en, test_de = save_raw_text(test_subset, "test")

    # 4. BPE 토크나이저 학습
    print("\nBPE 토크나이저 학습 중...")
    vocab_size = config["data"]["vocab_size"]

    # 학습 데이터 결합 (영어 + 독일어)
    combined_train = data_dir / "train_combined.txt"
    with open(combined_train, "w", encoding="utf-8") as out_f:
        for path in [train_en, train_de]:
            with open(path, encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    # SentencePiece 모델 학습
    spm_model_prefix = str(data_dir / "spm_bpe")
    spm.SentencePieceTrainer.train(
        input=str(combined_train),
        model_prefix=spm_model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        max_sentence_length=256,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print(f"BPE 모델 저장: {spm_model_prefix}.model")

    # 5. 토크나이징된 데이터 생성
    print("\n데이터 토크나이징 중...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{spm_model_prefix}.model")

    def tokenize_file(input_path: Path, output_path: Path):
        with (
            open(input_path, encoding="utf-8") as in_f,
            open(output_path, "w", encoding="utf-8") as out_f,
        ):
            for line in tqdm(in_f, desc=f"{input_path.name} 토크나이징"):
                tokens = sp.encode(line.strip(), out_type=str)
                out_f.write(" ".join(tokens) + "\n")

    # 모든 파일 토크나이징
    for prefix in ["train", "val", "test"]:
        for lang in ["en", "de"]:
            input_path = data_dir / f"{prefix}.{lang}"
            output_path = data_dir / f"{prefix}.tok.{lang}"
            tokenize_file(input_path, output_path)

    # 6. 데이터 통계 저장
    stats = {
        "vocab_size": vocab_size,
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_subset),
        "spm_model": f"{spm_model_prefix}.model",
        "prepared": True,
    }

    # 간단한 통계
    with open(train_en, encoding="utf-8") as f:
        train_lines = f.readlines()
        avg_len_en = sum(len(line.split()) for line in train_lines) / len(train_lines)

    with open(train_de, encoding="utf-8") as f:
        train_lines = f.readlines()
        avg_len_de = sum(len(line.split()) for line in train_lines) / len(train_lines)

    stats["avg_length"] = {
        "en": round(avg_len_en, 2),
        "de": round(avg_len_de, 2),
    }

    stats_path = data_dir / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n데이터 준비 완료!")
    print(f"통계 정보: {stats_path}")
    print(f"평균 문장 길이 - EN: {avg_len_en:.1f} 단어, DE: {avg_len_de:.1f} 단어")

    # 정리
    if not use_cache:
        combined_train.unlink()


def main():
    parser = argparse.ArgumentParser(description="WMT14 데이터 준비")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtx3090.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="캐시 파일 삭제",
    )
    args = parser.parse_args()

    prepare_wmt14_data(args.config, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
