#!/usr/bin/env python3
"""
WMT 2014 English-German 번역 데이터셋 다운로드 스크립트

RTX 3090 학습을 위한 서브셋 준비
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

import yaml
from tqdm import tqdm


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """URL에서 파일을 다운로드합니다."""
    if os.path.exists(dest_path):
        print(f"파일이 이미 존재합니다: {dest_path}")
        return

    # 파일 크기 확인
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("Content-Length", 0))

    # 다운로드 진행률 표시
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:

        def update_progress(_block_num, block_size, _total_size):
            pbar.update(block_size)

        urllib.request.urlretrieve(url, dest_path, reporthook=update_progress)


def extract_files(archive_path: str, extract_to: str):
    """압축 파일을 해제합니다."""
    print(f"압축 해제 중: {archive_path}")
    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith(".gz"):
        import gzip
        import shutil

        output_path = archive_path[:-3]  # .gz 제거
        with gzip.open(archive_path, "rb") as f_in, open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def prepare_wmt14_subset(config_path: str):
    """WMT14 데이터의 서브셋을 준비합니다."""
    # 설정 파일 로드
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== WMT 2014 En-De 데이터셋 다운로드 ===\n")

    # WMT14 데이터 URL들
    # 주의: 실제 WMT14 데이터는 여러 소스에서 제공됩니다
    # 여기서는 예시 URL을 사용합니다
    urls = {
        "training": [
            # Europarl v7
            "http://www.statmt.org/europarl/v7/de-en.tgz",
            # Common Crawl
            "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            # News Commentary
            "http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz",
        ],
        "dev": [
            # newstest2013 (validation set)
            "http://data.statmt.org/wmt14/test-full/newstest2013-src.en.sgm",
            "http://data.statmt.org/wmt14/test-full/newstest2013-ref.de.sgm",
        ],
        "test": [
            # newstest2014 (test set)
            "http://data.statmt.org/wmt14/test-full/newstest2014-deen.src.en.sgm",
            "http://data.statmt.org/wmt14/test-full/newstest2014-deen.ref.de.sgm",
        ],
    }

    # 다운로드 디렉토리 생성
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    print("주의: WMT14 전체 데이터셋은 매우 큽니다 (수 GB).")
    print(f"설정된 학습 데이터 크기: {config['data']['train_size']} 문장\n")

    # 간단한 다운로드 예시 (실제로는 더 복잡한 처리 필요)
    print("데이터 다운로드를 시작하려면 실제 WMT14 데이터 소스를 확인하세요.")
    print("대안으로 Hugging Face Datasets를 사용할 수 있습니다:")
    print("  pip install datasets")
    print("  from datasets import load_dataset")
    print('  dataset = load_dataset("wmt14", "de-en")\n')

    # 서브셋 준비를 위한 디렉토리 구조
    for split in ["train", "val", "test"]:
        (data_dir / split).mkdir(exist_ok=True)

    # 설정 파일에 데이터 경로 저장
    data_info = {
        "data_dir": str(data_dir),
        "train_size": config["data"]["train_size"],
        "val_size": config["data"]["val_size"],
        "test_size": config["data"]["test_size"],
        "prepared": False,
        "note": "Hugging Face datasets 라이브러리 사용을 권장합니다",
    }

    info_path = data_dir / "data_info.yaml"
    with open(info_path, "w") as f:
        yaml.dump(data_info, f, default_flow_style=False)

    print(f"\n데이터 정보가 {info_path}에 저장되었습니다.")
    print("\n다음 단계:")
    print("1. Hugging Face datasets로 WMT14 데이터 로드")
    print("2. BPE 토크나이저 학습")
    print("3. 데이터 전처리 및 서브셋 생성")


def main():
    parser = argparse.ArgumentParser(description="WMT14 데이터 다운로드")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rtx3090.yaml",
        help="설정 파일 경로",
    )
    args = parser.parse_args()

    prepare_wmt14_subset(args.config)


if __name__ == "__main__":
    main()
