"""
WMT14 데이터셋을 위한 메모리 효율적인 데이터 로더

RTX 3090에 최적화된 배치 처리 및 동적 패딩 지원
"""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class WMTDataset(Dataset):
    """메모리 효율적인 WMT 번역 데이터셋"""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer_path: str,
        max_length: int = 100,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.use_cache = use_cache

        # SentencePiece 토크나이저 로드
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(tokenizer_path)
        except ImportError:
            raise ImportError("sentencepiece가 필요합니다: uv pip install sentencepiece")

        # 특수 토큰 ID
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()

        # 데이터 로드
        self._load_data()

    def _load_data(self):
        """토크나이즈된 데이터를 메모리 효율적으로 로드"""
        src_path = self.data_dir / f"{self.split}.tok.en"
        tgt_path = self.data_dir / f"{self.split}.tok.de"

        # 캐시 확인
        cache_path = self.data_dir / f"{self.split}_cache.pt"
        if self.use_cache and cache_path.exists():
            print(f"캐시에서 {self.split} 데이터 로드 중...")
            cache = torch.load(cache_path)
            self.src_data = cache["src"]
            self.tgt_data = cache["tgt"]
            return

        print(f"{self.split} 데이터 로드 중...")
        self.src_data = []
        self.tgt_data = []

        with (
            open(src_path, encoding="utf-8") as src_f,
            open(tgt_path, encoding="utf-8") as tgt_f,
        ):
            for src_line, tgt_line in tqdm(zip(src_f, tgt_f), desc=f"{self.split} 로드"):
                # 토큰을 ID로 변환
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()

                # 길이 제한 적용 (BOS/EOS 포함해서 계산)
                if len(src_tokens) > self.max_length - 2 or len(tgt_tokens) > self.max_length - 2:
                    continue

                # BOS, EOS 토큰 추가
                src_ids = (
                    [self.bos_id]
                    + self.sp.piece_to_id(src_tokens[: self.max_length - 2])
                    + [self.eos_id]
                )
                tgt_ids = (
                    [self.bos_id]
                    + self.sp.piece_to_id(tgt_tokens[: self.max_length - 2])
                    + [self.eos_id]
                )

                self.src_data.append(torch.tensor(src_ids, dtype=torch.long))
                self.tgt_data.append(torch.tensor(tgt_ids, dtype=torch.long))

        # 캐시 저장
        if self.use_cache:
            print(f"캐시 저장 중: {cache_path}")
            torch.save(
                {"src": self.src_data, "tgt": self.tgt_data},
                cache_path,
            )

        print(f"{self.split} 데이터 로드 완료: {len(self.src_data)} 샘플")

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {
            "src": self.src_data[idx],
            "tgt": self.tgt_data[idx],
            "src_len": len(self.src_data[idx]),
            "tgt_len": len(self.tgt_data[idx]),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """동적 패딩을 적용하는 collate 함수"""
    # 소스와 타겟 분리
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    # 패딩 적용
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # 마스크 생성
    src_mask = (src_padded != pad_id).float()
    tgt_mask = (tgt_padded != pad_id).float()

    # Decoder를 위한 look-ahead 마스크
    tgt_seq_len = tgt_padded.size(1)
    tgt_attn_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "tgt_attn_mask": tgt_attn_mask,
    }


class DynamicBatchSampler:
    """비슷한 길이의 시퀀스를 함께 배치하는 샘플러"""

    def __init__(
        self,
        dataset: WMTDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 길이별로 인덱스 정렬
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: max(dataset[i]["src_len"], dataset[i]["tgt_len"]),
        )

    def __iter__(self):
        if self.shuffle:
            # 비슷한 길이끼리 섞기
            import random

            # 버킷으로 나누기
            buckets = []
            current_bucket = []

            for idx in self.sorted_indices:
                current_bucket.append(idx)
                if len(current_bucket) == self.batch_size * 10:  # 버킷 크기
                    random.shuffle(current_bucket)
                    buckets.extend(current_bucket)
                    current_bucket = []

            if current_bucket:
                random.shuffle(current_bucket)
                buckets.extend(current_bucket)

            indices = buckets
        else:
            indices = self.sorted_indices

        # 배치 생성
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataloaders(
    config: dict,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """학습, 검증, 테스트 데이터로더 생성"""
    data_dir = config["paths"]["data_dir"]
    data_stats_path = Path(data_dir) / "data_stats.json"

    # 데이터 통계 로드
    if not data_stats_path.exists():
        raise FileNotFoundError(
            "데이터가 준비되지 않았습니다. prepare_wmt14_data.py를 먼저 실행하세요."
        )

    with open(data_stats_path) as f:
        data_stats = json.load(f)

    tokenizer_path = data_stats["spm_model"]

    # 데이터셋 생성
    train_dataset = WMTDataset(
        data_dir,
        "train",
        tokenizer_path,
        max_length=config["data"]["max_length"],
    )

    val_dataset = WMTDataset(
        data_dir,
        "val",
        tokenizer_path,
        max_length=config["data"]["max_length"],
    )

    test_dataset = WMTDataset(
        data_dir,
        "test",
        tokenizer_path,
        max_length=config["data"]["max_length"],
    )

    # Collate 함수 (패딩 ID 포함)
    def collate_with_pad(batch):
        return collate_fn(batch, pad_id=train_dataset.pad_id)

    # 동적 배치 사용 여부
    if config["data"].get("use_dynamic_batching", True):
        # 동적 배치 샘플러
        train_sampler = DynamicBatchSampler(
            train_dataset, config["training"]["batch_size"], shuffle=True
        )
        val_sampler = DynamicBatchSampler(
            val_dataset, config["training"]["batch_size"], shuffle=False
        )
        test_sampler = DynamicBatchSampler(
            test_dataset, config["training"]["batch_size"], shuffle=False
        )

        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    else:
        # 일반 데이터로더
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_with_pad,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    print("\n데이터로더 생성 완료:")
    print(f"  학습: {len(train_dataset)} 샘플, {len(train_loader)} 배치")
    print(f"  검증: {len(val_dataset)} 샘플, {len(val_loader)} 배치")
    print(f"  테스트: {len(test_dataset)} 샘플, {len(test_loader)} 배치")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 테스트
    import yaml

    config_path = "configs/rtx3090_debug.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 데이터로더 생성 테스트
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)

        # 첫 번째 배치 확인
        batch = next(iter(train_loader))
        print("\n첫 번째 배치:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    except FileNotFoundError as e:
        print(e)
        print("\n먼저 다음 명령을 실행하세요:")
        print(f"  python scripts/prepare_wmt14_data.py --config {config_path}")
