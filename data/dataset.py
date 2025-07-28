"""
Dataset 클래스 구현

번역, 언어 모델링, 분류 등 다양한 태스크를 위한 데이터셋 클래스들을 구현합니다.
"""

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import BaseTokenizer


@dataclass
class DataCollatorForLanguageModeling:
    """언어 모델링을 위한 Data Collator

    MLM (Masked Language Modeling) 또는 CLM (Causal Language Modeling)을 위한
    배치 데이터를 준비합니다.
    """

    tokenizer: BaseTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """배치 처리"""
        # 패딩
        batch = self._pad_batch(examples)

        # MLM 처리
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])
        else:
            # CLM의 경우 input_ids를 그대로 labels로 사용
            batch["labels"] = batch["input_ids"].clone()

        return batch

    def _pad_batch(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """배치 패딩 처리"""
        # 가장 긴 시퀀스 길이 찾기
        max_length = max(len(ex["input_ids"]) for ex in examples)

        # pad_to_multiple_of 적용
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        # 패딩 적용
        batch = defaultdict(list)
        for example in examples:
            for key, value in example.items():
                if key == "input_ids":
                    # 패딩 추가
                    padded = value + [self.tokenizer.pad_token_id] * (max_length - len(value))
                    batch[key].append(padded)
                elif key == "attention_mask":
                    # Attention mask
                    padded = value + [0] * (max_length - len(value))
                    batch[key].append(padded)

        # 텐서로 변환
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        # Attention mask 생성 (없는 경우)
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()

        return batch

    def mask_tokens(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MLM을 위한 마스킹"""
        labels = inputs.clone()

        # 마스킹할 위치 선택 (특수 토큰 제외)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self._get_special_tokens_mask(labels)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 마스킹되지 않은 토큰은 loss 계산에서 제외

        # 80% - MASK 토큰으로 교체
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% - 랜덤 토큰으로 교체
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 나머지 10%는 변경하지 않음

        return inputs, labels

    def _get_special_tokens_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """특수 토큰 마스크 생성"""
        special_tokens_ids = [
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id,
        ]

        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in special_tokens_ids:
            if token_id is not None:
                special_tokens_mask |= inputs == token_id

        return special_tokens_mask


@dataclass
class DataCollatorForSeq2Seq:
    """Seq2Seq 태스크를 위한 Data Collator"""

    tokenizer: BaseTokenizer
    padding: bool | str = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """배치 처리"""
        # Source와 Target 분리
        sources = []
        targets = []

        for feature in features:
            if "source" in feature:
                sources.append(feature["source"])
            elif "input_ids" in feature:
                sources.append(feature["input_ids"])

            if "target" in feature:
                targets.append(feature["target"])
            elif "labels" in feature:
                targets.append(feature["labels"])

        # 패딩 처리
        batch = {}

        # Source 패딩
        if sources:
            max_source_length = max(len(s) for s in sources)
            if self.pad_to_multiple_of:
                max_source_length = (
                    (max_source_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
                ) * self.pad_to_multiple_of

            padded_sources = []
            source_masks = []

            for source in sources:
                padded = source + [self.tokenizer.pad_token_id] * (max_source_length - len(source))
                mask = [1] * len(source) + [0] * (max_source_length - len(source))
                padded_sources.append(padded)
                source_masks.append(mask)

            batch["input_ids"] = torch.tensor(padded_sources)
            batch["attention_mask"] = torch.tensor(source_masks)

        # Target 패딩
        if targets:
            max_target_length = max(len(t) for t in targets)
            if self.pad_to_multiple_of:
                max_target_length = (
                    (max_target_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
                ) * self.pad_to_multiple_of

            padded_targets = []

            for target in targets:
                padded = target + [self.label_pad_token_id] * (max_target_length - len(target))
                padded_targets.append(padded)

            batch["labels"] = torch.tensor(padded_targets)

        return batch


class TextDataset(Dataset):
    """기본 텍스트 데이터셋"""

    def __init__(
        self,
        file_path: str,
        tokenizer: BaseTokenizer,
        max_length: int = 512,
        truncation: bool = True,
    ):
        """
        Args:
            file_path: 텍스트 파일 경로 (한 줄에 하나의 텍스트)
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            truncation: Truncation 여부
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

        # 데이터 로드
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(line)

        logging.info(f"Loaded {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # 토큰화
        encoded = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.max_length, truncation=self.truncation
        )

        return {"input_ids": encoded, "attention_mask": [1] * len(encoded)}


class TranslationDataset(Dataset):
    """번역 데이터셋"""

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512,
        truncation: bool = True,
    ):
        """
        Args:
            src_file: 소스 언어 파일
            tgt_file: 타겟 언어 파일
            src_tokenizer: 소스 언어 토크나이저
            tgt_tokenizer: 타겟 언어 토크나이저
            max_source_length: 소스 최대 길이
            max_target_length: 타겟 최대 길이
            truncation: Truncation 여부
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.truncation = truncation

        # 데이터 로드
        self.examples = []
        with (
            open(src_file, encoding="utf-8") as sf,
            open(tgt_file, encoding="utf-8") as tf,
        ):
            for src_line, tgt_line in zip(sf, tf):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if src_line and tgt_line:
                    self.examples.append((src_line, tgt_line))

        logging.info(f"Loaded {len(self.examples)} translation pairs")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_text, tgt_text = self.examples[idx]

        # 소스 토큰화
        src_encoded = self.src_tokenizer.encode(
            src_text,
            add_special_tokens=True,
            max_length=self.max_source_length,
            truncation=self.truncation,
        )

        # 타겟 토큰화
        tgt_encoded = self.tgt_tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_target_length,
            truncation=self.truncation,
        )

        return {
            "source": src_encoded,
            "target": tgt_encoded[:-1],  # Decoder input (EOS 제외)
            "labels": tgt_encoded[1:],  # Decoder target (BOS 제외)
        }


class LanguageModelingDataset(Dataset):
    """언어 모델링 데이터셋 (CLM/MLM)"""

    def __init__(
        self,
        file_path: str,
        tokenizer: BaseTokenizer,
        block_size: int = 512,
        mlm: bool = False,
        overwrite_cache: bool = False,
    ):
        """
        Args:
            file_path: 텍스트 파일 경로
            tokenizer: 토크나이저
            block_size: 블록 크기 (시퀀스 길이)
            mlm: Masked Language Modeling 여부
            overwrite_cache: 캐시 덮어쓰기 여부
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mlm = mlm

        # 캐시 파일 경로
        cache_file = f"{file_path}.{tokenizer.__class__.__name__}.{block_size}.cache"

        # 캐시 확인
        if os.path.exists(cache_file) and not overwrite_cache:
            logging.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                import pickle

                self.examples = pickle.load(f)
        else:
            logging.info(f"Creating dataset from {file_path}")
            self.examples = self._create_examples(file_path)

            # 캐시 저장
            logging.info(f"Saving dataset to cache {cache_file}")
            with open(cache_file, "wb") as f:
                import pickle

                pickle.dump(self.examples, f)

        logging.info(f"Dataset contains {len(self.examples)} examples of block size {block_size}")

    def _create_examples(self, file_path: str) -> list[list[int]]:
        """블록 단위로 예제 생성"""
        # 전체 텍스트 토큰화
        tokenized_text = []

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    tokenized_text.extend(tokens)
                    # 문서 경계에 EOS 토큰 추가
                    if self.tokenizer.eos_token_id is not None:
                        tokenized_text.append(self.tokenizer.eos_token_id)

        # 블록으로 분할
        examples = []
        for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size):
            examples.append(tokenized_text[i : i + self.block_size])

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "attention_mask": [1] * len(self.examples[idx])}


class ClassificationDataset(Dataset):
    """분류 데이터셋"""

    def __init__(
        self,
        file_path: str,
        tokenizer: BaseTokenizer,
        max_length: int = 512,
        label_map: dict[str, int] | None = None,
    ):
        """
        Args:
            file_path: JSON 파일 경로 ({"text": "...", "label": "..."} 형식)
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            label_map: 레이블을 ID로 매핑하는 사전
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 데이터 로드
        with open(file_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        # 레이블 맵 생성
        if label_map is None:
            labels = sorted({item["label"] for item in self.data})
            self.label_map = {label: i for i, label in enumerate(labels)}
        else:
            self.label_map = label_map

        self.id_to_label = {v: k for k, v in self.label_map.items()}

        logging.info(f"Loaded {len(self.data)} examples with {len(self.label_map)} labels")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = self.label_map[item["label"]]

        # 토큰화
        encoded = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.max_length, truncation=True
        )

        return {"input_ids": encoded, "attention_mask": [1] * len(encoded), "labels": label}

    @property
    def num_labels(self):
        return len(self.label_map)


class DatasetSplitter:
    """데이터셋 분할 유틸리티"""

    @staticmethod
    def train_test_split(
        dataset: Dataset, test_size: float = 0.1, random_seed: int = 42
    ) -> tuple[Dataset, Dataset]:
        """학습/테스트 데이터셋 분할"""
        random.seed(random_seed)

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        split = int(len(dataset) * (1 - test_size))
        train_indices = indices[:split]
        test_indices = indices[split:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        return train_dataset, test_dataset

    @staticmethod
    def train_val_test_split(
        dataset: Dataset, val_size: float = 0.1, test_size: float = 0.1, random_seed: int = 42
    ) -> tuple[Dataset, Dataset, Dataset]:
        """학습/검증/테스트 데이터셋 분할"""
        random.seed(random_seed)

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        test_split = int(len(dataset) * (1 - test_size))
        val_split = int(test_split * (1 - val_size))

        train_indices = indices[:val_split]
        val_indices = indices[val_split:test_split]
        test_indices = indices[test_split:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        return train_dataset, val_dataset, test_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Any | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader 생성 헬퍼 함수"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
