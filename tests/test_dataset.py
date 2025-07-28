"""
Dataset 클래스 테스트
"""

import sys

sys.path.append(".")

import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from data.dataset import (
    ClassificationDataset,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DatasetSplitter,
    LanguageModelingDataset,
    TextDataset,
    TranslationDataset,
    create_dataloader,
)
from data.tokenizer import SimpleTokenizer


def create_sample_files():
    """테스트용 샘플 파일 생성"""
    # 텍스트 파일
    text_data = [
        "This is a sample text for testing.",
        "Machine learning is fascinating.",
        "Natural language processing with transformers.",
        "Deep learning models are powerful.",
        "Attention is all you need.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "Artificial intelligence is the future.",
    ]

    text_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    text_file.write("\n".join(text_data))
    text_file.close()

    # 번역 파일
    src_data = [
        "Hello, world!",
        "How are you?",
        "I love programming.",
        "Machine learning is amazing.",
        "See you tomorrow.",
    ]

    tgt_data = [
        "안녕하세요, 세계!",
        "어떻게 지내세요?",
        "저는 프로그래밍을 좋아합니다.",
        "머신러닝은 놀랍습니다.",
        "내일 봐요.",
    ]

    src_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_src.txt")
    src_file.write("\n".join(src_data))
    src_file.close()

    tgt_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_tgt.txt")
    tgt_file.write("\n".join(tgt_data))
    tgt_file.close()

    # 분류 데이터
    classification_data = [
        {"text": "This movie is great!", "label": "positive"},
        {"text": "I hate this product.", "label": "negative"},
        {"text": "The service was okay.", "label": "neutral"},
        {"text": "Absolutely fantastic experience!", "label": "positive"},
        {"text": "Terrible quality, very disappointed.", "label": "negative"},
        {"text": "It's fine, nothing special.", "label": "neutral"},
    ]

    class_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
    for item in classification_data:
        class_file.write(json.dumps(item) + "\n")
    class_file.close()

    return text_file.name, src_file.name, tgt_file.name, class_file.name


def test_text_dataset():
    """TextDataset 테스트"""
    print("=== TextDataset 테스트 ===\n")

    # 샘플 파일 생성
    text_file, _, _, _ = create_sample_files()

    # 토크나이저 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["This is a test.", "Machine learning", "Natural language"])

    # 데이터셋 생성
    dataset = TextDataset(file_path=text_file, tokenizer=tokenizer, max_length=20)

    print(f"데이터셋 크기: {len(dataset)}")

    # 샘플 확인
    print("\n샘플 데이터:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n샘플 {i + 1}:")
        print(f"  Input IDs: {sample['input_ids']}")
        print(f"  Attention Mask: {sample['attention_mask']}")
        print(f"  디코딩: {tokenizer.decode(sample['input_ids'])}")

    # 정리
    os.unlink(text_file)

    return dataset


def test_translation_dataset():
    """TranslationDataset 테스트"""
    print("\n=== TranslationDataset 테스트 ===\n")

    # 샘플 파일 생성
    _, src_file, tgt_file, _ = create_sample_files()

    # 토크나이저 생성
    src_tokenizer = SimpleTokenizer()
    src_tokenizer.build_vocab(["Hello", "How are you", "I love programming"])

    tgt_tokenizer = SimpleTokenizer()
    tgt_tokenizer.build_vocab(["안녕하세요", "어떻게 지내세요", "프로그래밍"])

    # 데이터셋 생성
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_source_length=20,
        max_target_length=20,
    )

    print(f"데이터셋 크기: {len(dataset)}")

    # 샘플 확인
    print("\n샘플 데이터:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n샘플 {i + 1}:")
        print(f"  Source: {src_tokenizer.decode(sample['source'], skip_special_tokens=False)}")
        print(f"  Target: {tgt_tokenizer.decode(sample['target'], skip_special_tokens=False)}")
        print(f"  Labels: {sample['labels']}")

    # 정리
    os.unlink(src_file)
    os.unlink(tgt_file)

    return dataset


def test_language_modeling_dataset():
    """LanguageModelingDataset 테스트"""
    print("\n=== LanguageModelingDataset 테스트 ===\n")

    # 샘플 파일 생성
    text_file, _, _, _ = create_sample_files()

    # 토크나이저 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(open(text_file).read().split())

    # CLM 데이터셋
    print("1. Causal Language Modeling Dataset:")
    clm_dataset = LanguageModelingDataset(
        file_path=text_file, tokenizer=tokenizer, block_size=15, mlm=False, overwrite_cache=True
    )

    print(f"   데이터셋 크기: {len(clm_dataset)}")
    print("   블록 크기: 15")

    sample = clm_dataset[0]
    print(f"   샘플 Input IDs: {sample['input_ids']}")
    print(f"   디코딩: {tokenizer.decode(sample['input_ids'])}")

    # MLM 데이터셋
    print("\n2. Masked Language Modeling Dataset:")
    mlm_dataset = LanguageModelingDataset(
        file_path=text_file, tokenizer=tokenizer, block_size=15, mlm=True, overwrite_cache=True
    )

    print(f"   데이터셋 크기: {len(mlm_dataset)}")

    # 정리
    os.unlink(text_file)
    cache_files = [f for f in os.listdir(".") if f.endswith(".cache")]
    for cf in cache_files:
        os.unlink(cf)

    return clm_dataset, mlm_dataset


def test_classification_dataset():
    """ClassificationDataset 테스트"""
    print("\n=== ClassificationDataset 테스트 ===\n")

    # 샘플 파일 생성
    _, _, _, class_file = create_sample_files()

    # 토크나이저 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["This movie is great", "I hate this", "It's fine"])

    # 데이터셋 생성
    dataset = ClassificationDataset(file_path=class_file, tokenizer=tokenizer, max_length=20)

    print(f"데이터셋 크기: {len(dataset)}")
    print(f"레이블 수: {dataset.num_labels}")
    print(f"레이블 맵: {dataset.label_map}")

    # 샘플 확인
    print("\n샘플 데이터:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n샘플 {i + 1}:")
        print(f"  Text: {tokenizer.decode(sample['input_ids'])}")
        print(f"  Label ID: {sample['labels']}")
        print(f"  Label: {dataset.id_to_label[sample['labels']]}")

    # 정리
    os.unlink(class_file)

    return dataset


def test_data_collators():
    """Data Collator 테스트"""
    print("\n=== Data Collator 테스트 ===\n")

    # 토크나이저 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["This is a test", "Machine learning", "Hello world"])

    # 1. MLM Collator 테스트
    print("1. DataCollatorForLanguageModeling (MLM):")
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 샘플 배치
    examples = [
        {"input_ids": [2, 11, 12, 13, 3], "attention_mask": [1, 1, 1, 1, 1]},
        {"input_ids": [2, 14, 15, 3], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [2, 16, 17, 18, 19, 3], "attention_mask": [1, 1, 1, 1, 1, 1]},
    ]

    batch = mlm_collator(examples)
    print(f"   배치 키: {list(batch.keys())}")
    print(f"   Input shape: {batch['input_ids'].shape}")
    print(f"   Labels shape: {batch['labels'].shape}")
    print(f"   마스킹된 토큰 수: {(batch['labels'] != -100).sum().item()}")

    # 2. Seq2Seq Collator 테스트
    print("\n2. DataCollatorForSeq2Seq:")
    seq2seq_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
    )

    # 샘플 배치
    seq2seq_examples = [
        {"source": [2, 11, 12, 3], "target": [2, 14, 15, 3]},
        {"source": [2, 16, 17, 18, 3], "target": [2, 19, 3]},
    ]

    seq2seq_batch = seq2seq_collator(seq2seq_examples)
    print(f"   배치 키: {list(seq2seq_batch.keys())}")
    print(f"   Input shape: {seq2seq_batch['input_ids'].shape}")
    print(f"   Labels shape: {seq2seq_batch['labels'].shape}")
    print(f"   Pad to multiple of 8: Input length = {seq2seq_batch['input_ids'].shape[1]}")


def test_dataset_splitter():
    """DatasetSplitter 테스트"""
    print("\n=== DatasetSplitter 테스트 ===\n")

    # 더미 데이터셋 생성
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.data = list(range(size))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset(100)

    # 1. Train/Test 분할
    print("1. Train/Test 분할 (90/10):")
    train_dataset, test_dataset = DatasetSplitter.train_test_split(dataset, test_size=0.1)
    print(f"   Train 크기: {len(train_dataset)}")
    print(f"   Test 크기: {len(test_dataset)}")

    # 2. Train/Val/Test 분할
    print("\n2. Train/Val/Test 분할 (80/10/10):")
    train_dataset, val_dataset, test_dataset = DatasetSplitter.train_val_test_split(
        dataset, val_size=0.1, test_size=0.1
    )
    print(f"   Train 크기: {len(train_dataset)}")
    print(f"   Val 크기: {len(val_dataset)}")
    print(f"   Test 크기: {len(test_dataset)}")


def test_dataloader_creation():
    """DataLoader 생성 테스트"""
    print("\n=== DataLoader 생성 테스트 ===\n")

    # 샘플 파일 생성
    text_file, _, _, _ = create_sample_files()

    # 토크나이저와 데이터셋 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["This is a test", "Machine learning"])

    dataset = TextDataset(text_file, tokenizer, max_length=20)

    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # DataLoader 생성
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

    print(f"DataLoader 배치 수: {len(dataloader)}")
    print("배치 크기: 4")

    # 첫 번째 배치 확인
    for batch in dataloader:
        print("\n첫 번째 배치:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        break

    # 정리
    os.unlink(text_file)


def visualize_dataset_statistics():
    """데이터셋 통계 시각화"""
    print("\n=== 데이터셋 통계 시각화 ===\n")

    # 더 많은 샘플 데이터 생성
    texts = [
        "Short text.",
        "This is a medium length sentence.",
        "Here we have a much longer sentence that contains many more words than the previous examples.",
        "AI is amazing!",
        "Natural language processing with deep learning models.",
        "The quick brown fox jumps over the lazy dog multiple times in this sentence.",
    ] * 10  # 반복하여 더 많은 데이터 생성

    text_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    text_file.write("\n".join(texts))
    text_file.close()

    # 토크나이저와 데이터셋 생성
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    dataset = TextDataset(text_file.name, tokenizer, max_length=100)

    # 시퀀스 길이 수집
    lengths = []
    for i in range(len(dataset)):
        sample = dataset[i]
        lengths.append(len(sample["input_ids"]))

    # 시각화
    plt.figure(figsize=(10, 6))

    # 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("시퀀스 길이")
    plt.ylabel("빈도")
    plt.title("시퀀스 길이 분포")
    plt.axvline(
        np.mean(lengths), color="red", linestyle="--", label=f"평균: {np.mean(lengths):.1f}"
    )
    plt.legend()

    # 박스 플롯
    plt.subplot(1, 2, 2)
    plt.boxplot(lengths)
    plt.ylabel("시퀀스 길이")
    plt.title("시퀀스 길이 박스 플롯")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/dataset_statistics.png", dpi=150)
    print("데이터셋 통계가 'outputs/dataset_statistics.png'에 저장되었습니다.")

    # 통계 출력
    print("\n시퀀스 길이 통계:")
    print(f"  평균: {np.mean(lengths):.2f}")
    print(f"  표준편차: {np.std(lengths):.2f}")
    print(f"  최소: {np.min(lengths)}")
    print(f"  최대: {np.max(lengths)}")
    print(f"  중앙값: {np.median(lengths):.0f}")

    # 정리
    os.unlink(text_file.name)


def test_mlm_masking_visualization():
    """MLM 마스킹 시각화"""
    print("\n=== MLM 마스킹 시각화 ===\n")

    # 토크나이저 생성
    tokenizer = SimpleTokenizer()
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    tokenizer.build_vocab(vocab)

    # MLM collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # 샘플 텍스트
    text = "the quick brown fox jumps over the lazy dog"
    input_ids = tokenizer.encode(text, add_special_tokens=True)

    # 여러 번 마스킹하여 통계 수집
    mask_counts = np.zeros(len(input_ids))
    num_trials = 1000

    for _ in range(num_trials):
        batch = collator([{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}])
        masked_positions = (batch["labels"][0] != -100).numpy()
        mask_counts += masked_positions

    mask_probabilities = mask_counts / num_trials

    # 시각화
    plt.figure(figsize=(12, 6))

    tokens = tokenizer.decode(input_ids, skip_special_tokens=False).split()
    x = np.arange(len(tokens))

    plt.bar(x, mask_probabilities, alpha=0.7)
    plt.xticks(x, tokens, rotation=45)
    plt.ylabel("마스킹 확률")
    plt.title(f"MLM 마스킹 확률 분포 ({num_trials}회 시행)")
    plt.axhline(y=0.15, color="red", linestyle="--", label="목표 확률 (0.15)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/mlm_masking_distribution.png", dpi=150)
    print("MLM 마스킹 분포가 'outputs/mlm_masking_distribution.png'에 저장되었습니다.")

    # 통계 출력
    print("\n마스킹 통계:")
    print(f"  특수 토큰 (<bos>, <eos>) 마스킹 확률: {mask_probabilities[[0, -1]].mean():.3f}")
    print(f"  일반 토큰 마스킹 확률: {mask_probabilities[1:-1].mean():.3f}")
    print(f"  전체 평균: {mask_probabilities.mean():.3f}")


if __name__ == "__main__":
    # 1. TextDataset 테스트
    text_dataset = test_text_dataset()

    # 2. TranslationDataset 테스트
    translation_dataset = test_translation_dataset()

    # 3. LanguageModelingDataset 테스트
    clm_dataset, mlm_dataset = test_language_modeling_dataset()

    # 4. ClassificationDataset 테스트
    classification_dataset = test_classification_dataset()

    # 5. Data Collator 테스트
    test_data_collators()

    # 6. Dataset Splitter 테스트
    test_dataset_splitter()

    # 7. DataLoader 생성 테스트
    test_dataloader_creation()

    # 8. 데이터셋 통계 시각화
    visualize_dataset_statistics()

    # 9. MLM 마스킹 시각화
    test_mlm_masking_visualization()

    print("\n모든 테스트가 완료되었습니다!")
