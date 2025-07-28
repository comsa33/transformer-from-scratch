"""
Tokenizer 테스트 및 시각화
"""

import sys

sys.path.append(".")

import os
import time

import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from data.tokenizer import BPETokenizer, SimpleTokenizer, WordPieceTokenizer

# 테스트용 텍스트 데이터
SAMPLE_TEXTS = [
    "Hello, world! This is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
    "Natural language processing with transformers.",
    "안녕하세요, 한국어 테스트입니다.",
    "I love programming in Python!",
    "Deep learning models are powerful.",
    "Attention is all you need.",
]

KOREAN_TEXTS = [
    "안녕하세요, 오늘 날씨가 좋네요.",
    "한국어 자연어 처리는 어렵습니다.",
    "트랜스포머 모델을 구현하고 있습니다.",
    "딥러닝은 정말 재미있어요!",
    "인공지능의 미래는 밝습니다.",
]


def test_simple_tokenizer():
    """Simple Tokenizer 테스트"""
    print("=== Simple Tokenizer 테스트 ===\n")

    # 토크나이저 생성
    tokenizer = SimpleTokenizer(lowercase=True)

    # 어휘 구축
    print("1. 어휘 구축")
    tokenizer.build_vocab(SAMPLE_TEXTS, min_freq=1, max_vocab_size=100)
    print(f"   어휘 크기: {tokenizer.vocab_size}")
    print(f"   특수 토큰: {tokenizer.special_tokens}")

    # 토큰화 테스트
    print("\n2. 토큰화 테스트")
    test_text = "Hello, world! This is a new test."
    tokens = tokenizer.tokenize(test_text)
    print(f"   원본: {test_text}")
    print(f"   토큰: {tokens}")

    # 인코딩/디코딩 테스트
    print("\n3. 인코딩/디코딩 테스트")
    ids = tokenizer.encode(test_text, add_special_tokens=True)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"   토큰 ID: {ids}")
    print(f"   디코딩: {decoded}")

    # 배치 처리 테스트
    print("\n4. 배치 처리 테스트")
    batch_texts = ["Hello world", "Test sentence", "Another example"]
    batch_encoded = tokenizer.batch_encode(batch_texts, padding=True, add_special_tokens=True)
    print(f"   배치 크기: {len(batch_encoded['input_ids'])}")
    print(f"   패딩된 길이: {len(batch_encoded['input_ids'][0])}")

    # Unknown 토큰 처리
    print("\n5. Unknown 토큰 처리")
    unknown_text = "This contains unknown words: cryptocurrency blockchain"
    unknown_ids = tokenizer.encode(unknown_text)
    print(f"   텍스트: {unknown_text}")
    print(f"   Unknown 토큰 수: {unknown_ids.count(tokenizer.unk_token_id)}")

    return tokenizer


def test_bpe_tokenizer():
    """BPE Tokenizer 테스트"""
    print("\n=== BPE Tokenizer 테스트 ===\n")

    # 토크나이저 생성
    tokenizer = BPETokenizer(vocab_size=100, min_frequency=2)

    # BPE 학습
    print("1. BPE 학습")
    start_time = time.time()
    tokenizer.learn_bpe(SAMPLE_TEXTS)
    learn_time = time.time() - start_time
    print(f"   학습 시간: {learn_time:.2f}초")
    print(f"   어휘 크기: {tokenizer.vocab_size}")
    print(f"   병합 규칙 수: {len(tokenizer.merges)}")

    # 병합 규칙 예시
    print("\n2. 병합 규칙 예시 (처음 10개)")
    for i, merge in enumerate(tokenizer.merges[:10]):
        print(f"   {i + 1}. '{merge[0]}' + '{merge[1]}' → '{merge[0]}{merge[1]}'")

    # 토큰화 테스트
    print("\n3. 토큰화 테스트")
    test_texts = ["hello world", "machine learning", "natural language"]
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"   '{text}' → {tokens}")

    # Subword 분할 효과
    print("\n4. Subword 분할 효과")
    words = ["learning", "learned", "learner", "unlearnable"]
    for word in words:
        tokens = tokenizer.tokenize(word)
        print(f"   '{word}' → {tokens}")

    return tokenizer


def test_wordpiece_tokenizer():
    """WordPiece Tokenizer 테스트"""
    print("\n=== WordPiece Tokenizer 테스트 ===\n")

    # 토크나이저 생성
    tokenizer = WordPieceTokenizer(vocab_size=200)

    # 어휘 구축
    print("1. WordPiece 어휘 구축")
    tokenizer.build_vocab(SAMPLE_TEXTS + KOREAN_TEXTS)
    print(f"   어휘 크기: {tokenizer.vocab_size}")

    # 기본 토큰화 테스트
    print("\n2. 기본 토큰화 테스트")
    test_text = "Hello, this is a test of WordPiece tokenization."
    basic_tokens = tokenizer.basic_tokenizer(test_text)
    wordpiece_tokens = tokenizer.tokenize(test_text)
    print(f"   원본: {test_text}")
    print(f"   기본 토큰: {basic_tokens}")
    print(f"   WordPiece: {wordpiece_tokens}")

    # Subword 처리
    print("\n3. Subword 처리")
    test_words = ["unbelievable", "preprocessing", "tokenization"]
    for word in test_words:
        tokens = tokenizer.tokenize(word)
        print(f"   '{word}' → {tokens}")

    # 한국어 처리
    print("\n4. 한국어 처리")
    korean_text = "안녕하세요, 트랜스포머입니다."
    korean_tokens = tokenizer.tokenize(korean_text)
    print(f"   원본: {korean_text}")
    print(f"   토큰: {korean_tokens}")

    # ## 접두사 확인
    print("\n5. ## 접두사 패턴")
    subword_tokens = [token for token in tokenizer.vocab if token.startswith("##")]
    print(f"   ## 토큰 수: {len(subword_tokens)}")
    print(f"   예시: {subword_tokens[:10]}")

    return tokenizer


def compare_tokenizers():
    """다양한 토크나이저 비교"""
    print("\n=== 토크나이저 비교 ===\n")

    # 토크나이저들 생성
    simple_tok = SimpleTokenizer()
    simple_tok.build_vocab(SAMPLE_TEXTS)

    bpe_tok = BPETokenizer(vocab_size=150)
    bpe_tok.learn_bpe(SAMPLE_TEXTS)

    wp_tok = WordPieceTokenizer(vocab_size=150)
    wp_tok.build_vocab(SAMPLE_TEXTS)

    # 테스트 문장들
    test_sentences = [
        "The transformer architecture is revolutionary.",
        "Unbelievable performance improvements!",
        "Machine learning preprocessing pipeline.",
    ]

    print("토큰화 결과 비교:")
    print("-" * 80)

    for sentence in test_sentences:
        print(f"\n원본: {sentence}")
        print(f"Simple:    {simple_tok.tokenize(sentence)}")
        print(f"BPE:       {bpe_tok.tokenize(sentence)}")
        print(f"WordPiece: {wp_tok.tokenize(sentence)}")

    # 토큰 수 비교
    print("\n\n토큰 수 비교:")
    print("-" * 50)
    print("Sentence | Simple | BPE | WordPiece")
    print("-" * 50)

    for i, sentence in enumerate(test_sentences):
        simple_len = len(simple_tok.tokenize(sentence))
        bpe_len = len(bpe_tok.tokenize(sentence))
        wp_len = len(wp_tok.tokenize(sentence))
        print(f"   {i + 1}     |   {simple_len:3d}  | {bpe_len:3d} |    {wp_len:3d}")


def test_special_tokens():
    """특수 토큰 처리 테스트"""
    print("\n=== 특수 토큰 처리 테스트 ===\n")

    tokenizer = SimpleTokenizer(special_tokens=["[CUSTOM1]", "[CUSTOM2]"])
    tokenizer.build_vocab(SAMPLE_TEXTS)

    # 특수 토큰 확인
    print("1. 특수 토큰 목록")
    for token in tokenizer.special_tokens:
        token_id = tokenizer.vocab.get(token, -1)
        print(f"   {token}: {token_id}")

    # 인코딩 with/without 특수 토큰
    text = "Hello world"
    print(f"\n2. 인코딩 비교 (텍스트: '{text}')")

    with_special = tokenizer.encode(text, add_special_tokens=True)
    without_special = tokenizer.encode(text, add_special_tokens=False)

    print(f"   특수 토큰 포함: {with_special}")
    print(f"   특수 토큰 제외: {without_special}")

    # 디코딩 비교
    print("\n3. 디코딩 비교")
    decoded_with = tokenizer.decode(with_special, skip_special_tokens=False)
    decoded_without = tokenizer.decode(with_special, skip_special_tokens=True)

    print(f"   특수 토큰 포함: {decoded_with}")
    print(f"   특수 토큰 제외: {decoded_without}")


def test_save_load():
    """토크나이저 저장/로드 테스트"""
    print("\n=== 토크나이저 저장/로드 테스트 ===\n")

    # 원본 토크나이저 생성
    original = BPETokenizer(vocab_size=100)
    original.learn_bpe(SAMPLE_TEXTS)

    # 저장
    save_dir = "./test_tokenizer"
    original.save_pretrained(save_dir)
    print(f"1. 토크나이저 저장 완료: {save_dir}")

    # 저장된 파일 확인
    print("\n2. 저장된 파일:")
    for file in os.listdir(save_dir):
        print(f"   - {file}")

    # 로드
    loaded = BPETokenizer.from_pretrained(save_dir)
    print("\n3. 토크나이저 로드 완료")

    # 동일성 확인
    test_text = "Testing save and load functionality"
    original_tokens = original.tokenize(test_text)
    loaded_tokens = loaded.tokenize(test_text)

    print("\n4. 동일성 확인")
    print(f"   원본 토큰: {original_tokens}")
    print(f"   로드 토큰: {loaded_tokens}")
    print(f"   동일함: {original_tokens == loaded_tokens}")

    # 정리
    import shutil

    shutil.rmtree(save_dir)


def visualize_token_distribution():
    """토큰 분포 시각화"""
    print("\n=== 토큰 분포 시각화 ===\n")

    # 더 많은 텍스트로 토크나이저 학습
    extended_texts = SAMPLE_TEXTS * 10  # 반복하여 더 많은 데이터 생성

    tokenizers = {
        "Simple": SimpleTokenizer(),
        "BPE": BPETokenizer(vocab_size=100),
        "WordPiece": WordPieceTokenizer(vocab_size=100),
    }

    # 각 토크나이저 학습
    tokenizers["Simple"].build_vocab(extended_texts)
    tokenizers["BPE"].learn_bpe(extended_texts)
    tokenizers["WordPiece"].build_vocab(extended_texts)

    # 토큰 길이 분포 수집
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, tokenizer) in enumerate(tokenizers.items()):
        token_lengths = []

        # 각 텍스트 토큰화하고 토큰 길이 수집
        for text in SAMPLE_TEXTS:
            tokens = tokenizer.tokenize(text)
            token_lengths.extend([len(token) for token in tokens])

        # 히스토그램 그리기
        ax = axes[idx]
        ax.hist(token_lengths, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("토큰 길이 (문자 수)")
        ax.set_ylabel("빈도")
        ax.set_title(f"{name} Tokenizer 토큰 길이 분포")
        ax.grid(True, alpha=0.3)

        # 통계 정보 추가
        avg_len = sum(token_lengths) / len(token_lengths)
        ax.axvline(avg_len, color="red", linestyle="--", label=f"평균: {avg_len:.1f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/tokenizer_length_distribution.png", dpi=150)
    print("토큰 길이 분포가 'outputs/tokenizer_length_distribution.png'에 저장되었습니다.")


def benchmark_tokenizers():
    """토크나이저 성능 벤치마크"""
    print("\n=== 토크나이저 성능 벤치마크 ===\n")

    # 큰 텍스트 데이터 생성
    " ".join(SAMPLE_TEXTS * 100)

    tokenizers = {
        "Simple": SimpleTokenizer(),
        "BPE": BPETokenizer(vocab_size=1000),
        "WordPiece": WordPieceTokenizer(vocab_size=1000),
    }

    # 학습
    print("1. 토크나이저 학습 시간")
    for name, tokenizer in tokenizers.items():
        start = time.time()

        if name == "Simple":
            tokenizer.build_vocab(SAMPLE_TEXTS * 10)
        elif name == "BPE":
            tokenizer.learn_bpe(SAMPLE_TEXTS * 10)
        else:
            tokenizer.build_vocab(SAMPLE_TEXTS * 10)

        train_time = time.time() - start
        print(f"   {name}: {train_time:.3f}초")

    # 토큰화 속도
    print("\n2. 토큰화 속도 (1000회 반복)")
    for name, tokenizer in tokenizers.items():
        start = time.time()

        for _ in range(1000):
            tokens = tokenizer.tokenize("This is a test sentence for benchmarking.")

        tokenize_time = time.time() - start
        print(f"   {name}: {tokenize_time:.3f}초 ({tokenize_time / 1000 * 1000:.1f}ms/call)")

    # 압축률 비교
    print("\n3. 압축률 비교")
    test_text = " ".join(SAMPLE_TEXTS)
    original_chars = len(test_text)

    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(test_text)
        num_tokens = len(tokens)
        compression = original_chars / num_tokens
        print(f"   {name}: {num_tokens} 토큰 (압축률: {compression:.2f} chars/token)")


if __name__ == "__main__":
    # 1. Simple Tokenizer 테스트
    simple_tokenizer = test_simple_tokenizer()

    # 2. BPE Tokenizer 테스트
    bpe_tokenizer = test_bpe_tokenizer()

    # 3. WordPiece Tokenizer 테스트
    wp_tokenizer = test_wordpiece_tokenizer()

    # 4. 토크나이저 비교
    compare_tokenizers()

    # 5. 특수 토큰 처리
    test_special_tokens()

    # 6. 저장/로드 테스트
    test_save_load()

    # 7. 토큰 분포 시각화
    visualize_token_distribution()

    # 8. 성능 벤치마크
    benchmark_tokenizers()

    print("\n모든 테스트가 완료되었습니다!")
