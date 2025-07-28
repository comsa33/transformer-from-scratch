"""
Tokenizer 구현

BPE (Byte Pair Encoding) 기반 토크나이저와 간단한 word-level 토크나이저를 구현합니다.
"""

import json
import os
import re
import unicodedata
from collections import Counter


class BaseTokenizer:
    """토크나이저 기본 클래스"""

    def __init__(
        self,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        mask_token: str = "<mask>",
        special_tokens: list[str] | None = None,
    ):
        """
        Args:
            unk_token: Unknown token
            pad_token: Padding token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            mask_token: Mask token (for MLM)
            special_tokens: 추가 특수 토큰들
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        # 특수 토큰 목록
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token, mask_token]
        if special_tokens:
            self.special_tokens.extend(special_tokens)

        # 어휘 사전
        self.vocab = {}
        self.inverse_vocab = {}

        # 특수 토큰 ID
        self.pad_token_id = None
        self.unk_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.mask_token_id = None

    def tokenize(self, text: str) -> list[str]:
        """텍스트를 토큰으로 분할"""
        raise NotImplementedError

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        """텍스트를 토큰 ID로 변환"""
        tokens = self.tokenize(text)

        # 특수 토큰 추가
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]

        # Truncation
        if truncation and max_length:
            tokens = tokens[:max_length]

        # 토큰을 ID로 변환
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]

        # Padding
        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """토큰 ID를 텍스트로 변환"""
        tokens = [self.inverse_vocab.get(id, self.unk_token) for id in ids]

        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]

        return self._tokens_to_text(tokens)

    def _tokens_to_text(self, tokens: list[str]) -> str:
        """토큰들을 텍스트로 결합"""
        return " ".join(tokens)

    def batch_encode(self, texts: list[str], **kwargs) -> dict[str, list[list[int]]]:
        """배치 인코딩"""
        encodings = [self.encode(text, **kwargs) for text in texts]

        # 패딩을 위한 최대 길이 찾기
        if kwargs.get("padding") and not kwargs.get("max_length"):
            max_len = max(len(enc) for enc in encodings)
            # 다시 인코딩 (패딩 적용)
            kwargs["max_length"] = max_len
            encodings = [self.encode(text, **kwargs) for text in texts]

        return {"input_ids": encodings}

    def batch_decode(self, batch_ids: list[list[int]], **kwargs) -> list[str]:
        """배치 디코딩"""
        return [self.decode(ids, **kwargs) for ids in batch_ids]

    def save_pretrained(self, save_directory: str):
        """토크나이저 저장"""
        os.makedirs(save_directory, exist_ok=True)

        # 어휘 저장
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # 설정 저장
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": self.special_tokens,
            "tokenizer_class": self.__class__.__name__,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """저장된 토크나이저 로드"""
        # 설정 로드
        config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        # 토크나이저 생성
        tokenizer = cls(
            unk_token=config["unk_token"],
            pad_token=config["pad_token"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            mask_token=config["mask_token"],
        )

        # 어휘 로드
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        with open(vocab_file, encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)

        tokenizer._build_inverse_vocab()
        return tokenizer

    def _build_inverse_vocab(self):
        """역 어휘 사전 구축"""
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # 특수 토큰 ID 설정
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)
        self.mask_token_id = self.vocab.get(self.mask_token, 4)

    @property
    def vocab_size(self) -> int:
        """어휘 크기"""
        return len(self.vocab)


class SimpleTokenizer(BaseTokenizer):
    """간단한 공백 기반 토크나이저"""

    def __init__(self, lowercase: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.lowercase = lowercase

        # 기본 전처리 패턴
        self.punct_pattern = re.compile(r"([.,!?;:])")

    def tokenize(self, text: str) -> list[str]:
        """텍스트를 토큰으로 분할"""
        if self.lowercase:
            text = text.lower()

        # 구두점 분리
        text = self.punct_pattern.sub(r" \1 ", text)

        # 공백으로 분할
        tokens = text.split()

        return tokens

    def build_vocab(self, texts: list[str], min_freq: int = 1, max_vocab_size: int | None = None):
        """텍스트로부터 어휘 구축"""
        # 토큰 빈도 계산
        token_freqs = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_freqs.update(tokens)

        # 특수 토큰 추가
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # 빈도순으로 정렬
        sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)

        # 어휘 구축
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_vocab_size and len(vocab) >= max_vocab_size:
                break
            if token not in vocab:
                vocab[token] = len(vocab)

        self.vocab = vocab
        self._build_inverse_vocab()


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding (BPE) 토크나이저"""

    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size_target = vocab_size
        self.min_frequency = min_frequency

        # BPE 관련
        self.merges = []  # BPE merge 규칙
        self.word_tokenizer = re.compile(r"\w+|[^\w\s]+")

    def _get_word_frequency(self, texts: list[str]) -> Counter:
        """단어 빈도 계산"""
        word_freq = Counter()
        for text in texts:
            words = self.word_tokenizer.findall(text.lower())
            word_freq.update(words)
        return word_freq

    def _get_pair_stats(self, vocab: dict[str, int]) -> Counter:
        """문자 쌍 통계 계산"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_pair(self, pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
        """가장 빈번한 쌍을 병합"""
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    def learn_bpe(self, texts: list[str]):
        """BPE 학습"""
        # 단어 빈도 계산
        word_freq = self._get_word_frequency(texts)

        # 초기 어휘 (문자 단위로 분할)
        vocab = {}
        for word, freq in word_freq.items():
            word_tokens = " ".join(list(word)) + " </w>"
            vocab[word_tokens] = freq

        # 특수 토큰으로 시작
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # 개별 문자 추가
        for word in vocab:
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        # BPE 병합
        num_merges = self.vocab_size_target - len(self.vocab)
        for _ in range(num_merges):
            pairs = self._get_pair_stats(vocab)
            if not pairs:
                break

            # 최소 빈도 체크
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break

            # 병합 수행
            vocab = self._merge_pair(best_pair, vocab)
            self.merges.append(best_pair)

            # 새 토큰 추가
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if len(self.vocab) >= self.vocab_size_target:
                break

        self._build_inverse_vocab()

    def tokenize(self, text: str) -> list[str]:
        """BPE를 사용한 토큰화"""
        tokens = []
        words = self.word_tokenizer.findall(text.lower())

        for word in words:
            # 단어를 문자로 분할
            word_tokens = list(word) + ["</w>"]

            # BPE 병합 적용
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge:
                        word_tokens = word_tokens[:i] + ["".join(merge)] + word_tokens[i + 2 :]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens

    def save_pretrained(self, save_directory: str):
        """BPE 토크나이저 저장"""
        super().save_pretrained(save_directory)

        # BPE merges 저장
        merges_file = os.path.join(save_directory, "merges.txt")
        with open(merges_file, "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """저장된 BPE 토크나이저 로드"""
        tokenizer = super().from_pretrained(pretrained_model_name_or_path)

        # BPE merges 로드
        merges_file = os.path.join(pretrained_model_name_or_path, "merges.txt")
        if os.path.exists(merges_file):
            tokenizer.merges = []
            with open(merges_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        tokenizer.merges.append(tuple(parts))

        return tokenizer


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece 토크나이저 (BERT 스타일)"""

    def __init__(
        self,
        vocab_size: int = 30000,
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        **kwargs,
    ):
        # BERT 스타일 특수 토큰
        kwargs["unk_token"] = unk_token
        kwargs["pad_token"] = kwargs.get("pad_token", "[PAD]")
        kwargs["bos_token"] = kwargs.get("bos_token", "[CLS]")
        kwargs["eos_token"] = kwargs.get("eos_token", "[SEP]")
        kwargs["mask_token"] = kwargs.get("mask_token", "[MASK]")

        super().__init__(**kwargs)
        self.vocab_size_target = vocab_size
        self.max_input_chars_per_word = max_input_chars_per_word
        self.never_split = set(self.special_tokens)

    def _tokenize_chinese_chars(self, text: str) -> str:
        """중국어 문자 주변에 공백 추가"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp: int) -> bool:
        """중국어 문자인지 확인"""
        return (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        )

    def _run_strip_accents(self, text: str) -> str:
        """악센트 제거"""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_control(self, char: str) -> bool:
        """제어 문자인지 확인"""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        return cat.startswith("C")

    def _is_whitespace(self, char: str) -> bool:
        """공백 문자인지 확인"""
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        return cat == "Zs"

    def _is_punctuation(self, char: str) -> bool:
        """구두점인지 확인"""
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        return cat.startswith("P")

    def basic_tokenizer(self, text: str) -> list[str]:
        """기본 토큰화 (단어 단위)"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = text.strip().split()
        split_tokens = []

        for token in orig_tokens:
            if token in self.never_split:
                split_tokens.append(token)
            else:
                token = token.lower()
                token = self._run_strip_accents(token)
                split_tokens.extend(self._run_split_on_punc(token))

        return split_tokens

    def _run_split_on_punc(self, text: str) -> list[str]:
        """구두점 기준 분할"""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []

        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def wordpiece_tokenizer(self, text: str) -> list[str]:
        """WordPiece 토큰화"""
        output_tokens = []

        for token in self.basic_tokenizer(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None

                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def tokenize(self, text: str) -> list[str]:
        """텍스트를 WordPiece 토큰으로 변환"""
        return self.wordpiece_tokenizer(text)

    def build_vocab(self, texts: list[str], min_freq: int = 2):
        """WordPiece 어휘 구축"""
        # 기본 토큰 빈도 계산
        word_freq = Counter()
        for text in texts:
            words = self.basic_tokenizer(text)
            word_freq.update(words)

        # 특수 토큰으로 시작
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}

        # 문자 수준 토큰 추가
        char_freq = Counter()
        for word, freq in word_freq.items():
            if freq >= min_freq:
                for char in word:
                    char_freq[char] += freq

        # 빈도순으로 문자 추가
        for char, _freq in char_freq.most_common():
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
            if len(self.vocab) >= self.vocab_size_target // 2:
                break

        # WordPiece 토큰 추가
        for word, freq in word_freq.most_common():
            if freq < min_freq:
                break

            # 단어를 subword로 분할
            for i in range(1, len(word)):
                for j in range(len(word) - i + 1):
                    subword = word[j : j + i]
                    if j > 0:
                        subword = "##" + subword

                    if subword not in self.vocab and len(self.vocab) < self.vocab_size_target:
                        self.vocab[subword] = len(self.vocab)

        self._build_inverse_vocab()


def create_tokenizer(tokenizer_type: str = "simple", **kwargs) -> BaseTokenizer:
    """토크나이저 생성 헬퍼 함수

    Args:
        tokenizer_type: 'simple', 'bpe', 'wordpiece' 중 선택
        **kwargs: 토크나이저별 추가 인자

    Returns:
        토크나이저 인스턴스
    """
    if tokenizer_type.lower() == "simple":
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type.lower() == "bpe":
        return BPETokenizer(**kwargs)
    elif tokenizer_type.lower() == "wordpiece":
        return WordPieceTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
