"""
Transformer 모델 설정 클래스
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 모델의 설정을 관리하는 클래스

    Attributes:
        d_model: 모델의 hidden dimension (기본값: 512)
        num_heads: Multi-head attention의 head 개수 (기본값: 8)
        num_encoder_layers: Encoder 레이어 개수 (기본값: 6)
        num_decoder_layers: Decoder 레이어 개수 (기본값: 6)
        d_ff: Feed-forward network의 hidden dimension (기본값: 2048)
        max_seq_length: 최대 시퀀스 길이 (기본값: 512)
        vocab_size: 어휘 크기 (필수)
        pad_token_id: Padding 토큰 ID (기본값: 0)
        dropout_rate: 일반 dropout 비율 (기본값: 0.1)
        attention_dropout_rate: Attention dropout 비율 (기본값: 0.1)
        activation: FFN 활성화 함수 (기본값: 'relu')
        layer_norm_eps: Layer normalization epsilon (기본값: 1e-6)
        initializer_range: 가중치 초기화 범위 (기본값: 0.02)
        use_bias: Linear 레이어에 bias 사용 여부 (기본값: True)
        share_embeddings: Encoder/Decoder 임베딩 공유 여부 (기본값: False)
    """

    # 모델 차원
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048

    # 시퀀스 관련
    max_seq_length: int = 512
    vocab_size: int | None = None
    pad_token_id: int = 0

    # Dropout
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # 기타 설정
    activation: str = "relu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_bias: bool = True
    share_embeddings: bool = False

    def __post_init__(self):
        """설정 값 검증"""
        if self.vocab_size is None:
            raise ValueError("vocab_size는 반드시 지정되어야 합니다.")

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model})은 num_heads ({self.num_heads})로 나누어떨어져야 합니다."
            )

        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(f"dropout_rate는 0과 1 사이여야 합니다. 현재 값: {self.dropout_rate}")

        if self.attention_dropout_rate < 0 or self.attention_dropout_rate > 1:
            raise ValueError(
                f"attention_dropout_rate는 0과 1 사이여야 합니다. "
                f"현재 값: {self.attention_dropout_rate}"
            )

    @property
    def d_k(self) -> int:
        """각 attention head의 dimension"""
        return self.d_model // self.num_heads

    @property
    def d_v(self) -> int:
        """각 attention head의 value dimension"""
        return self.d_model // self.num_heads
