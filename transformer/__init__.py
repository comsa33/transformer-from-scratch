"""
Transformer 모델 구현 패키지

이 패키지는 "Attention is All You Need" 논문의 Transformer 아키텍처를 구현합니다.
"""

from .config import TransformerConfig

# 아직 구현되지 않은 모듈들은 주석 처리
# from .models.transformer import Transformer
# from .models.encoder import TransformerEncoder
# from .models.decoder import TransformerDecoder

__version__ = "0.1.0"
__all__ = [
    "TransformerConfig",
    # "Transformer",
    # "TransformerEncoder",
    # "TransformerDecoder",
]
