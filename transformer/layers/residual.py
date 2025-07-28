"""
Residual Connection 구현

Transformer에서 각 sub-layer에 적용되는 residual connection을 구현합니다.
Layer normalization과 함께 사용되어 깊은 네트워크의 학습을 안정화합니다.
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from .normalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization

    각 sub-layer 주위에 residual connection을 추가합니다.
    원논문은 Post-Norm을 사용하지만, 많은 최신 구현은 Pre-Norm을 사용합니다.

    Post-Norm: x + Dropout(SubLayer(LayerNorm(x)))
    Pre-Norm: x + Dropout(SubLayer(LayerNorm(x)))

    Args:
        size: 입력의 feature dimension
        dropout: Dropout 비율
        norm_first: True면 Pre-Norm, False면 Post-Norm
    """

    def __init__(self, size: int, dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        self.size = size

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Residual connection 적용

        Args:
            x: 입력 텐서
            sublayer: 적용할 sub-layer 함수

        Returns:
            Residual connection이 적용된 출력
        """
        if self.norm_first:
            # Pre-Norm: LayerNorm -> SubLayer -> Dropout -> Add
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-Norm: SubLayer -> Dropout -> Add -> LayerNorm
            return self.norm(x + self.dropout(sublayer(x)))


class PreNormResidualConnection(nn.Module):
    """
    Pre-Normalization Residual Connection

    많은 최신 Transformer 구현에서 사용하는 방식입니다.
    학습이 더 안정적이고 수렴이 빠른 것으로 알려져 있습니다.

    Args:
        size: Feature dimension
        dropout: Dropout 비율
    """

    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        Pre-Norm residual connection

        Args:
            x: 입력 텐서
            sublayer: 적용할 sub-layer 모듈
            *args, **kwargs: sublayer에 전달할 추가 인자

        Returns:
            Residual connection이 적용된 출력
        """
        # Normalize first
        normalized = self.norm(x)

        # Apply sublayer
        output = sublayer(normalized, *args, **kwargs)

        # Apply dropout and residual connection
        return x + self.dropout(output)


class PostNormResidualConnection(nn.Module):
    """
    Post-Normalization Residual Connection

    원 Transformer 논문의 방식입니다.

    Args:
        size: Feature dimension
        dropout: Dropout 비율
    """

    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        Post-Norm residual connection

        Args:
            x: 입력 텐서
            sublayer: 적용할 sub-layer 모듈
            *args, **kwargs: sublayer에 전달할 추가 인자

        Returns:
            Residual connection이 적용된 출력
        """
        # Apply sublayer
        output = sublayer(x, *args, **kwargs)

        # Apply dropout and residual connection
        residual = x + self.dropout(output)

        # Normalize after
        return self.norm(residual)


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath)

    학습 시 일정 확률로 residual branch를 건너뜁니다.
    Vision Transformer 등에서 사용되는 정규화 기법입니다.

    Args:
        drop_prob: Drop probability
        scale_by_keep: True면 keep probability로 스케일링
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic depth 적용"""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        # Binary mask 생성
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        if self.scale_by_keep:
            x = x / keep_prob

        return x * random_tensor


class ResidualConnectionWithStochasticDepth(nn.Module):
    """
    Stochastic Depth를 포함한 Residual Connection

    Args:
        size: Feature dimension
        dropout: Dropout 비율
        drop_path: Stochastic depth 비율
        norm_first: Pre-norm vs Post-norm
    """

    def __init__(
        self, size: int, dropout: float = 0.1, drop_path: float = 0.0, norm_first: bool = True
    ):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """Residual connection with stochastic depth"""
        if self.norm_first:
            # Pre-norm
            output = sublayer(self.norm(x), *args, **kwargs)
            output = self.dropout(output)
            output = self.drop_path(output)
            return x + output
        else:
            # Post-norm
            output = sublayer(x, *args, **kwargs)
            output = self.dropout(output)
            output = self.drop_path(output)
            return self.norm(x + output)


class ScaleResidualConnection(nn.Module):
    """
    Scaled Residual Connection

    일부 모델에서는 residual branch에 스케일 factor를 적용합니다.

    Args:
        size: Feature dimension
        dropout: Dropout 비율
        scale: Residual branch 스케일 factor
        norm_first: Pre-norm vs Post-norm
    """

    def __init__(
        self, size: int, dropout: float = 0.1, scale: float = 1.0, norm_first: bool = True
    ):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, sublayer: nn.Module, *args, **kwargs) -> torch.Tensor:
        """Scaled residual connection"""
        if self.norm_first:
            # Pre-norm with scaling
            output = sublayer(self.norm(x), *args, **kwargs)
            return x + self.scale * self.dropout(output)
        else:
            # Post-norm with scaling
            output = sublayer(x, *args, **kwargs)
            return self.norm(x + self.scale * self.dropout(output))


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Residual Connection 테스트 ===\n")

    # 더미 sub-layer (선형 변환)
    class DummySubLayer(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.linear = nn.Linear(size, size)

        def forward(self, x):
            return self.linear(x)

    # 파라미터
    batch_size = 2
    seq_length = 10
    d_model = 64

    # 입력
    x = torch.randn(batch_size, seq_length, d_model)
    sublayer = DummySubLayer(d_model)

    # 다양한 Residual Connection 테스트
    print("1. Basic Residual Connection (Pre-Norm):")
    res_conn = ResidualConnection(d_model, norm_first=True)
    output = res_conn(x, lambda x: sublayer(x))
    print(f"   입력 shape: {x.shape}")
    print(f"   출력 shape: {output.shape}")
    print(f"   입력 norm: {x.norm():.4f}")
    print(f"   출력 norm: {output.norm():.4f}")

    print("\n2. Post-Norm Residual Connection:")
    post_norm = PostNormResidualConnection(d_model)
    output = post_norm(x, sublayer)
    print(f"   출력 norm: {output.norm():.4f}")

    print("\n3. Residual Connection with Stochastic Depth:")
    stochastic_res = ResidualConnectionWithStochasticDepth(d_model, drop_path=0.1)
    stochastic_res.train()  # Training mode에서만 drop이 발생
    output = stochastic_res(x, sublayer)
    print(f"   출력 norm: {output.norm():.4f}")

    print("\n4. Scaled Residual Connection:")
    scaled_res = ScaleResidualConnection(d_model, scale=0.5)
    output = scaled_res(x, sublayer)
    print(f"   출력 norm (scale=0.5): {output.norm():.4f}")
