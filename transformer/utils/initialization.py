"""
가중치 초기화 유틸리티

Transformer 모델의 가중치를 효과적으로 초기화하는 다양한 방법들을 제공합니다.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Union, Callable


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier uniform 초기화
    
    Args:
        tensor: 초기화할 텐서
        gain: 스케일링 factor
        
    Returns:
        초기화된 텐서
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier normal 초기화
    
    Args:
        tensor: 초기화할 텐서
        gain: 스케일링 factor
        
    Returns:
        초기화된 텐서
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def he_uniform_(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', 
                nonlinearity: str = 'relu') -> torch.Tensor:
    """He (Kaiming) uniform 초기화
    
    Args:
        tensor: 초기화할 텐서
        a: negative slope (LeakyReLU용)
        mode: 'fan_in' 또는 'fan_out'
        nonlinearity: 활성화 함수 종류
        
    Returns:
        초기화된 텐서
    """
    fan = _calculate_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


def he_normal_(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in',
               nonlinearity: str = 'relu') -> torch.Tensor:
    """He (Kaiming) normal 초기화
    
    Args:
        tensor: 초기화할 텐서
        a: negative slope (LeakyReLU용)
        mode: 'fan_in' 또는 'fan_out'
        nonlinearity: 활성화 함수 종류
        
    Returns:
        초기화된 텐서
    """
    fan = _calculate_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """정규 분포 초기화
    
    Args:
        tensor: 초기화할 텐서
        mean: 평균
        std: 표준편차
        
    Returns:
        초기화된 텐서
    """
    with torch.no_grad():
        tensor.normal_(mean, std)
    return tensor


def uniform_(tensor: torch.Tensor, a: float = 0.0, b: float = 1.0) -> torch.Tensor:
    """균등 분포 초기화
    
    Args:
        tensor: 초기화할 텐서
        a: 최소값
        b: 최대값
        
    Returns:
        초기화된 텐서
    """
    with torch.no_grad():
        tensor.uniform_(a, b)
    return tensor


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                     a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal 초기화
    
    정규 분포에서 [mean - a*std, mean + b*std] 범위를 벗어난 값들을 재샘플링
    
    Args:
        tensor: 초기화할 텐서
        mean: 평균
        std: 표준편차
        a: 하한 (표준편차 단위)
        b: 상한 (표준편차 단위)
        
    Returns:
        초기화된 텐서
    """
    with torch.no_grad():
        # 간단한 방법: normal 분포 후 clamp
        tensor.normal_(mean, std)
        lower = mean - abs(a) * std
        upper = mean + abs(b) * std
        tensor.clamp_(lower, upper)
    
    return tensor


def init_bert_params(module: nn.Module):
    """BERT 스타일 파라미터 초기화
    
    - Linear layers: normal(0, 0.02)
    - Embedding layers: normal(0, 0.02)
    - LayerNorm: bias=0, weight=1
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def init_gpt2_params(module: nn.Module, n_layers: int):
    """GPT-2 스타일 파라미터 초기화
    
    - Residual 레이어들은 1/sqrt(n_layers)로 스케일링
    - 나머지는 normal(0, 0.02)
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    
    # Residual connections을 위한 스케일링
    if hasattr(module, 'is_residual') and module.is_residual:
        with torch.no_grad():
            module.weight.data /= math.sqrt(n_layers)


def init_transformer_params(module: nn.Module, d_model: int, 
                          init_type: str = 'xavier_uniform'):
    """Transformer 논문 스타일 초기화
    
    Args:
        module: 초기화할 모듈
        d_model: 모델 차원
        init_type: 초기화 방법 ('xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal')
    """
    if isinstance(module, nn.Linear):
        if init_type == 'xavier_uniform':
            xavier_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            xavier_normal_(module.weight)
        elif init_type == 'he_uniform':
            he_uniform_(module.weight)
        elif init_type == 'he_normal':
            he_normal_(module.weight)
        else:
            # 기본값: Transformer 논문의 방법
            module.weight.data.uniform_(-1/math.sqrt(d_model), 1/math.sqrt(d_model))
        
        if module.bias is not None:
            module.bias.data.zero_()
    
    elif isinstance(module, nn.Embedding):
        normal_(module.weight, mean=0, std=d_model**-0.5)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple:
    """텐서의 fan_in과 fan_out 계산"""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


def _calculate_fan(tensor: torch.Tensor, mode: str) -> int:
    """텐서의 fan 계산"""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        return fan_in
    elif mode == 'fan_out':
        return fan_out
    elif mode == 'fan_avg':
        return (fan_in + fan_out) // 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """활성화 함수에 따른 gain 계산"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
    
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        else:
            negative_slope = param
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


class TransformerInitializer:
    """Transformer 모델을 위한 통합 초기화 클래스"""
    
    def __init__(self, d_model: int, n_layers: int, init_type: str = 'xavier_uniform'):
        """
        Args:
            d_model: 모델 차원
            n_layers: 레이어 수
            init_type: 초기화 타입
        """
        self.d_model = d_model
        self.n_layers = n_layers
        self.init_type = init_type
    
    def initialize(self, model: nn.Module):
        """모델 전체 초기화"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                self._init_linear(module)
            elif isinstance(module, nn.Embedding):
                self._init_embedding(module)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                self._init_norm(module)
        
        model.apply(_init_weights)
    
    def _init_linear(self, module: nn.Linear):
        """Linear 레이어 초기화"""
        if self.init_type == 'xavier_uniform':
            xavier_uniform_(module.weight)
        elif self.init_type == 'xavier_normal':
            xavier_normal_(module.weight)
        elif self.init_type == 'he_uniform':
            he_uniform_(module.weight)
        elif self.init_type == 'he_normal':
            he_normal_(module.weight)
        elif self.init_type == 'bert':
            normal_(module.weight, std=0.02)
        elif self.init_type == 'gpt2':
            normal_(module.weight, std=0.02)
            # Output projection은 추가 스케일링
            if hasattr(module, 'is_output_projection') and module.is_output_projection:
                with torch.no_grad():
                    module.weight /= math.sqrt(2 * self.n_layers)
        else:
            # 기본: Transformer 논문
            bound = 1 / math.sqrt(self.d_model)
            uniform_(module.weight, -bound, bound)
        
        if module.bias is not None:
            module.bias.data.zero_()
    
    def _init_embedding(self, module: nn.Embedding):
        """Embedding 레이어 초기화"""
        if self.init_type in ['bert', 'gpt2']:
            normal_(module.weight, std=0.02)
        else:
            normal_(module.weight, std=self.d_model**-0.5)
        
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    def _init_norm(self, module: Union[nn.LayerNorm, nn.GroupNorm]):
        """Normalization 레이어 초기화"""
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)