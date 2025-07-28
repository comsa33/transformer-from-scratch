"""
Positional Encoding 구현

Transformer는 RNN과 달리 순서 정보를 직접적으로 처리하지 않으므로,
입력 시퀀스의 위치 정보를 명시적으로 추가해야 합니다.

"Attention is All You Need" 논문에서는 sinusoidal 함수를 사용한
positional encoding을 제안했습니다.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    각 위치와 차원에 대해 고유한 sinusoidal 패턴을 생성합니다.
    이를 통해 모델이 토큰의 상대적/절대적 위치를 학습할 수 있습니다.
    
    수식:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    여기서:
        - pos: 시퀀스 내 위치 (0부터 시작)
        - i: 차원 인덱스
        - d_model: 모델의 차원
    
    Args:
        d_model: 모델의 차원 (임베딩 차원과 동일해야 함)
        max_seq_length: 최대 시퀀스 길이
        dropout: Dropout 비율 (default: 0.1)
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Positional encoding을 미리 계산하여 저장
        # shape: [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # 각 위치에 대한 인덱스 생성 [0, 1, 2, ..., max_seq_length-1]
        # shape: [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 주파수 항 계산
        # 논문의 수식: 1 / 10000^(2i/d_model)
        # = exp(log(1/10000) * 2i/d_model)
        # = exp(-log(10000) * 2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Sinusoidal 패턴 적용
        # 짝수 인덱스: sin 함수 사용
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 홀수 인덱스: cos 함수 사용
        # d_model이 홀수인 경우를 처리하기 위해 슬라이싱 조정
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # [max_seq_length, d_model] -> [1, max_seq_length, d_model]
        # 배치 차원 추가
        pe = pe.unsqueeze(0)
        
        # Buffer로 등록 (학습되지 않는 파라미터)
        # 모델 저장/로드 시 함께 저장되지만 gradient는 계산되지 않음
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서에 positional encoding을 더합니다.
        
        Args:
            x: 입력 텐서 [batch_size, seq_length, d_model]
            
        Returns:
            Positional encoding이 더해진 텐서 [batch_size, seq_length, d_model]
        """
        # 입력 시퀀스 길이만큼의 positional encoding을 가져옴
        # x.size(1)은 시퀀스 길이
        pe = self.pe[:, :x.size(1), :]
        
        # 입력에 positional encoding 더하기
        # Broadcasting이 자동으로 처리됨
        x = x + pe
        
        # Dropout 적용
        return self.dropout(x)
    
    def get_encoding(self, seq_length: int) -> torch.Tensor:
        """
        특정 길이의 positional encoding을 반환합니다.
        디버깅이나 시각화에 유용합니다.
        
        Args:
            seq_length: 시퀀스 길이
            
        Returns:
            Positional encoding 텐서 [1, seq_length, d_model]
        """
        return self.pe[:, :seq_length, :]


class LearnablePositionalEncoding(nn.Module):
    """
    학습 가능한 Positional Encoding (선택적 구현)
    
    일부 모델에서는 sinusoidal 대신 학습 가능한 positional encoding을 사용합니다.
    
    Args:
        d_model: 모델의 차원
        max_seq_length: 최대 시퀀스 길이
        dropout: Dropout 비율
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 학습 가능한 positional embedding
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서에 학습 가능한 positional encoding을 더합니다.
        
        Args:
            x: 입력 텐서 [batch_size, seq_length, d_model]
            
        Returns:
            Positional encoding이 더해진 텐서
        """
        # 입력 시퀀스 길이만큼의 positional encoding을 가져옴
        pe = self.pe[:, :x.size(1), :]
        
        # 입력에 positional encoding 더하기
        x = x + pe
        
        return self.dropout(x)


def create_sinusoidal_positions(n_pos: int, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding을 생성하는 독립 함수
    
    Args:
        n_pos: 위치의 개수
        dim: 차원
        
    Returns:
        Positional encoding 텐서 [n_pos, dim]
    """
    position = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
    dim_t = torch.arange(dim, dtype=torch.float).unsqueeze(0)
    div_term = 1 / (10000 ** (2 * (dim_t // 2) / dim))
    
    pos_embedding = torch.zeros(n_pos, dim)
    pos_embedding[:, 0::2] = torch.sin(position * div_term[:, 0::2])
    pos_embedding[:, 1::2] = torch.cos(position * div_term[:, 1::2])
    
    return pos_embedding


if __name__ == "__main__":
    # 테스트 코드
    import matplotlib.pyplot as plt
    
    # Positional Encoding 생성
    d_model = 512
    max_len = 100
    
    pe = PositionalEncoding(d_model=d_model, max_seq_length=max_len, dropout=0.0)
    
    # 더미 입력 생성
    dummy_input = torch.zeros(1, max_len, d_model)
    
    # Positional encoding 적용
    output = pe(dummy_input)
    
    # 처음 몇 개 차원의 패턴 시각화
    encoding = pe.get_encoding(max_len).squeeze(0).numpy()
    
    plt.figure(figsize=(15, 5))
    
    # 처음 4개 차원의 sinusoidal 패턴 표시
    plt.subplot(1, 2, 1)
    for i in range(4):
        plt.plot(encoding[:, i], label=f'dim {i}')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Positional Encoding - First 4 Dimensions')
    plt.legend()
    
    # 전체 패턴 히트맵
    plt.subplot(1, 2, 2)
    plt.imshow(encoding[:50, :128].T, aspect='auto', cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png')
    print("Positional encoding visualization saved to 'positional_encoding_visualization.png'")
    
    # Shape 확인
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Positional encoding shape: {encoding.shape}")