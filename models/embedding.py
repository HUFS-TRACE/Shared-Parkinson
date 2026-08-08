"""정규화·위치 인코딩 — 모달을 가리지 않는 기본 블록.

두 모달 모두 여기를 거쳐 "평균 0·분산 1의 무단위 값"이 되므로, 뒤쪽 공유 인코더는
입력이 펜 압력인지 멜 밴드인지 알 방법이 없다.
"""
import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """윈도우 자체 통계로 채널별 정규화 (Reversible Instance Normalization).

    train fold에서 통계를 fit해 test에 적용하는 방식과 달리, 윈도우마다 자기
    통계를 쓰므로 **구조적으로 누수가 없다.** 모달마다 단위가 달라도(헤르츠 vs
    필압) 여기를 지나면 무단위가 되어 같은 인코더에 넣을 수 있다.
    """

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # 처음엔 항등 변환이 되도록 초기화
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class PositionalEncoding(nn.Module):
    """sin/cos 고정 위치 인코딩 (학습 파라미터 없음).

    학습형 위치 임베딩 대신 고정형을 쓰는 이유는 두 가지다.
      · 모달마다 토큰 수가 다른데(필기 39 · 음성 15) 고정형은 길이에 무관하다
      · 파라미터가 0이라 "인코더를 공유한다"는 비율이 희석되지 않는다
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]
