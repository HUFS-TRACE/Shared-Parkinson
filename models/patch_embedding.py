"""모달 전용 입구 — 시계열을 패치로 잘라 공유 인코더가 먹을 토큰으로 만든다.

    필기 [B,  6, 2000] → (B*6,  39, d_model)
    음성 [B, 80,  200] → (B*80, 15, d_model)

채널을 배치 축에 접어 넣는 **channel-independent** 방식이라, 채널 수가 다른 모달이
같은 인코더를 통과할 수 있다. 인코더가 보는 것은 "토큰 수열"뿐이고 그 수열이 몇 개
묶여 왔는지는 무관하다. 토큰 개수가 39 vs 15로 달라도 어텐션은 길이에 무관하다.
"""
import torch.nn as nn

from .embedding import PositionalEncoding, RevIN


class PatchEmbedding(nn.Module):
    """(B, C, T) → (B*C, num_patches, d_model)

    Args:
        num_channels: 필기 6(펜 센서) · 음성 80(멜 밴드)
        seq_len:      필기 2000(2초 × 1000Hz) · 음성 200(2초 × 100프레임)
        patch_len:    패치 길이. stride와 함께 토큰 수를 정한다
        stride:       패치 간격. patch_len의 절반이면 50% 겹침
    """

    def __init__(self, num_channels, seq_len, patch_len, stride, d_model, dropout=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        if self.num_patches < 1:
            raise ValueError(
                f"seq_len({seq_len})이 너무 짧습니다. "
                f"patch_len({patch_len})·stride({stride})를 확인하세요."
            )

        self.revin = RevIN(num_channels)
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, _ = x.shape
        x = self.revin(x)                                    # (B, C, T)
        x = x.unfold(dimension=-1,
                     size=self.patch_len,
                     step=self.stride)                       # (B, C, P, patch_len)
        x = x.reshape(b * c, self.num_patches, self.patch_len)
        x = self.proj(x)                                     # (B*C, P, d_model)
        x = self.pos_enc(x)
        return self.dropout(x)
