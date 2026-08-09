"""모달 전용 출구 — 인코더 표현을 정상/환자 로짓으로.

채널 수가 모달마다 다르므로(필기 6 · 음성 80) 입력 차원 `C*d_model`도 다르다.
분류 헤드는 어차피 모달별로 따로 둘 수밖에 없다.
"""
import torch.nn as nn


class ModalityClassifier(nn.Module):
    """(B, C, d_model) → (B, num_classes)

    패치 축은 SharedPatchTST에서 평균 풀링으로 이미 접혔고, 여기서는 채널을
    이어붙여 한 번에 분류한다. 채널별로 따로 판정한 뒤 투표하지 않는 이유는,
    채널 간 상관(예: 펜 압력과 기울기가 함께 흔들림)이 진단 신호이기 때문이다.
    """

    def __init__(self, num_channels, d_model, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                                    # (B, C*d_model)
            nn.Dropout(dropout),
            nn.Linear(num_channels * d_model, num_classes),
        )

    def forward(self, z):
        return self.net(z)
