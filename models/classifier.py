"""모달 전용 출구 — 인코더 표현을 정상/환자 로짓으로.

채널 수가 모달마다 다르므로(필기 6 · 음성 80) 입력 차원 `C*d_model`도 다르다.
분류 헤드는 어차피 모달별로 따로 둘 수밖에 없다.

왜 헤드가 세 종류인가
    PatchEmbedding이 채널을 배치 축에 접어 넣으므로, 인코더는 채널을 서로 무관한
    시계열로 본다. 채널이 다시 만나는 곳은 이 분류기 하나뿐이다.
    "그 하나로 채널 간 상관을 감당할 수 있는가"가 회의에서 나온 물음이고,
    답하려면 서로 다른 헤드를 같은 조건에서 맞대야 한다.

        linear  Σ_c W_c·z_c                 채널 상호작용 없음 · 용량 낮음  (기준)
        perch   Σ_c MLP(z_c)  (가중치 공유)  채널 상호작용 없음 · 용량 높음
        mlp     MLP(concat_c z_c)           채널 상호작용 있음 · 용량 높음

    linear는 z에 대해 1차식이라 z_c·z_c' 같은 곱셈 항이 없다. 채널을 더할 뿐이다.
    mlp는 은닉층에서 채널이 섞이므로 상호작용을 표현할 수 있다.

    perch가 왜 필요한가
        mlp가 이겼을 때 그것이 "채널이 섞여서"인지 "그냥 파라미터가 많아서"인지
        가려야 한다. perch는 용량은 늘리되 채널은 끝까지 따로 두는 대조군이다.

            mlp > perch > linear   상호작용도 용량도 기여한다
            mlp ≈ perch > linear   용량 때문이지 상호작용 때문이 아니다
            셋이 비슷           →  linear 하나로 충분하다

    perch의 MLP는 채널끼리 가중치를 공유한다. 채널마다 따로 두면 음성(80채널)에서
    파라미터가 인코더보다 커져 비교가 용량 싸움으로 변한다.
"""
import torch.nn as nn

HEADS = ("linear", "perch", "mlp")


class ModalityClassifier(nn.Module):
    """(B, C, d_model) → (B, num_classes)

    패치 축은 SharedPatchTST에서 평균 풀링으로 이미 접혔다.
    """

    def __init__(self, num_channels, d_model, num_classes=2, dropout=0.2,
                 head="linear", hidden=32):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"모르는 헤드: {head} (가능: {HEADS})")
        self.head = head

        if head == "linear":
            self.net = nn.Sequential(
                nn.Flatten(),                                # (B, C*d_model)
                nn.Dropout(dropout),
                nn.Linear(num_channels * d_model, num_classes),
            )
        elif head == "mlp":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(num_channels * d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:                                                # perch
            # 채널마다 같은 MLP를 통과시킨 뒤 로짓을 더한다. 채널은 끝까지 안 섞인다.
            self.per_channel = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, z):
        if self.head == "perch":
            return self.per_channel(z).sum(dim=1)            # (B, C, K) → (B, K)
        return self.net(z)
