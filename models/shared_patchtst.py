"""
Shared PatchTST — 음성·필기가 트랜스포머 인코더를 공유하는 분류 모델.

    필기 [B,  6, 2000] → 필기 patch_embedding ┐
                                              ├→ ★공유 인코더★ → 모달별 classifier
    음성 [B, 80,  200] → 음성 patch_embedding ┘

왜 공유 인코더인가
    두 데이터셋에 공통 피험자가 0명이라 late fusion은 "좋아졌다"를 잴 수 없다
    (한 사람의 P필기·P음성이 동시에 있어야 하는데 그런 사람이 없다).
    학습 시점에 가중치를 공유하면, 평가는 여전히 모달 하나씩 하면서
    "모달 B에서 배운 것이 모달 A의 성능을 올리는가"를 A/B로 잴 수 있다.
    학습이 끝나면 필기만 넣어도 예측이 나오므로 짝 데이터가 필요한 순간이 없다.

무엇이 공유되나
    임베딩을 지나면 두 모달 모두 "d_model 차원 토큰의 수열"이 되어, 인코더는
    입력이 펜 압력인지 멜 밴드인지 알 방법이 없다. 어텐션은 토큰끼리 내적하고
    FFN은 토큰마다 차원을 가공할 뿐이며, 수열 길이가 달라도(39 vs 15) 무관하다.
    d64·6층 기준 전체 파라미터의 약 91%가 공유된다.

⚠️ 비교 시 주의
    필기 최적은 d64/L6, 음성 최적은 d128/L2로 서로 다르다. 인코더를 공유하려면
    하나로 통일해야 하는데, 그러면 한쪽은 자기 최적을 못 쓴다. 성능이 떨어졌을 때
    "공유 탓"인지 "설정 탓"인지 섞이지 않으려면, 비교 대상 단독 모델도
    **같은 설정으로 다시 돌려야** 한다.
        ✅ 공유(d64 L6) vs 단독(d64 L6)
        ❌ 공유(d64 L6) vs 단독(d128 L2)

Early Exit은 아직 포함하지 않는다. 기본 모델이 먼저다.
"""
import torch.nn as nn

from .classifier import ModalityClassifier
from .patch_embedding import PatchEmbedding


class SharedPatchTST(nn.Module):
    """모달별 임베딩·분류기 + 공유 트랜스포머 인코더.

    Args:
        specs: {모달이름: dict(num_channels=, seq_len=, patch_len=, stride=)}
            예) {"hw":    dict(num_channels=6,  seq_len=2000, patch_len=100, stride=50),
                 "voice": dict(num_channels=80, seq_len=200,  patch_len=24,  stride=12)}

    forward(x, modality)로 한 번에 한 모달씩 흘린다. 두 모달을 한 텐서에 담지 않는
    이유는 애초에 담을 수 없기 때문이다 — 채널 수도 길이도 다르고, 같은 사람의
    두 모달이 0명이라 채널 축으로 붙일 수도 없다.
    """

    def __init__(
        self,
        specs,
        num_classes=2,
        d_model=64,
        n_heads=4,
        n_layers=6,
        d_ff=128,
        dropout=0.2,
        head_dropout=0.2,
    ):
        super().__init__()
        if not specs:
            raise ValueError("specs가 비어 있습니다. 최소 한 모달이 필요합니다.")

        self.modalities = list(specs)
        self.d_model = d_model
        self.n_layers = n_layers

        self.embeddings = nn.ModuleDict({
            name: PatchEmbedding(
                num_channels=s["num_channels"], seq_len=s["seq_len"],
                patch_len=s["patch_len"], stride=s["stride"],
                d_model=d_model, dropout=dropout)
            for name, s in specs.items()
        })

        # ★ 공유 구간 — 모달을 가리지 않는 유일한 부분
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,               # pre-LN (학습 안정)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifiers = nn.ModuleDict({
            name: ModalityClassifier(
                num_channels=s["num_channels"], d_model=d_model,
                num_classes=num_classes, dropout=head_dropout)
            for name, s in specs.items()
        })

    def forward(self, x, modality):
        if modality not in self.embeddings:
            raise KeyError(f"모르는 모달: {modality} (가능: {self.modalities})")
        b, c, _ = x.shape
        z = self.embeddings[modality](x)         # (B*C, P, d_model)
        z = self.encoder(z)                      # ★ 공유
        z = z.mean(dim=1).reshape(b, c, -1)      # 패치 평균 → (B, C, d_model)
        return self.classifiers[modality](z)

    # ────────────────────────── 리포팅 ──────────────────────────

    def param_report(self):
        """공유/전용 파라미터 수. 공유 비율이 낮으면 "인코더를 공유했다"는 주장이 약해진다."""
        shared = sum(p.numel() for p in self.encoder.parameters())
        own = {
            m: sum(p.numel() for p in self.embeddings[m].parameters())
               + sum(p.numel() for p in self.classifiers[m].parameters())
            for m in self.modalities
        }
        total = shared + sum(own.values())
        return dict(shared=shared, own=own, total=total,
                    shared_ratio=shared / total if total else 0.0)

    def describe(self):
        r = self.param_report()
        lines = [
            f"SharedPatchTST  d_model={self.d_model} · {self.n_layers}층 "
            f"· 모달 {len(self.modalities)}개",
            f"  공유 인코더        {r['shared']:>9,}  ({r['shared_ratio'] * 100:.1f}%)",
        ]
        for m in self.modalities:
            emb = self.embeddings[m]
            lines.append(
                f"  {m:<8} 전용      {r['own'][m]:>9,}   "
                f"(채널 {emb.num_channels} · 패치 {emb.patch_len}/{emb.stride}"
                f" → 토큰 {emb.num_patches}개)"
            )
        lines.append(f"  {'합계':<8}          {r['total']:>9,}")
        return "\n".join(lines)
