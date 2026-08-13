import torch.nn as nn

from .classifier import ModalityClassifier
from .patch_embedding import PatchEmbedding


class SharedEarlyExitPatchTST(nn.Module):
    """
    Shared PatchTST + layer-wise Early Exit.

    각 Transformer Encoder Layer 뒤에
    모달별 classifier를 하나씩 둔다.
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
            raise ValueError("specs가 비어 있습니다.")

        self.modalities = list(specs)
        self.d_model = d_model
        self.n_layers = n_layers

        # -------------------------
        # 1. 모달별 Patch Embedding
        # -------------------------
        self.embeddings = nn.ModuleDict({
            name: PatchEmbedding(
                num_channels=s["num_channels"],
                seq_len=s["seq_len"],
                patch_len=s["patch_len"],
                stride=s["stride"],
                d_model=d_model,
                dropout=dropout,
            )
            for name, s in specs.items()
        })

        # -------------------------
        # 2. 공유 Transformer Encoder
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # -------------------------
        # 3. 모달별 + Layer별 Exit
        # -------------------------
        self.exit_classifiers = nn.ModuleDict({
            name: nn.ModuleList([
                ModalityClassifier(
                    num_channels=s["num_channels"],
                    d_model=d_model,
                    num_classes=num_classes,
                    dropout=head_dropout,
                )
                for _ in range(n_layers)
            ])
            for name, s in specs.items()
        })

    def forward(self, x, modality):
        """
        학습용 forward.

        모든 Layer의 logits를 반환한다.

        반환:
            [
                exit1_logits,
                exit2_logits,
                ...
                exitN_logits
            ]
        """

        if modality not in self.embeddings:
            raise KeyError(
                f"모르는 모달: {modality} "
                f"(가능: {self.modalities})"
            )

        b, c, _ = x.shape

        # 모달별 embedding
        z = self.embeddings[modality](x)

        outputs = []

        # Transformer를 한 층씩 통과
        for i, layer in enumerate(self.encoder.layers):

            z = layer(z)

            # 패치 방향 평균
            pooled = z.mean(dim=1).reshape(b, c, -1)

            # 해당 Layer 전용 classifier
            logits = self.exit_classifiers[modality][i](pooled)

            outputs.append(logits)

        return outputs

    def forward_to_exit(self, x, modality, exit_layer):
        """
        특정 Early Exit까지만 실제로 계산한다.

        exit_layer:
            1 → Layer 1까지만
            2 → Layer 2까지만
            ...
            6 → Layer 6까지

        MFLOPs 측정할 때 사용할 함수.
        """

        if not 1 <= exit_layer <= self.n_layers:
            raise ValueError(
                f"exit_layer는 1~{self.n_layers}이어야 합니다."
            )

        b, c, _ = x.shape

        z = self.embeddings[modality](x)

        for i, layer in enumerate(self.encoder.layers):

            z = layer(z)

            if i + 1 == exit_layer:
                pooled = z.mean(dim=1).reshape(b, c, -1)

                return self.exit_classifiers[modality][i](pooled)

        raise RuntimeError("Exit에 도달하지 못했습니다.")