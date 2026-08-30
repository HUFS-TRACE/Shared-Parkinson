"""난이도 구간별 깊이 이득 — Early Exit이 성립할 조건이 있는가.

왜 이것을 재나
    Early Exit의 전제는 "쉬운 표본은 얕게, 어려운 표본은 깊게"다. 그러려면
    **깊이의 이득이 표본마다 달라야** 한다. 모든 표본에서 이득이 같으면 깊이를
    고정하는 편이 낫고, 적응할 여지가 없다.

    파킨슨에서는 누가 어려운 표본인지 알 수 없었다. 모델이 틀린 사람을 어렵다고
    부르면 순환이기 때문이다. VitalDB는 다르다 — 라벨이 심박 60을 기준으로
    갈리므로 **경계까지의 거리가 모델과 무관하게 정해진 난이도**다.

        난이도 = |hr - 60|      서맥 라벨은 경계가 하나다

무엇을 비교하나
    L1·L2·L3·L6을 각각 독립 학습한 체크포인트가 이미 있다. 같은 윈도우에 대해
    네 모델의 예측을 모두 받아, 난이도 구간별로 깊이 이득을 나눈다.

        쉬운 구간에서 이득 ~ 0, 어려운 구간에서 이득 > 0
            -> 적응할 여지가 있다. Early Exit의 조건이 성립한다.
        모든 구간에서 이득이 비슷
            -> 깊이를 고정하는 편이 낫다. 조건이 성립하지 않는다.

    ⚠️ 한 클래스가 95%를 넘는 구간은 정확도가 의미를 잃으므로 해석에서 뺀다.
       서맥 라벨에서 경계로부터 20 이상 떨어진 구간이 그렇다(양성률 0.4%).
       다 정상이라고 찍어도 99.6%가 나온다.

재학습하지 않는다
    저장된 가중치로 추론만 다시 한다. fold마다 모델이 다르므로 그 fold의 test에만
    쓴다 — 다른 fold에 쓰면 학습에 본 데이터가 섞인다.

실행
    python analyze_difficulty.py
    python analyze_difficulty.py --seeds 42 1 7 0 123 --bins 3 10 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.shared_patchtst import SharedPatchTST
from utils.dataset import WindowDataset, load_modality, subject_kfold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DEPTHS = [("L1", "vdb_vdb_depth_L1"), ("L2", "vdb_vdb_depth_L2"),
          ("L3", "vdb_vdb_depth_L3"), ("L6", "vdb_vdb_ecg")]


def predict(tag, seed, fold, X, spec, enc, n_layers, device, bs=256):
    """저장된 가중치로 추론만. 없으면 None."""
    ck = RES / f"model_{tag}_s{seed}_f{fold}.pt"
    if not ck.exists():
        return None
    e = dict(enc)
    e["n_layers"] = n_layers
    net = SharedPatchTST({"ecg": spec}, num_classes=2, **e).to(device)
    net.load_state_dict(torch.load(ck, map_location=device))
    net.eval()
    out = []
    with torch.no_grad():
        for xb, in DataLoader(torch.utils.data.TensorDataset(
                torch.from_numpy(X)), batch_size=bs):
            out.append(torch.softmax(net(xb.to(device), "ecg"), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description="난이도 구간별 깊이 이득")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 1, 7, 0, 123])
    ap.add_argument("--bins", type=float, nargs="*", default=[3, 10, 20],
                    help="경계까지의 거리 구간 경계. 기본 <3 / 3~10 / 10~20 / >=20")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
    enc = mcfg["shared_encoder"]

    z = np.load(cfg["data"]["ecg_path"], allow_pickle=True)
    X, y, sid, hr = (np.asarray(z["X"], dtype=np.float32), z["y"].astype(np.int64),
                     z["subject_id"], np.asarray(z["hr"], dtype=np.float32))
    spec = dict(num_channels=X.shape[1], seq_len=X.shape[2], **mcfg["modalities"]["ecg"])

    # 라벨이 hr < 60 에서 갈리므로 경계까지의 거리가 곧 난이도다
    diff = np.abs(hr - 60.0)
    edges = [0.0] + list(args.bins) + [np.inf]
    names = ([f"매우어려움 <{args.bins[0]:.0f}"]
             + [f"{a:.0f}~{b:.0f}" for a, b in zip(args.bins, args.bins[1:])]
             + [f"쉬움 >={args.bins[-1]:.0f}"])

    print(f"윈도우 {len(y):,} · 피험자 {len(np.unique(sid)):,} · 시드 {args.seeds}")
    print(f"난이도 = |hr - 60|  ·  구간 {names}\n")

    # (시드, 깊이) -> 전체 윈도우에 대한 확률. fold마다 그 fold의 test만 채운다.
    P = {}
    for seed in args.seeds:
        folds = subject_kfold(sid, y, n_splits=cfg["split"]["n_folds"],
                              val_size=cfg["split"]["val_size"], seed=seed,
                              stratified=cfg["split"].get("stratified", False),
                              shuffle=True)
        for lab, tag in DEPTHS:
            p = np.full(len(y), np.nan, dtype=np.float32)
            L = 6 if lab == "L6" else int(lab[1])
            for f, (_, _, te) in enumerate(folds):
                q = predict(tag, seed, f, X[te], spec, enc, L, args.device)
                if q is None:
                    break
                p[te] = q
            if not np.isnan(p).all():
                P[(seed, lab)] = p
        print(f"  seed {seed}: " + " ".join(
            lab for lab, _ in DEPTHS if (seed, lab) in P))

    if not P:
        raise SystemExit("체크포인트가 없습니다. --keep-ckpt 로 학습해야 합니다.")

    print(f"\n{'구간':>12}{'윈도우':>9}{'양성률':>8}"
          + "".join(f"{lab:>9}" for lab, _ in DEPTHS)
          + f"{'L6-L1':>9}{'부호':>8}")

    rows = []
    for i, nm in enumerate(names):
        m = (diff >= edges[i]) & (diff < edges[i + 1])
        acc = {}
        for lab, _ in DEPTHS:
            per = [np.mean((P[(s, lab)][m] >= 0.5).astype(int) == y[m])
                   for s in args.seeds if (s, lab) in P]
            acc[lab] = np.array(per)
        d = acc["L6"] - acc["L1"]
        sg = "모두 +" if (d > 0).all() else "모두 -" if (d < 0).all() else "엇갈림"
        print(f"{nm:>12}{m.sum():>9,}{y[m].mean():>8.1%}"
              + "".join(f"{acc[lab].mean():>9.4f}" for lab, _ in DEPTHS)
              + f"{d.mean():>+9.4f}{sg:>8}")
        rows.append((nm, m.sum(), d, y[m].mean()))

    print()
    print("  읽는 법")

    # 한 클래스가 95%를 넘는 구간은 정확도가 의미를 잃는다. 다 찍어도 그만큼 나온다.
    ok = [(nm, n, d) for nm, n, d, pr in rows if 0.05 < pr < 0.95]
    for nm, _, _, pr in rows:
        if not 0.05 < pr < 0.95:
            print(f"     [{nm}] 양성률 {pr:.1%} — 한 클래스로 찍어도 "
                  f"{max(pr, 1 - pr):.1%}라 정확도가 의미 없다. 해석에서 뺀다.")
    if len(ok) < 2:
        print("     비교할 구간이 부족하다.")
        return

    g = np.array([d.mean() for _, _, d in ok])
    print(f"     남은 구간 {[nm for nm, _, _ in ok]}  "
          f"이득 {g.min():+.4f} ~ {g.max():+.4f}")
    if g.max() - g.min() < 0.005:
        print("     -> 난이도가 달라도 깊이 이득이 변하지 않는다. 적응할 여지가 없다.")
        print("        표본마다 다른 깊이를 쓸 이유가 없어 Early Exit의 전제가 깨진다.")
    elif g[0] > g[-1] + 0.01:
        print("     -> 어려운 구간일수록 이득이 크다. 적응할 여지가 있다.")
    else:
        print("     -> 구간별 이득이 순서를 이루지 않는다. 시드별 부호를 함께 볼 것.")
    print()


if __name__ == "__main__":
    main()
