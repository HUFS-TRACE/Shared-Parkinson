"""단순 통계 특징 + 얕은 분류기 — 트랜스포머가 필요한 태스크인가.

왜 재나
    깊이 축에서 L1~L6이 구별되지 않았다(전부 엇갈림). 두 가지로 읽을 수 있다.

        (가) 이 태스크는 저수준 특징만으로 풀린다 → 계층적 표현이 필요 없다
        (나) 61명·5시드로는 층 간 차이를 검출할 검정력이 없다

    둘을 가르려면 **바닥**을 재야 한다. 윈도우마다 평균·표준편차 같은 통계량만 뽑아
    로지스틱 회귀로 풀었을 때 PatchTST와 비슷하면 (가)이고, 크게 낮으면 트랜스포머가
    실제로 무언가를 배우고 있다는 뜻이라 (나) 쪽으로 기운다.

    이것은 "우리 모델이 좋다/나쁘다"를 보려는 것이 아니라, **태스크의 난이도 바닥**을
    재는 것이다. 발표에서 "왜 층이 필요 없나"에 답하려면 이 값이 있어야 한다.

무엇을 뽑나
    채널마다 8개. 시계열 분류에서 관례적으로 쓰는 최소 집합이다.

        평균 · 표준편차 · 최소 · 최대 · 중앙 · IQR
        1차 차분의 표준편차   (얼마나 급하게 변하나 — 떨림)
        0 교차율              (진동 빈도)

    필기 6채널이면 48차원, 음성 20밴드면 160차원이 된다.
    주파수 특징(FFT)은 넣지 않았다. 바닥을 재는 것이 목적이므로 가장 단순한 쪽을 쓴다.

어떻게 맞대나
    **같은 fold를 쓴다.** train_test_split을 새로 하면 비교가 성립하지 않는다.
    `subject_kfold(shuffle=True)`로 train.py와 같은 분할을 만들고, 표준화도
    train fold에서만 fit한다.

    판정도 같은 방식이다 — 윈도우 확률을 사람 단위로 평균(soft voting)한다.

실행
    python baseline_simple.py                    # 필기·음성 · 시드 5개
    python baseline_simple.py --modalities hw
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.dataset import load_modality, subject_kfold  # noqa: E402

SEEDS = (42, 1, 7, 0, 123)
MEL20 = list(range(0, 80, 4))


def features(X):
    """[N, C, T] → [N, C*8]. 채널마다 통계량 8개."""
    d = np.diff(X, axis=2)
    q1, med, q3 = np.percentile(X, [25, 50, 75], axis=2)
    f = np.stack([
        X.mean(axis=2),
        X.std(axis=2),
        X.min(axis=2),
        X.max(axis=2),
        med,
        q3 - q1,                                   # IQR
        d.std(axis=2),                             # 급변 정도 (떨림)
        (np.diff(np.signbit(X - X.mean(axis=2, keepdims=True)),
                 axis=2) != 0).mean(axis=2),       # 0 교차율
    ], axis=2)                                     # [N, C, 8]
    return f.reshape(len(X), -1)


def subject_acc(prob, y, sid, idx_test):
    """사람 단위 soft voting 정확도. train.py와 같은 방식."""
    s = sid[idx_test]
    ok = []
    for u in np.unique(s):
        m = s == u
        ok.append(int(prob[m].mean() >= 0.5) == int(y[idx_test][m][0]))
    return float(np.mean(ok))


def run(name, X, y, sid, seeds, n_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    F = features(X)
    print(f"  특징 {F.shape[1]}차원 ({X.shape[1]}채널 × 8)")

    out = []
    for seed in seeds:
        folds = subject_kfold(sid, y, n_splits=n_folds, val_size=0.1,
                              seed=seed, shuffle=True)
        accs, aucs, wacc = [], [], []
        for tr, va, te in folds:
            sc = StandardScaler().fit(F[tr])       # train fold에서만 fit
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(F[tr]), y[tr])
            p = clf.predict_proba(sc.transform(F[te]))[:, 1]
            accs.append(subject_acc(p, y, sid, te))
            aucs.append(roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan)
            wacc.append(((p >= 0.5).astype(int) == y[te]).mean())
        out.append([np.mean(accs), np.nanmean(aucs), np.mean(wacc)])
    a = np.array(out)
    print(f"  {'seed':>6}{'피험자':>10}{'AUC':>10}{'윈도우':>10}")
    for s, r in zip(seeds, a):
        print(f"  {s:>6}{r[0]:>10.4f}{r[1]:>10.4f}{r[2]:>10.4f}")
    print(f"  {'평균':>6}{a[:, 0].mean():>10.4f}{a[:, 1].mean():>10.4f}"
          f"{a[:, 2].mean():>10.4f}")
    return a.mean(axis=0)


def main():
    ap = argparse.ArgumentParser(description="단순 통계 특징 기준선")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    ap.add_argument("--modalities", default="hw,voice")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # PatchTST 쪽 실측값 (새 분할 · 시드 5개). 맞대기 위해 여기 적어 둔다.
    REF = {"hw": (0.8236, 0.8430, 0.6900), "voice": (0.6615, 0.7230, 0.6490)}

    got = {}
    for m in args.modalities.split(","):
        path = ROOT / cfg["data"][f"{m}_path"]
        if not path.exists():
            path = Path(cfg["data"][f"{m}_path"])
        if not path.exists():
            print(f"[{m}] {path} 없음 — 건너뜁니다\n")
            continue
        X, y, sid = load_modality(str(path))
        if m == "voice":
            X = np.ascontiguousarray(X[:, MEL20, :])   # PatchTST 쪽과 같은 mel20
        print("=" * 62)
        print(f"{m}  X{tuple(X.shape)}  피험자 {len(np.unique(sid))}명")
        print("=" * 62)
        got[m] = run(m, X, y, sid, args.seeds)
        print()

    print("=" * 62)
    print("PatchTST와 비교 — 태스크의 난이도 바닥은 어디인가")
    print("=" * 62)
    print(f"  {'':8}{'지표':>8}{'단순특징':>10}{'PatchTST':>10}{'차이':>9}")
    for m, v in got.items():
        for i, k in enumerate(("피험자", "AUC", "윈도우")):
            r = REF[m][i]
            print(f"  {m if i == 0 else '':8}{k:>8}{v[i]:>10.4f}{r:>10.4f}{v[i] - r:>+9.4f}")
    print()
    print("  단순 특징이 비슷하면 → 저수준 특징만으로 풀리는 태스크다.")
    print("     계층적 표현이 필요 없으므로 층이 무의미한 것이 자연스럽다.")
    print("  단순 특징이 크게 낮으면 → 트랜스포머가 실제로 무언가를 배우고 있다.")
    print("     그렇다면 층 간 차이를 검출하지 못한 것은 검정력 문제일 수 있다.")


if __name__ == "__main__":
    main()
