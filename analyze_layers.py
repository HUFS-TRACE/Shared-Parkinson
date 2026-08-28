"""층별 표현 분석 — 층이 정보를 더하는가.

무엇을 묻나
    깊이 축에서 L1~L6이 구별되지 않았다(전부 엇갈림). 왜 그런지를 표현 수준에서 본다.
    학습은 필요 없다 — `train.py --save-repr`가 만든 npz만 읽는다.

네 가지를 잰다

    (1) 층별 linear probe          ★ 주장
        z_i 로 로지스틱 회귀를 학습해 피험자 정확도를 낸다.
        층을 지나며 평평하면 **층이 정보를 더하지 않는다**는 직접 증거다.

    (2) CKA (인접 + 전체 행렬)      설명
        인접 층 표현이 얼마나 비슷한가. 1.0에 가까우면 표현이 안 바뀐다.

        ⚠️ CKA만으로는 약하다. 표현이 비슷해도 마지막에 선형 분리가 좋아질 수 있고,
        달라도 그 변화가 라벨과 무관할 수 있다. (1)이 주장이고 (2)는 그 이유다.

    (3) 모달 간 CKA (공유 모델에서만)
        같은 층에서 필기 표현과 음성 표현이 얼마나 닮았나.
        어느 층에서 두 모달이 갈라지는지 보인다. 공유 인코더가 실제로 무엇을
        공유하는지에 대한 직접 관찰이고, 우리가 여태 한 번도 안 본 것이다.

    (4) 층 간 예측 뒤집힘
        L1에서 맞힌 사람이 L6에서 틀리는 비율, 그 반대 비율.
        둘 다 낮으면 층이 예측을 바꾸지 않는다는 뜻이다. (1)을 사람 단위로 쪼갠 것.

CKA를 어떻게 계산하나
    선형 CKA를 쓴다. 특징 차원(384~5120)이 표본 수보다 크거나 비슷하므로
    그램 행렬(N×N) 대신 **특징 공분산 쪽 항등식**을 쓴다.

        CKA(X, Y) = ||Xᵀ Y||_F² / ( ||Xᵀ X||_F · ||Yᵀ Y||_F )

    두 행렬을 열 평균 중심화한 뒤 계산한다. 표본 수가 많으면 무작위로 줄인다.

실행
    python analyze_layers.py                       # results/repr_*.npz 전부
    python analyze_layers.py --tag sw2_base_hw     # 특정 태그만
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
MAX_N = 4000          # CKA 표본 상한. 이보다 많으면 무작위로 줄인다


def linear_cka(X, Y, rng=None):
    """선형 CKA. X, Y는 [N, D_x], [N, D_y] — 같은 표본을 다른 특징으로 본 것."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) > MAX_N:
        take = (rng or np.random.default_rng(0)).choice(len(X), MAX_N, replace=False)
        X, Y = X[take], Y[take]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


def probe(Zte, yte, sidte, foldte, Zva, yva, sidva, foldva):
    """층 하나의 표현으로 로지스틱 회귀 → (피험자 정확도, AUC).

    **같은 fold의 모델이 만든 val로 학습해 test로 평가한다.** fold마다 모델이
    따로 학습되므로 fold 0의 384차원과 fold 1의 384차원은 같은 좌표계가 아니다 —
    다른 네트워크가 만든 벡터공간이다. test를 fold를 넘어 섞어 학습하면 좌표계가
    다른 것을 하나로 취급하는 셈이라 신호가 사라진다(실측: 정확도가 우연 수준으로
    떨어졌다). val→같은 fold test로 좁히면 이 문제가 없다 — 캘리브레이션을
    val에서 하는 것과 같은 이유다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    accs, aucs = [], []
    for f in np.unique(foldte):
        tr = foldva == f
        te = foldte == f
        if len(np.unique(yva[tr])) < 2 or len(np.unique(yte[te])) < 2:
            continue
        sc = StandardScaler().fit(Zva[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(Zva[tr]), yva[tr])
        p = clf.predict_proba(sc.transform(Zte[te]))[:, 1]
        s = sidte[te]
        ok = [int(p[s == u].mean() >= 0.5) == int(yte[te][s == u][0])
              for u in np.unique(s)]
        accs.append(np.mean(ok))
        aucs.append(roc_auc_score(yte[te], p))
    return float(np.mean(accs)), float(np.mean(aucs))


def subject_pred(Zte, yte, sidte, foldte, Zva, yva, sidva, foldva):
    """층 하나의 표현으로 사람별 예측. (4) 뒤집힘 계산용 — {사람: (예측, 정답)}"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = {}
    for f in np.unique(foldte):
        tr = foldva == f
        te = foldte == f
        if len(np.unique(yva[tr])) < 2:
            continue
        sc = StandardScaler().fit(Zva[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Zva[tr]), yva[tr])
        p = clf.predict_proba(sc.transform(Zte[te]))[:, 1]
        s = sidte[te]
        for u in np.unique(s):
            out[u] = (int(p[s == u].mean() >= 0.5), int(yte[te][s == u][0]))
    return out


def analyze_one(path, rng):
    d = np.load(path, allow_pickle=True)
    R, y, sid, fold = d["repr"], d["y"], d["subject_id"], d["fold"]
    has_val = "val_repr" in d
    L = R.shape[0]
    print("=" * 76)
    print(f"{Path(path).name}   {L}층 × {R.shape[1]}윈도우 × {R.shape[2]}차원 "
          f"· 피험자 {len(np.unique(sid))}명")
    print("=" * 76)

    if not has_val:
        print("  ⚠️ val_repr이 없는 옛 파일입니다 — probe를 건너뜁니다 "
              "(fold를 넘어 test를 섞으면 좌표계가 안 맞아 우연 수준이 나옵니다).")

    # ── (1) 층별 probe + (2) 인접 CKA ──
    print(f"  {'층':>4}{'probe 피험자':>14}{'probe AUC':>12}{'다음 층과 CKA':>16}")
    accs, preds = [], []
    for i in range(L):
        cka = (linear_cka(R[i], R[i + 1], rng) if i + 1 < L else np.nan)
        cell = f"{cka:.4f}" if not np.isnan(cka) else "—"
        if has_val:
            Z = R[i].astype(np.float32)
            Zv = d["val_repr"][i].astype(np.float32)
            a, u = probe(Z, y, sid, fold, Zv, d["val_y"], d["val_subject_id"], d["val_fold"])
            accs.append(a)
            preds.append(subject_pred(Z, y, sid, fold, Zv, d["val_y"],
                                      d["val_subject_id"], d["val_fold"]))
            print(f"  {i + 1:>4}{a:>14.4f}{u:>12.4f}{cell:>16}")
        else:
            print(f"  {i + 1:>4}{'—':>14}{'—':>12}{cell:>16}")

    if accs:
        d1 = accs[-1] - accs[0]
        print(f"\n  L1 → L{L} probe 변화 {d1:+.4f}"
              + ("   → 층이 정보를 더하지 않는다" if abs(d1) < 0.02 else ""))

    # ── (2) 전체 CKA 행렬 ──
    print(f"\n  CKA 행렬")
    M = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            M[i, j] = M[j, i] = linear_cka(R[i], R[j], rng)
    print("     " + "".join(f"{f'L{j+1}':>8}" for j in range(L)))
    for i in range(L):
        print(f"  L{i + 1:<2}" + "".join(f"{M[i, j]:>8.3f}" for j in range(L)))
    off = M[np.triu_indices(L, k=1)]
    print(f"  비대각 평균 {off.mean():.3f} · 최소 {off.min():.3f}"
          + ("   → 층을 지나도 표현이 거의 그대로다" if off.min() > 0.8 else ""))

    # ── (4) 뒤집힘 ──
    if not preds:
        return dict(accs=accs, cka=M, path=Path(path).name)
    a, b = preds[0], preds[-1]
    common = sorted(set(a) & set(b))
    if common:
        a_ok = np.array([a[u][0] == a[u][1] for u in common])
        b_ok = np.array([b[u][0] == b[u][1] for u in common])
        lost = (a_ok & ~b_ok).sum()
        gained = (~a_ok & b_ok).sum()
        print(f"\n  예측 뒤집힘 (L1 → L{L}, {len(common)}명)")
        print(f"    L1 맞힘 → L{L} 틀림 : {lost}명 ({lost / len(common) * 100:.1f}%)")
        print(f"    L1 틀림 → L{L} 맞힘 : {gained}명 ({gained / len(common) * 100:.1f}%)")
        if lost + gained <= len(common) * 0.1:
            print(f"    → 층이 예측을 거의 바꾸지 않는다")
    return dict(accs=accs, cka=M, path=Path(path).name)


def cross_modal(paths, rng):
    """(3) 같은 층에서 필기 표현과 음성 표현이 얼마나 닮았나 (공유 모델).

    두 모달의 윈도우 수·차원이 다르므로 표본을 짝지을 수 없다. 대신 각 모달의
    표현을 **특징 축에서 요약**해 층별로 비교한다 — 사람 단위 평균 표현을 만들고,
    두 모달의 피험자 집합이 겹치지 않으므로 표본 간 유사도 대신 **표현 공간의
    2차 통계(고유값 스펙트럼)**를 비교한다.
    """
    if len(paths) != 2:
        return
    A = np.load(paths[0], allow_pickle=True)["repr"]
    B = np.load(paths[1], allow_pickle=True)["repr"]
    L = min(A.shape[0], B.shape[0])
    print("=" * 76)
    print("모달 간 표현 구조 비교 (공유 인코더) — 같은 층에서 두 모달이 닮았나")
    print("=" * 76)
    print("  두 모달은 피험자가 겹치지 않아 표본을 짝지을 수 없다.")
    print("  그래서 표현 공간의 고유값 스펙트럼(정규화)을 비교한다.")
    print(f"\n  {'층':>4}{'필기 유효차원':>14}{'음성 유효차원':>14}{'스펙트럼 거리':>14}")
    for i in range(L):
        out = []
        for R in (A[i], B[i]):
            X = np.asarray(R, dtype=np.float64)
            if len(X) > MAX_N:
                X = X[rng.choice(len(X), MAX_N, replace=False)]
            X = X - X.mean(axis=0, keepdims=True)
            s = np.linalg.svd(X, compute_uv=False) ** 2
            s = s / s.sum()
            # 유효 차원 = 참여율 (participation ratio). 표현이 몇 방향을 쓰는가
            out.append((1.0 / (s ** 2).sum(), s[:20]))
        pa, sa = out[0]
        pb, sb = out[1]
        n = min(len(sa), len(sb))
        dist = np.abs(sa[:n] - sb[:n]).sum()
        print(f"  {i + 1:>4}{pa:>14.1f}{pb:>14.1f}{dist:>14.4f}")
    print("\n  유효 차원이 층을 지나며 줄면 표현이 저차원으로 붕괴하는 것이다.")
    print("  두 모달의 스펙트럼 거리가 커지면 층을 지나며 갈라지는 것이다.")


def main():
    ap = argparse.ArgumentParser(description="층별 표현 분석")
    ap.add_argument("--tag", help="파일명에 이 문자열이 든 것만")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(p for p in RES.glob("repr_*.npz")
                   if "smoke" not in p.name and (not args.tag or args.tag in p.name))
    if not files:
        raise SystemExit("results/repr_*.npz 가 없습니다. "
                         "train.py --save-repr 로 먼저 만드세요.")
    rng = np.random.default_rng(args.seed)

    got = defaultdict(list)
    for f in files:
        r = analyze_one(f, rng)
        # 같은 학습(tag)의 두 모달을 묶어 모달 간 비교에 쓴다
        stem = f.stem[len("repr_"):]
        modal, _, tail = stem.partition("_")
        got[tail].append((modal, f))
        print()

    for tail, pairs in got.items():
        if len(pairs) == 2:
            order = {m: p for m, p in pairs}
            if "hw" in order and "voice" in order:
                cross_modal([order["hw"], order["voice"]], rng)
                print()

    print("=" * 76)
    print("읽는 법")
    print("=" * 76)
    print("  probe가 층을 지나며 평평하다   → 층이 정보를 더하지 않는다 (주장)")
    print("  CKA가 1.0에 가깝다             → 표현이 안 바뀐다 (그 이유)")
    print("  뒤집힘이 적다                  → 층이 예측을 안 바꾼다 (사람 단위 확인)")
    print("  ⚠️ CKA만으로는 약하다. probe가 주장이고 CKA는 설명이다.")


if __name__ == "__main__":
    main()
