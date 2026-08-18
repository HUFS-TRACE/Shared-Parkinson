"""유보가 왜 환자 쪽으로 쏠리는가.

무슨 문제인가
    판정 보류는 사람 평균 확률이 0.5 근처인 사람을 뺀다. 그런데 필기에서 보류된
    사람의 환자 비율이 전체(42.6%)보다 +32~35%p 높다. 스크리닝에서 이건 최악의
    실패 모드다 — 정확도는 오르지만 오른 이유가 "어려운 환자를 빼서"이기 때문이다.

    음성에서는 같은 쏠림이 없다. 방법이 원래 편향된 것이라면 양쪽에서 나와야 한다.
    그러니 필기에만 있는 무언가가 원인이다.

무엇을 의심하나 — 세 갈래
    (1) 모델이 환자를 덜 확신한다
        파킨슨 필기는 정도가 연속적이라, 초기 환자는 정상과 구별이 어렵다.
        그렇다면 환자가 경계 근처에 몰리는 것은 데이터의 성질이지 방법의 결함이 아니다.

    (2) 윈도우 수가 많으면 평균이 0.5로 끌려간다
        사람 판정은 윈도우 확률의 평균이다. 윈도우가 많을수록 평균은 전체 평균으로
        수렴해 0.5에 가까워진다(큰 수의 법칙). 그런데 필기 환자는 1인당 윈도우가
        정상의 2.4배다. 그러면 **환자라서가 아니라 윈도우가 많아서** 경계에 몰린다.
        이건 순전히 기계적인 인공물이고, 반드시 고쳐야 한다.

        음성은 1인당 윈도우 비가 1.10배로 거의 같다. 쏠림이 없다는 사실이
        이 가설과 맞아떨어진다.

    (3) 판정 자체가 비대칭이다
        민감도가 특이도보다 낮으면 환자가 원래 경계 근처에 있다는 뜻이다.

어떻게 가르나
    (2)를 확인하는 방법은 윈도우 수를 맞춰 보는 것이다. 모든 사람에게서 같은 개수만
    뽑아 평균을 다시 내면, 윈도우 수 차이가 사라진다. 그래도 쏠림이 남으면 (1)이고,
    사라지면 (2)다.

실행
    python analyze_abstain_bias.py
    python analyze_abstain_bias.py --rate 0.2 --match 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sequential_exit import calibrate, logit, sigmoid  # noqa: E402
from verify_abstain import fit_band  # noqa: E402


def per_subject(p, y, sid, idx, n_match=None):
    """사람별 (평균확률, 정답, 윈도우수).

    n_match를 주면 사람마다 시간축 균등 간격으로 그 개수만 뽑아 평균한다.
    윈도우 수 차이를 없애 (2)번 가설을 검증하기 위한 것이다.
    """
    mp, ty, nw = [], [], []
    for s in np.unique(sid):
        m = np.nonzero(sid == s)[0]
        m = m[np.argsort(idx[m], kind="stable")]        # 녹음 시간 순
        nw.append(len(m))
        if n_match and len(m) > n_match:
            m = m[np.linspace(0, len(m) - 1, n_match).astype(int)]
        mp.append(float(np.mean(p[m])))
        ty.append(int(y[m][0]))
    return np.array(mp), np.array(ty), np.array(nw)


def spearman(a, b):
    """순위 상관. 단조 관계만 보므로 분포 모양에 둔감하다."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def look(path, rate, n_match):
    d = np.load(path, allow_pickle=True)
    T, c = calibrate(d)
    p = sigmoid(logit(d["prob"]) / T - c)
    mp, ty, nw = per_subject(p, d["y"], d["subject_id"], d["idx"])
    margin = np.abs(mp - 0.5)
    pred = (mp >= 0.5).astype(int)
    prev = ty.mean()

    print("=" * 78)
    print(f"{Path(path).name}   피험자 {len(ty)}명 · PD 비율 {prev:.1%}")
    print("=" * 78)

    # ── (3) 판정이 비대칭인가 ──
    sens = float(pred[ty == 1].mean())
    spec = float(1 - pred[ty == 0].mean())
    print(f"  민감도 {sens:.3f} · 특이도 {spec:.3f}"
          + ("   ← 환자를 더 많이 놓친다" if sens < spec - 0.05 else ""))

    # ── (1) 환자가 경계에 몰리는가 ──
    m1, m0 = margin[ty == 1], margin[ty == 0]
    print(f"\n  경계까지 거리 |p-0.5|   환자 중앙 {np.median(m1):.4f} · "
          f"정상 중앙 {np.median(m0):.4f}")

    # ── (2) 윈도우 수가 원인인가 ──
    print(f"\n  1인당 윈도우   환자 중앙 {int(np.median(nw[ty == 1]))} · "
          f"정상 중앙 {int(np.median(nw[ty == 0]))} "
          f"({np.median(nw[ty == 1]) / np.median(nw[ty == 0]):.2f}배)")
    r = spearman(nw, margin)
    print(f"  윈도우 수 ↔ 경계거리 순위상관 {r:+.3f}"
          + ("   ← 많을수록 경계에 가깝다(평균이 0.5로 수렴)" if r < -0.2 else
             "   ← 뚜렷한 관계 없음" if abs(r) <= 0.2 else ""))

    # ── 윈도우 수를 맞추면 쏠림이 사라지나 ──
    band = fit_band(d, T, c, rate)
    hold = margin < band
    rows = [("원본", hold, mp)]
    if n_match:
        mp2, ty2, _ = per_subject(p, d["y"], d["subject_id"], d["idx"], n_match)
        assert (ty2 == ty).all(), "사람 순서가 어긋났다"
        h2 = np.abs(mp2 - 0.5) < np.quantile(np.abs(mp2 - 0.5), hold.mean())
        rows.append((f"윈도우 {n_match}개로 맞춤", h2, mp2))

    print(f"\n  보류율 목표 {rate:.0%} (val 밴드 {band:.4f})")
    print(f"  {'조건':>20}{'보류':>7}{'보류집단 PD비율':>17}{'전체와 차이':>13}")
    out = {}
    for lab, h, m in rows:
        if h.sum() == 0:
            continue
        gap = ty[h].mean() - prev
        out[lab] = gap
        print(f"  {lab:>20}{int(h.sum()):>6}명{ty[h].mean():>16.1%}{gap:>+13.1%}p")
    return out


def main():
    ap = argparse.ArgumentParser(description="유보 편향의 원인 가리기")
    ap.add_argument("--probs", nargs="*", default=None)
    ap.add_argument("--rate", type=float, default=0.20, help="목표 보류율")
    ap.add_argument("--match", type=int, default=80,
                    help="윈도우 수를 이 개수로 맞춰 다시 계산 (0이면 안 함)")
    args = ap.parse_args()

    files = args.probs or sorted(
        p for p in (ROOT / "results").glob("probs_*.npz") if "smoke" not in p.name)
    agg = {}
    for f in files:
        agg[Path(f).name] = look(f, args.rate, args.match or None)
        print()

    print("=" * 78)
    print("종합 — 윈도우 수를 맞추면 쏠림이 줄어드는가")
    print("=" * 78)
    print(f"  {'파일':<44}{'원본':>10}{'맞춤':>10}{'변화':>10}")
    for name, o in agg.items():
        if len(o) < 2:
            continue
        a, b = list(o.values())
        print(f"  {name:<44}{a:>+9.1%}p{b:>+9.1%}p{b - a:>+9.1%}p")
    print("\n  맞춘 뒤 쏠림이 0 근처로 줄면, 원인은 환자의 병세가 아니라")
    print("  '환자에게 윈도우가 많아 평균이 0.5로 끌려간 것'이다 — 고칠 수 있는 결함이다.")


if __name__ == "__main__":
    main()
