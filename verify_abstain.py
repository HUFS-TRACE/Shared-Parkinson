"""판정 보류가 정말 작동하는지 검증한다.

왜 필요한가
    sequential_exit.py의 run_abstain은 "보류하고 남은 사람의 정확도가 올랐다"까지만
    보여준다. 그런데 애매한 사람을 아무렇게나 빼도 정확도는 오른다. 오른다는 사실
    자체는 보류 규칙이 쓸 만하다는 증거가 되지 못한다.

    세 가지를 확인해야 주장이 선다.

    (1) 보류된 사람이 실제로 틀리는 사람인가
        보류 집단을 억지로 판정했을 때 정확도가 우연 수준(50%)이면, 규칙이 정말
        "모델이 모르는 사람"을 골라낸 것이다. 보류 집단도 80%씩 맞힌다면 멀쩡한
        사람을 버려 남은 쪽 평균만 올린 셈이다.

    (2) 보류가 한쪽 클래스만 걸러내지 않는가
        환자만 보류되면 남은 집단은 정상 위주가 되고, 정확도가 오른 것처럼 보이지만
        실제로는 환자를 놓친 것이다. 스크리닝에서 이건 최악의 실패다.
        보류 집단의 HC:PD가 전체 비율과 비슷해야 한다.

    (3) 필기·음성 양쪽에서 재현되는가
        한쪽에서만 되면 데이터 특성이지 방법이 아니다.

밴드를 정하는 방법
    test를 보고 밴드를 고르면 테스트셋 피팅이다. 목표 보류율을 먼저 정하고,
    **val에서** 그 보류율이 나오는 밴드를 구해 test에 그대로 적용한다.
    val은 fold마다 사람이 겹치므로 (fold, 사람)을 단위로 센다.

실행
    python verify_abstain.py
    python verify_abstain.py --rates 0.1 0.2 0.3
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sequential_exit import calibrate, logit, sigmoid  # noqa: E402

RATES = (0.10, 0.20, 0.30)


def by_subject(p, y, sid):
    """윈도우 확률 → 사람별 (평균확률, 정답). 판정 단위는 사람이다."""
    out = []
    for s in np.unique(sid):
        m = sid == s
        out.append((float(np.mean(p[m])), int(y[m][0])))
    mp = np.array([a for a, _ in out])
    ty = np.array([b for _, b in out])
    return mp, ty


def fit_band(d, T, c, rate):
    """val에서 목표 보류율이 나오는 밴드를 구한다. val이 없으면 밴드를 못 정한다.

    fold마다 같은 사람이 다시 나오므로 (fold, 사람)을 각각 한 건으로 센다.
    사람 단위로 합치면 fold 간 평균이 섞여 test와 조건이 달라진다.
    """
    if "val_prob" not in d:
        return None
    p = sigmoid(logit(d["val_prob"]) / T - c)
    # subject_id는 모달마다 dtype이 다르다(필기 문자열, 음성 정수) — 문자열로 맞춘다
    key = np.char.add(np.char.add(d["val_fold"].astype(str), "|"),
                      d["val_subject_id"].astype(str))
    mp, _ = by_subject(p, d["val_y"], key)
    return float(np.quantile(np.abs(mp - 0.5), rate))


def check(path, rates):
    d = np.load(path, allow_pickle=True)
    T, c = calibrate(d)
    p = sigmoid(logit(d["prob"]) / T - c)
    mp, ty = by_subject(p, d["y"], d["subject_id"])

    margin = np.abs(mp - 0.5)
    pred = (mp >= 0.5).astype(int)
    base = float(np.mean(pred == ty))
    n = len(ty)
    prev = float(np.mean(ty))

    print("=" * 78)
    print(Path(path).name)
    print("=" * 78)
    print(f"  피험자 {n}명 (PD {int(ty.sum())} / HC {n - int(ty.sum())}, PD 비율 {prev:.1%})")
    print(f"  캘리브레이션 (val): T={T:.2f}  c={c:+.3f}")
    print(f"  [기준] 전원 판정: 정확도 {base:.4f} ({int(np.sum(pred == ty))}/{n}명)")

    print(f"\n  {'목표':>6}{'밴드(val)':>11}{'실제보류':>9}{'판정 정확도':>13}{'기준대비':>10}"
          f"{'보류집단 정확도':>16}{'보류집단 PD비율':>16}")
    rows = []
    for r in rates:
        band = fit_band(d, T, c, r)
        if band is None:
            print("  val이 없어 밴드를 정할 수 없습니다 — 건너뜁니다.")
            return None
        hold = margin < band
        keep = ~hold
        if keep.sum() == 0 or hold.sum() == 0:
            print(f"  {r:>5.0%}{band:>11.4f}   (한쪽이 비어 건너뜀)")
            continue

        acc_keep = float(np.mean(pred[keep] == ty[keep]))
        # (1) 보류 집단을 억지로 판정했을 때. 우연 수준이어야 규칙이 제 일을 한 것이다.
        acc_hold = float(np.mean(pred[hold] == ty[hold]))
        # (2) 보류가 한쪽 클래스로 쏠리는가.
        prev_hold = float(np.mean(ty[hold]))

        rows.append(dict(rate=r, band=band, n_hold=int(hold.sum()),
                         real=hold.mean(), acc_keep=acc_keep,
                         delta=acc_keep - base, acc_hold=acc_hold,
                         prev_hold=prev_hold))
        print(f"  {r:>5.0%}{band:>11.4f}{hold.mean()*100:>8.1f}%{acc_keep:>13.4f}"
              f"{acc_keep - base:>+10.4f}{acc_hold:>15.4f} "
              f"{prev_hold:>15.1%}")

    # ── 판정 ──
    print()
    for row in rows:
        tag = f"  보류율 {row['real']:.0%} ({row['n_hold']}명):"
        # 보류 집단이 우연 수준(±10%p)이면 "모델이 모르는 사람"을 고른 것이다
        if row["acc_hold"] <= 0.60:
            v1 = f"보류집단 {row['acc_hold']:.0%} — 우연 수준. 규칙이 모르는 사람을 골랐다"
        elif row["acc_hold"] < base:
            v1 = f"보류집단 {row['acc_hold']:.0%} — 남은 쪽보다 낮지만 우연보다는 높다"
        else:
            v1 = f"⚠️ 보류집단 {row['acc_hold']:.0%} — 맞힐 사람을 버리고 있다"
        # 보류 집단 PD 비율이 전체와 15%p 넘게 벌어지면 편향이다
        gap = row["prev_hold"] - prev
        v2 = ("클래스 쏠림 없음" if abs(gap) <= 0.15
              else f"⚠️ PD 비율이 전체보다 {gap:+.0%}p — {'환자' if gap > 0 else '정상'} 쪽이 걸러진다")
        print(f"{tag} {v1}")
        print(f"  {'':>{len(tag)-2}}  {v2}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="판정 보류 검증")
    ap.add_argument("--probs", nargs="*", default=None)
    ap.add_argument("--rates", nargs="*", type=float, default=list(RATES))
    args = ap.parse_args()

    files = args.probs or sorted(
        p for p in (ROOT / "results").glob("probs_*.npz") if "smoke" not in p.name)
    if not files:
        raise SystemExit("확률 파일이 없습니다. train.py --save-probs 로 먼저 만드세요.")

    agg = defaultdict(list)
    for f in files:
        rows = check(f, args.rates)
        if rows:
            modal = Path(f).name.split("_")[1]
            for r in rows:
                agg[(modal, r["rate"])].append(r)
        print()

    # ── (3) 시드·모달에 걸쳐 재현되는가 ──
    print("=" * 78)
    print("종합 — 시드에 걸쳐 재현되는가")
    print("=" * 78)
    print(f"  {'모달':>6}{'목표':>7}{'시드':>5}{'보류율':>9}{'기준대비':>11}{'보류집단 정확도':>17}")
    for (modal, rate), rs in sorted(agg.items()):
        dl = np.array([r["delta"] for r in rs])
        hd = np.array([r["acc_hold"] for r in rs])
        rl = np.array([r["real"] for r in rs])
        sign = "모두 +" if (dl > 0).all() else ("모두 -" if (dl < 0).all() else "엇갈림")
        print(f"  {modal:>6}{rate:>6.0%}{len(rs):>5}{rl.mean()*100:>8.1f}%"
              f"{dl.mean():>+9.4f} {sign:>7}{hd.mean():>10.4f}"
              f"  ({hd.min():.2f}~{hd.max():.2f})")
    print("\n  기준대비가 모든 시드에서 +여야 보류가 재현된다고 말할 수 있다.")
    print("  보류집단 정확도가 0.5 근처여야 '모르는 사람을 골랐다'는 주장이 선다.")


if __name__ == "__main__":
    main()
