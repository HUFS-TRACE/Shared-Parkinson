"""
센서(채널) 축 결과 집계 — 몇 개를 켜야 하고, 그 이득이 재현되는가.

왜 AUC를 먼저 보나
    피험자 정확도는 61명이라 한 명이 1.6%p로 양자화된다. 채널을 하나 더 켰을 때
    생기는 진짜 변화보다 이 계단이 커서, 순위가 쉽게 뒤집힌다. AUC는 그 양자화가
    없어 채널 축의 계단을 훨씬 안정적으로 드러낸다.

무엇을 확인하나
    (1) 채널 수를 늘릴수록 좋아지는가, 어디서 포화되는가
    (2) 전체(6채널)를 이기는 부분집합이 있는가 — 있다면 과적합의 증거다
    (3) 그 순서가 seed에 걸쳐 재현되는가

실행
    python analyze_channel.py                 # 중요도순
    python analyze_channel.py --reverse       # 역순 대조군
"""
import argparse
import csv
import glob
import math
import statistics as st
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CH_NAME = {0: "마이크", 1: "그립", 2: "압력", 3: "TiltX", 4: "TiltY", 5: "TiltZ"}
CH_ORDER = [3, 2, 1, 4, 0, 5]
MFLOPS = {1: 18.2, 2: 36.3, 3: 54.5, 4: 72.7, 5: 90.9, 6: 109.0}
KEYS = (("roc_auc", "AUC"), ("subject_acc", "피험자"), ("window_acc", "윈도우"))


def load(tag):
    """{채널 수: {seed: row}}"""
    out = {}
    for f in glob.glob(str(ROOT / f"results/ch*{tag}_d64L6_s*.csv")):
        name = Path(f).stem                       # ch3_d64L6_s42
        k = int(name.split("_")[0][2:].rstrip("r"))
        s = int(name.split("_s")[-1])
        out.setdefault(k, {})[s] = next(csv.DictReader(open(f, encoding="utf-8")))
    return out


def cell(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f}±{st.stdev(vals):.3f}"


def main():
    ap = argparse.ArgumentParser(description="센서 축 결과 집계")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()
    tag = "r" if args.reverse else ""
    order = CH_ORDER[::-1] if args.reverse else CH_ORDER

    data = load(tag)
    if not data:
        raise SystemExit(f"results/ch*{tag}_d64L6_s*.csv 가 없습니다. "
                         f"channel_exit.py 를 먼저 돌리세요.")
    seeds = sorted({s for v in data.values() for s in v})

    print("=" * 78)
    print(f"센서 축 — {'역순' if args.reverse else '중요도순'} 누적  (seed {seeds})")
    print("=" * 78)
    print(f"  {'채널':>4}{'추가 센서':<10}{'MFLOPs':>8}{'절감':>7}"
          + "".join(f"{n:>16}" for _, n in KEYS))
    agg = {}
    for k in sorted(data):
        rows = data[k]
        vals = {key: [float(r[key]) for r in rows.values()] for key, _ in KEYS}
        agg[k] = vals
        print(f"  {k:>4}{CH_NAME[order[k-1]]:<10}{MFLOPS[k]:>8.1f}"
              f"{(1 - MFLOPS[k]/MFLOPS[6])*100:>6.0f}%"
              + "".join(f"{cell(vals[key]):>16}" for key, _ in KEYS)
              + f"   (n={len(rows)})")

    if 6 not in agg:
        return
    full = agg[6]

    # ── 전체를 이기는 부분집합이 있는가 ──
    print(f"\n  전체(6채널) 대비  —  같은 seed끼리 짝지어 비교")
    print(f"  {'채널':>4}{'절감':>7}" + "".join(f"{n + ' 차이':>14}{'t':>7}" for _, n in KEYS))
    for k in sorted(agg):
        if k == 6:
            continue
        line = f"  {k:>4}{(1 - MFLOPS[k]/MFLOPS[6])*100:>6.0f}%"
        for key, _ in KEYS:
            d = [float(data[k][s][key]) - float(data[6][s][key])
                 for s in seeds if s in data[k] and s in data[6]]
            if len(d) < 2:
                line += f"{(d[0]*100 if d else 0):>+13.1f}%p{'—':>7}"
                continue
            m, sd = st.mean(d), st.stdev(d)
            t = m / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
            star = "★" if abs(t) > 2.78 else ""
            line += f"{m*100:>+13.1f}%p{t:>6.2f}{star}"
        print(line)
    print(f"\n  ★ = |t| > 2.78 (n=3, 양측 α=0.05).  +면 부분집합이 전체보다 낫다.")

    # ── 포화 지점 ──
    auc = {k: st.mean(v["roc_auc"]) for k, v in agg.items()}
    best = max(auc, key=auc.get)
    print(f"\n  {'─'*74}")
    print(f"  AUC 최고: {best}채널 ({auc[best]:.4f}) · "
          f"연산 {MFLOPS[best]:.1f} MFLOPs ({(1-MFLOPS[best]/MFLOPS[6])*100:.0f}% 절감)")
    if best < 6:
        print(f"  → 전체(6채널 {auc[6]:.4f})보다 {(auc[best]-auc[6])*100:+.1f}%p 높다. "
              f"채널을 더 켜는 것이 손해라는 뜻이고, 61명 규모에서 과적합의 신호다.")

    # 계단이 어디서 생기나
    ks = sorted(auc)
    jumps = [(ks[i], auc[ks[i]] - auc[ks[i-1]]) for i in range(1, len(ks))]
    big = max(jumps, key=lambda x: x[1])
    print(f"  가장 큰 상승: {big[0]-1}→{big[0]}채널에서 {big[1]*100:+.1f}%p "
          f"({CH_NAME[order[big[0]-1]]} 추가)")

    # ── seed별 최적점이 흩어지는가 ──
    print(f"\n  seed별 AUC 최고 채널 수")
    for s in seeds:
        per = {k: float(data[k][s]["roc_auc"]) for k in sorted(data) if s in data[k]}
        if per:
            b = max(per, key=per.get)
            print(f"    seed {s:<4} {b}채널 ({per[b]:.4f})")
    print(f"\n  최적점이 seed마다 다르면 '몇 채널을 쓰라'고 권장할 수 없다.")


if __name__ == "__main__":
    main()
