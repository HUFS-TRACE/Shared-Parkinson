"""캐스케이드 — 센서 축으로 Early Exit을 만들 수 있는가.

왜 이걸 재나
    센서 축(3채널이면 6채널과 동등)은 절감이 성립한 유일한 축이지만, **Early Exit이
    아니다.** 학습 전에 채널을 잘라 모든 사람에게 똑같이 3채널을 쓰므로 입력마다
    계산이 달라지지 않는다. 그건 model compression이지 dynamic inference가 아니다.

    이것을 Early Exit으로 바꾸는 자연스러운 방법이 캐스케이드다.

        1단계  3채널로 판정한다                     연산 0.5x
        2단계  애매한 사람만 6채널로 다시 본다       그 사람들에 한해 +1.0x

    입력마다 계산량이 달라지므로 이건 진짜 Early Exit이다. 성립하려면 조건이 하나 있다 —
    **6채널이 3채널이 어려워한 사람을 더 잘 풀어야 한다.** 그래야 올려보낼 이유가 있다.

무엇을 맞대나
    3채널 단독 · 6채널 단독 · 캐스케이드. 셋 다 같은 피험자·같은 fold다
    (GroupKFold가 결정적이라 시드가 달라도 test 명단이 같다 — 이슈 #15).

임계값
    0.5로 자르면 시드마다 확률 눈금이 밀려 민감도가 0.35~0.96으로 흔들린다
    (analyze_threshold.py 참고). 그래서 학습 유병률(26/61)로 임계값을 맞춘다.

실행
    python analyze_cascade.py
    python analyze_cascade.py --escalate 0.2 0.3 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_abstain_bias import per_subject  # noqa: E402
from sequential_exit import calibrate, logit, sigmoid  # noqa: E402
from verify_abstain import fit_band  # noqa: E402

PREV = 26 / 61          # 학습 데이터의 환자 비율. test를 보고 정한 값이 아니다
COST = {"3ch": 0.5, "6ch": 1.0}     # 채널 수에 거의 정비례한다(channel-independent 패칭)


def load(path):
    d = np.load(path, allow_pickle=True)
    T, c = calibrate(d)
    p = sigmoid(logit(d["prob"]) / T - c)
    mp, ty, _ = per_subject(p, d["y"], d["subject_id"], d["idx"])
    return d, mp, ty, np.unique(d["subject_id"])


def main():
    ap = argparse.ArgumentParser(description="3채널 → 6채널 캐스케이드")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 7])
    ap.add_argument("--escalate", nargs="*", type=float, default=[0.2, 0.3, 0.5],
                    help="2단계로 올려보낼 비율")
    ap.add_argument("--three", default="results/probs_hw_sw_ab_hw3_s{seed}.npz")
    ap.add_argument("--six", default="results/probs_hw_sw_class_weight_cwwindow_s{seed}.npz")
    args = ap.parse_args()

    rows = {f: [] for f in args.escalate}
    solo = []
    hard = {f: [] for f in args.escalate}

    for s in args.seeds:
        p3, p6 = ROOT / args.three.format(seed=s), ROOT / args.six.format(seed=s)
        if not (p3.exists() and p6.exists()):
            raise SystemExit(f"확률 파일이 없습니다: {p3.name} / {p6.name}")
        _, m3, y3, s3 = load(p3)
        _, m6, y6, s6 = load(p6)
        if not ((s3 == s6).all() and (y3 == y6).all()):
            raise SystemExit("두 모델의 피험자가 다릅니다 — 같은 fold여야 비교됩니다")

        t3, t6 = np.quantile(m3, 1 - PREV), np.quantile(m6, 1 - PREV)
        pred3, pred6 = (m3 >= t3).astype(int), (m6 >= t6).astype(int)
        solo.append([(pred3 == y3).mean(), (pred6 == y3).mean()])

        margin = np.abs(m3 - t3)
        for f in args.escalate:
            up = margin < np.quantile(margin, f)      # 3채널이 애매해한 사람
            casc = np.where(up, pred6, pred3)
            rows[f].append([(casc == y3).mean(), COST["3ch"] + f * COST["6ch"]])
            # 올려보낸 사람만 떼어 본다 — 6채널이 정말 더 잘 푸는가
            hard[f].append([(pred3[up] == y3[up]).mean(), (pred6[up] == y3[up]).mean()])

    a = np.array(solo)
    print("=" * 70)
    print(f"단독  (시드 {args.seeds} · 임계값은 유병률 {PREV:.1%}로 맞춤)")
    print("=" * 70)
    print(f"  3채널  정확도 {a[:, 0].mean():.4f}   연산 {COST['3ch']:.2f}x")
    print(f"  6채널  정확도 {a[:, 1].mean():.4f}   연산 {COST['6ch']:.2f}x")

    print("\n" + "=" * 70)
    print("캐스케이드 — 3채널로 먼저 보고, 애매한 사람만 6채널로")
    print("=" * 70)
    print(f"  {'승급':>6}{'정확도':>10}{'연산':>9}{'3채널 대비':>12}{'6채널 대비':>12}")
    for f in args.escalate:
        r = np.array(rows[f])
        print(f"  {f:>5.0%}{r[:, 0].mean():>10.4f}{r[:, 1].mean():>8.2f}x"
              f"{r[:, 0].mean() - a[:, 0].mean():>+12.4f}"
              f"{r[:, 0].mean() - a[:, 1].mean():>+12.4f}")

    print("\n" + "=" * 70)
    print("올려보낸 사람만 떼어 보면 — 6채널이 정말 더 잘 푸는가")
    print("=" * 70)
    print(f"  {'승급':>6}{'3채널':>10}{'6채널':>10}{'차이':>10}")
    gains = []
    for f in args.escalate:
        h = np.array(hard[f])
        g = h[:, 1].mean() - h[:, 0].mean()
        gains.append(g)
        print(f"  {f:>5.0%}{h[:, 0].mean():>10.3f}{h[:, 1].mean():>10.3f}{g:>+10.3f}")

    print()
    if max(gains) <= 0.02:
        print("  → 6채널이 어려운 사람을 더 잘 푸는 것이 아니다. **올려보낼 곳이 없다.**")
        print("     캐스케이드가 성립하려면 2단계가 1단계의 실패를 건져야 하는데,")
        print("     여기서는 두 모델이 같은 사람에게서 같이 틀린다.")
    else:
        print(f"  → 6채널이 어려운 사람에서 {max(gains):+.3f} 낫다. 캐스케이드에 근거가 있다.")

    best = min(args.escalate, key=lambda f: -np.array(rows[f])[:, 0].mean())
    if np.array(rows[best])[:, 0].mean() <= a[:, 0].mean():
        print("\n  ⚠️ 어떤 승급 비율에서도 3채널 단독을 못 이긴다.")
        print("     3채널이 더 정확하면서 더 싸므로(0.50x), 캐스케이드를 쓸 이유가 없다.")


if __name__ == "__main__":
    main()
