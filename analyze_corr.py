"""채널 간 상관 — 채널을 독립으로 다뤄도 되는가.

왜 재나
    PatchEmbedding이 채널을 배치 축에 접는다(channel-independent).

        x.reshape(b * c, num_patches, patch_len)

    이 순간부터 인코더는 채널 6개를 "서로 무관한 시계열 6개"로 본다. 채널이 다시
    만나는 곳은 마지막 분류기 하나뿐이다. 채널들이 실제로 서로 무관하다면 잃는 것이
    없지만, 강하게 얽혀 있다면 그 구조를 인코더가 통째로 못 본다.

    회의에서 나온 지적이 이것이고, 판단은 데이터가 해야 한다.

무엇을 기준으로 판단하나
    윈도우 하나 안에서 채널끼리의 피어슨 상관을 재고, 윈도우 전체에 걸쳐 평균한다.
    부호는 상쇄되므로 절대값을 쓴다.

        평균 |r| 이 낮고 0.3 넘는 쌍이 드물다   → 독립 취급이 정당하다
        평균 |r| 이 높고 0.3 넘는 쌍이 흔하다   → 인코더가 놓치는 구조가 있다

    0.3은 관례적인 "약한 상관"의 하한이다. 절대 기준이라기보다 두 모달을 같은 자로
    재기 위한 눈금이다.

음성은 인접 밴드를 따로 본다
    멜 밴드는 주파수 순으로 늘어서 있어 이웃끼리 겹친다. 포먼트 하나가 여러 밴드에
    동시에 걸리므로 인접 상관이 높은 것은 당연하다. 문제는 그것이 "당연해서 괜찮다"가
    아니라, 그만큼 **중복된 채널을 80개나 인코더에 밀어 넣고 있다**는 뜻이라는 데 있다.

실행
    python analyze_corr.py
    python analyze_corr.py --max-windows 2000     # 표본을 줄여 빠르게
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
STRONG = 0.3          # "약한 상관"의 관례적 하한. 두 모달을 같은 자로 재기 위한 눈금


def mean_abs_corr(X, rng, max_windows):
    """[N, C, T] → 윈도우 평균 |상관| 행렬 [C, C].

    윈도우마다 상관을 내고 나서 평균한다. 전체를 이어붙여 한 번에 내면 사람·과제가
    섞여 "같은 사람 안에서 채널이 함께 움직이는가"라는 물음이 흐려진다.
    """
    n = len(X)
    take = rng.choice(n, min(max_windows, n), replace=False)
    C = X.shape[1]
    acc = np.zeros((C, C))
    used = 0
    for i in take:
        x = X[i].astype(np.float64)
        sd = x.std(axis=1)
        if (sd < 1e-8).any():          # 상수 채널이 있으면 상관이 정의되지 않는다
            continue
        acc += np.abs(np.corrcoef(x))
        used += 1
    if not used:
        raise SystemExit("상관을 낼 수 있는 윈도우가 없습니다(상수 채널만 존재).")
    return acc / used, used


def report(name, X, rng, max_windows, labels=None):
    R, used = mean_abs_corr(X, rng, max_windows)
    C = R.shape[0]
    iu = np.triu_indices(C, k=1)         # 대각과 아래 삼각은 중복이라 뺀다
    off = R[iu]

    print("=" * 74)
    print(f"{name}  채널 {C}개 · 윈도우 {used}개 표본 (전체 {len(X)}개)")
    print("=" * 74)
    print(f"  평균 |r| {off.mean():.3f}   중앙 {np.median(off):.3f}   최대 {off.max():.3f}")
    n_strong = int((off > STRONG).sum())
    print(f"  |r| > {STRONG} 인 쌍: {n_strong}/{len(off)} ({n_strong / len(off) * 100:.1f}%)")

    if C <= 10:
        head = labels or [f"ch{i}" for i in range(C)]
        print("\n  " + "".join(f"{h:>8}" for h in [""] + head))
        for i in range(C):
            print(f"  {head[i]:>8}" + "".join(
                f"{R[i, j]:>8.3f}" if i != j else f"{'—':>8}" for j in range(C)))
        k = np.argmax(off)
        a, b = iu[0][k], iu[1][k]
        print(f"\n  가장 얽힌 쌍: {head[a]} ↔ {head[b]}  |r|={off.max():.3f}")
    else:
        # 밴드가 많으면 행렬을 다 못 보여준다. 거리별로 요약한다.
        print("\n  밴드 간 거리별 평균 |r|")
        for d in (1, 2, 4, 8, 16, 32):
            if d >= C:
                break
            v = np.array([R[i, i + d] for i in range(C - d)])
            print(f"    거리 {d:>2}: {v.mean():.3f}")

    # ── 판정 ──
    if off.mean() < 0.10 and n_strong / len(off) < 0.05:
        print(f"\n  → 채널 독립 취급이 데이터로 뒷받침된다 "
              f"(평균 {off.mean():.3f}, 강한 쌍 {n_strong / len(off) * 100:.1f}%)")
    else:
        print(f"\n  ⚠️ 채널이 서로 얽혀 있다 (평균 {off.mean():.3f}, "
              f"강한 쌍 {n_strong / len(off) * 100:.1f}%)")
        print(f"     인코더는 채널을 따로 보므로 이 구조를 못 쓴다. "
              f"분류기 하나가 이걸 다 감당하는 셈이다.")
    return off.mean(), n_strong / len(off)


def main():
    ap = argparse.ArgumentParser(description="채널 간 상관 측정")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    ap.add_argument("--max-windows", type=int, default=3000,
                    help="표본 윈도우 수. 전체를 다 돌 필요는 없다")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    # 필기 채널 이름 — analyze_channel.py의 중요도 순서와 같은 인덱스다
    HW = ["마이크", "그립", "압력", "TiltX", "TiltY", "TiltZ"]

    out = {}
    for m, labels in (("hw", HW), ("voice", None)):
        path = ROOT / cfg["data"][f"{m}_path"]
        if not path.exists():
            path = Path(cfg["data"][f"{m}_path"])
        if not path.exists():
            print(f"[{m}] {path} 없음 — 건너뜁니다\n")
            continue
        X = np.load(path, allow_pickle=True)["X"]
        lab = labels if (labels and X.shape[1] == len(labels)) else None
        out[m] = report(m, X, rng, args.max_windows, lab)
        print()

    if len(out) == 2:
        a, b = out["hw"][0], out["voice"][0]
        print("=" * 74)
        print(f"두 모달 비교: 필기 {a:.3f}  vs  음성 {b:.3f}  ({b / a:.1f}배)")
        print("=" * 74)
        print("  같은 인코더를 쓰는데 한쪽만 독립 가정이 맞는다면, 그 인코더가")
        print("  두 모달에 똑같이 적합하다고 말할 수 없다.")


if __name__ == "__main__":
    main()
