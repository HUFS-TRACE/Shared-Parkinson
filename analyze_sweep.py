"""스윕 결과 집계 — 설정을 바꾼 것이 시드 노이즈보다 큰가.

왜 시드를 짝지어 보나
    같은 설정을 시드만 바꿔 돌려도 피험자 정확도가 1.0~2.7%p 흔들린다. 설정 하나를
    바꾼 뒤 정확도가 2%p 올랐다고 해서 그 설정이 나은 것이 아니다.

    시드를 짝지으면(같은 시드끼리 기준-실험을 뺀다) 시드가 만든 흔들림이 상쇄된다.
    남는 것이 설정의 효과다. 그래서 차이의 부호가 시드마다 일관되는지를 먼저 본다.

        3시드 모두 같은 부호  → 설정 효과라고 말할 수 있다
        부호가 엇갈림        → 노이즈와 구별되지 않는다

    t값도 같이 낸다. 시드 3개면 자유도 2라 |t| > 4.30이어야 α=0.05에서 유의하다.
    표본이 3개뿐이라 t는 참고용이고, 부호 일관성이 더 실질적인 근거다.

왜 AUC를 함께 보나
    피험자 정확도는 사람 수로 양자화된다(필기 61명이면 1명 = 1.6%p, 음성 231명이면
    0.4%p). 필기는 이 계단이 커서 작은 변화가 안 보이거나 과장된다. AUC는 그 계단이
    없어 방향을 더 안정적으로 드러낸다. 둘이 어긋나면 결론을 보류한다.

실행
    python analyze_sweep.py                    # results/sw_*.csv 전부
    python analyze_sweep.py --axis voice_stride
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
T_CRIT = 4.303          # 자유도 2, 양측 α=0.05
METRICS = (("subject_acc_soft", "피험자"), ("roc_auc", "AUC"), ("window_acc", "윈도우"))

# run_sweep.py의 BASELINE과 같아야 한다
BASELINE = {
    "base": None,          # 기준선 자체 — 맞댈 대상이 없다
    "base_v1": None,
    "d_model": "hw-voice_d64L6_s{seed}.csv",
    "hw_patch": "hw_d64L6_s{seed}.csv",
    "hw_stride": "hw_d64L6_s{seed}.csv",
    "voice_patch": "sw_voice_mel_mel20_s{seed}.csv",
    "voice_stride": "voice_d64L6_s{seed}.csv",
    "head": "hw_d64L6_s{seed}.csv",
    "window_cap": "hw_d64L6_s{seed}.csv",
    "voice_mel": "voice_d64L6_s{seed}.csv",
    "ab": "hw_d64L6_s{seed}.csv",
    "class_weight": "hw_d64L6_s{seed}.csv",
    "depth": "hw_d64L6_s{seed}.csv",
    "d_model_v2": "sw_shared_v2_s9mel20_s{seed}.csv",
    "voice_v2": "voice_d64L6_s{seed}.csv",
    "shared_v2": "hw-voice_d64L6_s{seed}.csv",
}
# 축이 어느 모달을 건드리는가 — 기준 CSV에 두 모달이 다 있을 때 고르기 위해
AXIS_MODAL = {"hw_patch": "hw", "hw_stride": "hw", "head": "hw",
              "window_cap": "hw", "ab": "hw", "voice_v2": "voice", "depth": "hw",
              "class_weight": "hw",
              "voice_patch": "voice", "voice_stride": "voice", "voice_mel": "voice"}


def read(path, modality=None):
    """CSV → {모달: 행}. modality를 주면 그 모달 행만."""
    rows = {r["modality"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    if modality:
        return rows.get(modality)
    return rows


def paired_t(d):
    """차이 배열 → t값. 표준편차가 0이면 t가 정의되지 않는다."""
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def collect(tag="sw"):
    """<tag>_<축>_<설정>_s<시드>.csv 를 (축, 설정) → {시드: 경로} 로 모은다."""
    got = defaultdict(dict)
    for f in sorted(RES.glob(f"{tag}_*.csv")):
        stem = f.stem[len(tag) + 1:]
        head, _, seed = stem.rpartition("_s")
        for ax in sorted(BASELINE, key=len, reverse=True):
            if head.startswith(ax + "_"):
                got[(ax, head[len(ax) + 1:])][int(seed)] = f
                break
    return got


def main():
    ap = argparse.ArgumentParser(description="스윕 결과 집계")
    ap.add_argument("--axis", nargs="*", default=None)
    ap.add_argument("--tag", default="sw",
                    help="읽을 결과 파일 접두사. run_sweep.py의 --tag와 같아야 한다")
    ap.add_argument("--rebase", action="store_true",
                    help="비교 기준을 base 축 산출물로 본다. run_sweep --rebase 와 짝")
    args = ap.parse_args()

    got = collect(args.tag)
    # 분할을 바꿨으면 옛 기준선 CSV를 쓸 수 없다. base 축 산출물로 갈아탄다.
    if args.rebase:
        from run_sweep import REBASE
        for ax, base in REBASE.items():
            BASELINE[ax] = f"{args.tag}_{base}_s{{seed}}.csv"
    if not got:
        raise SystemExit(f"results/{args.tag}_*.csv 가 없습니다.")

    axes = sorted({ax for ax, _ in got} & set(args.axis or BASELINE))
    axes = [a for a in axes if BASELINE.get(a)]      # 기준선 자체(base)는 맞댈 것이 없다
    for ax in axes:
        modal = AXIS_MODAL.get(ax)
        print("=" * 84)
        print(f"[{ax}]  기준: {BASELINE[ax].format(seed='*')}"
              + (f"  · 모달 {modal}" if modal else ""))
        print("=" * 84)

        for (a, label), seeds in sorted(got.items()):
            if a != ax:
                continue
            diffs, base_v, new_v, used = defaultdict(list), defaultdict(list), defaultdict(list), []
            for seed, path in sorted(seeds.items()):
                bpath = RES / BASELINE[ax].format(seed=seed)
                if not bpath.exists():
                    continue
                # 공유 모델(d_model 축)은 두 모달이 다 나오므로 각각 본다
                mods = [modal] if modal else list(read(path))
                for m in mods:
                    b, n = read(bpath, m), read(path, m)
                    if not b or not n:
                        continue
                    for key, _ in METRICS:
                        diffs[(m, key)].append(float(n[key]) - float(b[key]))
                        base_v[(m, key)].append(float(b[key]))
                        new_v[(m, key)].append(float(n[key]))
                used.append(seed)

            if not diffs:
                print(f"  {label}: 기준 CSV가 없어 비교 불가")
                continue

            for m in sorted({k[0] for k in diffs}):
                print(f"\n  ── {label}"
                      + (f" ({m})" if not modal else "")
                      + f"  시드 {sorted(set(used))} ──")
                print(f"  {'지표':>8}{'기준':>10}{'실험':>10}{'차이':>10}"
                      f"{'부호':>10}{'t':>9}")
                for key, lab in METRICS:
                    d = np.array(diffs[(m, key)])
                    sign = ("모두 +" if (d > 0).all() else
                            "모두 -" if (d < 0).all() else "엇갈림")
                    t = paired_t(d)
                    star = " ★" if abs(t) > T_CRIT else ""
                    print(f"  {lab:>8}{np.mean(base_v[(m, key)]):>10.4f}"
                          f"{np.mean(new_v[(m, key)]):>10.4f}{d.mean():>+10.4f}"
                          f"{sign:>9}{t:>9.2f}{star}")

                acc = np.array(diffs[(m, "subject_acc_soft")])
                auc = np.array(diffs[(m, "roc_auc")])
                if (acc > 0).all() and (auc > 0).all():
                    print(f"     → 3시드 모두 정확도·AUC가 함께 올랐다. 설정 효과로 볼 만하다.")
                elif (acc < 0).all() and (auc < 0).all():
                    print(f"     → 3시드 모두 함께 내렸다. 이 설정은 나쁘다.")
                elif (acc > 0).all() != (auc > 0).all():
                    print(f"     ⚠️ 정확도와 AUC가 어긋난다 — 결론을 보류한다.")
                else:
                    print(f"     → 부호가 엇갈린다. 시드 노이즈와 구별되지 않는다.")
        print()

    print("=" * 84)
    print(f"부호가 3시드 모두 같아야 설정 효과라고 말할 수 있다. "
          f"★는 |t| > {T_CRIT} (자유도 2, α=0.05).")


if __name__ == "__main__":
    main()
