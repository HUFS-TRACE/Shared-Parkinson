"""PhysioNet 2019 패혈증 ragged npz → 우리 학습이 읽는 윈도우 npz.

민규가 만든 전처리 산출물을 변환한다. 원본은 환자별 시계열을 세로로 이어붙인
ragged 배열이라 우리 load_modality()가 요구하는 (N, C, T) 고정 길이가 아니다.

    X (1,552,210, 102)   환자 i 구간 = X[offsets[i]:offsets[i+1]]
    y (1,552,210,)       시간 단위 라벨
    split (40,336,)      0=train 1=val 2=test  (환자 단위, 층화됨)

윈도우와 라벨을 어떻게 정하나
    시각 t에서 **직전 W시간**을 입력으로 보고 **그 시각의 라벨** y[t]를 맞힌다.
    패혈증 조기 예측의 표준 구성이다. 관측 길이가 W보다 짧은 앞부분은 버린다.

        입력  X[t-W+1 : t+1]      (W, 102) -> 전치해 (102, W)
        라벨  y[t]

    stride로 t를 건너뛴다. stride 1이면 155만 윈도우라 그대로는 못 돌린다.

민규가 밝힌 조건 셋을 그대로 지킨다
    1. split 배열을 그대로 쓴다. 정규화 통계를 그 train으로 뽑았으므로 다시
       나누면 누수다. --keep-split 이 기본이고, 학습은 --fixed-split 으로 돈다.
    2. RevIN을 꺼야 한다. 이미 z-score 돼 있어 윈도우마다 평균을 다시 빼면
       "심박 130과 70"의 차이가 사라진다. 학습에서 --no-revin 을 쓴다.
    3. 시간 단위 양성이 1.8%다. class_weight로 다루되 정확도는 보지 않는다.

우리 기준으로 다시 하려면(B안)
    --renorm 을 주면 값 34채널을 원본 스케일로 되돌린다(X*std+mean). 그러면
    fold별로 다시 정규화할 수 있어 GroupKFold x 시드 5개가 성립한다.
    마스크·경과시간 68채널은 정규화와 무관하므로 건드리지 않는다.
    ⚠️ 클리핑(1.0 퍼센타일)은 되돌릴 수 없다.

실행
    python sepsis_to_npz.py --hours 24
    python sepsis_to_npz.py --hours 12 --stride 6
    python sepsis_to_npz.py --hours 24 --renorm --out-suffix _raw
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC = Path.home() / "Downloads" / "sepsis_physionet2019.npz"
DST = ROOT / "dataset" / "processed" / "sepsis"


def main():
    ap = argparse.ArgumentParser(description="패혈증 ragged npz → 윈도우 npz")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--hours", type=int, default=24, help="윈도우 길이(시간)")
    ap.add_argument("--stride", type=int, default=12, help="시각을 몇 시간씩 건너뛰나")
    ap.add_argument("--max-per-subject", type=int, default=4,
                    help="환자당 윈도우 상한. 긴 환자가 학습을 지배하는 것을 막는다")
    ap.add_argument("--max-subjects", type=int,
                    help="환자 상한. 분할 비율을 유지한 채 줄인다")
    ap.add_argument("--renorm", action="store_true",
                    help="값 34채널을 원본 스케일로 되돌린다(B안용)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    z = np.load(args.src, allow_pickle=True)
    X, y, off, split = z["X"], z["y"], z["offsets"], z["split"]
    pid, ch = z["patient_ids"], z["channel_names"]
    rng = np.random.default_rng(args.seed)
    W, S = args.hours, args.stride

    keep = np.arange(len(split))
    if args.max_subjects and args.max_subjects < len(keep):
        # 분할 비율을 유지한다. 통째로 무작위로 뽑으면 train/val/test가 흔들린다.
        sel = []
        for c in (0, 1, 2):
            idx = np.nonzero(split == c)[0]
            n = round(args.max_subjects * len(idx) / len(split))
            sel.append(rng.permutation(idx)[:n])
        keep = np.sort(np.concatenate(sel))
        print(f"환자 {len(split):,} → {len(keep):,}명 (분할 비율 유지)")

    Xw, yw, sw, spw = [], [], [], []
    short = 0
    for i in keep:
        a, b = off[i], off[i + 1]
        n = b - a
        if n < W:
            short += 1
            continue
        # 시각 t는 W-1부터. 뒤에서부터 stride로 뽑아 최근 구간을 우선 남긴다.
        ts = np.arange(n - 1, W - 2, -S)[:args.max_per_subject][::-1]
        for t in ts:
            Xw.append(X[a + t - W + 1: a + t + 1])
            yw.append(y[a + t])
        sw += [str(pid[i])] * len(ts)
        spw += [int(split[i])] * len(ts)

    Xw = np.stack(Xw).astype(np.float32).transpose(0, 2, 1)   # (N, C, T)
    yw = np.asarray(yw, dtype=np.int64)
    sw = np.asarray(sw)
    spw = np.asarray(spw, dtype=np.int8)

    if args.renorm:
        # 값 34채널만 되돌린다. 마스크·경과시간은 정규화 대상이 아니었다.
        m, sd = z["norm_mean"].astype(np.float32), z["norm_std"].astype(np.float32)
        Xw[:, :34, :] = Xw[:, :34, :] * sd[None, :, None] + m[None, :, None]
        print("값 34채널을 원본 스케일로 되돌렸다 (클리핑은 복원 불가)")

    args.dst.mkdir(parents=True, exist_ok=True)
    out = args.dst / f"sepsis_{W}h{args.out_suffix}.npz"
    np.savez_compressed(out, X=Xw, y=yw, subject_id=sw, split=spw,
                        channel_names=ch)

    print(f"\n[{out.name}]  ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"  X {Xw.shape} · 환자 {len(np.unique(sw)):,}명 · 길이 부족으로 버림 {short:,}명")
    print(f"  양성 {yw.mean():.2%}  ·  불균형 {(1 - yw.mean()) / max(yw.mean(), 1e-9):.0f}:1")
    print(f"  {'split':>8}{'환자':>9}{'윈도우':>9}{'양성률':>9}")
    for c, nm in ((0, "train"), (1, "val"), (2, "test")):
        mk = spw == c
        print(f"  {nm:>8}{len(np.unique(sw[mk])):>9,}{mk.sum():>9,}{yw[mk].mean():>9.2%}")
    print(f"\n  토큰 수: patch/stride를 p/s로 두면 ({W}-p)//s+1")
    for p, st in ((2, 1), (4, 2), (6, 3)):
        if W >= p:
            print(f"    patch {p} stride {st} → {(W - p) // st + 1}개")


if __name__ == "__main__":
    main()
