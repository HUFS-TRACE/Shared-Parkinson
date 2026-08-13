"""
음성 npz에서 과제(task) 부분집합만 뽑는다.

왜 필요한가
    voice_windows_1.5s.npz에는 모음(a) 9,500 조각과 DDK(ddk) 1,445 조각이 함께 들어
    있는데, train.py는 task를 무시하고 전부 학습한다(utils/dataset.py). 작업지시서
    §2-3은 "a만 사용, ddk 제외"였으므로 그 조건으로 실험하려면 걸러내야 한다.

    데이터를 새로 만드는 것이 아니라 이미 있는 npz에서 행을 고르는 것뿐이라,
    파일을 따로 저장해 두는 대신 이 스크립트를 남긴다. 같은 결과를 5초에 얻는다.

⚠️ 다만 ddk를 빼는 것이 정답은 아니다
    실측하면 ddk를 포함한 쪽이 낫다(seed 3개).

        지표      a만(9,500)   ddk 포함(10,945)   차이      t
        윈도우    .6190        .6726              +5.4%p   2.34
        피험자    .6174        .6756              +5.8%p   2.19
        AUC       .7131        .7352              +2.2%p   2.75

    지시서는 "발화 성격이 달라 섞으면 교란된다"고 보았으나, 채널 독립 구조라
    모델 안에서 뒤엉키지 않고 데이터가 15% 늘어난 효과가 더 컸던 것으로 보인다.
    이 스크립트는 지시서 조건을 재현하기 위한 것이지 권장 설정이 아니다.

실행
    python preprocessing/voice/filter_task.py                    # a만 (지시서 조건)
    python preprocessing/voice/filter_task.py --tasks a,ddk      # 전부 (기본 npz와 동일)
    python preprocessing/voice/filter_task.py --tasks ddk        # DDK만
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset/processed/voice/voice_windows_1.5s.npz"


def main():
    ap = argparse.ArgumentParser(description="음성 npz 과제별 필터")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--tasks", default="a", help="쉼표 구분. 예: a / a,ddk / ddk")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"입력 없음: {args.src}\n  git lfs pull 로 먼저 받으세요.")

    keep = [t.strip() for t in args.tasks.split(",")]
    d = np.load(args.src, allow_pickle=True)
    m = np.isin(d["task"], keep)
    if not m.any():
        raise SystemExit(f"task {keep}에 해당하는 조각이 없습니다. "
                         f"(있는 값: {sorted(set(d['task'].tolist()))})")

    X, y, sid, task = d["X"][m], d["y"][m], d["subject_id"][m], d["task"][m]
    out = args.out or args.src.with_name(
        f"{args.src.stem}_{'-'.join(keep)}.npz")

    print(f"입력  {args.src.name}  {d['X'].shape}  "
          f"task {dict(Counter(d['task'].tolist()))}")
    print(f"선택  {keep}  →  {X.shape}  라벨 {np.bincount(y).tolist()} (0=HC, 1=PD)")

    # 화자가 줄었는지 — 특정 과제만 수행한 사람이 있으면 통째로 빠진다
    n_all, n_sel = len(np.unique(d["subject_id"])), len(np.unique(sid))
    print(f"화자  {n_all} → {n_sel}명"
          + (f"  ⚠️ {n_all - n_sel}명이 빠졌습니다" if n_sel < n_all else ""))

    bad = [s for s in np.unique(sid) if len(np.unique(y[sid == s])) > 1]
    if bad:
        raise SystemExit(f"❌ 한 화자가 두 라벨을 가집니다: {bad[:5]} — 원본을 확인하세요.")
    print(f"라벨 충돌 0건")

    np.savez_compressed(out, X=X, y=y, subject_id=sid, task=task)
    print(f"\n저장  {out}  ({out.stat().st_size/1e6:.0f} MB)")
    # 레포 밖 경로면 relative_to가 터진다 — 그때는 절대 경로를 그대로 쓴다
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"사용  python train.py --modalities voice --voice-path {shown}")


if __name__ == "__main__":
    main()
