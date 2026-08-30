"""VitalDB case별 npz → 우리 학습이 읽는 모달별 npz 2개.

왜 변환이 필요한가
    vitaldb_preprocess.py는 사례마다 파일 하나를 만들고 두 모달을 한 파일에 담는다.
    우리 load_modality()는 모달별 파일 하나에서 X · y · subject_id 세 키를 읽는다.

        case_00001.npz  ecg (446,1,3750) · ppg (446,1,3750) · label · hr · subjectid
                            ↓
        ecg_windows_30s.npz    X (N,1,3750) · y (N,) · subject_id (N,)
        voice... 가 아니라 ppg_windows_30s.npz  동일 구조

subject_id에 caseid가 아니라 subjectid를 넣는다
    한 환자가 수술을 여러 번 받아 case가 여러 건인 경우가 있다(6,156건 / 5,886명).
    caseid로 나누면 같은 사람이 train과 test에 걸쳐 누수가 된다. 파킨슨에서
    신호 파일 66개를 Person_ID로 묶어 61명으로 만든 것과 같은 처리다.

split 열은 읽지 않는다
    전처리가 만든 고정 분할 하나로는 시드 5개 GroupKFold를 돌릴 수 없다(이슈 #15).
    subject_id만 남기고 분할은 학습할 때 다시 뽑는다.

라벨을 두 벌 만들 수 있다 (--label)
    both   60~100 정상 / 밖 비정상.  전처리 노트북 원본 정의.
    brady  hr >= 60 정상 / hr < 60 서맥.  경계가 하나뿐이다.

    both는 경계가 60과 100 두 개라 "쉬운 표본"(어느 경계에서도 20 이상 떨어진 것)이
    2.8%밖에 안 나온다. 심박이 80 근처여야 하는데 수술 중에는 드물기 때문이다.
    난이도를 조절해가며 비교하려면 쉬운 쪽 표본이 있어야 하는데 그게 없다.
    게다가 라벨 1에 서맥과 빈맥이 함께 들어가 결정 경계가 끊어진 두 구간이 된다.

    빈맥(hr > 100)으로 가르면 양성이 4.9%뿐이라 19:1로 치우친다. 마취 중에는
    심박이 떨어지는 쪽이 흔해서다. 서맥이 21.6%로 균형이 낫고(3.6:1) 쉬운 표본도
    2.8% -> 26.7%로 늘어난다.

hr을 함께 저장한다
    학습에는 쓰지 않는다 — 라벨이 hr에서 나오므로 입력에 넣으면 답을 주는 셈이다.
    난이도 min(|hr-60|, |hr-100|)를 분석 단계에서 계산하려고 남긴다. 이것이 없으면
    "쉬운 표본이 정말 먼저 나가는가"를 잴 수 없고, 그것이 이 데이터를 쓰는 이유다.

실행
    python vitaldb_to_npz.py                       # 전부
    python vitaldb_to_npz.py --max-subjects 260    # 파킨슨 음성(231명) 규모로 맞춤
    python vitaldb_to_npz.py --cap 40              # 사람당 윈도우 상한
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dataset" / "vitaldb" / "processed_cases"
DST = ROOT / "dataset" / "processed" / "vitaldb"
MODALITIES = ("ecg", "ppg")


def cap_per_subject(sid, cap, rng):
    """사람당 최대 cap개만 남긴다. 시간 순서를 유지한 채 균등 간격으로 뽑는다.

    앞에서 자르지 않는 이유는 수술 후반부 구간을 통째로 버리지 않기 위해서다.
    train.py의 _cap_positions와 같은 방식이다.
    """
    keep = []
    for u in np.unique(sid):
        w = np.nonzero(sid == u)[0]
        keep.append(w if len(w) <= cap else
                    w[np.linspace(0, len(w) - 1, cap).astype(int)])
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser(description="VitalDB case npz → 모달별 npz")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--max-subjects", type=int,
                    help="피험자 상한. 파킨슨과 규모를 맞출 때 쓴다(음성 231명)")
    ap.add_argument("--cap", type=int,
                    help="사람당 윈도우 상한. VitalDB는 사례당 300~450개라 "
                         "전체를 담으면 파킨슨의 100배가 넘는다")
    ap.add_argument("--label", choices=["both", "brady"], default="both",
                    help="both=60~100 정상(원본) · brady=hr<60 서맥. "
                         "brady는 경계가 하나라 난이도 스윕이 가능하다")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(args.src.glob("case_*.npz"))
    if not files:
        raise SystemExit(f"{args.src} 에 case_*.npz 가 없습니다. "
                         f"먼저 vitaldb_preprocess.py 를 실행하세요.")

    rng = np.random.default_rng(args.seed)

    # 1차 통과 — 피험자 목록만 모은다. 상한을 걸려면 어떤 사람을 남길지
    # 먼저 정해야 하고, 그래야 신호를 한 번만 읽는다.
    case_subject = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            case_subject[f] = str(z["subjectid"])
    subjects = sorted(set(case_subject.values()))
    print(f"case {len(files)}개 · 피험자 {len(subjects)}명")
    if len(files) != len(subjects):
        print(f"  -> case가 피험자보다 많다. subjectid로 묶는다 "
              f"(수술 2회 이상 {len(files) - len(subjects)}건)")

    if args.max_subjects and len(subjects) > args.max_subjects:
        pick = set(rng.permutation(subjects)[:args.max_subjects])
        files = [f for f in files if case_subject[f] in pick]
        print(f"  -> 피험자 {args.max_subjects}명으로 제한 (case {len(files)}건)")

    args.dst.mkdir(parents=True, exist_ok=True)
    for m in MODALITIES:
        X, y, sid, hr, phr = [], [], [], [], []
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                n = len(z["label"])
                X.append(z[m])
                y.append(z["label"])
                sid.append(np.full(n, str(z["subjectid"])))
                hr.append(z["hr"])
                phr.append(z["ppg_estimated_hr"])

        X = np.concatenate(X).astype(np.float32)
        y = np.concatenate(y).astype(np.int64)
        sid = np.concatenate(sid)
        hr = np.concatenate(hr).astype(np.float32)
        phr = np.concatenate(phr).astype(np.float32)

        if args.cap:
            keep = cap_per_subject(sid, args.cap, rng)
            X, y, sid, hr, phr = X[keep], y[keep], sid[keep], hr[keep], phr[keep]

        # 라벨은 hr에서 다시 만든다 — 재전처리 없이 정의만 바꿀 수 있다
        if args.label == "brady":
            y = (hr < 60).astype(np.int64)

        suffix = "" if args.label == "both" else f"_{args.label}"
        out = args.dst / f"{m}_windows_30s{suffix}.npz"
        np.savez_compressed(out, X=X, y=y, subject_id=sid, hr=hr,
                            ppg_estimated_hr=phr)

        c = Counter(y.tolist())
        # 라벨이 hr에서 나오므로, 경계까지의 거리가 곧 난이도다
        diff = (np.minimum(np.abs(hr - 60), np.abs(hr - 100))
                if args.label == "both" else np.abs(hr - 60))
        print(f"\n[{m}] {out.name}  ({out.stat().st_size / 1e6:.0f} MB)")
        print(f"  X {X.shape} · 피험자 {len(np.unique(sid))}명")
        print(f"  라벨  정상 {c[0]:,} ({c[0] / len(y):.1%}) · "
              f"비정상 {c[1]:,} ({c[1] / len(y):.1%})  "
              f"불균형 {max(c.values()) / max(min(c.values()), 1):.1f}:1")
        print(f"  난이도 (경계까지 거리)")
        for lo, hi, name in [(0, 3, "매우 어려움"), (3, 10, "어려움"),
                             (10, 20, "보통"), (20, 999, "쉬움")]:
            k = int(((diff >= lo) & (diff < hi)).sum())
            print(f"    {name:8} {lo:>2}~{hi if hi < 999 else '':<3} "
                  f"{k:>8,} ({k / len(diff):5.1%})")

    print(f"\nconfigs/config.yaml 에 경로를 추가하고 아래처럼 쓴다")
    print(f"  python train.py --modalities ecg --max-windows 40")
    print(f"  python train.py --modalities ecg,ppg --max-windows 40   # 공유 인코더")


if __name__ == "__main__":
    main()
