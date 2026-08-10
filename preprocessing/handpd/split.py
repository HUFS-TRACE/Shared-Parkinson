"""
피험자 단위 분할 + 누수 방지 정규화 + 나이 교란 진단.

preprocess.py가 만든 windows_*.npz를 받아
  - StratifiedGroupKFold로 subject_id 기준 분할 (같은 사람이 train·test에 못 감)
  - 정규화는 train fold에서만 fit → test에 적용
  - 나이-only 베이스라인으로 "나이빨" 감별

실행
    python preprocessing/handpd/split.py --npz dataset/processed/handpd/windows_2s.npz \
        --data-root dataset/processed/handpd
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding="utf-8")

N_SPLITS = 5
SEED = 42


def fit_channel_stats(X):
    """채널별 mean·std (float64 누적). ⚠️ train fold에만 fit."""
    x = X.astype(np.float64)
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def subject_age_map(root):
    """메타데이터에서 subject_id → 나이."""
    age = {}
    for f in (glob.glob(os.path.join(root, "HealthySignal", "*.txt"))
              + glob.glob(os.path.join(root, "PatientSignal", "*.txt"))):
        with open(f) as fh:
            head = fh.read(600)
        pid = re.search(r"#<Person_ID_Number>(.*?)</", head)
        a = re.search(r"#<Age>(\d+)</", head)
        if pid and a:
            sid = ("P_" if "Patient" in f else "H_") + pid.group(1)
            age[sid] = int(a.group(1))
    return age


def report_fold_diagnostics(subject_id, y, splits, age_arr):
    """fold별 나이 gap과 나이-only 베이스라인.

    정상군 43.4세 vs 환자군 58.0세로 14.6세 차이가 난다. 모델이 90%를 맞혀도
    "병을 배운 것"인지 "나이를 배운 것"인지 구분되지 않으므로, 나이 하나만으로
    로지스틱 회귀를 돌려 하한선을 만든다.

        모델 정확도 − 나이-only 정확도 = 나이 위에 얹은 순수 질병 기여분

    ⚠️ 평균으로 보면 안 된다. fold마다 나이차가 5.2~27.0세로 크게 다르므로
    fold별로 대조해야 한다. 나이차가 8세 미만인 fold가 가장 정직한 조건이다.
    """
    maj = np.bincount(y).argmax()
    print("fold 진단 (test set) — 나이 교란 감별")
    print("  나이-only = 나이만으로 로지스틱 회귀. 나이차가 클수록 높게 나오는 게 정상.")
    print("  → 실제 모델이 이 baseline을 fold마다 얼마나 넘느냐가 순수 질병 기여분.\n")
    print("%5s %14s %14s %8s %11s %11s" % (
        "fold", "환자(n,나이)", "정상(n,나이)", "나이차", "나이-only", "다수클래스"))

    for fold, (tr, te) in enumerate(splits):
        subs = np.unique(subject_id[te])
        lab = {s: y[subject_id == s][0] for s in subs}
        p_age = [age_arr[subject_id == s][0] for s in subs if lab[s] == 1]
        h_age = [age_arr[subject_id == s][0] for s in subs if lab[s] == 0]
        gap = np.mean(p_age) - np.mean(h_age)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(age_arr[tr].reshape(-1, 1), y[tr])
        age_acc = lr.score(age_arr[te].reshape(-1, 1), y[te])
        maj_acc = (y[te] == maj).mean()

        flag = "  나이 쏠림" if gap > 20 else ("  정직" if gap < 8 else "")
        print("%5d %6d,%5.1f세 %6d,%5.1f세 %6.1f세 %9.1f%% %10.1f%%%s" % (
            fold, len(p_age), np.mean(p_age), len(h_age), np.mean(h_age),
            gap, age_acc * 100, maj_acc * 100, flag))
    print()


def cap_per_subject(idx, subject_id, max_per_subject, rng):
    """train에서 피험자별 윈도우 상한 (긴 파일 과표현 완화).

    P13 한 명이 약 211 윈도우로 평균(약 30)의 7배다. 사람 단위 분할로 대부분
    완화되지만, 필요하면 상한을 걸 수 있다. 기본은 끄고 정보를 다 살린다.
    """
    if not max_per_subject:
        return idx
    keep = []
    for s in np.unique(subject_id[idx]):
        s_idx = idx[subject_id[idx] == s]
        if len(s_idx) > max_per_subject:
            s_idx = rng.choice(s_idx, max_per_subject, replace=False)
        keep.append(s_idx)
    return np.concatenate(keep)


def make_splits(subject_id, y, n_splits=N_SPLITS, seed=SEED):
    """subject_id를 group으로 하는 StratifiedGroupKFold 인덱스 목록."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.arange(len(y)), y, groups=subject_id))


def main():
    ap = argparse.ArgumentParser(description="피험자 단위 분할 + 누수 검증")
    ap.add_argument("--npz", default="dataset/processed/handpd/windows_2s.npz")
    ap.add_argument("--data-root", default="dataset/processed/handpd",
                    help="나이 메타를 읽을 signal 폴더 위치")
    ap.add_argument("--n-folds", type=int, default=N_SPLITS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-win-per-subject", type=int, default=None,
                    help="train에서 피험자별 윈도우 상한 (기본: 제한 없음)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X, y, subject_id = d["X"], d["y"], d["subject_id"]
    print(f"윈도우 {X.shape} | subject {len(np.unique(subject_id))}명 | "
          f"라벨 {np.bincount(y).tolist()} (0=정상, 1=환자)\n")

    splits = make_splits(subject_id, y, args.n_folds, args.seed)
    rng = np.random.default_rng(args.seed)

    age_map = subject_age_map(args.data_root)
    missing = set(subject_id) - set(age_map)
    if missing:
        sys.exit(f"나이 정보를 찾지 못한 subject: {sorted(missing)[:5]} ...")
    age_arr = np.array([age_map[s] for s in subject_id], dtype=float)

    report_fold_diagnostics(subject_id, y, splits, age_arr)

    for fold, (tr, te) in enumerate(splits):
        tr = cap_per_subject(tr, subject_id, args.max_win_per_subject, rng)
        assert not set(subject_id[tr]) & set(subject_id[te]), \
            f"fold {fold}: subject 누수"
        mean, std = fit_channel_stats(X[tr])            # train에서만 fit
        assert np.isfinite(mean).all() and np.isfinite(std).all()

    print("전 fold subject 누수 0 · 정규화 train-only fit 확인")
    print("→ 학습 연결: Xtr=(X[tr]-mean)/std, Xte=(X[te]-mean)/std")


if __name__ == "__main__":
    main()
