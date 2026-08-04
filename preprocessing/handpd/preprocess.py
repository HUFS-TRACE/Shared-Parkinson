"""
NewHandPD 6채널 시계열 → PatchTST 입력용 전처리.

파이프라인
    1 파싱·라벨 → 2 글리치 제거 → 3 이상치 점검 → 4 윈도잉 → 5 저장

정규화는 여기서 하지 않는다. z-score 통계를 전체 데이터로 내면 test 정보가
train으로 새어들어가므로, **분할 후 train fold에서만 fit**해야 한다.
따라서 이 스크립트는 raw 윈도우를 저장하고, 정규화 함수만 제공한다.
분할은 split.py가 subject_id 배열로 수행한다.

실행
    python preprocessing/handpd/preprocess.py --data-root dataset/handpd --seconds 2

산출
    dataset/handpd/windows_2s.npz
        X          [N, 6, T]  float32   raw 윈도우
        y          [N]        int       0=정상, 1=환자
        subject_id [N]        str       "H_12" / "P_7"  ← 사람 단위 분할 키
        task       [N]        str       circle / spiral / meander / diadochokinesis
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 콘솔에서 한글 깨짐 방지

SIGNAL_DIRS = ["HealthySignal", "PatientSignal"]
SAMPLE_RATE = 1000          # Hz
GLITCH_ROWS = 2             # 앞 N행 = 센서 초기화 시 튀는 값, 제거
MIN_DURATION_S = 1.0        # 이보다 짧으면 윈도우를 하나도 못 만들어 제외

# 6채널 (SIBGRAPI 2016 기준)
#   ch1 마이크 · ch2 그립력 · ch3 펜촉 축압력 · ch4~6 기울기 X/Y/Z
# 주의: 데이터셋에 x·y 좌표(궤적)는 없다. "어디를 지나갔나"가 아니라
#       "어떻게 쥐고 눌렀나"를 재는 신호다.
CHANNELS = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]

# Object 코드 → 과제명. 환자군 파일은 c_spiral로 오타나 있어 둘 다 매핑한다.
TASK_MAP = {
    "a_circ_p": "circle", "b_circ_a": "circle",
    "c_spira": "spiral", "c_spiral": "spiral",
    "d_mea": "meander",
    "e_diador": "diadochokinesis", "f_diadol": "diadochokinesis",
}


# ── 1·2. 파싱 + 라벨 + subject_id + 글리치 제거 ──────────────────────

def parse_file(path):
    """→ (meta dict, data ndarray[T, 6]). 앞 GLITCH_ROWS행 제거 완료."""
    with open(path) as f:
        lines = f.readlines()
    meta_end = next(i for i, l in enumerate(lines) if l.strip() == "#</meta>")
    meta = {}
    for l in lines[:meta_end]:
        m = re.match(r"#<(\w+)>(.*)</\w+>", l.strip())
        if m:
            meta[m.group(1)] = m.group(2)
    body = pd.read_csv(path, skiprows=meta_end + 1, sep="\t", header=None).values
    return meta, body[GLITCH_ROWS:]


def file_record(path):
    """파일 1개 → 메타 레코드.

    subject_id는 반드시 메타데이터의 Person_ID_Number로 만든다. 파일명 번호
    (-H1~-H38, -P1~-P32)로 만들면 66개가 되지만 실제 피험자는 61명이다.
    환자군에 같은 사람이 다른 번호로 들어간 경우가 있다.
        P1 = P4,  P9 = P10,  P15 = P17,  P2 = P25 = P26
    파일명 기준으로 분할하면 같은 사람이 train과 test에 나뉘어 누수가 생긴다.
    """
    meta, body = parse_file(path)
    group = "P" if "Patient" in path else "H"
    return {
        "path": path,
        "file": os.path.basename(path),
        "label": 1 if group == "P" else 0,
        "subject_id": f"{group}_{meta.get('Person_ID_Number')}",
        "task": TASK_MAP.get(meta.get("Object"), meta.get("Object")),
        "object_index": meta.get("Object_Index"),
        "n_samples": len(body),
        "duration_s": len(body) / SAMPLE_RATE,
    }


# ── 3. 이상치 점검 ─────────────────────────────────────────────────

def scan_dataset(root):
    """전 파일 스캔 → manifest + 리포트.

    길이가 3~106초로 편차가 크지만 **자동 제외하지 않는다.**
    가장 긴 파일(circA-P13, 106초)을 확인해보면 6채널 동시 정지 구간이 0샘플이고
    압력이 끝까지 유지된다. 센서 오류가 아니라 실제로 느리게 그린 것이며,
    이는 서동증(bradykinesia)의 직접적인 증거다. 자동 필터를 걸면 질병 신호가
    가장 뚜렷한 샘플을 버리게 된다. 그래서 리포트만 출력하고 판단은 사람이 한다.
    """
    files = []
    for d in SIGNAL_DIRS:
        files += glob.glob(os.path.join(root, d, "*.txt"))
    if not files:
        sys.exit(f"signal 파일을 찾지 못했습니다: {root}/{{{','.join(SIGNAL_DIRS)}}}/*.txt")

    manifest = pd.DataFrame([file_record(f) for f in files])

    print(f"총 파일 {len(manifest)} | 고유 subject {manifest.subject_id.nunique()}명")
    print(f"라벨 분포 {manifest.label.value_counts().to_dict()} (0=정상, 1=환자)")
    print("\n과제별 파일 수:")
    print(manifest.task.value_counts().to_string())

    print("\n길이(초) 분위수:")
    print(manifest.duration_s.describe(percentiles=[.01, .05, .5, .95, .99]).round(2).to_string())

    print("\n가장 긴 파일 5개 (서동증 후보 — 자동 제외하지 않음):")
    print(manifest.nlargest(5, "duration_s")[["file", "task", "duration_s"]]
          .to_string(index=False))

    excluded = manifest[manifest.duration_s < MIN_DURATION_S]
    print(f"\n제외 대상 (< {MIN_DURATION_S}초, 윈도우 0개): {len(excluded)}개")
    if len(excluded):
        print(excluded[["file", "duration_s"]].to_string(index=False))

    manifest["keep"] = manifest.duration_s >= MIN_DURATION_S
    return manifest


# ── 4. 윈도잉 ──────────────────────────────────────────────────────

def make_windows(manifest, win_len, hop):
    """슬라이딩 윈도우로 자르고 메타를 같은 인덱스로 태깅.

    트랜스포머는 고정 길이 입력이 필요한데 파일 길이가 3~106초로 제각각이다.
    패딩(최대 길이 기준)은 짧은 파일이 대부분 빈 값이 되고, 자르기(최소 길이 기준)는
    긴 파일 대부분을 버려 서동증 신호를 잃는다. 윈도잉은 손실 없이 샘플 수도 늘린다.

    50% 겹침을 두는 이유는 획이 윈도우 경계에 걸려 두 동강 나는 것을 막기 위해서다.
    다만 겹침은 이웃 윈도우만 이어줄 뿐 전체 그림의 장거리 흐름은 살리지 못한다.
    그건 윈도우 길이로 조절해야 한다.

    → X[N, 6, win_len], y[N], subject_id[N], task[N]
    """
    X, y, subj, task = [], [], [], []
    for row in manifest[manifest.keep].itertuples():
        _, body = parse_file(row.path)
        for start in range(0, len(body) - win_len + 1, hop):
            X.append(body[start:start + win_len].T)      # [6, win_len]
            y.append(row.label)
            subj.append(row.subject_id)
            task.append(row.task)
    return (np.asarray(X, dtype=np.float32), np.asarray(y),
            np.asarray(subj), np.asarray(task))


# ── 정규화 (분할 후 train fold에서만 fit) ───────────────────────────

def fit_channel_stats(X_train):
    """train 윈도우에서 채널별 mean·std. ⚠️ train에만 fit할 것.

    float64로 누적한다. 마이크 채널은 값이 3.0 근처에 몰려 있고 분산이 작아,
    float32로 2,800만 개를 합산하면 누적 오차가 커져 정규화 검증이 실패한다.
    """
    x = X_train.astype(np.float64)
    mean = x.mean(axis=(0, 2), keepdims=True)      # [1, 6, 1]
    std = x.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X, mean, std):
    """train 통계를 test 포함 모든 세트에 적용."""
    return (X - mean) / std


# ── 5. 저장 + 검증 ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NewHandPD 전처리")
    ap.add_argument("--data-root", default="dataset/handpd",
                    help="HealthySignal/ PatientSignal/ 이 있는 폴더")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="윈도우 길이(초). 1/2/4초 비교 결과 2초가 최적")
    ap.add_argument("--overlap", type=float, default=0.5, help="겹침 비율")
    ap.add_argument("--out", default=None, help="출력 npz 경로")
    args = ap.parse_args()

    win_len = int(args.seconds * SAMPLE_RATE)
    hop = max(1, int(win_len * (1 - args.overlap)))
    out = args.out or os.path.join(args.data_root,
                                   f"windows_{args.seconds:g}s.npz")

    print("=" * 60, "\n[1-3] 파싱 · 글리치 제거 · 이상치 점검\n" + "=" * 60)
    manifest = scan_dataset(args.data_root)

    print("\n" + "=" * 60,
          f"\n[4] 윈도잉 (win={win_len} hop={hop})\n" + "=" * 60)
    X, y, subj, task = make_windows(manifest, win_len, hop)
    print(f"윈도우 {X.shape} (N, 채널, 시간) | 라벨 {np.bincount(y).tolist()}")
    print(f"기여 subject {len(np.unique(subj))}명")

    print("\n" + "=" * 60,
          "\n[5] 저장 (raw — 정규화는 분할 후)\n" + "=" * 60)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, X=X, y=y, subject_id=subj, task=task)
    print(f"저장: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    # ── 무결성 검증 ──
    assert len(X) == len(y) == len(subj) == len(task), "배열 길이 불일치"
    assert X.shape[1:] == (6, win_len), f"윈도우 shape 오류: {X.shape}"
    assert not np.isnan(X).any(), "NaN 존재"
    m, s = fit_channel_stats(X)
    ch_means = apply_zscore(X, m, s).astype(np.float64).mean(axis=(0, 2))
    assert np.abs(ch_means).max() < 1e-2, f"정규화 후 채널평균≈0 아님: {ch_means}"
    print("\n검증 통과 — 정렬 · shape · NaN · 정규화")


if __name__ == "__main__":
    main()
