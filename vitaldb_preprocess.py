"""VitalDB ECG/PPG 30초 윈도우 전처리 — 시연이 만든 Colab 노트북의 로컬 실행판.

전처리 로직은 원본 노트북(vitaldb_preprocess.ipynb) 그대로다. 바꾼 것은 셋뿐이다.
    - Google Drive 마운트 제거, BASE_DIR을 로컬 경로로
    - 셀 실행 대신 CLI 인자
    - 원본 데이터는 VitalDB API에서 사례별로 받는다(로컬 원본 불필요)

산출물도 원본과 같다.
    case_manifest.csv          caseid · subjectid · split
    preprocessing_summary.csv  사례별 처리 결과
    processed_cases/case_NNNNN.npz
        ecg (N,1,3750) · ppg (N,1,3750) · hr · ppg_estimated_hr
        label · start_sec · caseid · subjectid · split

⚠️ split 열은 우리 학습에 쓰지 않는다. 고정 분할 하나로는 시드 5개 GroupKFold를
   돌릴 수 없기 때문이다(이슈 #15). npz의 subjectid로 다시 나눈다.

실행
    pip install vitaldb
    python vitaldb_preprocess.py --cases 20            # 앞에서 20개만
    python vitaldb_preprocess.py --cases 0             # 전체
    python vitaldb_preprocess.py --cases 50 --start 20 # 이어서
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent / "dataset" / "vitaldb"
PROCESSED_DIR = BASE_DIR / "processed_cases"
MANIFEST_PATH = BASE_DIR / "case_manifest.csv"
SUMMARY_PATH = BASE_DIR / "preprocessing_summary.csv"

ECG_TRACK = "SNUADC/ECG_II"
PPG_TRACK = "SNUADC/PLETH"
HR_TRACK = "Solar8000/HR"
TRACKS = [ECG_TRACK, PPG_TRACK, HR_TRACK]

TARGET_FS = 125
WINDOW_SECONDS = 30
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS      # 3750
RANDOM_SEED = 42


# ── 전처리 유틸 (원본 셀 10) ────────────────────────────────────────────
def fill_nan_linear(x):
    """소량 NaN을 선형 보간한다. 유효값이 2개 미만이면 None."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return None
    indices = np.arange(len(x))
    return np.interp(indices, indices[valid], x[valid])


def bandpass_filter(x, lowcut, highcut, fs, order=4):
    """Butterworth 대역통과 필터 (zero-phase)."""
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def zscore_normalize(x):
    """평균 0, 표준편차 1. 분산이 거의 없으면 None.

    윈도우마다 자기 통계를 쓰므로 fold를 넘는 누수가 없다. 우리 모델의 RevIN과
    같은 방식이라 중복이지만 무해하다.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return None
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


# ── 윈도우 전처리 (원본 셀 12) ──────────────────────────────────────────
def preprocess_window(window: pd.DataFrame):
    ecg_raw = window["ecg"].to_numpy(dtype=np.float64)
    ppg_raw = window["ppg"].to_numpy(dtype=np.float64)
    hr_raw = window["hr"].to_numpy(dtype=np.float64)

    if np.isfinite(ecg_raw).mean() < 0.99:
        return None
    if np.isfinite(ppg_raw).mean() < 0.99:
        return None

    valid_hr = hr_raw[np.isfinite(hr_raw)]
    if len(valid_hr) == 0:
        return None
    median_hr = float(np.median(valid_hr))
    if median_hr < 25 or median_hr > 220:
        return None

    finite_ecg = ecg_raw[np.isfinite(ecg_raw)]
    clipping_ratio = np.mean(np.abs(finite_ecg) >= 4.8)
    if clipping_ratio > 0.001:
        return None

    ecg = fill_nan_linear(ecg_raw)
    ppg = fill_nan_linear(ppg_raw)
    if ecg is None or ppg is None:
        return None

    if np.std(ecg) < 0.01:
        return None
    if np.std(ppg) < 0.1 or np.ptp(ppg) < 1.0:
        return None
    if np.max(np.abs(np.diff(ecg))) > 3.0:
        return None

    try:
        ecg_filtered = bandpass_filter(ecg, 0.5, 40.0, TARGET_FS)
        ppg_filtered = bandpass_filter(ppg, 0.5, 8.0, TARGET_FS)
    except ValueError:
        return None
    if np.std(ecg_filtered) < 1e-5 or np.std(ppg_filtered) < 1e-5:
        return None

    prominence = max(0.2 * np.std(ppg_filtered), 1e-5)
    peaks, _ = find_peaks(ppg_filtered, distance=int(TARGET_FS * 0.3),
                          prominence=prominence)
    duration_seconds = len(ppg_filtered) / TARGET_FS
    ppg_estimated_hr = len(peaks) * 60.0 / duration_seconds
    if ppg_estimated_hr < 30 or ppg_estimated_hr > 220:
        return None
    # ⚠️ 맥파 봉우리로 센 심박이 모니터 심박과 30 이상 벌어지면 버린다.
    #    품질 관리로는 타당하나, 남는 데이터가 "봉우리 세기가 이미 통하는" 쪽으로
    #    치우친다. 라벨이 그 심박에서 나오므로 과제가 더 쉬워지는 방향이다.
    #    난이도 축 실험을 할 때 이 선택 편향을 감안해야 한다.
    if abs(ppg_estimated_hr - median_hr) > 30:
        return None

    ecg_normalized = zscore_normalize(ecg_filtered)
    ppg_normalized = zscore_normalize(ppg_filtered)
    if ecg_normalized is None or ppg_normalized is None:
        return None

    label = 0 if 60 <= median_hr <= 100 else 1

    return {"ecg": ecg_normalized, "ppg": ppg_normalized, "hr": median_hr,
            "ppg_estimated_hr": ppg_estimated_hr, "label": label}


# ── 사례 처리 (원본 셀 14) ──────────────────────────────────────────────
def make_result_record(caseid, subjectid, split, status, total_windows=0,
                       saved_windows=0, normal_windows=0, abnormal_windows=0,
                       file_path="", error=""):
    return {"caseid": int(caseid), "subjectid": str(subjectid), "split": str(split),
            "status": status, "total_windows": total_windows,
            "saved_windows": saved_windows, "normal_windows": normal_windows,
            "abnormal_windows": abnormal_windows, "file": file_path, "error": error}


def process_case(caseid, subjectid, split, overwrite=False):
    import vitaldb

    caseid = int(caseid)
    subjectid = str(subjectid)
    split = str(split)
    output_path = PROCESSED_DIR / f"case_{caseid:05d}.npz"

    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=False) as saved:
            saved_labels = saved["label"]
            total_windows = (int(saved["total_windows"])
                             if "total_windows" in saved.files else len(saved_labels))
        return make_result_record(
            caseid=caseid, subjectid=subjectid, split=split, status="saved",
            total_windows=total_windows, saved_windows=len(saved_labels),
            normal_windows=int(np.sum(saved_labels == 0)),
            abnormal_windows=int(np.sum(saved_labels == 1)),
            file_path=str(output_path))

    vital_file = vitaldb.VitalFile(caseid, TRACKS)
    data = vital_file.to_numpy(TRACKS, interval=1 / TARGET_FS)

    if data is None:
        return make_result_record(caseid, subjectid, split, status="no_data")
    if len(data) < WINDOW_SIZE:
        return make_result_record(caseid, subjectid, split, status="too_short")

    data_df = pd.DataFrame(data, columns=["ecg", "ppg", "hr"])
    total_windows = len(data_df) // WINDOW_SIZE

    ecg_windows, ppg_windows = [], []
    hr_values, ppg_hr_values, labels, start_seconds = [], [], [], []

    for start in range(0, len(data_df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        result = preprocess_window(data_df.iloc[start:start + WINDOW_SIZE])
        if result is None:
            continue
        ecg_windows.append(result["ecg"])
        ppg_windows.append(result["ppg"])
        hr_values.append(result["hr"])
        ppg_hr_values.append(result["ppg_estimated_hr"])
        labels.append(result["label"])
        start_seconds.append(start / TARGET_FS)

    if len(labels) == 0:
        return make_result_record(caseid, subjectid, split, status="no_valid_windows",
                                  total_windows=total_windows)

    # PatchTST 입력 형태: [N, 1, 3750]
    ecg_array = np.stack(ecg_windows)[:, None, :].astype(np.float32)
    ppg_array = np.stack(ppg_windows)[:, None, :].astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)

    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        ecg=ecg_array, ppg=ppg_array,
        hr=np.asarray(hr_values, dtype=np.float32),
        ppg_estimated_hr=np.asarray(ppg_hr_values, dtype=np.float32),
        label=label_array,
        start_sec=np.asarray(start_seconds, dtype=np.float32),
        total_windows=np.int32(total_windows), fs=np.int32(TARGET_FS),
        caseid=np.int32(caseid), subjectid=np.array(subjectid),
        split=np.array(split))
    temporary_path.replace(output_path)

    return make_result_record(
        caseid=caseid, subjectid=subjectid, split=split, status="saved",
        total_windows=total_windows, saved_windows=len(label_array),
        normal_windows=int(np.sum(label_array == 0)),
        abnormal_windows=int(np.sum(label_array == 1)),
        file_path=str(output_path))


def update_summary(new_records):
    if len(new_records) == 0:
        if SUMMARY_PATH.exists():
            return pd.read_csv(SUMMARY_PATH, dtype={"subjectid": str})
        return pd.DataFrame()

    new_df = pd.DataFrame(new_records)
    if SUMMARY_PATH.exists():
        old_df = pd.read_csv(SUMMARY_PATH, dtype={"subjectid": str})
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["caseid"], keep="last")
    else:
        combined = new_df
    combined = combined.sort_values("caseid")
    combined.to_csv(SUMMARY_PATH, index=False)
    return combined


# ── 사례 목록과 분할 (원본 셀 6·8) ──────────────────────────────────────
def build_manifest():
    """VitalDB API에서 세 트랙이 모두 있는 사례를 고르고 환자 단위로 나눈다.

    분할은 원본 노트북 그대로 두되(재현성), 학습에는 쓰지 않는다.
    같은 환자가 여러 split에 걸치지 않게 하는 부분만 우리에게 의미가 있다.
    """
    if MANIFEST_PATH.exists():
        print(f"기존 분할 사용: {MANIFEST_PATH.name}")
        return pd.read_csv(MANIFEST_PATH, dtype={"subjectid": str})

    print("VitalDB API에서 사례 목록을 받는 중...")
    case_info = pd.read_csv("https://api.vitaldb.net/cases")
    track_info = pd.read_csv("https://api.vitaldb.net/trks")

    selected = track_info[track_info["tname"].isin(set(TRACKS))]
    counts = selected.groupby("caseid")["tname"].nunique()
    valid_caseids = counts[counts == len(TRACKS)].index.tolist()

    valid_cases = (case_info[case_info["caseid"].isin(valid_caseids)]
                   [["caseid", "subjectid"]].dropna(subset=["subjectid"])
                   .drop_duplicates().copy())
    valid_cases["caseid"] = valid_cases["caseid"].astype(int)
    valid_cases["subjectid"] = valid_cases["subjectid"].astype(str)
    print(f"사례 {len(valid_cases)}개 · 고유 환자 {valid_cases['subjectid'].nunique()}명")

    subject_ids = valid_cases["subjectid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(subject_ids)
    n = len(subject_ids)
    train_subjects = set(subject_ids[:int(n * 0.70)])
    validation_subjects = set(subject_ids[int(n * 0.70):int(n * 0.85)])

    def assign(sid):
        if sid in train_subjects:
            return "train"
        return "validation" if sid in validation_subjects else "test"

    manifest = valid_cases.copy()
    manifest["split"] = manifest["subjectid"].apply(assign)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"새 분할 저장: {MANIFEST_PATH.name}")
    return manifest


def run(manifest, start_index=0, run_cases=20, overwrite=False):
    run_manifest = (manifest.iloc[start_index:].copy() if not run_cases else
                    manifest.iloc[start_index:start_index + run_cases].copy())

    completed = set()
    if SUMMARY_PATH.exists() and not overwrite:
        prev = pd.read_csv(SUMMARY_PATH)
        done = prev[prev["status"].isin(
            ["saved", "too_short", "no_data", "no_valid_windows"])]
        completed = set(done["caseid"].astype(int).tolist())

    run_manifest = run_manifest[~run_manifest["caseid"].isin(completed)]
    print(f"이번에 처리할 사례: {len(run_manifest)}개")

    pending = []
    for row in tqdm(run_manifest.itertuples(index=False), total=len(run_manifest)):
        try:
            record = process_case(row.caseid, row.subjectid, row.split, overwrite)
        except Exception as error:
            record = make_result_record(row.caseid, row.subjectid, row.split,
                                        status="error", error=str(error))
        pending.append(record)
        if len(pending) >= 5:
            update_summary(pending)
            pending = []
    if pending:
        update_summary(pending)

    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH, dtype={"subjectid": str})
    return pd.DataFrame()


def summarize(summary):
    if summary.empty:
        print("처리된 사례가 없습니다.")
        return
    ok = summary[summary["status"] == "saved"]
    print("\n" + "=" * 60)
    print(f"  status: {summary['status'].value_counts().to_dict()}")
    if ok.empty:
        return
    tot, sav = ok["total_windows"].sum(), ok["saved_windows"].sum()
    nor, abn = ok["normal_windows"].sum(), ok["abnormal_windows"].sum()
    print(f"  윈도우  총 {tot:,} -> 저장 {sav:,}  (품질 탈락 {100 * (1 - sav / tot):.1f}%)")
    print(f"  라벨    정상 {nor:,} ({nor / (nor + abn):.1%}) · "
          f"비정상 {abn:,} ({abn / (nor + abn):.1%})")
    print(f"  피험자  {ok['subjectid'].nunique():,}명 · 사례 {len(ok):,}건")
    print(f"  파킨슨 필기 13,474 윈도우 대비 {sav / 13474:.1f}배 "
          f"— 학습에는 --max-windows 필수")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="VitalDB ECG/PPG 전처리 (로컬 실행판)")
    ap.add_argument("--cases", type=int, default=20,
                    help="처리할 사례 수. 0이면 전체 (기본 20)")
    ap.add_argument("--start", type=int, default=0, help="시작 인덱스")
    ap.add_argument("--overwrite", action="store_true", help="기존 npz를 다시 만든다")
    args = ap.parse_args()

    try:
        import vitaldb  # noqa: F401
    except ImportError:
        raise SystemExit("vitaldb 가 없습니다.  pip install vitaldb")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"저장 위치: {BASE_DIR}")
    summarize(run(build_manifest(), args.start, args.cases, args.overwrite))


if __name__ == "__main__":
    main()
