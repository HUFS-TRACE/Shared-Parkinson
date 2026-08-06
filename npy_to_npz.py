from pathlib import Path
import re
import numpy as np

# convert.py가 있는 현재 폴더
INPUT_DIR = Path(__file__).resolve().parent

# 변환된 npz 파일을 저장할 폴더
OUTPUT_DIR = INPUT_DIR / "a_npz"
OUTPUT_DIR.mkdir(exist_ok=True)

# 허용 파일명:
# 82-pd-a-01.npy
# 101-hc-a-02.npy
filename_pattern = re.compile(
    r"^(\d+)-(pd|hc)-a(?:-(\d+))?$",
    re.IGNORECASE
)

npy_files = sorted(INPUT_DIR.glob("*.npy"))

print("검색 폴더:", INPUT_DIR)
print("찾은 npy 파일 수:", len(npy_files))
print()

success_count = 0
error_count = 0

for npy_path in npy_files:
    result = filename_pattern.fullmatch(npy_path.stem)

    if result is None:
        print("❌ 파일명 형식 오류:", npy_path.name)
        error_count += 1
        continue

    subject_text, class_name, sample_number = result.groups()

    # 원본 npy 데이터
    X = np.load(npy_path)

    # PD는 1, HC는 0
    if class_name.lower() == "pd":
        y = np.array(1, dtype=np.int64)
    else:
        y = np.array(0, dtype=np.int64)

    # 파일명 맨 앞 숫자
    subject_id = np.array(int(subject_text), dtype=np.int64)

    # 이번 폴더는 모두 모음 a
    task = np.array("a")

    # 파일명은 유지하고 확장자만 npz로 변경
    output_path = OUTPUT_DIR / f"{npy_path.stem}.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        subject_id=subject_id,
        task=task
    )

    print(
        f"✅ {npy_path.name} → {output_path.name} "
        f"| X={X.shape}, y={y.item()}, "
        f"subject_id={subject_id.item()}, task={task.item()}"
    )

    success_count += 1

print("\n===== 변환 결과 =====")
print("변환 성공:", success_count)
print("변환 실패:", error_count)
print("저장 폴더:", OUTPUT_DIR)