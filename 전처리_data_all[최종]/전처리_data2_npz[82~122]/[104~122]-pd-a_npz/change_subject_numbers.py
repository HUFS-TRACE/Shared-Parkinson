from pathlib import Path
import re
import csv

# 이 파이썬 파일이 들어 있는 폴더
folder = Path(__file__).parent

# 새 번호 시작값
START_NEW_ID = 104

# False: 바뀔 이름만 미리 확인
# True: 실제 파일명 변경
EXECUTE = True

# 예: 193-hc-a-01.npz
pattern = re.compile(r"^(?P<subject>\d+)-(?P<rest>.+)\.npz$", re.IGNORECASE)

matched_files = []
skipped_files = []

# 현재 폴더의 .npz 파일 확인
for old_path in folder.iterdir():

    if not old_path.is_file() or old_path.suffix.lower() != ".npz":
        continue

    match = pattern.match(old_path.name)

    if not match:
        skipped_files.append(old_path.name)
        continue

    old_subject_id = int(match.group("subject"))
    rest_name = match.group("rest")

    matched_files.append({
        "old_path": old_path,
        "old_subject_id": old_subject_id,
        "rest_name": rest_name,
    })

if not matched_files:
    print("변경할 .npz 파일을 찾지 못했습니다.")
    raise SystemExit

# 기존 사람 번호를 작은 순서대로 정렬
unique_old_ids = sorted({
    item["old_subject_id"]
    for item in matched_files
})

# 193 → 82, 194 → 83, 195 → 84 ...
subject_mapping = {
    old_id: START_NEW_ID + index
    for index, old_id in enumerate(unique_old_ids)
}

rename_plan = []

for item in matched_files:
    new_subject_id = subject_mapping[item["old_subject_id"]]

    # 맨 앞 숫자만 바꾸고 나머지는 그대로 유지
    new_name = f"{new_subject_id}-{item['rest_name']}.npz"
    new_path = folder / new_name

    rename_plan.append({
        **item,
        "new_subject_id": new_subject_id,
        "new_path": new_path,
    })

# 변경 후 파일명 중복 확인
new_names = [item["new_path"].name for item in rename_plan]

if len(new_names) != len(set(new_names)):
    print("오류: 변경 후 같은 파일명이 생깁니다.")
    print("파일 이름은 변경하지 않았습니다.")
    raise SystemExit

# 변경될 이름이 이미 존재하는지 확인
for item in rename_plan:
    if item["new_path"].exists():
        print(f"오류: 이미 존재하는 파일 → {item['new_path'].name}")
        print("파일 이름은 변경하지 않았습니다.")
        raise SystemExit

# 사람 번호와 음성 순서대로 정렬
rename_plan.sort(
    key=lambda item: (
        item["old_subject_id"],
        item["old_path"].name
    )
)

print("=" * 80)
print("파일 이름 변경 미리보기")
print("=" * 80)

for item in rename_plan:
    print(
        f"{item['old_path'].name}"
        f" → "
        f"{item['new_path'].name}"
    )

print("\n사람 번호 대응표")

for old_id in unique_old_ids:
    print(f"{old_id} → {subject_mapping[old_id]}")

print(f"\n전체 사람 수: {len(unique_old_ids)}명")
print(f"변경 대상 파일 수: {len(rename_plan)}개")

if skipped_files:
    print("\n형식이 달라 제외된 파일")

    for file_name in skipped_files:
        print(" -", file_name)

if EXECUTE:
    # 변경 내역 저장
    csv_path = folder / "subject_number_mapping.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "old_subject_id",
            "new_subject_id",
            "old_filename",
            "new_filename",
        ])

        for item in rename_plan:
            writer.writerow([
                item["old_subject_id"],
                item["new_subject_id"],
                item["old_path"].name,
                item["new_path"].name,
            ])

    # 실제 파일명 변경
    for item in rename_plan:
        item["old_path"].rename(item["new_path"])

    print("\n파일 이름 변경 완료!")
    print("변경 내역 저장:", csv_path.name)

else:
    print("\n현재는 미리보기 상태입니다.")
    print("결과가 맞으면 EXECUTE = True로 바꾸고 다시 실행하세요.")