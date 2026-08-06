"""
subject_id 정정 — 같은 사람이 두 개의 id를 갖는 문제를 바로잡는다.

무엇이 잘못됐나
    원본에서 아래 두 파일은 **같은 사람(0034번)** 이다.
        HC_A1_0034.npy       모음 '아'
        HC_PATAKA_0034.npy   파타카(DDK)
    그런데 npy_to_npz.py가 폴더별로 돌면서 파일명 맨 앞 숫자를 subject_id로
    쓰는 바람에, 모음은 전처리_data3_npz(id 82~192)로, 파타카는
    전처리_data4_npz(id 234~334)로 들어가 한 사람이 서로 다른 두 id를 받았다.

왜 고쳐야 하나
    StratifiedGroupKFold(groups=subject_id)는 id를 사람이라고 믿고 자른다.
    한 사람이 두 id로 갈려 있으면 한 조각은 train, 다른 조각은 test로 갈 수
    있고(5-fold면 약 80% 확률), 그러면 학습 때 들은 목소리를 test에서 다시
    듣는다. 정확도가 실제보다 높게 나오고 그 숫자로 내린 판정은 오염된다.
    NewHandPD에서 P1=P4를 잡았던 것과 같은 유형이다.

어떻게 고치나
    원본 파일명 맨 뒤 숫자가 사람 번호다(parkinson-audio-preprocessing/README).

    ⚠️ 번호는 **클래스 안에서만 유일하다.** README가 HC 명단과 PD 명단을 따로
       나열한 것이 그 뜻이다. 실제로 전처리_audio의 HC 68번과 전처리_pataka의
       PD 68번은 서로 다른 사람이다. 그래서 사람을 식별하는 키는 번호 하나가
       아니라 **(클래스, 번호)** 쌍이어야 한다.
       (번호만으로 묶었다가 라벨 충돌 2건으로 걸러낸 실수다)

    사람번호 오름차순 순서대로 id가 붙었다는 가설을 **파일 수와 클래스로 검산**한
    뒤, data4의 id를 같은 사람의 data3 id로 되돌린다. 검산이 하나라도 어긋나면
    매핑을 내보내지 않는다 — 잘못 합치는 것이 안 합치는 것보다 나쁘기 때문이다.

    데이터 파일은 건드리지 않는다. 매핑 표(JSON)만 내보내고, 이후 윈도잉
    단계에서 canonical_subject_id()로 적용한다. 되돌릴 수 있게 두려는 것이다.

실행
    python fix_subject_id.py                    # 검산 + 매핑 저장
    python fix_subject_id.py --verify-only      # 검산만
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
NPZ_ROOT_DEFAULT = HERE / "전처리_data_all"
SRC_DEFAULT = HERE.parent / "parkinson-audio-preprocessing"
MAP_PATH_DEFAULT = HERE / "voice_subject_id_map.json"

# npz 폴더 ↔ 원본 폴더. 같은 사람이 걸쳐 있는 쌍만 적는다.
#   data1(81명·1인1파일)·data2(41명·이름으로 이미 병합됨)는 중복이 없어 제외.
PAIR = dict(원본="전처리_audio", 중복="전처리_pataka",
            npz원본="전처리_data3_npz", npz중복="전처리_data4_npz")

NPZ_NAME = re.compile(r"^(\d+)-(pd|hc)-", re.IGNORECASE)
SRC_NAME = re.compile(r"^(HC|PD)_[A-Z0-9]+_(\d+)$", re.IGNORECASE)


def scan_npz(folder):
    """npz 폴더 → {subject_id: (파일 수, 라벨)}"""
    cnt, lab = Counter(), {}
    for f in sorted(folder.rglob("*.npz")):
        m = NPZ_NAME.match(f.stem)
        if not m:
            raise ValueError(f"npz 파일명 형식 예상 밖: {f.name}")
        sid = int(m.group(1))
        cnt[sid] += 1
        lab[sid] = 0 if m.group(2).lower() == "hc" else 1
    return cnt, lab


def scan_src(folder):
    """원본 npy 폴더 → {(클래스, 번호): 파일 수}

    사람 키를 (클래스, 번호)로 잡는 이유는 모듈 docstring 참고. 번호만 쓰면
    HC 68과 PD 68이 한 사람으로 합쳐진다.
    """
    cnt = Counter()
    for f in sorted(folder.rglob("*.npy")):
        m = SRC_NAME.match(f.stem)
        if not m:
            raise ValueError(f"원본 파일명 형식 예상 밖: {f.name}")
        cnt[(m.group(1).upper(), int(m.group(2)))] += 1
    return cnt


def derive(src_folder, npz_folder, label):
    """원본 사람 ↔ npz subject_id 대응을 세우고 검산한다.

    가설: 한 수집분 안에서 사람번호 오름차순으로 id가 붙었다.
    전제: 그 수집분 안에서는 번호가 클래스에 걸쳐 유일하다(먼저 확인한다).
    검산: 순위별로 (파일 수, 클래스)가 모두 일치해야 한다. 하나라도 어긋나면
          정렬 가설이 틀렸다는 뜻이므로 즉시 중단한다.
    """
    s_cnt = scan_src(src_folder)
    n_cnt, n_lab = scan_npz(npz_folder)
    print(f"\n[{label}] 원본 {src_folder.name} · npz {npz_folder.name}")
    n_hc = sum(1 for c, _ in s_cnt if c == "HC")
    print(f"  원본 사람 {len(s_cnt)}명 (HC {n_hc} / PD {len(s_cnt) - n_hc})"
          f" / npz subject_id {len(n_cnt)}개")

    # 번호가 클래스에 걸쳐 겹치면 "번호 오름차순"이 사람을 유일하게 정하지 못한다
    by_num = defaultdict(set)
    for cls, num in s_cnt:
        by_num[num].add(cls)
    dup = {n: c for n, c in by_num.items() if len(c) > 1}
    if dup:
        raise SystemExit(f"  ❌ 이 수집분 안에서 번호가 HC·PD에 겹침: {list(dup)[:5]}"
                         f" — 번호 오름차순 대응이 성립하지 않는다")

    if len(s_cnt) != len(n_cnt):
        raise SystemExit(f"  ❌ 인원 수 불일치 ({len(s_cnt)} vs {len(n_cnt)}) — 대응 불가")

    persons = sorted(s_cnt, key=lambda k: k[1])       # 번호 오름차순
    sids = sorted(n_cnt)
    bad_cnt = [(p, s) for p, s in zip(persons, sids) if s_cnt[p] != n_cnt[s]]
    bad_lab = [(p, s) for p, s in zip(persons, sids)
               if (0 if p[0] == "HC" else 1) != n_lab[s]]
    print(f"  파일 수 일치 {len(persons) - len(bad_cnt)}/{len(persons)}   "
          f"클래스 일치 {len(persons) - len(bad_lab)}/{len(persons)}")
    if bad_cnt or bad_lab:
        print(f"  ❌ 정렬 대응 가설 실패. 파일수 {bad_cnt[:3]}, 클래스 {bad_lab[:3]}")
        raise SystemExit("  → 매핑을 만들지 않는다. 사람 대응을 수동으로 확인할 것.")
    print(f"  ✅ 정렬 대응 성립  (예: 원본 {persons[:3]} ↔ id {sids[:3]})")
    return dict(zip(sids, persons))          # subject_id → (클래스, 번호)


def main():
    ap = argparse.ArgumentParser(description="음성 subject_id 중복 정정")
    ap.add_argument("--npz-root", type=Path, default=NPZ_ROOT_DEFAULT)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT,
                    help="parkinson-audio-preprocessing 저장소 경로 (원본 파일명 필요)")
    ap.add_argument("--out", type=Path, default=MAP_PATH_DEFAULT)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    for p in (args.npz_root, args.src):
        if not p.exists():
            raise SystemExit(f"경로 없음: {p}")

    print("=" * 68)
    print("음성 subject_id 중복 정정 — data3(모음) ↔ data4(파타카)")
    print("=" * 68)

    sid2person_a = derive(args.src / PAIR["원본"],
                          args.npz_root / PAIR["npz원본"], "모음")
    sid2person_p = derive(args.src / PAIR["중복"],
                          args.npz_root / PAIR["npz중복"], "파타카")

    # 같은 사람인지 확인 — 파타카 화자가 모음 화자에 포함돼야 합친다
    person_a, person_p = set(sid2person_a.values()), set(sid2person_p.values())
    only_p = person_p - person_a
    print(f"\n[대조] 모음 {len(person_a)}명 · 파타카 {len(person_p)}명 · "
          f"교집합 {len(person_a & person_p)}명 · 파타카 전용 {len(only_p)}명")
    if only_p:
        print(f"  ℹ️ 파타카에만 있는 {len(only_p)}명은 그대로 둔다(합칠 상대가 없음)")

    person2sid_a = {p: s for s, p in sid2person_a.items()}
    remap = {s: person2sid_a[p] for s, p in sid2person_p.items() if p in person2sid_a}
    print(f"  → data4 id {len(remap)}개를 data3 id로 합침  "
          f"(예: " + ", ".join(f"{a}→{b}" for a, b in sorted(remap.items())[:4]) + ")")

    # ── 전체 id에 대한 최종 매핑 (바뀌지 않는 id는 자기 자신) ──
    all_cnt, all_lab = Counter(), {}
    for folder in sorted(args.npz_root.iterdir()):
        if folder.is_dir():
            c, l = scan_npz(folder)
            all_cnt.update(c)
            all_lab.update(l)
    canon = {sid: remap.get(sid, sid) for sid in sorted(all_cnt)}

    # ── 사후 검증 ──
    print("\n" + "=" * 68)
    print("사후 검증")
    print("=" * 68)
    merged_lab = defaultdict(set)
    merged_cnt = Counter()
    for sid, n in all_cnt.items():
        merged_lab[canon[sid]].add(all_lab[sid])
        merged_cnt[canon[sid]] += n
    conflict = {k: v for k, v in merged_lab.items() if len(v) > 1}
    print(f"  정정 전 subject_id : {len(all_cnt)}개")
    print(f"  정정 후 실제 사람  : {len(merged_lab)}명   "
          f"({len(all_cnt) - len(merged_lab)}명이 두 id로 갈려 있었음)")
    print(f"  한 사람이 두 라벨을 갖는 경우: {len(conflict)}건 "
          f"{'✅' if not conflict else '❌ ' + str(list(conflict)[:5])}")
    if conflict:
        raise SystemExit("  → 라벨 충돌. 매핑이 틀렸다는 뜻이므로 저장하지 않는다.")

    c = np.array(sorted(merged_cnt.values()))
    print(f"  사람당 파일 수: 최소 {c.min()} / 중앙 {int(np.median(c))} / 최대 {c.max()}")
    n_people = len(merged_lab)
    print(f"  5-fold 기준 fold당 test ≈ {n_people / 5:.0f}명 → 1명 ≈ "
          f"{100 / (n_people / 5):.1f}%p")

    if args.verify_only:
        print("\n--verify-only: 저장하지 않음")
        return

    payload = dict(
        note="음성 subject_id 정정표. data4(파타카)의 id를 같은 사람의 "
             "data3(모음) id로 합친다. 나머지 id는 자기 자신.",
        derived_from=dict(npz_root=str(args.npz_root), src=str(args.src)),
        n_ids_before=len(all_cnt), n_people_after=len(merged_lab),
        n_merged=len(remap),
        map={str(k): v for k, v in canon.items()},
    )
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n매핑 저장: {args.out}  ({len(canon)}개 id)")
    print("  사용법: from fix_subject_id import canonical_subject_id")
    print("          sid = canonical_subject_id(sid)   # 윈도잉 단계에서 적용")


# ───────────────────────── 하위 단계에서 쓰는 헬퍼 ─────────────────────────

_MAP_CACHE = None


def load_map(path=MAP_PATH_DEFAULT):
    """정정표를 읽어 {원래 id: 정식 id} dict로 반환 (캐시)."""
    global _MAP_CACHE
    if _MAP_CACHE is None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} 가 없습니다. 먼저 `python fix_subject_id.py`를 실행하세요.")
        _MAP_CACHE = {int(k): int(v)
                      for k, v in json.loads(p.read_text(encoding="utf-8"))["map"].items()}
    return _MAP_CACHE


def canonical_subject_id(subject_id, path=MAP_PATH_DEFAULT):
    """원래 subject_id → 사람 단위로 합쳐진 정식 id. 스칼라·배열 모두 받는다.

    윈도잉해서 npz로 쌓기 **직전에** 적용해야 한다. 그래야 fold 분할이
    사람 단위로 정확히 걸린다.
    """
    m = load_map(path)
    arr = np.asarray(subject_id)
    if arr.ndim == 0:
        return m.get(int(arr), int(arr))
    return np.array([m.get(int(s), int(s)) for s in arr.ravel()]).reshape(arr.shape)


if __name__ == "__main__":
    main()
