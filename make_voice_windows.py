"""
음성 가변 길이 npz → 고정 윈도우 npz. 필기 windows_2s.npz와 같은 형식으로 맞춘다.

왜 필요한가
    전처리 결과가 파일마다 길이가 다르다(0.56 ~ 34.57초, 중앙 4.40초).
    트랜스포머는 고정 길이 입력을 받으므로 그대로는 못 물린다.
    필기가 preprocess.py에서 하는 일을 음성에 대해 하는 스크립트다.

왜 2초인가
    파일 길이를 실측해 정했다. 중앙이 4.40초라 2초는 여유가 있다.

        윈도우   버려질 파일        남는 세그먼트
          1초     2개 (0.1%)        17,293
        ★ 2초    55개 (3.3%)         7,378
          3초   168개 (10.1%)        4,081
          4초   614개 (37.0%)        2,448

    4초는 37%를 버려 논외다. 1초와 2초 사이는 예전 스윕에서 T=64 vs T=126이
    74.3% vs 75.0%로 갈리지 않았다(0.7%p). 반면 seed 노이즈는 5.8%p로 8배 크다.
    노이즈가 신호보다 크면 스윕해도 나오는 것은 노이즈다. 그래서 길이는 고정하고
    그 시간을 seed 반복에 쓴다.

    2초를 고른 이유는 셋이다.
      · 필기도 2초라 "같은 시간"으로 맞으면 모달 간 비교에 군더더기가 없다
      · DDK(파타카)는 초당 5~7음절이라 2초는 돼야 리듬 붕괴가 한 윈도우에 담긴다
      · 원본 전처리 코드(박민규)의 권장값도 window_sec=2.0 이다

왜 짧은 파일을 버리지 않고 패딩하나
    2초 미만이 전체로는 3.3%지만 **data1에만 몰려 있다** — 81명 중 27명(33%)이다.
    버리면 그 27명은 세그먼트가 0개가 되어 화자 자체가 사라진다. 화자가 줄면
    fold당 인원이 줄고 seed 노이즈가 커진다. -80dB(로그멜의 무음)로 뒤를 채워
    살린다. 모델은 그 구간을 "소리 없음"으로 학습한다.

subject_id
    fix_subject_id.py의 정정표를 적용한다. 같은 사람이 모음/파타카에서 서로 다른
    id를 받은 건이 99명 있어, 그대로 두면 5-fold에서 약 80% 확률로 한 사람이
    train과 test에 걸친다. 정정표가 없으면 --no-canon으로 건너뛸 수 있으나
    그때 나온 정확도는 낙관 편향이 섞인 값이다.

실행
    python make_voice_windows.py                       # 2초, 50% 겹침
    python make_voice_windows.py --seconds 1           # 다른 길이가 필요하면
    python make_voice_windows.py --no-canon            # 정정표 없이 (권장하지 않음)

산출
    voice_windows_2s.npz
        X          [N, 80, 200] float32   로그멜 (정규화 안 됨 — 분할 후 train에서만)
        y          [N]          int64     0=HC, 1=PD
        subject_id [N]          int64     정정 적용된 화자 id  ← 분할 키
        task       [N]          <U8       'a' 등
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
FRAMES_PER_SEC = 100          # 원본 전처리 hop 10ms
SILENCE_DB = -80.0            # 로그멜의 무음. 패딩 값


def iter_npz(root):
    """npz 하나씩 → (X[80,T], y, subject_id, task). LFS 포인터는 건너뛴다."""
    for f in sorted(root.rglob("*.npz")):
        if f.stat().st_size < 1024:                     # LFS 포인터(200바이트 남짓)
            yield f, None
            continue
        d = np.load(f, allow_pickle=True)
        yield f, (d["X"], int(d["y"]), int(d["subject_id"]), str(d["task"]))


def cut(x, win, hop):
    """[80, T] → [[80, win], ...]. 짧으면 -80dB로 뒤를 채워 1개를 만든다."""
    T = x.shape[1]
    if T < win:
        return [np.pad(x, ((0, 0), (0, win - T)), constant_values=SILENCE_DB)]
    return [x[:, s:s + win] for s in range(0, T - win + 1, hop)]


def main():
    ap = argparse.ArgumentParser(description="음성 고정 윈도우 생성")
    ap.add_argument("--root", type=Path, default=HERE / "전처리_data_all")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-canon", action="store_true",
                    help="subject_id 정정표를 적용하지 않는다 (누수 위험)")
    args = ap.parse_args()

    win = int(args.seconds * FRAMES_PER_SEC)
    hop = max(1, int(win * (1 - args.overlap)))
    out = args.out or HERE / f"voice_windows_{args.seconds:g}s.npz"

    if not args.root.exists():
        raise SystemExit(f"경로 없음: {args.root}")

    canon = (lambda s: s)
    if not args.no_canon:
        from fix_subject_id import canonical_subject_id
        canon = lambda s: int(canonical_subject_id(s))

    print("=" * 66)
    print(f"음성 윈도잉  {args.seconds:g}초({win}프레임) · 겹침 {args.overlap:.0%}(hop {hop})")
    print("=" * 66)

    X, y, sid, task = [], [], [], []
    n_file = n_pad = n_skip = 0
    lost = defaultdict(int)          # LFS 미다운로드 폴더별 집계

    for f, rec in iter_npz(args.root):
        if rec is None:
            n_skip += 1
            lost[f.parents[1].name] += 1
            continue
        x, lab, s, t = rec
        if x.ndim != 2:
            raise SystemExit(f"shape 예상 밖 {f.name}: {x.shape}")
        n_file += 1
        if x.shape[1] < win:
            n_pad += 1
        for seg in cut(x.astype(np.float32), win, hop):
            X.append(seg); y.append(lab); sid.append(canon(s)); task.append(t)

    if n_skip:
        print(f"\n⚠️ LFS 미다운로드 {n_skip}개를 건너뜀 — 실제 데이터가 아닙니다")
        for k, v in sorted(lost.items()):
            print(f"     {k}: {v}개")
        print("     git lfs pull 로 받은 뒤 다시 실행하세요.")
    if not X:
        raise SystemExit("읽은 파일이 없습니다.")

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    sid = np.asarray(sid, dtype=np.int64)
    task = np.asarray(task)

    print(f"\n원본 파일 {n_file}개 (그중 {win}프레임 미만 {n_pad}개는 -80dB 패딩)")
    print(f"세그먼트 {X.shape}  라벨 {np.bincount(y).tolist()} (0=HC, 1=PD)")

    # ── 검증 ──
    print("\n" + "=" * 66)
    print("검증")
    print("=" * 66)
    people = np.unique(sid)
    lab_of = defaultdict(set)
    for s, l in zip(sid, y):
        lab_of[s].add(l)
    conflict = [s for s, v in lab_of.items() if len(v) > 1]
    print(f"  화자 {len(people)}명")
    print(f"  한 화자가 두 라벨: {len(conflict)}건 "
          f"{'OK' if not conflict else 'FAIL ' + str(conflict[:5])}")
    if conflict:
        raise SystemExit("  → 라벨이 충돌합니다. subject_id 정정표를 확인하세요.")

    per = np.array(sorted(Counter(sid.tolist()).values()))
    print(f"  화자당 세그먼트: 최소 {per.min()} / 중앙 {int(np.median(per))} "
          f"/ 최대 {per.max()}")
    if per.max() > 20 * np.median(per):
        print("     ⚠️ 편차가 큽니다 — 학습 때 화자 가중 샘플러를 쓰는 편이 좋습니다")
    print(f"  5-fold 기준 fold당 test ≈ {len(people)/5:.0f}명 "
          f"→ 1명 ≈ {100/(len(people)/5):.1f}%p")
    assert len(X) == len(y) == len(sid) == len(task), "배열 길이 불일치"
    assert X.shape[1:] == (80, win), f"shape 오류 {X.shape}"
    assert np.isfinite(X).all(), "NaN/Inf 존재"

    np.savez_compressed(out, X=X, y=y, subject_id=sid, task=task)
    print(f"\n저장: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print("  정규화는 하지 않았습니다 — 분할 후 train fold에서만 fit하세요.")


if __name__ == "__main__":
    main()
