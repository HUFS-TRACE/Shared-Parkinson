"""d_model · patch_len · stride를 **한 번에 하나씩만** 바꿔 돌린다 (OFAT).

왜 하나씩인가
    셋을 동시에 바꾸면 정확도가 변했을 때 무엇 때문인지 못 가른다. 특히 patch_len과
    stride는 둘 다 토큰 개수를 바꾸므로, 같이 움직이면 "토큰이 많아서"인지 "한 토큰이
    담는 시간이 길어서"인지 구별되지 않는다. 그래서 축을 이렇게 갈랐다.

        patch 축   겹침 50%를 고정하고 patch_len만 (stride = patch/2)
                   → 한 토큰이 담는 시간이 바뀐다
        stride 축  patch_len을 고정하고 stride만
                   → 샘플링 촘촘함(토큰 수)만 바뀐다
        d_model 축 patch·stride를 고정하고 d_model만

    d_ff는 128로 고정한다. 보통 d_model의 2배로 같이 키우지만, 그러면 d_model 효과와
    d_ff 효과가 섞인다. 축 하나만 움직이는 것이 이 실험의 요점이다.

왜 patch/stride는 모달 단독으로 도나
    인코더가 공유라, 음성 패칭을 바꾸면 인코더가 달라지고 그 인코더로 판정하는 필기
    정확도까지 따라 움직인다. 공유 모델에서 두 모달 패칭을 동시에 스윕하면 축이
    다시 섞인다. 그래서 패칭은 단독 모델에서 깨끗하게 재고, 이긴 설정만 공유로 옮겨
    확인한다. d_model은 공유 파라미터라 처음부터 공유 모델에서 잰다.

꼬리 손실
    unfold는 딱 떨어지지 않는 꼬리를 잘라낸다. 아래 설정은 음성 기준(p24/s12,
    150프레임 중 144만 사용 = 6프레임 손실)을 빼고 전부 손실 0으로 골랐다.
    기준 자체가 손실을 안고 있으므로, 음성 stride 9는 스윕인 동시에 그 손실의
    교정이기도 하다.

시드
    시드마다 피험자 정확도가 1.0~2.7%p 흔들린다(3시드 실측). 설정 간 차이가 그보다
    작으면 단일 시드로는 못 가린다. 그래서 모든 설정을 같은 시드 3개로 돌려
    시드를 짝지어 비교한다.

실행
    python run_sweep.py --dry                 # 무엇을 돌릴지만 출력
    python run_sweep.py --axis voice_stride   # 한 축만
    python run_sweep.py                       # 전부 (오래 걸림 — --dry로 먼저 확인)

    이미 결과 CSV가 있으면 건너뛴다. 중간에 끊겨도 다시 돌리면 이어서 간다.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

# 시드는 **미리 정해 둔다.** 3개를 돌려 보고 4·5번째를 고르면 결과를 보고 고르는 것이
# 되어 리뷰에서 문제가 된다. "널리 쓰이는 값 중 작은 것부터"라는 규칙으로 정했다.
SEEDS = (42, 1, 7, 0, 123)
SEEDS_CORE = (42, 1, 7)          # 1차로 돌릴 3개

# 4칸 간격 20밴드. 인접 밴드 |r|=0.64, 거리 4에서 0.45로 떨어진다(analyze_corr.py).
MEL20 = ",".join(str(i) for i in range(0, 80, 4))

# (축, 라벨, 모달, 추가인자)  — 기준 설정은 이미 돌렸으므로 넣지 않는다
AXES = {
    # ── 기준선 ──
    # 분할을 바꾸면(--split-shuffle) 기준선도 다시 세워야 한다. 기존 hw_d64L6_*.csv
    # 등은 옛 분할에서 나온 값이라 섞어 쓸 수 없다. 확률까지 저장해 유보·캐스케이드·
    # 임계값 분석이 재학습 없이 돌게 한다.
    "base": [
        ("hw", "hw", ["--save-probs"]),                                # 필기 단독 6채널
        ("hw3", "hw", ["--hw-channels", "3,2,1", "--save-probs"]),     # 센서 축 · 캐스케이드
        ("voice", "voice", ["--voice-patch", "24", "--voice-stride", "9",
                            "--voice-channels", MEL20, "--save-probs"]),
        ("shared", "hw,voice", ["--voice-patch", "24", "--voice-stride", "9",
                                "--voice-channels", MEL20, "--save-probs"]),
    ],
    # ③ 깊이 축 — 층을 몇 개 써야 하는가. 기준은 base/hw (L6)다.
    #
    #   기존 t=0.51("정적 L1 ≈ L6")은 다른 레포에서 옛 분할로 낸 값이다.
    #   7절에서 "옛 분할 값은 신뢰할 수 없다"고 쓰면서 3절이 그 값을 쓰면 자기모순이다.
    #   세 축을 같은 조건에서 재기 위해 이 레포·새 분할로 다시 잰다.
    #
    #   Early Exit의 전제는 "층을 더 쌓으면 나아진다"이다. 그것이 성립하지 않으면
    #   층을 골라 멈출 이유가 없다. 그 전제를 직접 확인하는 축이다.
    "depth": [
        ("L1", "hw", ["--n-layers", "1", "--save-probs"]),
        ("L2", "hw", ["--n-layers", "2"]),
        ("L3", "hw", ["--n-layers", "3"]),
    ],
    # ①·⑦의 "고치기 전" 기준선. 음성 80밴드 p24/s12 — 꼬리 6프레임을 버리던 설정이다.
    # 회당 22분으로 비싸서 tier 2로 미뤘다. tier 1 주장에는 필요하지 않다.
    "base_v1": [
        ("voice1", "voice", ["--save-probs"]),
    ],
    # d_model만. 공유 파라미터라 공유 모델에서 잰다 (기준 d64)
    "d_model": [
        ("d32", "hw,voice", ["--d-model", "32"]),
        ("d128", "hw,voice", ["--d-model", "128"]),
    ],
    # 필기 patch (겹침 50% 고정). 기준 p100/s50 = 39토큰
    "hw_patch": [
        ("p50", "hw", ["--hw-patch", "50", "--hw-stride", "25"]),     # 79토큰
        ("p200", "hw", ["--hw-patch", "200", "--hw-stride", "100"]),  # 19토큰
    ],
    # 필기 stride (patch 100 고정)
    "hw_stride": [
        ("s25", "hw", ["--hw-patch", "100", "--hw-stride", "25"]),    # 77토큰 겹침75%
        ("s100", "hw", ["--hw-patch", "100", "--hw-stride", "100"]),  # 20토큰 겹침0
    ],
    # 음성 patch (겹침 50% 고정). 기준 p24/s12 = 11토큰
    # p48/s24로 키우면 꼬리가 6프레임 잘리므로 p50/s25를 쓴다(손실 0, 토큰 5)
    # 겹침 50% 고정, patch_len만. 채널은 mel20으로 고정한다 — 80밴드로 돌면
    # p12(24토큰)만 회당 40분이라 축 하나에 2시간이 든다. mel20은 성능이 같고 1/4이다.
    # 그래서 비교 기준도 같은 mel20에 기준 패칭(p24/s12)인 sw_voice_mel_mel20이다.
    "voice_patch": [
        ("p12", "voice", ["--voice-patch", "12", "--voice-stride", "6",
                          "--voice-channels", MEL20]),                      # 24토큰
        ("p50", "voice", ["--voice-patch", "50", "--voice-stride", "25",
                          "--voice-channels", MEL20]),                      # 5토큰
    ],
    # 음성 stride (patch 24 고정). s9는 꼬리 손실 0 — ①번 문제의 교정안이다
    "voice_stride": [
        ("s9", "voice", ["--voice-patch", "24", "--voice-stride", "9"]),    # 15토큰 손실0
        ("s25", "voice", ["--voice-patch", "25", "--voice-stride", "25"]),  # 6토큰 겹침0
    ],
    # ④ 분류기 하나로 채널 상관을 감당하는가. 필기 단독에서 잰다.
    # 음성으로 하면 mlp 헤드가 165k 파라미터라 공유 비율이 91%→52%로 떨어져,
    # "공유 인코더" 비교가 아니라 헤드 용량 싸움이 된다.
    "head": [
        ("perch", "hw", ["--head", "perch"]),   # 용량↑ 상호작용 없음 (대조군)
        ("mlp", "hw", ["--head", "mlp"]),       # 용량↑ 상호작용 있음
    ],
    # ⑤ 사람당 윈도우 상한. 필기 중앙 163 / 최대 801이고, 환자가 1인당 2.4배 많다.
    # 상한 150이면 윈도우 라벨비가 1:1.77 → 1:0.86이 되어 사람 비율(1:0.74)에 붙는다.
    "window_cap": [
        ("w150", "hw", ["--max-windows", "150"]),
        ("w80", "hw", ["--max-windows", "80"]),
    ],
    # ⑦ 멜밴드 축소. 인접 밴드 |r|=0.64라 80개는 같은 것을 여러 번 넣는 셈이다.
    # 4칸 간격 20개(거리 4의 |r|=0.45)와 8칸 간격 10개를 본다.
    "voice_mel": [
        ("mel20", "voice", ["--voice-channels", ",".join(str(i) for i in range(0, 80, 4))]),
        ("mel10", "voice", ["--voice-channels", ",".join(str(i) for i in range(0, 80, 8))]),
    ],
    # ⑧ A+B — 3채널(TiltX·압력·그립)에서도 유보가 작동하는가.
    # 확률을 저장해야 verify_abstain.py가 재학습 없이 읽는다.
    "ab": [
        ("hw3", "hw", ["--hw-channels", "3,2,1", "--save-probs"]),
    ],
    # ② 유보가 환자 쪽으로 쏠리는 원인. 필기 단독 3팔, 전부 확률 저장.
    #
    #   진단: 민감도가 낮은 시드일수록 보류집단 환자 비율이 높았다(0.35→+35%p,
    #   0.96→-2.6%p). 원인은 class_weights가 **윈도우 라벨**을 세는 데 있다.
    #   필기는 환자 윈도우가 2.1배라 가중치가 HC 1.384 : PD 0.783으로 뒤집힌다.
    #   사람 기준으로는 환자가 26 대 35로 소수인데도 그렇다.
    #
    #   cw_window 는 기존과 같은 설정을 확률까지 저장해 다시 돌린 것이다. 기존
    #   hw_d64L6_*.csv에는 확률이 없어 편향을 잴 수 없으므로, 같은 자로 재려면
    #   이 팔이 있어야 한다.
    "class_weight": [
        ("cwwindow", "hw", ["--class-weight", "window", "--save-probs"]),
        ("cwsubject", "hw", ["--class-weight", "subject", "--save-probs"]),
        ("cap150", "hw", ["--max-windows", "150", "--save-probs"]),
    ],
    # ①+⑦ 합본. 두 축이 따로는 이겼는데 같이 걸면 어떤지는 따로 재야 한다.
    #   stride 9  꼬리 손실 0 (토큰 11 → 15)
    #   mel20     4칸 간격 20밴드 (인접 |r|=0.64라 80개는 중복)
    # 둘 다 "버리는 것을 줄이거나 중복을 줄인다"는 같은 방향이라 겹칠 수 있다.
    "voice_v2": [
        ("s9mel20", "voice", ["--voice-patch", "24", "--voice-stride", "9",
                              "--voice-channels", MEL20, "--save-probs"]),
    ],
    # d_model을 v2 기준에서 다시 잰다. 앞의 d_model 축은 v1(음성 80밴드)에서 쟀는데,
    # v2가 새 기준선이 됐으므로 그 위에서도 같은 결론이 나오는지 확인해야 한다.
    # 음성이 20밴드라 회당 25분 -> 9분으로 줄어든다.
    "d_model_v2": [
        ("d32", "hw,voice", ["--d-model", "32", "--voice-patch", "24", "--voice-stride", "9",
                             "--voice-channels", MEL20]),
        ("d128", "hw,voice", ["--d-model", "128", "--voice-patch", "24", "--voice-stride", "9",
                              "--voice-channels", MEL20]),
    ],
    # 위 설정으로 공유 모델을 다시 세운다. 기준선 전체가 이 값으로 갱신된다.
    # 필기 쪽은 손대지 않으므로 "음성 설정만 바뀌었을 때 공유 효과가 어떻게 되는가"를
    # 그대로 읽을 수 있다.
    "shared_v2": [
        ("s9mel20", "hw,voice", ["--voice-patch", "24", "--voice-stride", "9",
                                 "--voice-channels", MEL20, "--save-probs"]),
    ],
}

# 패칭 축은 단독 모델이라 비교 기준도 단독이어야 한다. 이미 있는 CSV들이다.
BASELINE = {
    "base": None,          # 기준선 자체 — 맞댈 대상이 없다
    "base_v1": None,
    "d_model": "hw-voice_d64L6_s{seed}.csv",
    "hw_patch": "hw_d64L6_s{seed}.csv",
    "hw_stride": "hw_d64L6_s{seed}.csv",
    "voice_patch": "sw_voice_mel_mel20_s{seed}.csv",
    "voice_stride": "voice_d64L6_s{seed}.csv",
    "head": "hw_d64L6_s{seed}.csv",
    "window_cap": "hw_d64L6_s{seed}.csv",
    "voice_mel": "voice_d64L6_s{seed}.csv",
    "ab": "hw_d64L6_s{seed}.csv",        # A(3채널)의 비교 대상은 6채널이다
    "class_weight": "hw_d64L6_s{seed}.csv",
    "depth": "hw_d64L6_s{seed}.csv",
    "d_model_v2": "sw_shared_v2_s9mel20_s{seed}.csv",
    "voice_v2": "voice_d64L6_s{seed}.csv",
    "shared_v2": "hw-voice_d64L6_s{seed}.csv",
}


# 끊어서 돌리기 위한 묶음. 앞 tier가 뒤 tier의 전제다 — 뒤 tier의 비교 기준이 앞에 있다.
TIERS = {
    1: ["base"],                                                  # 논문 주 주장 전부
    0: ["depth"],                                                 # 깊이 축 (3절)
    2: ["base_v1", "voice_stride", "voice_mel", "d_model_v2"],     # ①⑦ + d_model
    3: ["head", "window_cap", "hw_patch", "hw_stride"],            # "튜닝은 영향 없다"
}

# 분할을 바꾸면(--split-shuffle) 옛 기준선 CSV를 쓸 수 없다. 그때 무엇과 맞대나.
# --rebase 를 주면 BASELINE 대신 이 표를 쓴다.
REBASE = {
    "hw_patch": "base_hw", "hw_stride": "base_hw",
    "head": "base_hw", "window_cap": "base_hw", "class_weight": "base_hw",
    "ab": "base_hw",                       # 3채널의 상대는 6채널
    "depth": "base_hw",                    # L1·L2·L3 의 상대는 L6이다
    "voice_stride": "base_v1_voice1",      # ① 고치기 전 설정과 맞댄다
    "voice_mel": "base_v1_voice1",         # ⑦ 도 마찬가지
    "voice_patch": "base_voice",
    "d_model": "base_shared", "d_model_v2": "base_shared",
    "voice_v2": "base_v1_voice1", "shared_v2": "base_shared",
}

# 실측 회당 소요(분). ETA용이라 정확할 필요는 없고 감만 주면 된다.
MINUTES = {"base": 6, "base_v1": 22, "depth": 3, "ab": 3, "voice_stride": 6, "voice_mel": 4,
           "d_model_v2": 11, "head": 5, "window_cap": 3, "hw_patch": 7,
           "hw_stride": 7, "d_model": 30, "voice_patch": 6, "class_weight": 5,
           "voice_v2": 6, "shared_v2": 9}


def out_name(axis, label, seed, tag):
    """축·설정·시드로 파일명을 정한다.

    train.py가 붙이는 자동 이름을 여기서 재현하려 들면, 인자를 하나 추가할 때마다
    양쪽을 같이 고쳐야 하고 어긋나면 조용히 덮어쓴다. --out으로 못박는 편이 안전하다.

    tag가 접두사다. **분할을 바꾸면 반드시 새 tag를 써야 한다** — 파일명이 겹치면
    "이미 있음"으로 건너뛰고, 두 분할의 값이 한 표에 섞여 짝 t-검정이 무효가 된다.
    """
    return f"{tag}_{axis}_{label}_s{seed}.csv"


def fmt(minutes):
    m = int(round(minutes))
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"


def main():
    ap = argparse.ArgumentParser(description="OFAT 스윕 — tier로 끊어 돌린다")
    ap.add_argument("--tier", nargs="*", type=int, choices=sorted(TIERS),
                    help="묶음 단위로 돌린다. 여러 개 주면 순서대로")
    ap.add_argument("--axis", nargs="*", choices=list(AXES),
                    help="축을 직접 지정 (--tier와 같이 쓰면 합집합)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS_CORE),
                    help="기본은 1차 3개. 미리 정해 둔 전체는 SEEDS 참고")
    ap.add_argument("--tag", default="sw",
                    help="결과 파일 접두사. 분할을 바꾸면 반드시 새 값을 줄 것")
    ap.add_argument("--split-shuffle", action="store_true",
                    help="시드가 fold 분할까지 바꾸게 한다 (이슈 #15)")
    ap.add_argument("--rebase", action="store_true",
                    help="비교 기준을 base 축 산출물로 바꾼다. --split-shuffle과 같이 쓴다")
    ap.add_argument("--dry", action="store_true", help="계획만 출력하고 돌리지 않는다")
    args = ap.parse_args()

    axes = [a for t in sorted(args.tier or []) for a in TIERS[t]]
    axes += [a for a in (args.axis or []) if a not in axes]
    if not axes:
        axes = list(AXES)

    # 분할이 다른 결과가 같은 파일명으로 섞이는 것이 이 스크립트에서 가장 위험하다.
    # 조용히 잘못되기 때문에 아예 실행을 막는다.
    if args.split_shuffle and args.tag == "sw":
        raise SystemExit(
            "--split-shuffle 을 쓰면서 --tag 를 기본값으로 두면, 옛 분할 결과를\n"
            "'이미 있음'으로 건너뛰고 두 분할의 값이 한 표에 섞입니다.\n"
            "짝 t-검정이 조용히 무효가 됩니다. --tag sw2 처럼 새 접두사를 주세요.")

    base_of = REBASE if args.rebase else None
    jobs = []
    for ax in axes:
        for label, modal, extra in AXES[ax]:
            for seed in args.seeds:
                csv = ROOT / "results" / out_name(ax, label, seed, args.tag)
                jobs.append((ax, label, modal, extra, seed, csv))

    todo = [j for j in jobs if not j[5].exists()]
    print(f"tier {args.tier or '-'} · 축 {len(axes)}개 · 시드 {args.seeds} · tag={args.tag}"
          + ("  [분할 섞기 ON]" if args.split_shuffle else ""))
    print(f"총 {len(jobs)}개 · 이미 있음 {len(jobs) - len(todo)}개 · 돌릴 것 {len(todo)}개"
          f" · 예상 {fmt(sum(MINUTES.get(j[0], 6) for j in todo))}")

    # tier별 소요를 미리 보여준다 — 어디서 끊을지 정하기 위해
    if len(axes) > 1:
        print()
        for t, tax in sorted(TIERS.items()):
            mine = [j for j in todo if j[0] in tax]
            if mine:
                print(f"  tier {t}: {len(mine):>3}회  "
                      f"{fmt(sum(MINUTES.get(j[0], 6) for j in mine)):>7}"
                      f"   {', '.join(tax)}")
        etc = [j for j in todo if not any(j[0] in tax for tax in TIERS.values())]
        if etc:
            print(f"  기타  : {len(etc):>3}회  "
                  f"{fmt(sum(MINUTES.get(j[0], 6) for j in etc)):>7}")
    if not todo:
        print("\n모두 완료돼 있습니다.")
        return

    if base_of:
        miss = sorted({ax for ax in axes if ax in base_of for sd in args.seeds
                       if not (ROOT / "results" /
                               f"{args.tag}_{base_of[ax]}_s{sd}.csv").exists()})
        if miss:
            print(f"\n⚠️ 비교 기준이 아직 없는 축: {miss}")
            print("   tier 1(base)을 먼저 돌리면 채워집니다. 학습 자체는 진행됩니다.")

    print("\n중단은 Ctrl+C. 끝난 회차는 CSV로 남고, 같은 명령으로 다시 돌리면 이어서 갑니다.")
    done = 0.0
    for i, (ax, label, modal, extra, seed, csv) in enumerate(todo, 1):
        cmd = [sys.executable, str(ROOT / "train.py"),
               "--modalities", modal, "--seed", str(seed), "--out", str(csv)] + extra
        if args.split_shuffle:
            cmd.append("--split-shuffle")
        left = sum(MINUTES.get(j[0], 6) for j in todo[i - 1:])
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(todo)}] {ax}/{label} seed {seed} → {csv.name}"
              f"   (남은 예상 {fmt(left)})")
        print("  " + " ".join(cmd[1:]))
        if args.dry:
            continue
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            # 쓰다 만 CSV가 남으면 다음 실행이 "완료"로 착각한다
            if csv.exists():
                csv.unlink()
                print(f"\n  중단 — 쓰다 만 {csv.name} 을 지웠습니다.")
            print(f"  {i - 1}/{len(todo)}회 완료 ({fmt(done)}). 같은 명령으로 이어서 갑니다.")
            return
        if r.returncode != 0:
            if csv.exists():
                csv.unlink()
            print(f"  ❌ 실패 (코드 {r.returncode}) — 여기서 멈춥니다")
            return
        done += (time.time() - t0) / 60
        print(f"  ✅ {(time.time() - t0) / 60:.1f}분  (누적 {fmt(done)})")

    print(f"\n끝났습니다. 총 {fmt(done)}")
    print(f"집계: python analyze_sweep.py --tag {args.tag}"
          + (" --rebase" if args.rebase else ""))


if __name__ == "__main__":
    main()
