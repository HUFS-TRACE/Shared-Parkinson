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
SEEDS = (42, 1, 7)

# 4칸 간격 20밴드. 인접 밴드 |r|=0.64, 거리 4에서 0.45로 떨어진다(analyze_corr.py).
MEL20 = ",".join(str(i) for i in range(0, 80, 4))

# (축, 라벨, 모달, 추가인자)  — 기준 설정은 이미 돌렸으므로 넣지 않는다
AXES = {
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
    "d_model_v2": "sw_shared_v2_s9mel20_s{seed}.csv",
    "voice_v2": "voice_d64L6_s{seed}.csv",
    "shared_v2": "hw-voice_d64L6_s{seed}.csv",
}


def out_name(axis, label, seed):
    """축·설정·시드로 파일명을 직접 정한다.

    train.py가 붙이는 자동 이름을 여기서 재현하려 들면, 인자를 하나 추가할 때마다
    양쪽을 같이 고쳐야 하고 어긋나면 조용히 덮어쓴다. --out으로 못박는 편이 안전하다.
    """
    return f"sw_{axis}_{label}_s{seed}.csv"


def main():
    ap = argparse.ArgumentParser(description="patch·stride·d_model OFAT 스윕")
    ap.add_argument("--axis", nargs="*", default=list(AXES), choices=list(AXES))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--dry", action="store_true", help="명령만 출력하고 돌리지 않는다")
    args = ap.parse_args()

    jobs = []
    for ax in args.axis:
        for label, modal, extra in AXES[ax]:
            for seed in args.seeds:
                csv = ROOT / "results" / out_name(ax, label, seed)
                jobs.append((ax, label, modal, extra, seed, csv))

    todo = [j for j in jobs if not j[5].exists()]
    print(f"총 {len(jobs)}개 · 이미 있음 {len(jobs) - len(todo)}개 · 돌릴 것 {len(todo)}개")
    if not todo:
        print("모두 완료돼 있습니다.")
        return

    missing = {ax for ax in args.axis
               for s in args.seeds
               if not (ROOT / "results" / BASELINE[ax].format(seed=s)).exists()}
    if missing:
        print(f"⚠️ 비교 기준 CSV가 없는 축: {sorted(missing)} — 스윕해도 맞댈 대상이 없습니다")

    for ax, label, modal, extra, seed, csv in todo:
        cmd = [sys.executable, str(ROOT / "train.py"),
               "--modalities", modal, "--seed", str(seed),
               "--out", str(csv)] + extra
        print(f"\n{'=' * 70}\n[{ax}/{label}] seed {seed}  → {csv.name}\n  "
              + " ".join(cmd[1:]))
        if args.dry:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"  ❌ 실패 (코드 {r.returncode}) — 여기서 멈춥니다")
            return
        print(f"  ✅ {(time.time() - t0) / 60:.1f}분")

    print("\n끝났습니다. 축마다 기준과 시드를 짝지어 비교하세요.")


if __name__ == "__main__":
    main()
