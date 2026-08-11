"""
센서(채널) 축 Early Exit — 몇 개의 센서만 켜도 판정이 되는가.

세 번째 축이다
    깊이 축(층 건너뛰기)은 절감 상한이 층 수(6배)이고 실측에서 층 자체가 무의미했다.
    시간 축(윈도우 개수)은 상한이 크지만 확률 품질이 받쳐주지 못했다.
    센서 축은 성격이 다르다 — 절감이 추론 시점이 아니라 **하드웨어 단계**에서 온다.

    필기는 채널 독립(channel-independent) 패칭이라 채널을 배치 축에 접어 넣는다.
    따라서 연산량이 채널 수에 거의 정비례한다. 6채널 중 2개만 쓰면 약 1/3이다.
    게다가 센서를 덜 쓰면 기기 원가와 전력도 함께 내려간다. 스크리닝 기기를
    싸게 만드는 것이 목표라면 이 축이 가장 직접적이다.

왜 채널을 학습 전에 자르나
    RevIN은 채널별 affine 파라미터를 갖고, 분류기 입력은 num_channels x d_model이다.
    학습된 모델에서 채널을 빼면 두 곳 모두 차원이 어긋난다. 마스킹(0으로 채우기)은
    차원은 지키지만 연산량이 줄지 않아 절감 주장을 못 한다.

    그래서 부분집합마다 **독립 학습**한다. 그러면 "같은 설정에서 채널만 다르다"가
    성립해, 정확도 차이를 채널 탓으로 돌릴 수 있다. E1(깊이 대조군)에서 백본이
    어긋나 비교가 무효가 됐던 것과 같은 실수를 피하려는 것이다.

채널 순서는 어디서 왔나
    RandomForest 중요도(792파일 실측)를 쓴다. 절대값은 누수가 있어 신뢰할 수 없지만
    **상대 순위**는 쓸 수 있다.

        TiltX(0.211) > 압력(0.204) > 그립(0.183) > TiltY(0.157) > 마이크(0.123) ~ TiltZ(0.122)

    누적 순서로 넣어 "중요한 것부터 켜면 몇 개에서 포화되는가"를 본다.
    ⚠️ 이 순위는 RandomForest 기준이라 트랜스포머에서 다를 수 있다. 그래서 역순
    대조군(--reverse)도 함께 돌려, 순서가 결과를 만드는지 확인해야 한다.

실행
    python channel_exit.py --dry-run              # MFLOPs·파라미터 표만 (학습 없음)
    python channel_exit.py --seeds 42,1,7         # 누적 스윕
    python channel_exit.py --reverse --seeds 42   # 역순 대조군
"""
import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

# 필기 6채널 (SIBGRAPI 2016). 0-based 인덱스.
CH_NAME = {0: "마이크", 1: "그립", 2: "압력", 3: "TiltX", 4: "TiltY", 5: "TiltZ"}
# RandomForest 중요도 내림차순. 상대 순위만 쓴다.
CH_ORDER = [3, 2, 1, 4, 0, 5]


def measure(n_channels, seq_len=2000, patch_len=100, stride=50,
            d_model=64, n_layers=6, d_ff=None, n_heads=4):
    """채널 n개일 때 샘플 1개당 MFLOPs와 파라미터 수.

    채널을 배치 축에 접으므로 인코더 연산은 채널 수에 정비례한다. 분류기만
    num_channels x d_model 입력이라 채널 수에 비례해 파라미터가 늘어난다.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from models.shared_patchtst import SharedPatchTST

    specs = {"hw": dict(num_channels=n_channels, seq_len=seq_len,
                        patch_len=patch_len, stride=stride)}
    model = SharedPatchTST(specs, num_classes=2, d_model=d_model,
                           n_layers=n_layers, n_heads=n_heads,
                           d_ff=d_ff or d_model * 2)
    # ⚠️ train() 상태로 재야 한다. eval()이면 nn.TransformerEncoder가 fused
    #    fast-path로 빠져 카운터가 인코더 연산을 통째로 놓친다(실제로 겪음).
    model.cpu().train()
    x = torch.randn(1, n_channels, seq_len)
    with FlopCounterMode(display=False) as f:
        model(x, "hw")
    n_par = sum(p.numel() for p in model.parameters())
    return f.get_total_flops() / 1e6, n_par


def run(channels, seed, tag):
    """train.py를 서브프로세스로 호출. 학습 로직을 복제하지 않으려는 것이다."""
    cmd = [sys.executable, "train.py", "--modalities", "hw",
           "--hw-channels", ",".join(str(c) for c in channels),
           "--seed", str(seed),
           "--out", f"results/ch{len(channels)}{tag}_d64L6_s{seed}.csv"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"학습 실패: 채널 {channels} seed {seed}")
    for line in r.stdout.splitlines():
        if line.startswith("hw "):
            return line
    return "(결과 줄을 못 찾음)"


def main():
    ap = argparse.ArgumentParser(description="센서 축 Early Exit 스윕")
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--reverse", action="store_true",
                    help="중요도 역순으로 누적 — 순서가 결과를 만드는지 확인하는 대조군")
    ap.add_argument("--dry-run", action="store_true", help="MFLOPs 표만, 학습 없음")
    args = ap.parse_args()

    order = CH_ORDER[::-1] if args.reverse else CH_ORDER
    tag = "r" if args.reverse else ""

    print("=" * 76)
    print(f"센서 축 — {'역순' if args.reverse else '중요도순'} 누적")
    print("=" * 76)
    print(f"  순서: " + " → ".join(f"{CH_NAME[c]}" for c in order))

    full_f, full_p = measure(6)
    print(f"\n  {'채널 수':>7}{'추가된 센서':<12}{'MFLOPs':>10}{'전체 대비':>10}{'파라미터':>10}")
    plan = []
    for k in range(1, 7):
        sel = order[:k]
        f, p = measure(k)
        plan.append((sel, f, p))
        print(f"  {k:>7}  {CH_NAME[order[k-1]]:<12}{f:>10.1f}{f/full_f*100:>9.1f}%{p:>10,}")

    if args.dry_run:
        print("\n  --dry-run: 학습하지 않음")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\n  학습 {len(plan)}구성 x seed {len(seeds)}개 = {len(plan)*len(seeds)}회")
    print(f"  (필기 단독 1회 약 4분 → 총 {len(plan)*len(seeds)*4}분 예상)\n")

    for sel, f, _ in plan:
        for seed in seeds:
            print(f"  ── 채널 {len(sel)}개 {[CH_NAME[c] for c in sel]} seed {seed} "
                  f"({f:.0f} MFLOPs)")
            print(f"     {run(sel, seed, tag)}")

    print(f"\n  결과 CSV: results/ch*{tag}_d64L6_s*.csv")
    print(f"  집계는 analyze_channel.py 또는 CSV를 직접 읽어 비교할 것.")


if __name__ == "__main__":
    main()
