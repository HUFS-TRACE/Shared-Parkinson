"""판정 임계값을 어디에 두어야 하는가 — 0.5는 왜 틀렸나.

무엇을 발견했나
    필기에서 확률 파일 18개를 보면 사람 단위 AUC는 0.888~0.964로 안정적인데
    민감도는 0.346~0.962로 흔들린다. 순위는 잘 매기는데 자르는 지점이 시드마다
    다르다는 뜻이다. 실제로 확률의 중앙값이 시드별로 0.153 / 0.409 / 0.516이다.

    0.5로 자르면 확률이 낮은 쪽으로 쏠린 시드에서는 거의 모두를 정상으로 찍는다.
    그래서 환자가 경계 근처에 남고, 판정 보류가 환자를 먼저 걷어낸다.
    (민감도 ↔ 유보 편중 순위상관 -0.787, n=18)

세 가지 임계값을 맞대 본다
    fixed     0.5 고정                       (기존)
    val_j     val에서 Youden J 최대          val 라벨을 쓴다
    prior     확률 상위 p%를 환자로          p = 학습 유병률(26/61). 라벨을 안 쓴다

    prior는 test 확률의 **분포**를 본다(라벨은 안 본다). 한 사람씩 판정할 때는
    쓸 수 없고, 코호트를 한 번에 스크리닝할 때만 성립한다. 그 조건을 밝히고 써야 한다.
    val 확률의 분위수로 임계값을 정해 test에 옮기는 변형도 넣었다(prior_val) —
    val은 조기 종료에 쓰인 집합이라 확률이 낙관적으로 몰려 잘 옮겨가지 않는다.

실행
    python analyze_threshold.py
    python analyze_threshold.py --prev 0.426
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_abstain_bias import per_subject  # noqa: E402
from sequential_exit import calibrate, logit, sigmoid  # noqa: E402


def val_subjects(d, T, c):
    """val 확률을 사람 단위로. fold마다 사람이 겹치므로 (fold, 사람)이 단위다."""
    p = sigmoid(logit(d["val_prob"]) / T - c)
    key = np.char.add(np.char.add(d["val_fold"].astype(str), "|"),
                      d["val_subject_id"].astype(str))
    u = np.unique(key)
    return (np.array([p[key == s].mean() for s in u]),
            np.array([int(d["val_y"][key == s][0]) for s in u]))


def thresholds(mp, vmp, vty, prev):
    """이름 → 임계값."""
    cand = np.unique(vmp)
    youden = max(cand, key=lambda t: (vmp[vty == 1] >= t).mean()
                 + (vmp[vty == 0] < t).mean()) if len(cand) else 0.5
    return {
        "fixed": 0.5,
        "val_j": float(youden),
        "prior_val": float(np.quantile(vmp, 1 - prev)),
        "prior": float(np.quantile(mp, 1 - prev)),
    }


def main():
    ap = argparse.ArgumentParser(description="판정 임계값 비교")
    ap.add_argument("--probs", nargs="*", default=None)
    ap.add_argument("--prev", type=float, default=26 / 61,
                    help="학습 데이터의 환자 비율. test를 보고 정한 값이 아니다")
    args = ap.parse_args()

    files = args.probs or sorted(
        p for p in (ROOT / "results").glob("probs_hw_*.npz") if "smoke" not in p.name)
    if not files:
        raise SystemExit("확률 파일이 없습니다.")

    names = ("fixed", "val_j", "prior_val", "prior")
    acc = {k: [] for k in names}
    sen = {k: [] for k in names}
    spe = {k: [] for k in names}
    aucs = []

    print(f"{'실험':<32}{'AUC':>7}" + "".join(f"{k:>11}" for k in names))
    for f in files:
        d = np.load(f, allow_pickle=True)
        T, c = calibrate(d)
        p = sigmoid(logit(d["prob"]) / T - c)
        mp, ty, _ = per_subject(p, d["y"], d["subject_id"], d["idx"])
        vmp, vty = val_subjects(d, T, c)
        aucs.append(roc_auc_score(ty, mp))

        line = f"{Path(f).stem.split('probs_hw_')[-1]:<32}{aucs[-1]:>7.3f}"
        for k, t in thresholds(mp, vmp, vty, args.prev).items():
            pred = (mp >= t).astype(int)
            acc[k].append(float((pred == ty).mean()))
            sen[k].append(float(pred[ty == 1].mean()))
            spe[k].append(float(1 - pred[ty == 0].mean()))
            line += f"{acc[k][-1]:>11.3f}"
        print(line)

    print(f"\nAUC(사람) {np.mean(aucs):.4f}  범위 {min(aucs):.3f}~{max(aucs):.3f}"
          f"  ← 순위 능력은 안정적이다")
    print(f"\n{'임계값':>11}{'정확도':>9}{'민감도':>9}{'특이도':>9}{'민감도 범위':>18}")
    for k in names:
        a, s = np.array(acc[k]), np.array(sen[k])
        print(f"{k:>11}{a.mean():>9.4f}{s.mean():>9.3f}{np.mean(spe[k]):>9.3f}"
              f"{f'{s.min():.3f}~{s.max():.3f}':>18}")

    base = np.array(acc["fixed"])
    print()
    for k in names[1:]:
        d = np.array(acc[k]) - base
        print(f"  {k:>10} vs fixed: {d.mean():+.4f}  "
              f"(오름 {(d > 0).sum()}/{len(d)} · 내림 {(d < 0).sum()}/{len(d)})")
    print("\n  prior는 test 확률 분포를 본다 — 코호트를 한 번에 볼 때만 쓸 수 있다.")
    print("  val 기반 두 방법이 지는 것은 val이 조기 종료에 쓰여 확률이 낙관적이기 때문이다.")


if __name__ == "__main__":
    main()
