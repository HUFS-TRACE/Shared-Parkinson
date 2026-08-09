"""
결과 집계 — 단독 vs 공유를 모달별·seed별로 맞대어 판정한다.

    python test.py                          # results/ 안의 CSV를 전부 읽어 비교
    python test.py --result-dir results     # 경로 지정

⚠️ 판정 기준에 관한 메모
    seed가 몇 개 안 되면 t 검정의 검정력이 매우 낮다. 그래서 t값 하나로 결론내지
    않고 아래 셋을 함께 본다.

      ① 부호 일관성 — seed 전부에서 같은 방향인가
      ② seed 간 t ≥ 2
      ③ 개선폭 > **공유 모델 자체의** seed 변동폭

    ③이 특히 중요하다. 단독의 seed 변동폭이 작다는 것만으로 개선을 주장하면
    공유 쪽 불안정성을 놓친다. 공유는 두 모달의 기울기가 한 인코더로 들어와
    단독보다 불안정하다 (실측: 음성 AUC 단독 0.004 vs 공유 0.017).
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Windows 한국어 로캘에서 출력 리다이렉트 시 cp949가 되어 이모지에서 죽는다
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
METRICS = [("roc_auc", "ROC-AUC"), ("subject_acc", "피험자 정확도"),
           ("window_acc", "윈도우 정확도")]


def read_csv(path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cols = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        r = dict(zip(cols, line.split(",")))
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        rows.append(r)
    return rows


def collect(result_dir):
    """{(모달, 종류): {seed: row}} — 종류는 solo / share.

    파일명에서 모달 구성과 seed를 읽는다. 예)
        voice_d64L6_s42.csv      → 음성 단독
        hw-voice_d64L6_s42.csv   → 2모달 공유
    """
    out = defaultdict(dict)
    for p in sorted(Path(result_dir).glob("*.csv")):
        if p.name.startswith("smoke"):
            continue
        m = re.search(r"_s(\d+)\.csv$", p.name)
        seed = int(m.group(1)) if m else 42
        for row in read_csv(p):
            modality = row.get("modality")
            if not isinstance(modality, str):
                continue
            # 한 파일에 두 모달이 있으면 공유 실행이다
            kind = "share" if p.name.split("_")[0].count("-") else "solo"
            out[(modality, kind)][seed] = row
    return out


def report(modality, table):
    solo, share = table.get((modality, "solo"), {}), table.get((modality, "share"), {})
    seeds = sorted(set(solo) & set(share))
    print("\n" + "=" * 72)
    print(f"  {modality.upper()}   단독 seed {sorted(solo)} · 공유 seed {sorted(share)}"
          f"  → 비교 가능 {seeds}")
    print("=" * 72)
    if not seeds:
        print("  비교 가능한 seed가 없습니다.")
        return

    for key, label in METRICS:
        a = np.array([solo[s][key] for s in seeds])
        b = np.array([share[s][key] for s in seeds])
        d = b - a
        print(f"\n  [{label}]")
        print(f"    {'seed':>6}{'단독':>10}{'공유':>10}{'차이':>10}")
        for s, x, y in zip(seeds, a, b):
            print(f"    {s:>6}{x:>10.4f}{y:>10.4f}{y - x:>+10.4f}")
        print(f"    {'평균':>6}{a.mean():>10.4f}{b.mean():>10.4f}{d.mean():>+10.4f}")

        sign = ("일관(+)" if (d > 0).all() else "일관(−)" if (d < 0).all()
                else f"엇갈림 ({int((d > 0).sum())}+/{int((d < 0).sum())}−)")
        share_sp = b.max() - b.min()
        print(f"    부호 {sign} · seed 변동폭 단독 {a.max() - a.min():.4f}"
              f" / **공유 {share_sp:.4f}**")

        if len(seeds) >= 2:
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 0 else float("inf")
            verdict = ("효과 있음" if abs(t) >= 2 else
                       "판단 보류" if abs(t) >= 1 else "노이즈에 묻힘")
            print(f"    seed 간 t = {t:+.2f} (n={len(d)}) ⇒ {verdict}")
            if share_sp >= abs(d.mean()):
                print(f"    ⚠️ 공유 변동폭({share_sp:.4f})이 평균 개선폭"
                      f"({abs(d.mean()):.4f}) 이상 — 개선을 주장할 수 없다")


def main():
    ap = argparse.ArgumentParser(description="단독 vs 공유 결과 집계")
    ap.add_argument("--result-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    if not args.result_dir.exists():
        raise SystemExit(f"결과 폴더가 없습니다: {args.result_dir}\n"
                         f"먼저 python train.py 로 실행 결과를 만드세요.")

    table = collect(args.result_dir)
    if not table:
        raise SystemExit(f"{args.result_dir} 에 읽을 CSV가 없습니다.")

    print("=" * 72)
    print("공유 인코더 집계 — 단독 vs 공유 (같은 설정·같은 fold)")
    print("=" * 72)
    for m in sorted({k[0] for k in table}):
        report(m, table)

    print("\n" + "=" * 72)
    print("""읽는 법
  · ① 부호 일관 + ② t ≥ 2 + ③ 개선폭 > 공유 변동폭 — 셋 다 만족해야 '효과 있음'
  · seed가 적으면 t 검정력이 낮다. t < 2 라도 '효과 없음'이 아니라 '판정 불가'다.
  · 피험자 정확도는 표본이 작은 모달에서 지표로 쓰기 어렵다
    (필기 fold당 12명 → 1명 8.3%p · 음성 fold당 47명 → 1명 2.1%p).""")


if __name__ == "__main__":
    main()
