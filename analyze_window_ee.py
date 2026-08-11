"""
윈도우 개수 축 Early Exit — 사전 점검 (게이트).

무엇을 하나
    한 사람의 윈도우를 1개씩 순서대로 넣다가 확신이 서면 나머지를 안 보고 끝내는
    구조가 **이 데이터에서 성립하는지**를 학습 없이 판정한다. 확률 파일만 읽으므로
    수초에 끝난다. 여기서 통과해야 본 실험(sequential_exit.py)이 의미가 있다.

왜 먼저 확인해야 하나
    순차 판정은 "증거를 모으면 경계에 닿는다"를 전제한다. 그런데 어떤 사람의 윈도우가
    **일관되게 틀린 쪽**을 가리키면, 증거를 모을수록 틀린 경계에 빨리 닿는다.
    그런 사람이 많으면 조기 종료는 정확도를 깎으면서 시간만 줄이는 장치가 된다.

    윈도우별 증거를 로그오즈로 두고 사람마다 평균을 낸다.

        e_i = logit(p_i) - c          한 윈도우가 주는 증거 (c = 사전확률 보정)
        d_s = mean(e_i)               그 사람의 드리프트
        n*_s = A / |d_s|              판정에 필요한 윈도우 수 (Wald 근사)

    부호가 정답과 맞고 |d_s|가 크면 적게 봐도 맞힌다. 부호가 뒤집힌 사람이 곧 위험군이다.

유형 분류
    A 몰표   부호 일치 · n* ≤ 10   → 빨리 끝난다. 시간 절감이 여기서 나온다
    B 접전   부호 일치 · n* > 10   → 오래 봐야 한다. 유보 대상
    C 반대   부호 불일치           → 확신을 갖고 틀린다. 조기 종료가 해로운 쪽

사전 등록 판정 기준 (결과를 보기 전에 고정한다)
    C 유형 비율 0~5%    진행 — 본 실험을 계획대로
    C 유형 비율 5~12%   조건부 — 유보 옵션을 필수로. 명목 오류율 보장은 주장하지 않는다
    C 유형 비율 12% 초과 재설계 — 정지 규칙이 아니라 확률 품질 쪽 문제

실행
    python analyze_window_ee.py                       # results/probs_*.npz 전부
    python analyze_window_ee.py --probs results/probs_hw_....npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EPS = 1e-6
NSTAR_CUTS = (5, 10, 20)          # n* 민감도 확인용


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def per_subject(prob, y, subject_id, c, bound):
    """피험자별 드리프트와 필요 윈도우 수.

    c는 사전확률 보정이다. 클래스 비율이 한쪽으로 기울면 logit의 중심이 0이 아니라
    그쪽으로 밀려, 보정 없이는 모든 사람이 같은 방향으로 흐른다.
    """
    e = logit(prob) - c
    rows = []
    for s in np.unique(subject_id):
        m = subject_id == s
        d = float(np.mean(e[m]))
        true = int(y[m][0])
        want = +1 if true == 1 else -1          # 환자면 증거가 +로 흘러야 맞다
        rows.append(dict(
            subject=str(s), n_windows=int(m.sum()), drift=d, y=true,
            agree=(np.sign(d) == want),
            n_star=(bound / abs(d) if abs(d) > EPS else np.inf),
        ))
    return rows


def classify(rows, cut=10):
    A = [r for r in rows if r["agree"] and r["n_star"] <= cut]
    B = [r for r in rows if r["agree"] and r["n_star"] > cut]
    C = [r for r in rows if not r["agree"]]
    return A, B, C


def report(path, alpha=0.05):
    d = np.load(path, allow_pickle=True)
    prob, y, sid = d["prob"], d["y"], d["subject_id"]

    # 사전확률 보정은 val에서 정한다. test에서 맞추면 테스트셋 피팅이다.
    c = float(np.mean(logit(d["val_prob"]))) if "val_prob" in d else 0.0
    bound = float(np.log((1 - alpha) / alpha))       # Wald 경계 (α=β 대칭)

    rows = per_subject(prob, y, sid, c, bound)
    n = len(rows)
    print("=" * 72)
    print(f"{Path(path).name}")
    print("=" * 72)
    print(f"  윈도우 {len(prob)}개 · 피험자 {n}명 · "
          f"사람당 윈도우 중앙 {int(np.median([r['n_windows'] for r in rows]))}개")
    print(f"  사전확률 보정 c = {c:+.3f} (val 기준) · 경계 A = {bound:.3f} (α={alpha})")

    # ── 유형 분류 (n* 기준 민감도 확인) ──
    print(f"\n  {'n* 기준':>8}{'A 몰표':>12}{'B 접전':>12}{'C 반대':>12}{'C 비율':>10}")
    for cut in NSTAR_CUTS:
        A, B, C = classify(rows, cut)
        print(f"  {cut:>8}{len(A):>12}{len(B):>12}{len(C):>12}{len(C)/n*100:>9.1f}%")

    A, B, C = classify(rows, 10)
    frac = len(C) / n

    # ── 드리프트 분포 ──
    dr = np.array([abs(r["drift"]) for r in rows])
    print(f"\n  |드리프트| 분위: "
          + "  ".join(f"{p}%={np.percentile(dr, p):.2f}" for p in (10, 25, 50, 75, 90)))
    # 단봉이면 사람마다 명확도가 비슷하다는 뜻이고, 그러면 적응적 종료의 이득이 없다.
    lo, hi = np.percentile(dr, [25, 75])
    print(f"  → 사분위 범위 {hi - lo:.2f}  "
          + ("(넓다 — 사람마다 명확도가 달라 적응적 종료에 이득이 있다)"
             if hi - lo > 0.5 else
             "(좁다 — 명확도가 비슷해 고정 개수가 최적에 가깝다)"))

    # ── 드리프트 크기가 신뢰도의 대리 지표인가 ──
    q = np.array_split(np.array(sorted(rows, key=lambda r: abs(r["drift"]))), 4)
    print(f"\n  |드리프트| 4분위별 정답 방향 일치율 (신뢰도 대리 지표 검증)")
    for i, g in enumerate(q, 1):
        acc = np.mean([r["agree"] for r in g])
        print(f"    {i}분위 (|d| {abs(g[0]['drift']):.2f}~{abs(g[-1]['drift']):.2f}) "
              f"  일치 {acc*100:>5.1f}%  ({len(g)}명)")

    # ── 사전 등록 기준 판정 ──
    verdict = ("진행" if frac <= 0.05 else
               "조건부 진행" if frac <= 0.12 else "재설계")
    print(f"\n  {'─'*68}")
    print(f"  C 유형 {len(C)}명 / {n}명 = {frac*100:.1f}%   →   판정: {verdict}")
    if verdict == "조건부 진행":
        print("    유보 옵션과 비대칭 경계를 필수로. 명목 오류율 보장은 주장하지 않는다.")
    elif verdict == "재설계":
        print("    정지 규칙이 아니라 확률 품질 쪽 문제. 캘리브레이션부터 점검할 것.")
    if C:
        worst = sorted(C, key=lambda r: -abs(r["drift"]))[:5]
        print(f"    가장 확신 있게 틀리는 피험자: "
              + ", ".join(f"{r['subject']}(d={r['drift']:+.2f})" for r in worst))
        # 이 사람들은 윈도우를 더 봐도 틀린 쪽으로 더 확신한다. 유보 대상이다.
    return dict(file=Path(path).name, n=n, c=c, frac_C=frac, verdict=verdict, rows=rows)


def main():
    ap = argparse.ArgumentParser(description="윈도우 개수 축 Early Exit 사전 점검")
    ap.add_argument("--probs", nargs="*", default=None,
                    help="확률 npz 경로. 생략하면 results/probs_*.npz 전부")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    files = args.probs or sorted(
        p for p in (ROOT / "results").glob("probs_*.npz") if "smoke" not in p.name)
    if not files:
        raise SystemExit("확률 파일이 없습니다. train.py --save-probs 로 먼저 만드세요.")

    out = [report(f, args.alpha) for f in files]

    if len(out) > 1:
        print("\n" + "=" * 72)
        print("종합 — 시드·모달에 걸쳐 C 유형이 재현되는가")
        print("=" * 72)
        print(f"  {'파일':<46}{'피험자':>7}{'C 비율':>9}{'판정':>12}")
        for r in out:
            print(f"  {r['file']:<46}{r['n']:>7}{r['frac_C']*100:>8.1f}%{r['verdict']:>12}")
        f = [r["frac_C"] for r in out]
        print(f"\n  C 비율 평균 {np.mean(f)*100:.1f}% · 범위 "
              f"{min(f)*100:.1f}~{max(f)*100:.1f}%p")
        print("  → 시드마다 크게 다르면 그 자체가 결과다(정지 규칙이 불안정하다는 뜻).")


if __name__ == "__main__":
    main()
