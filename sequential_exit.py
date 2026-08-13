"""
윈도우 개수 축 Early Exit — 본 실험.

무엇을 하나
    한 사람의 윈도우를 1개씩 순서대로 넣다가 확신이 충분히 서면 나머지를 안 보고
    판정을 끝낸다. 임계값을 여러 개로 바꿔가며 "평균 몇 개를 봤을 때 정확도가
    얼마인가"를 그린다. 전부 본 경우와 비교해 몇 %만 봐도 성능이 유지되는지 본다.

    깊이 축(층 건너뛰기)과 다른 축이다. 깊이는 절감 상한이 층 수(6배)로 막히지만,
    윈도우 개수는 사람당 중앙 220개라 상한이 훨씬 크다.

판정 규칙 — 두 가지를 비교한다
    (1) 확신 임계값 (max softmax 계열)
        누적 평균 확률이 thr을 넘거나 1-thr 밑으로 내려가면 종료.
        직관적이지만 오류율을 사전에 통제하지 못한다.

    (2) SPRT (순차확률비검정)
        증거를 로그오즈로 누적해 경계 ±A에 닿으면 종료. A = log((1-α)/α).
        α를 정하면 오류율 상한이 이론적으로 정해진다는 점이 (1)과 다르다.

        S_k = Σ (logit(p_i) - c)        k개까지의 누적 증거
        S_k ≥ +A  → 환자로 판정하고 종료
        S_k ≤ -A  → 정상으로 판정하고 종료
        상한까지 못 닿으면 마지막 부호로 판정 (강제 종료)

캘리브레이션
    조기 종료는 확률의 과신에 특히 취약하다. 과신하면 경계에 너무 빨리 닿아
    적은 근거로 틀린 판정을 내린다. 그래서 온도 T와 사전확률 c를 **val에서만** fit한다.
    test에서 맞추면 테스트셋 피팅이라 결과가 낙관 편향된다.

핵심 지표는 피험자 단위
    윈도우 정확도는 이 실험에서 의미가 없다. "평균 윈도우 몇 개로 몇 명을 맞혔나"가
    답해야 할 질문이다.

실행
    python sequential_exit.py                        # results/probs_*.npz 전부
    python sequential_exit.py --probs results/probs_hw_....npz --no-plot
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EPS = 1e-6
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
ALPHAS = (0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ────────────────────────── 캘리브레이션 (val에서만) ──────────────────────────

def fit_temperature(prob, y, grid=np.linspace(0.5, 5.0, 91)):
    """NLL을 최소화하는 온도 T. logit을 T로 나눠 과신을 눌러 준다.

    T > 1 이면 원래 확률이 과신이었다는 뜻이다. 조기 종료는 과신에 취약하므로
    이 보정이 정확도를 좌우한다.
    """
    z = logit(prob)
    best, bestT = np.inf, 1.0
    for T in grid:
        p = sigmoid(z / T)
        nll = -np.mean(y * np.log(np.clip(p, EPS, 1)) +
                       (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
        if nll < best:
            best, bestT = nll, T
    return float(bestT)


def calibrate(d):
    """val에서 (T, c)를 구한다. c는 증거의 중심을 0으로 옮기는 사전확률 보정."""
    if "val_prob" not in d:
        return 1.0, 0.0
    vp, vy = d["val_prob"], d["val_y"]
    T = fit_temperature(vp, vy)
    c = float(np.mean(logit(vp) / T))
    return T, c


# ────────────────────────── 순차 판정 ──────────────────────────

def order_windows(idx):
    """윈도우를 넣는 순서. 원래 인덱스 오름차순 = 녹음 시간 순서다.

    무작위로 섞지 않는 이유는 실제 운용을 흉내기 때문이다. 사람이 과제를 수행하는
    동안 윈도우가 시간 순으로 들어오므로, 그 순서대로 판정해야 "몇 초 만에 끝나나"가
    의미를 갖는다.
    """
    return np.argsort(idx, kind="stable")


def run_sprt(prob, y, sid, idx, T, c, A):
    """SPRT. → (예측, 정답, 본 윈도우 수, 강제종료 여부) 피험자별."""
    z = logit(prob) / T - c
    preds, trues, seen, forced = [], [], [], []
    for s in np.unique(sid):
        m = sid == s
        ev = z[m][order_windows(idx[m])]
        S = np.cumsum(ev)
        hit = np.nonzero(np.abs(S) >= A)[0]
        if len(hit):
            k = int(hit[0]) + 1
            preds.append(int(S[hit[0]] > 0)); forced.append(False)
        else:
            k = len(S)
            preds.append(int(S[-1] > 0)); forced.append(True)
        trues.append(int(y[m][0])); seen.append(k)
    return (np.array(preds), np.array(trues),
            np.array(seen), np.array(forced))


def run_threshold(prob, y, sid, idx, T, c, thr):
    """누적 평균 확률이 thr / 1-thr 밖으로 나가면 종료 (확신 임계값 방식)."""
    p = sigmoid(logit(prob) / T - c)
    preds, trues, seen, forced = [], [], [], []
    for s in np.unique(sid):
        m = sid == s
        pp = p[m][order_windows(idx[m])]
        run = np.cumsum(pp) / np.arange(1, len(pp) + 1)
        hit = np.nonzero((run >= thr) | (run <= 1 - thr))[0]
        if len(hit):
            k = int(hit[0]) + 1
            preds.append(int(run[hit[0]] >= thr)); forced.append(False)
        else:
            k = len(run)
            preds.append(int(run[-1] >= 0.5)); forced.append(True)
        trues.append(int(y[m][0])); seen.append(k)
    return (np.array(preds), np.array(trues),
            np.array(seen), np.array(forced))


def run_fixed(prob, y, sid, idx, T, c, k):
    """앞 k개만 보고 판정 — 적응적 종료의 대조군.

    이 대조군이 없으면 "적응적으로 멈춰서 절감했다"를 말할 수 없다. 같은 개수를
    그냥 앞에서 자른 것과 정확도가 같다면, 필요한 것은 정지 규칙이 아니라
    "몇 개면 충분하다"는 상수 하나다. 그때는 Early Exit이라는 장치 자체가 불필요하다.

    사람마다 윈도우 수가 달라 k개보다 적은 사람은 있는 것을 다 쓴다.
    """
    p = sigmoid(logit(prob) / T - c)
    preds, trues, seen = [], [], []
    for s in np.unique(sid):
        m = sid == s
        pp = p[m][order_windows(idx[m])][:k]
        preds.append(int(np.mean(pp) >= 0.5))
        trues.append(int(y[m][0]))
        seen.append(len(pp))
    return np.array(preds), np.array(trues), np.array(seen)


def run_abstain(prob, y, sid, T, c, band):
    """확신이 애매한 사람을 판정 보류로 뺀다.

    게이트에서 C 유형(확신을 갖고 틀리는 사람)이 13%였다. 이들이 오류의 대부분을
    만든다면, 애매한 사람을 빼고 나머지에서 정확도가 오르는지 볼 만하다.
    실제 스크리닝은 "애매하면 의사에게"가 정당한 설계이므로 보류가 실패가 아니다.

    band = 0.10 이면 사람 평균 확률이 0.40~0.60인 사람을 보류한다.
    → (보류율, 판정한 사람에서의 정확도)
    """
    p = sigmoid(logit(prob) / T - c)
    dec, hold = [], 0
    for s in np.unique(sid):
        m = sid == s
        mp = float(np.mean(p[m]))
        if abs(mp - 0.5) < band:
            hold += 1
            continue
        dec.append((int(mp >= 0.5), int(y[m][0])))
    n = len(np.unique(sid))
    acc = float(np.mean([a == b for a, b in dec])) if dec else float("nan")
    return hold / n, acc, len(dec)


def run_fixed_random(prob, y, sid, idx, T, c, k, seed=0):
    """앞 k개가 아니라 **무작위 k개**를 본다 — 순서 편향 대조군.

    시간 순서로 넣는 것이 실제 운용을 흉내지만, 과제 시작 부분이 불안정하면
    앞쪽 윈도우가 손해일 수 있다. 무작위가 더 좋으면 그 편향이 있다는 뜻이다.
    """
    rng = np.random.default_rng(seed)
    p = sigmoid(logit(prob) / T - c)
    preds, trues = [], []
    for s in np.unique(sid):
        m = sid == s
        pp = p[m]
        take = rng.choice(len(pp), min(k, len(pp)), replace=False)
        preds.append(int(np.mean(pp[take]) >= 0.5))
        trues.append(int(y[m][0]))
    return np.array(preds), np.array(trues)


def baseline_all(prob, y, sid, T, c):
    """전부 본 경우 — 비교 기준. soft voting(평균 확률)으로 판정한다."""
    p = sigmoid(logit(prob) / T - c)
    preds, trues, seen = [], [], []
    for s in np.unique(sid):
        m = sid == s
        preds.append(int(np.mean(p[m]) >= 0.5))
        trues.append(int(y[m][0]))
        seen.append(int(m.sum()))
    return np.array(preds), np.array(trues), np.array(seen)


# ────────────────────────── 보고 ──────────────────────────

def sweep(path, plot=True):
    d = np.load(path, allow_pickle=True)
    prob, y, sid, idx = d["prob"], d["y"], d["subject_id"], d["idx"]
    T, c = calibrate(d)

    bp, bt, bseen = baseline_all(prob, y, sid, T, c)
    base_acc, total = float(np.mean(bp == bt)), int(bseen.sum())
    n_sub = len(bseen)

    print("=" * 78)
    print(Path(path).name)
    print("=" * 78)
    print(f"  피험자 {n_sub}명 · 윈도우 {len(prob)}개 "
          f"(사람당 중앙 {int(np.median(bseen))}개)")
    print(f"  캘리브레이션 (val): 온도 T={T:.2f}  사전확률 c={c:+.3f}"
          + ("   ← T>1: 원래 확률이 과신이었다" if T > 1.05 else ""))
    print(f"\n  [기준] 전부 본 경우: 피험자 정확도 {base_acc:.4f} "
          f"({int(np.sum(bp == bt))}/{n_sub}명), 윈도우 {total}개")

    rows = []
    for name, params, fn in (("SPRT", ALPHAS, "sprt"),
                             ("임계값", THRESHOLDS, "thr")):
        print(f"\n  ── {name} ──")
        print(f"  {'파라미터':>9}{'평균 윈도우':>12}{'사용률':>9}"
              f"{'피험자 정확도':>14}{'기준 대비':>10}{'강제종료':>10}")
        for v in params:
            if fn == "sprt":
                A = float(np.log((1 - v) / v))
                pr, tr, seen, forced = run_sprt(prob, y, sid, idx, T, c, A)
                label = f"α={v:g}"
            else:
                pr, tr, seen, forced = run_threshold(prob, y, sid, idx, T, c, v)
                label = f"thr={v:g}"
            acc = float(np.mean(pr == tr))
            use = seen.sum() / total
            rows.append(dict(method=name, param=v, mean_seen=float(seen.mean()),
                             use=use, acc=acc, delta=acc - base_acc,
                             forced=float(forced.mean())))
            print(f"  {label:>9}{seen.mean():>12.1f}{use*100:>8.1f}%"
                  f"{acc:>14.4f}{acc - base_acc:>+10.4f}{forced.mean()*100:>9.1f}%")

    # ── ⭐ 고정 개수 대조군 — 적응적 종료가 정말 이득인가 ──
    print(f"\n  ── 고정 개수 (대조군) ──")
    print(f"  {'앞 k개':>9}{'평균 윈도우':>12}{'사용률':>9}"
          f"{'피험자 정확도':>14}{'기준 대비':>10}")
    fixed = {}
    for k in (1, 2, 3, 5, 10, 20, 30, 50, 90, 130, 185, 220):
        pr, tr, seen = run_fixed(prob, y, sid, idx, T, c, k)
        acc = float(np.mean(pr == tr))
        fixed[k] = (float(seen.mean()), acc)
        print(f"  {k:>9}{seen.mean():>12.1f}{seen.sum()/total*100:>8.1f}%"
              f"{acc:>14.4f}{acc - base_acc:>+10.4f}")

    # 적응적 설정마다 "같은 개수를 그냥 앞에서 자른 것"과 맞대 본다
    print(f"\n  ── 같은 평균 개수에서 적응적 vs 고정 ──")
    print(f"  {'적응적 설정':>16}{'평균개수':>10}{'적응적':>10}{'고정':>10}{'적응 이득':>10}")
    gains = []
    for r in sorted(rows, key=lambda r: r["mean_seen"]):
        k = max(1, int(round(r["mean_seen"])))
        pr, tr, seen = run_fixed(prob, y, sid, idx, T, c, k)
        fa = float(np.mean(pr == tr))
        gains.append(r["acc"] - fa)
        label = f"{r['method']} {r['param']:g}"
        print(f"  {label:>16}"
              f"{r['mean_seen']:>10.1f}{r['acc']:>10.4f}{fa:>10.4f}"
              f"{r['acc'] - fa:>+10.4f}")
    g = np.array(gains)
    print(f"\n  적응 이득 평균 {g.mean():+.4f} · 범위 {g.min():+.4f}~{g.max():+.4f}")
    if g.mean() <= 0.005:
        print(f"  ⚠️ 적응적 종료가 고정 개수보다 나은 것이 없다(평균 {g.mean():+.2%}).")
        print(f"     → 필요한 것은 정지 규칙이 아니라 '몇 개면 충분한가'라는 상수 하나다.")
    else:
        print(f"  적응적 종료가 같은 개수에서 평균 {g.mean():+.2%} 더 맞힌다.")

    # ── 성능이 유지되는 최소 사용률 ──
    ok = [r for r in rows if r["delta"] >= -0.01]        # 1%p 이내 손실 허용
    if ok:
        best = min(ok, key=lambda r: r["use"])
        print(f"\n  ⭐ 정확도 손실 1%p 이내에서 가장 적게 본 설정")
        print(f"     {best['method']} {best['param']:g} — 평균 {best['mean_seen']:.1f}개"
              f"({best['use']*100:.1f}%)로 정확도 {best['acc']:.4f} "
              f"({best['delta']:+.4f})")
        print(f"     → 윈도우를 {(1-best['use'])*100:.1f}% 덜 보고 같은 성능")
    else:
        print(f"\n  ⚠️ 어떤 설정도 기준 정확도의 1%p 이내에 들지 못했다.")
        print(f"     조기 종료가 이 데이터에서 정확도를 깎는다는 뜻이다.")

    if plot:
        draw(rows, base_acc, n_sub, Path(path).stem)
    return rows, base_acc


def draw(rows, base_acc, n_sub, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, color, mark in (("SPRT", "#1e5aa8", "o"), ("임계값", "#c0392b", "s")):
        r = [x for x in rows if x["method"] == name]
        ax.plot([x["mean_seen"] for x in r], [x["acc"] for x in r],
                marker=mark, color=color, lw=1.8, ms=6, label=name)
        for x in r[::2]:
            ax.annotate(f"{x['param']:g}", (x["mean_seen"], x["acc"]),
                        fontsize=7.5, color=color,
                        xytext=(0, -12), textcoords="offset points", ha="center")

    ax.axhline(base_acc, ls="--", color="#555", lw=1.4)
    ax.text(ax.get_xlim()[1], base_acc, f" 전부 본 경우 {base_acc:.3f}",
            va="center", fontsize=9, color="#555")
    ax.axhspan(base_acc - 0.01, base_acc + 0.01, color="#888", alpha=0.10)

    ax.set_xlabel("평균으로 본 윈도우 개수 (사람당)", fontsize=11)
    ax.set_ylabel("피험자 단위 정확도", fontsize=11)
    ax.set_title(f"윈도우 개수 축 Early Exit — {stem}  (피험자 {n_sub}명)",
                 fontsize=12.5, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="종료 규칙", fontsize=9.5)
    fig.tight_layout()
    out = ROOT / "results" / f"window_ee_{stem}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  곡선 저장: {out.name}")


def main():
    ap = argparse.ArgumentParser(description="윈도우 개수 축 Early Exit 본 실험")
    ap.add_argument("--probs", nargs="*", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    files = args.probs or sorted(
        p for p in (ROOT / "results").glob("probs_*.npz") if "smoke" not in p.name)
    if not files:
        raise SystemExit("확률 파일이 없습니다. train.py --save-probs 로 먼저 만드세요.")

    allrows = []
    for f in files:
        r, base = sweep(f, plot=not args.no_plot)
        allrows.append((Path(f).stem, r, base))

    # ── 시드에 걸쳐 재현되는가 ──
    if len(allrows) > 1:
        print("\n" + "=" * 78)
        print("시드 종합 — 절감폭이 재현되는가")
        print("=" * 78)
        print(f"  {'파일':<40}{'기준 acc':>10}{'1%p내 최소사용률':>18}")
        for stem, rows, base in allrows:
            ok = [r for r in rows if r["delta"] >= -0.01]
            cell = f"{min(r['use'] for r in ok)*100:.1f}%" if ok else "없음"
            print(f"  {stem:<40}{base:>10.4f}{cell:>18}")
        print("\n  시드마다 크게 다르면 절감폭을 단일 숫자로 보고하면 안 된다.")


if __name__ == "__main__":
    main()
