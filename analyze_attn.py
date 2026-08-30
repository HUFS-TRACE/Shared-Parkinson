"""층별 어텐션 집계 — 층이 '어디를' 보는가.

analyze_layers.py가 "표현이 어떻게 바뀌나"(probe·CKA)를 재는 데 반해, 여기서는
"시선이 어디로 향하나"를 잰다. 표현은 패치 축을 평균해 저장하므로 7번 패치가
22번을 보든 8번을 보든 평균 뒤에는 구별되지 않는다. 인접 CKA 0.97이
"아무것도 안 섞었다"인지 "섞었는데 결과가 비슷하다"인지도 갈리지 않는다.

여섯 가지를 낸다
    ① 층별 어텐션 거리   sum_j w_ij * |i-j|.  작으면 국소, 크면 전역.
                         비전에서는 층이 깊어질수록 단조 증가한다.
    ② 실효 패치 수       exp(엔트로피). "실질적으로 몇 개를 보나".
    ③ 층 간 어텐션 상관  L1의 PxP 표와 L2의 표가 닮았나.
    ④ 잡음 바닥         다른 시드의 같은 층끼리 상관. ③이 이보다 높아야 의미가
                         있다. CKA 값(0.97)과 직접 견주면 틀린다 — 지표가 다르다.
    ⑤ 입력 의존성       표본마다 어텐션이 얼마나 달라지나. 거리(스칼라)만 보면
                         "같은 범위를 본다"와 "같은 곳을 본다"가 구별되지 않는다.
    ⑥ 로짓 표준편차     엔트로피 증가가 '고르게 편다'인지 'softmax가 물러진
                         부산물'인지 가른다. 로짓이 함께 작아지면 후자다.

판정은 지금까지와 같다 — 시드 5개의 부호 일관성이 먼저고 t는 참고다
(자유도 4, 임계 2.776).

실행
    python analyze_attn.py
    python analyze_attn.py --tag sw2 --modality hw
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RES = Path(__file__).resolve().parent / "results"
T_CRIT = 2.776          # 자유도 4, 양측 alpha=0.05
NAMES = {"hw": "필기", "voice": "음성", "ecg": "심전도", "ppg": "맥파"}


def paired_t(d):
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def sign_label(d):
    return "모두 +" if (d > 0).all() else "모두 -" if (d < 0).all() else "엇갈림"


def corr_mat(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def map_corr(A):
    """층 간 어텐션 맵 상관. A는 (L, H, P, P) — 헤드를 평균해 층끼리 비교한다.

    헤드를 평균하는 이유는 헤드 번호가 층 사이에서 대응되지 않기 때문이다.
    L1의 0번 헤드와 L2의 0번 헤드가 같은 일을 한다는 보장이 없다.
    """
    M = A.mean(axis=1).reshape(A.shape[0], -1)
    M = M - M.mean(axis=1, keepdims=True)
    n = M / np.linalg.norm(M, axis=1, keepdims=True)
    return n @ n.T


def report(m, files):
    print("=" * 78)
    print(f"  {NAMES.get(m, m)}  ·  시드 {len(files)}개")
    print("=" * 78)

    Z, dists, ents, maps, seeds = [], [], [], [], []
    for f in files:
        z = np.load(f)
        Z.append(z)
        seeds.append(int(re.search(r"_s(\d+)\.npz", f.name).group(1)))
        dists.append(z["dist"].mean(axis=0))
        ents.append(np.exp(z["ent"]).mean(axis=0))     # 실효 패치 수
        maps.append(z["attn_map"])
    D, E = np.array(dists), np.array(ents)
    P, L = maps[0].shape[-1], D.shape[1]
    unif = (P * P - 1) / (3 * P)                       # 균등하게 볼 때의 거리

    print(f"\n  패치 {P}개 · 층 {L}개 (균등하게 볼 때 거리 {unif:.2f} · 실효 {P}개)")
    print(f"\n  {'층':>4}{'어텐션 거리':>14}{'균등 대비':>11}"
          f"{'실효 패치':>12}{'전체 대비':>11}")
    for i in range(L):
        print(f"  {'L' + str(i + 1):>4}{D[:, i].mean():>14.2f}"
              f"{D[:, i].mean() / unif:>10.0%}"
              f"{E[:, i].mean():>12.1f}{E[:, i].mean() / P:>10.0%}")

    print(f"\n  {'L1 → L' + str(L):>10}{'변화':>10}{'부호':>10}{'t':>8}")
    for lab, A in (("거리", D), ("실효 패치", E)):
        d = A[:, -1] - A[:, 0]
        t = paired_t(d)
        print(f"  {lab:>10}{d.mean():>+10.2f}{sign_label(d):>9}{t:>8.2f}"
              + (" ★" if abs(t) > T_CRIT else ""))

    # ⑥ 엔트로피 교란 — 로짓이 작아지면 softmax가 물러져 엔트로피가 저절로 오른다
    if "logit_std" in Z[0].files:
        LS = np.array([z["logit_std"] for z in Z])
        print(f"\n  {'층':>4}{'로짓 표준편차':>15}{'변동비율':>12}")
        for i in range(L):
            vr = np.mean([np.linalg.norm(z["attn_std"][i]) /
                          np.linalg.norm(z["attn_map"][i]) for z in Z])
            print(f"  {'L' + str(i + 1):>4}{LS[:, i].mean():>15.2f}{vr:>12.3f}")
        dl, de = LS[:, -1] - LS[:, 0], E[:, -1] - E[:, 0]
        print(f"\n  로짓 {dl.mean():+.2f} ({sign_label(dl)})  ·  "
              f"실효 패치 {de.mean():+.2f} ({sign_label(de)})")
        if (dl < 0).all() and (de > 0).all():
            print("     !! 로짓이 줄면서 엔트로피가 올랐다 — '고르게 편다'가 아니라")
            print("        softmax가 물러진 부산물일 수 있다. 진술을 낮춰야 한다.")
        elif abs(dl.mean()) < 0.05 * LS[:, 0].mean():
            print("     -> 로짓은 그대로인데 엔트로피만 올랐다. 실제로 고르게 편다.")

    # ⑤ 입력 의존성 — 표본별 맵이 서로 얼마나 다른가
    if "samples" in Z[0].files and Z[0]["samples"].size:
        print("\n  입력 의존성 (같은 층, 다른 표본끼리의 맵 상관)")
        print(f"  {'층':>4}{'표본 간 상관':>15}   해석")
        rng = np.random.default_rng(0)
        for i in range(L):
            S = Z[0]["samples"][:, i].mean(axis=1).astype(np.float32)
            pick = rng.permutation(len(S))[:60]
            r = float(np.mean([corr_mat(S[a], S[b])
                               for a in pick[:30] for b in pick[30:]]))
            note = ("입력마다 크게 다름" if r < 0.5 else
                    "입력에 거의 무관 — 고정 패턴" if r > 0.9 else "부분적으로 다름")
            print(f"  {'L' + str(i + 1):>4}{r:>15.3f}   {note}")

    # ③④ 층 간 상관과 잡음 바닥
    C = np.mean([map_corr(a) for a in maps], axis=0)
    adj = np.mean([C[i, i + 1] for i in range(L - 1)])
    Am = np.array([a.mean(axis=1) for a in maps])
    floor = np.mean([corr_mat(Am[i, l], Am[j, l]) for l in range(L)
                     for i in range(len(Am)) for j in range(len(Am)) if i < j])

    print(f"\n  층 간 어텐션 상관 — 인접 {adj:.3f} · L1↔L{L} {C[0, -1]:.3f}"
          f"  (잡음 바닥 {floor:.3f})")
    print("     " + "".join(f"{'L' + str(j + 1):>8}" for j in range(L)))
    for i in range(L):
        print(f"  L{i + 1} " + "".join(f"{C[i, j]:>8.3f}" for j in range(L)))
    if C[0, -1] < floor:
        print(f"     -> L1↔L{L}이 잡음 바닥보다 낮다. '덜 닮았다'가 아니라 무관하다.")

    print("\n  읽는 법")
    if abs(D[:, -1].mean() - D[:, 0].mean()) < 0.05 * unif:
        print("     거리가 평평하다 → 층이 계속 같은 범위를 본다.")
    elif (D[:, -1] - D[:, 0] > 0).all():
        print("     거리가 오른다 → 층이 깊어질수록 넓게 본다 (비전과 같은 방향).")
    else:
        print("     거리가 엇갈린다 → 시드 노이즈와 구별되지 않는다.")
    print()


def main():
    ap = argparse.ArgumentParser(description="층별 어텐션 집계")
    ap.add_argument("--tag", default="sw2")
    ap.add_argument("--modality", nargs="*", default=["hw", "voice", "ecg", "ppg"])
    args = ap.parse_args()

    for m in args.modality:
        files = sorted(RES.glob(f"attn_{m}_{args.tag}_*s*.npz"),
                       key=lambda p: int(re.search(r"_s(\d+)\.npz", p.name).group(1)))
        if not files:
            print(f"[{m}] 파일 없음 — attn_{m}_{args.tag}_*s*.npz")
            continue
        report(m, files)


if __name__ == "__main__":
    main()
