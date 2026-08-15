"""
공유 인코더 학습 — 음성·필기를 함께 학습하고, 평가는 모달별로 따로 한다.

한 스텝의 흐름
    필기 배치 → 필기 임베딩 → 공유 인코더 → 필기 분류기 → 손실 ┐
                                                                ├→ 합산 → backward 1회
    음성 배치 → 음성 임베딩 → 공유 인코더 → 음성 분류기 → 손실 ┘

    두 기울기가 같은 인코더로 흘러들어간다. 필기 배치로 인코더를 고치고 음성
    배치로 또 고치는 식이 아니라, **한 번의 backward에서 두 신호가 합쳐져**
    인코더를 함께 다듬는다. 각자 배치를 따로 넣으므로 짝 데이터가 필요 없다.

평가
    학습이 끝나면 필기만 넣어도 예측이 나온다(음성에서 배운 것은 이미 가중치
    안에 있다). 평가 단위는 여전히 모달 하나다 — 같은 fold·같은 사람에 대해
    단독 모델과 숫자를 직접 맞대면 A/B 비교가 그대로 성립한다.

실행
    python train.py --smoke-test                      # 합성 데이터로 파이프라인 점검
    python train.py                                   # configs/ 기본값으로 2모달
    python train.py --modalities voice                # 음성 단독 기준선
    python train.py --modalities hw --seed 7          # 필기 단독, seed 지정

⚠️ Early Exit은 아직 포함하지 않는다. 기본 모델이 먼저다.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Windows 한국어 로캘에서 출력을 파일로 리다이렉트하면 stdout이 cp949가 되어
# 이모지·일부 기호에서 UnicodeEncodeError로 죽는다. 결과 저장 뒤에 터지면
# 로그만 잘려 원인을 찾기 어려우므로 처음부터 UTF-8로 고정한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.shared_patchtst import SharedPatchTST
from utils.dataset import (PairedBatches, class_weights, load_modality,
                           make_loaders, subject_kfold, summarize, synthetic)
from utils.metrics import evaluate_predictions, fold_class_balance

ROOT = Path(__file__).resolve().parent

# 스모크 테스트용 — 실제와 같은 채널 수·길이를 쓰되 피험자만 줄인다.
# shape이 달라야 "채널 수가 다른 모달이 한 인코더를 통과하는가"가 검증된다.
SMOKE_SPEC = dict(hw=dict(n_subjects=20, n_channels=6, seq_len=2000),
                  voice=dict(n_subjects=24, n_channels=80, seq_len=200))


def parse_args():
    p = argparse.ArgumentParser(description="Shared PatchTST 학습 (Early Exit 없음)")
    p.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    p.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    p.add_argument("--modalities", default="hw,voice", help="쉼표 구분. hw / voice / hw,voice")
    p.add_argument("--hw-path"); p.add_argument("--voice-path")
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float); p.add_argument("--patience", type=int)
    p.add_argument("--n-folds", type=int); p.add_argument("--seed", type=int)
    p.add_argument("--d-model", type=int); p.add_argument("--n-layers", type=int)
    # 패칭 스윕용. d_model은 인코더가 공유라 모달 공통이지만, patch/stride는
    # 모달별 입구(PatchEmbedding)에 있어 따로 준다.
    p.add_argument("--hw-patch", type=int); p.add_argument("--hw-stride", type=int)
    p.add_argument("--voice-patch", type=int); p.add_argument("--voice-stride", type=int)
    p.add_argument("--head", choices=["linear", "perch", "mlp"],
                   help="분류 헤드. 채널 상관을 분류기 하나로 감당할 수 있는지 보는 "
                        "대조군 (models/classifier.py 주석 참고)")
    p.add_argument("--class-weight", choices=["window", "subject", "none"],
                   default="window",
                   help="손실 클래스 가중치를 무엇으로 세나. window(기존)는 필기에서 "
                        "환자 윈도우가 2.1배라 가중치가 뒤집힌다 (utils/dataset.py 주석)")
    p.add_argument("--max-windows", type=int,
                   help="사람당 윈도우 상한. 녹음이 긴 사람이 학습을 지배하는 것을 막는다. "
                        "넘치면 시간축 균등 간격으로 뽑는다")
    p.add_argument("--pair-mode", choices=["longest", "shortest"])
    p.add_argument("--stratified", action="store_true",
                   help="StratifiedGroupKFold 사용 (기본 GroupKFold는 층화 안 함)")
    p.add_argument("--smoke-test", action="store_true",
                   help="합성 데이터로 파이프라인만 점검 (실제 npz 불필요)")
    p.add_argument("--out", help="결과 CSV 경로. 기본은 results/<태그>.csv")
    p.add_argument("--hw-channels",
                   help="필기 채널 부분집합 (0-based, 쉼표). 예: 3,2,1 "
                        "→ TiltX·압력·그립만. 센서 축 연산량 실험용")
    p.add_argument("--voice-channels",
                   help="음성 멜밴드 부분집합 (0-based, 쉼표)")
    p.add_argument("--save-probs", action="store_true",
                   help="윈도우별 P(환자)를 저장한다. 시간 축 Early Exit(윈도우 개수)의 "
                        "입력이며, 이후 단계는 재학습 없이 이 파일만 읽는다")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(args):
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mcfg = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    # CLI가 YAML을 덮어쓴다
    for key, section in (("epochs", "train"), ("batch_size", "train"), ("lr", "train"),
                         ("patience", "train"), ("pair_mode", "train"),
                         ("n_folds", "split"), ("seed", "split")):
        v = getattr(args, key, None)
        if v is not None:
            cfg[section][key] = v
    if args.stratified:
        cfg["split"]["stratified"] = True
    if args.hw_path:
        cfg["data"]["hw_path"] = args.hw_path
    if args.voice_path:
        cfg["data"]["voice_path"] = args.voice_path
    if args.d_model:
        mcfg["shared_encoder"]["d_model"] = args.d_model
    if args.n_layers:
        mcfg["shared_encoder"]["n_layers"] = args.n_layers
    if args.head:
        mcfg["shared_encoder"]["head"] = args.head
    for m in mcfg["modalities"]:
        for cli, key in (("patch", "patch_len"), ("stride", "stride")):
            v = getattr(args, f"{m}_{cli}", None)
            if v:
                mcfg["modalities"][m][key] = v
    return cfg, mcfg


def gather_data(args, cfg):
    """{모달: dict(X=, y=, subject_id=)}"""
    data = {}
    for i, m in enumerate(args.modalities):
        if args.smoke_test:
            X, y, sid = synthetic(**SMOKE_SPEC[m], win_per_subject=8,
                                  seed=cfg["split"]["seed"] + i)
        else:
            path = cfg["data"][f"{m}_path"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[{m}] {path} 가 없습니다. configs/config.yaml의 경로를 고치거나 "
                    f"--{m}-path로 지정하세요. 파이프라인만 확인하려면 --smoke-test.")
            X, y, sid = load_modality(path)

        # 센서(채널) 축 실험 — 채널 부분집합만 남긴다.
        # RevIN의 채널별 affine과 분류기의 num_channels·d_model 입력이 채널 수에
        # 묶여 있어, 학습된 모델에서 채널을 빼는 것은 불가능하다. 그래서 부분집합을
        # **학습 전에** 잘라 독립 학습한다. 그러면 "같은 설정에서 채널만 다르다"가
        # 성립해 정확도 차이를 채널 탓으로 돌릴 수 있다.
        sel = getattr(args, f"{m}_channels", None)
        if sel:
            idx = [int(c) for c in sel.split(",")]
            if max(idx) >= X.shape[1] or min(idx) < 0:
                raise SystemExit(f"[{m}] 채널 번호가 범위를 벗어남: {idx} "
                                 f"(0~{X.shape[1] - 1})")
            print(f"  [{m}] 채널 {len(idx)}/{X.shape[1]}개 선택: {idx}")
            X = np.ascontiguousarray(X[:, idx, :])

        data[m] = dict(X=X, y=y, subject_id=sid)
    return data


def cap_train_windows(folds, data, n):
    """train 인덱스만 사람당 n개로 줄인다. val·test는 건드리지 않는다.

    왜 상한이 필요한가
        필기는 사람이 35:26(정상:환자)인데 윈도우는 4,867:8,607로 비율이 뒤집힌다.
        환자가 느리게 그려 녹음이 길고, 그래서 윈도우가 1인당 2.4배 나온다.
        질병 신호가 데이터 불균형으로 새어 들어가는 셈이다. 손실 가중치
        (class_weights)로는 못 잡는다 — 그건 라벨 비율만 보고 사람은 안 본다.

    왜 train에만 거는가
        test까지 자르면 평가 대상 자체가 달라져, 상한을 건 모델과 안 건 모델을
        **다른 자로** 재게 된다. 윈도우 정확도와 AUC가 특히 그렇다. 상한은 학습
        데이터의 성질을 고치려는 것이므로 train에만 걸어야 비교가 성립한다.

    왜 앞에서 N개가 아니라 균등 간격인가
        과제 앞부분만 남기면 "천천히 시작해 점점 흔들린다" 같은 후반부 신호를
        통째로 버린다. 원래 인덱스 순서가 녹음 시간 순이므로 그 위에서 균등하게 뽑는다.
    """
    for m, per_fold in folds.items():
        sid = data[m]["subject_id"]
        new, before, after = [], 0, 0
        for tr, va, te in per_fold:
            keep = []
            for s in np.unique(sid[tr]):
                w = tr[sid[tr] == s]                 # 원래 순서 = 녹음 시간 순
                keep.append(w if len(w) <= n else
                            w[np.linspace(0, len(w) - 1, n).astype(int)])
            keep = np.sort(np.concatenate(keep))
            before, after = before + len(tr), after + len(keep)
            new.append((keep, va, te))
        folds[m] = new
        print(f"  [{m}] train 윈도우 상한 {n}개 → fold 합계 {before}개에서 {after}개로 "
              f"({after / before * 100:.0f}%) · val·test는 그대로")
    return folds


def build_specs(data, mcfg):
    specs = {}
    for m, d in data.items():
        pc = mcfg["modalities"][m]
        specs[m] = dict(num_channels=d["X"].shape[1], seq_len=d["X"].shape[2],
                        patch_len=pc["patch_len"], stride=pc["stride"])
    return specs


def eval_modality(model, modality, loader, criterion, device):
    """(loss, preds, labels, probs) — 한 모달을 그 모달 분류기로 평가."""
    model.eval()
    total, preds, labels, probs = 0.0, [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X, modality)
            total += criterion(logits, y).item() * X.size(0)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y.cpu())
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return (total / len(loader.dataset),
            torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy())


def save_probs(saved, data, tag, result_dir):
    """윈도우별 P(환자)를 모달마다 npz로 저장. 시간 축 Early Exit의 입력이다.

    이후 단계(사전점검 · 캘리브레이션 · 순차판정)는 이 파일만 읽고 재학습하지 않는다.
    한 사람의 윈도우를 순서대로 넣어 판정하는 구조라, 윈도우가 어느 피험자의
    몇 번째인지 알아야 한다. 그래서 subject_id와 원래 인덱스를 함께 남긴다.
    """
    for m, s in saved.items():
        if not s["prob"]:
            continue
        cat = {k: np.concatenate(v) for k, v in s.items()}
        idx = cat["idx"]
        assert len(np.unique(idx)) == len(idx), \
            f"{m}: test 인덱스 중복 — fold 분할이 깨졌다"
        out = result_dir / f"probs_{m}_{tag}.npz"
        np.savez_compressed(
            out,
            prob=cat["prob"], idx=idx, fold=cat["fold"],
            y=data[m]["y"][idx], subject_id=data[m]["subject_id"][idx],
            val_prob=cat["val_prob"], val_idx=cat["val_idx"],
            val_fold=cat["val_fold"], val_y=data[m]["y"][cat["val_idx"]],
            val_subject_id=data[m]["subject_id"][cat["val_idx"]],
        )
        n_dup = len(cat["val_idx"]) - len(np.unique(cat["val_idx"]))
        print(f"  확률 저장: {out.name}  "
              f"(test {len(idx)} / val {len(cat['val_idx'])}, val 중복 {n_dup})")


def train_fold(model, train_loaders, val_loaders, criteria, cfg, device, ckpt):
    """공동 학습 1 fold. 조기중단 기준은 **모달 val loss의 합**.

    합으로 거는 이유는 인코더가 하나뿐이라 모달별로 다른 시점에 멈출 수 없기
    때문이다. 한 모달이 먼저 과적합해도 다른 쪽이 내려가면 합이 계속 줄어 안
    멈춘다 — 공유 구조의 구조적 타협이고, 모달별 val loss를 따로 찍어 확인한다.
    """
    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                  weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    paired = PairedBatches(train_loaders, mode=t["pair_mode"])

    best, no_improve, best_epoch, n_epoch = float("inf"), 0, 0, 0
    for epoch in range(1, t["epochs"] + 1):
        n_epoch = epoch
        model.train()
        for batches in paired:
            optimizer.zero_grad()
            loss = 0.0
            for m, (Xb, yb) in batches.items():
                # 손실을 합산해 backward는 한 번만 → 옵티마이저 스텝 수가 단독
                # 학습과 같아진다 (utils/dataset.PairedBatches 주석 참고)
                loss = loss + criteria[m](model(Xb.to(device), m), yb.to(device))
            loss.backward()
            optimizer.step()

        val = {m: eval_modality(model, m, val_loaders[m], criteria[m], device)[0]
               for m in val_loaders}
        total_val = sum(val.values())
        scheduler.step(total_val)

        if total_val < best:
            best, no_improve, best_epoch = total_val, 0, epoch
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
            if no_improve >= t["patience"]:
                break

    model.load_state_dict(torch.load(ckpt))
    return model, n_epoch, best_epoch


def main():
    args = parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    cfg, mcfg = load_config(args)
    seed = cfg["split"]["seed"]

    result_dir = ROOT / cfg["output"]["result_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    # 패칭을 CLI로 바꿨으면 파일명에 남긴다. 안 그러면 스윕 결과가 같은 이름으로
    # 서로를 덮어써서, 마지막 설정 하나만 남고도 알아채지 못한다.
    patch_tag = "".join(
        f"_{m}p{mcfg['modalities'][m]['patch_len']}s{mcfg['modalities'][m]['stride']}"
        for m in args.modalities
        if getattr(args, f"{m}_patch", None) or getattr(args, f"{m}_stride", None))
    opt_tag = (f"_h{args.head}" if args.head else "") + \
              (f"_w{args.max_windows}" if args.max_windows else "")
    out = Path(args.out) if args.out else result_dir / (
        f"{'smoke_' if args.smoke_test else ''}"
        f"{'-'.join(args.modalities)}_d{mcfg['shared_encoder']['d_model']}"
        f"L{mcfg['shared_encoder']['n_layers']}{patch_tag}{opt_tag}_s{seed}.csv")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    data = gather_data(args, cfg)
    specs = build_specs(data, mcfg)
    print(f"device={args.device}  seed={seed}"
          f"{'  [합성 데이터 · 스모크 테스트]' if args.smoke_test else ''}")
    for m, d in data.items():
        print("  " + summarize(m, d["X"], d["y"], d["subject_id"]))

    enc = mcfg["shared_encoder"]
    probe = SharedPatchTST(specs, num_classes=mcfg["num_classes"], **enc)
    print("\n" + probe.describe())
    rep = probe.param_report()
    del probe
    torch.manual_seed(seed)          # probe가 소비한 RNG를 되돌린다

    folds = {m: subject_kfold(d["subject_id"], d["y"], n_splits=cfg["split"]["n_folds"],
                              val_size=cfg["split"]["val_size"], seed=seed,
                              stratified=cfg["split"]["stratified"])
             for m, d in data.items()}

    if args.max_windows:
        folds = cap_train_windows(folds, data, args.max_windows)

    # GroupKFold는 층화를 하지 않으므로 fold별 클래스 비율을 찍어둔다
    print("\nfold별 클래스 분포 [정상, 환자]")
    for m in data:
        for r in fold_class_balance(data[m]["y"], folds[m]):
            print(f"  {m:<6} fold {r['fold']}  train {r['train']}  test {r['test']}")

    print(f"\n{cfg['split']['n_folds']}-fold 학습 시작 "
          f"(pair-mode={cfg['train']['pair_mode']}, 최대 {cfg['train']['epochs']} epoch)\n")

    acc = {m: {k: [] for k in
               ("window_acc", "subject_acc", "subject_acc_soft", "roc_auc")} for m in data}
    epochs_used = []
    # 시간 축 Early Exit용 윈도우 확률. test는 fold를 가로질러 전체를 정확히 한 번씩
    # 덮지만, val은 fold마다 새로 떼므로 겹칠 수 있다. 그래서 fold 태그를 함께 남긴다.
    saved = {m: {k: [] for k in ("prob", "idx", "fold",
                                 "val_prob", "val_idx", "val_fold")} for m in data}
    ckpt = result_dir / f"_tmp_{os.getpid()}.pt"

    for fold in range(cfg["split"]["n_folds"]):
        loaders = {m: make_loaders(data[m]["X"], data[m]["y"], folds[m][fold],
                                   cfg["train"]["batch_size"]) for m in data}
        criteria = {
            m: nn.CrossEntropyLoss(
                weight=torch.tensor(
                    class_weights(data[m]["y"][folds[m][fold][0]],
                                  subject_id=data[m]["subject_id"][folds[m][fold][0]],
                                  mode=args.class_weight),
                    dtype=torch.float32).to(device),
                label_smoothing=cfg["train"]["label_smoothing"])
            for m in data
        }
        model = SharedPatchTST(specs, num_classes=mcfg["num_classes"], **enc).to(device)
        model, n_ep, best_ep = train_fold(
            model, {m: l[0] for m, l in loaders.items()},
            {m: l[1] for m, l in loaders.items()}, criteria, cfg, device, ckpt)
        epochs_used.append(n_ep)

        print(f"[fold {fold + 1}/{cfg['split']['n_folds']}]  {n_ep} epoch (best {best_ep})")
        for m in data:
            _, preds, labels, probs = eval_modality(
                model, m, loaders[m][2], criteria[m], device)
            te = folds[m][fold][2]
            met = evaluate_predictions(preds, probs, labels, data[m]["subject_id"][te])
            for k, v in met.items():
                acc[m][k].append(v)

            if args.save_probs:
                # 캘리브레이션(온도·사전확률 보정)은 val에서 정해야 한다. test에서
                # 맞추면 테스트셋 피팅이고, fold마다 모델이 다르므로 fold별로 따로 구한다.
                _, _, _, vp = eval_modality(model, m, loaders[m][1], criteria[m], device)
                va = folds[m][fold][1]
                s = saved[m]
                s["prob"].append(probs);  s["idx"].append(te)
                s["fold"].append(np.full(len(te), fold))
                s["val_prob"].append(vp); s["val_idx"].append(va)
                s["val_fold"].append(np.full(len(va), fold))
            print(f"    {m:<6} 윈도우 {met['window_acc']:.3f}   "
                  f"피험자 hard {met['subject_acc']:.3f} / soft {met['subject_acc_soft']:.3f}"
                  f"   AUC {met['roc_auc']:.3f}")

    ckpt.unlink(missing_ok=True)

    if args.save_probs:
        save_probs(saved, data, out.stem, result_dir)

    print(f"\n{'모달':<8}{'윈도우acc':>16}{'피험자 hard':>18}{'피험자 soft':>18}{'AUC':>16}")
    rows = []
    for m in data:
        cells = [f"{np.mean(acc[m][k]):.3f} ± {np.std(acc[m][k]):.3f}"
                 for k in ("window_acc", "subject_acc", "subject_acc_soft", "roc_auc")]
        print(f"{m:<8}{cells[0]:>16}{cells[1]:>18}{cells[2]:>18}{cells[3]:>16}")
        rows.append(dict(
            modality=m,
            n_subjects=len(np.unique(data[m]["subject_id"])), n_windows=len(data[m]["y"]),
            d_model=enc["d_model"], n_layers=enc["n_layers"],
            patch_len=specs[m]["patch_len"], stride=specs[m]["stride"],
            **{k: round(float(np.mean(acc[m][k])), 4) for k in acc[m]},
            **{f"{k}_std": round(float(np.std(acc[m][k])), 4) for k in acc[m]},
            mean_epochs=round(float(np.mean(epochs_used)), 1),
            params_shared=rep["shared"], params_own=rep["own"][m],
            shared_ratio=round(rep["shared_ratio"], 4),
            pair_mode=cfg["train"]["pair_mode"], seed=seed))

    cols = list(rows[0])
    out.write_text(",".join(cols) + "\n"
                   + "".join(",".join(str(r[c]) for c in cols) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"\n수치 저장: {out}")

    n_min = min(len(np.unique(data[m]["subject_id"])) for m in data)
    print(f"  ⚠️ 피험자가 가장 적은 모달이 {n_min}명 → 1명 ≈ {100 / n_min:.1f}%p. "
          f"단일 seed·개별 fold는 과대해석 금지.")
    print(f"  ⚠️ 같은 설정(d{enc['d_model']}/L{enc['n_layers']})으로 돌린 단독 모델과만 "
          f"비교할 것. 설정이 다르면 공유 효과와 설정 효과가 섞인다.")


if __name__ == "__main__":
    main()
