"""
데이터 로딩·분할 — 모달별 npz를 각각 읽고, 학습 때 짝지어 흘린다.

⚠️ 데이터를 물리적으로 합치지 않는다
    필기  X[13474,  6, 2000]   6채널(펜 센서) · 2000 타임스텝
    음성  X[ 7658, 80,  200]  80채널(멜 밴드) · 200 프레임

    · 샘플 축으로 잇기   → 채널 6≠80, 길이 2000≠200이라 배열이 안 맞는다
    · 채널 축으로 잇기   → 같은 사람의 필기+음성이 0명이라 짝지을 수 없다
    · 리샘플해서 맞추기  → 펜 압력을 멜 밴드로 바꾸는 건 의미가 없다

    "공유"는 데이터가 아니라 **모델 안(인코더 가중치)에서만** 일어난다.
    여기서는 두 로더를 나란히 굴리기만 한다.

fold는 모달마다 독립적으로 만든다(필기 61명 / 음성 235명). fold i끼리 짝지어 함께
학습하고, 평가는 각자의 test 피험자에 대해 각자의 분류기로 한다.
"""
import numpy as np
import torch
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold)
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_modality(path):
    """npz → (X, y, subject_id).

    필요한 키는 X · y · subject_id 세 개다(task는 있으면 무시). X는 float32로
    캐스팅해 둔다 — fold 인덱싱(X[tr])이 어차피 복사본을 만드는데 원본이
    float64면 그 복사본도 float64라 메모리를 두 배로 쓴다.
    """
    d = np.load(path, allow_pickle=True)
    for k in ("X", "y", "subject_id"):
        if k not in d:
            raise KeyError(f"{path}에 '{k}'가 없습니다. (있는 키: {list(d.keys())})")
    return (np.asarray(d["X"], dtype=np.float32),
            np.asarray(d["y"]).astype(np.int64),
            np.asarray(d["subject_id"]))


def summarize(name, X, y, subject_id):
    per = np.array([np.sum(subject_id == s) for s in np.unique(subject_id)])
    return (f"{name:<8} X{tuple(X.shape)}  정상 {(y == 0).sum()} / 환자 {(y == 1).sum()}"
            f"  피험자 {len(np.unique(subject_id))}명"
            f"  (1인당 최소 {per.min()} / 중앙 {int(np.median(per))} / 최대 {per.max()})")


def subject_kfold(subject_id, y, n_splits=5, val_size=0.1, seed=42, stratified=False):
    """피험자 단위 K-Fold. 같은 사람이 train/val/test에 걸치지 않는다.

    ⚠️ `stratified=False`(기본)인 GroupKFold는 **층화를 하지 않는다.** fold마다
       정상/환자 비율이 흔들릴 수 있고, 피험자가 적으면 한 fold가 단일 클래스가
       되는 것도 가능하다(실제로 합성 데이터에서 겪었다 — 정확도가 정확히 0.000).
       실데이터에서도 fold별 클래스 비율은 한 번 확인하는 편이 좋다.
    """
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if stratified else GroupKFold(n_splits=n_splits))
    folds = []
    for train_val_idx, test_idx in splitter.split(np.arange(len(y)), y, groups=subject_id):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_rel, va_rel = next(gss.split(train_val_idx, y[train_val_idx],
                                        groups=subject_id[train_val_idx]))
        train_idx, val_idx = train_val_idx[tr_rel], train_val_idx[va_rel]

        # 누수 검증 — 세 집합의 피험자가 겹치면 안 된다
        assert not (set(subject_id[train_idx]) & set(subject_id[val_idx]))
        assert not (set(subject_id[train_idx]) & set(subject_id[test_idx]))
        assert not (set(subject_id[val_idx]) & set(subject_id[test_idx]))
        folds.append((train_idx, val_idx, test_idx))
    return folds


def make_loaders(X, y, fold, batch_size):
    """(train, val, test) DataLoader. train만 shuffle."""
    tr, va, te = fold
    return (
        DataLoader(WindowDataset(X[tr], y[tr]), batch_size=batch_size, shuffle=True),
        DataLoader(WindowDataset(X[va], y[va]), batch_size=batch_size),
        DataLoader(WindowDataset(X[te], y[te]), batch_size=batch_size),
    )


def class_weights(y_train, n_classes=2, subject_id=None, mode="window"):
    """역빈도 가중치. 분모의 n_classes 덕에 평균이 1.0이 되어 손실 스케일이 유지되므로
    learning rate를 다시 잡을 필요가 없다.

    mode="window" (기존)
        윈도우 라벨을 센다. 필기에서는 이것이 뒤집혀 있다.

            윈도우  HC 4,867 : PD 8,607   → 가중치 HC 1.384 : PD 0.783
            사람    HC    35 : PD    26   → 실제로는 환자가 소수

        환자가 느리게 그려 녹음이 길고 윈도우가 1인당 2.1배 나온다. 그래서 모델은
        환자를 "다수"로 보고 가중치를 깎는데, 사람 기준으로는 환자가 소수다.
        결과적으로 판정이 정상 쪽으로 밀려 민감도가 떨어진다(실측 0.35~0.96).
        그 여파가 판정 보류에서 드러난다 — 경계에 몰린 환자가 먼저 보류된다.

    mode="subject" (교정)
        사람당 1표로 센다. 판정 단위가 사람이므로 가중치도 사람을 세는 것이 맞다.
        윈도우가 몇 개 나왔는지는 녹음 길이의 문제이지 유병률이 아니다.

    mode="none"
        가중치를 주지 않는다. 위 둘의 대조군.
    """
    if mode == "none":
        return np.ones(n_classes, dtype=float)
    if mode == "subject":
        if subject_id is None:
            raise ValueError("mode='subject'에는 subject_id가 필요합니다.")
        # 한 사람은 라벨이 하나이므로 첫 윈도우의 라벨을 그 사람의 표로 쓴다
        _, first = np.unique(subject_id, return_index=True)
        y_train = y_train[first]
    cnt = np.bincount(y_train, minlength=n_classes)
    return cnt.sum() / (n_classes * np.maximum(cnt, 1))


def _cycle(loader):
    """로더를 무한 반복. shuffle=True면 한 바퀴 돌 때마다 다시 섞인다."""
    while True:
        for batch in loader:
            yield batch


class PairedBatches:
    """모달별 로더를 한 스텝에 하나씩 짝지어 내보낸다.

    길이가 다르다 — 필기 13,474 vs 음성 7,658 윈도우. 두 정책이 있다.

      longest  (기본) 가장 긴 로더 기준으로 한 epoch을 돌고 짧은 쪽은 순환시킨다.
               큰 모달의 데이터를 버리지 않는다. 대신 작은 모달이 반복 노출된다.
      shortest zip과 같다. 짧은 쪽에 맞춰 끊어 큰 모달의 데이터가 버려지므로,
               단독 기준선과의 비교가 성립하지 않는다.

    ⭐ 옵티마이저 스텝 수에 관한 메모
       한 스텝에서 두 모달의 손실을 **합산해 backward를 한 번만** 호출하므로,
       epoch당 스텝 수가 longest 정책에서 필기 단독 학습과 같아진다.
       "공유 모델은 인코더 업데이트가 2배라 비교가 불공정하다"는 교란이
       이 구현에서는 생기지 않는다. (모달을 번갈아 backward 하면 2배가 된다)

       다만 스텝당 인코더가 보는 **토큰 수**는 여전히 다르다. batch 64 기준
       필기는 64×6=384 수열, 음성은 64×80=5,120 수열이 통과한다. 음성이 13배
       무거운 이유이고, CPU 실행 시간을 지배한다.
    """

    def __init__(self, loaders, mode="longest"):
        if mode not in ("longest", "shortest"):
            raise ValueError(f"mode는 longest 또는 shortest (받은 값: {mode})")
        self.loaders = loaders
        self.mode = mode
        lens = [len(l) for l in loaders.values()]
        self.n_steps = max(lens) if mode == "longest" else min(lens)

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        iters = {m: _cycle(l) for m, l in self.loaders.items()}
        for _ in range(self.n_steps):
            yield {m: next(it) for m, it in iters.items()}


# ───────────────────────── 스모크 테스트용 합성 데이터 ─────────────────────────

def synthetic(n_subjects, n_channels, seq_len, win_per_subject=8, seed=0):
    """파이프라인 점검용 가짜 시계열. 실제 데이터 없이 학습이 도는지 확인한다.

    ⚠️ 두 가지를 지켜야 점검이 의미 있다.

    ① 라벨을 **주파수**에 심는다. RevIN이 윈도우마다 평균 0·분산 1로 정규화하므로
       "환자는 진폭이 크다" 같은 신호는 지워져 학습되지 않는다.

    ② 라벨을 피험자 번호 순서와 **어긋나게 섞는다.** `s % 2`로 주면 GroupKFold
       (층화 없음)가 홀/짝을 그대로 fold로 갈라 train이 전부 정상·test가 전부
       환자가 된다. 그러면 정확도가 0.000으로 나와 점검이 무의미해진다(실제로 겪음).
    """
    rng = np.random.default_rng(seed)

    labels = np.array([i % 2 for i in range(n_subjects)])
    rng.shuffle(labels)

    X, y, sid = [], [], []
    for s in range(n_subjects):
        label = int(labels[s])
        freq = 3.0 if label == 0 else 11.0
        t = np.linspace(0, 1, seq_len, dtype=np.float32)
        for _ in range(win_per_subject):
            phase = rng.uniform(0, 2 * np.pi, size=(n_channels, 1)).astype(np.float32)
            wave = np.sin(2 * np.pi * freq * t[None, :] + phase)
            X.append(wave + rng.normal(0, 0.4, size=(n_channels, seq_len)).astype(np.float32))
            y.append(label)
            sid.append(f"S{s:03d}")
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.int64),
            np.array(sid))
