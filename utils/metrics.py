"""평가 지표 — 윈도우 단위와 피험자 단위를 구분해서 본다.

지표 선택에 관한 경고
    윈도우는 수천 개지만 **실제 사람은 수십~수백 명**이다. 일반화 성능의 정직한
    지표는 피험자 단위지만, 표본이 작으면 1명이 차지하는 비중이 커져 노이즈가 된다.

        필기  61명 · fold당 test 12명  →  1명 = 8.3%p   ← 지표로 쓰기 어렵다
        음성 235명 · fold당 test 47명  →  1명 = 2.1%p   ← 상대적으로 안정적

    실제로 필기에서 피험자 정확도가 seed에 따라 8.1%p 흔들려, 관측된 개선폭과
    같은 크기였다. **필기에서는 ROC-AUC를 보는 편이 안전하다.**
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def subject_vote_hard(pred, subject_id, y_true):
    """윈도우 예측(0/1) → 피험자 단위 다수결 정확도.

    각 윈도우가 한 표씩. 확신도 0.51이든 0.99든 같은 1표다.
    """
    uniq = np.unique(subject_id)
    correct = sum(int(round(pred[subject_id == s].mean()) == y_true[subject_id == s][0])
                  for s in uniq)
    return correct / len(uniq)


def subject_vote_soft(probs, subject_id, y_true, threshold=0.5):
    """윈도우 확률 → 피험자 단위 평균 확률 판정.

    확신도가 표의 무게에 반영된다. hard와 갈리는 예:
      윈도우 10개 중 6개가 P=0.55(환자), 4개가 P=0.05(정상)
        hard → 6:4로 환자,  soft → 평균 0.35로 정상
    """
    uniq = np.unique(subject_id)
    correct = sum(int((probs[subject_id == s].mean() >= threshold)
                      == bool(y_true[subject_id == s][0]))
                  for s in uniq)
    return correct / len(uniq)


def evaluate_predictions(preds, probs, labels, subject_id):
    """윈도우 예측 묶음 → 지표 dict."""
    return dict(
        window_acc=float((preds == labels).mean()),
        subject_acc=subject_vote_hard(preds, subject_id, labels),
        subject_acc_soft=subject_vote_soft(probs, subject_id, labels),
        roc_auc=(float(roc_auc_score(labels, probs))
                 if len(np.unique(labels)) > 1 else float("nan")),
    )


def fold_class_balance(y, folds):
    """fold별 train/test 클래스 분포. GroupKFold가 층화를 안 하므로 확인용.

    한 fold의 test가 단일 클래스가 되면 AUC가 nan이 되고 정확도가 무의미해진다.
    """
    rows = []
    for i, (tr, _, te) in enumerate(folds):
        rows.append(dict(fold=i,
                         train=np.bincount(y[tr], minlength=2).tolist(),
                         test=np.bincount(y[te], minlength=2).tolist()))
    return rows


def summarize_runs(values):
    """여러 fold/seed 값 → (평균, 표준편차) 문자열."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std():.3f}"
