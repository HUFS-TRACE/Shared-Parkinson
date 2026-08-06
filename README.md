# Shared PatchTST 기반 경량 파킨슨병 스크리닝 모델

## 📌 프로젝트 목표

본 프로젝트는 **음성(Voice)** 데이터와 **필압(HandPD)** 데이터를 하나의 **Shared PatchTST Encoder**를 통해 학습하는 경량 파킨슨병 스크리닝 모델을 개발하는 것을 목표로 한다.

음성과 필압에 대해 각각 별도의 모델을 학습하는 것이 아니라, **모달리티별 입력 임베딩(Embedding)은 분리**하고, **Transformer Encoder는 공유(Shared Encoder)** 하여 하나의 모델로 학습한다.

또한 Layer 기반 Early Exit이 아닌, **Input-based Early Exit** 방식을 적용하여 입력 데이터를 순차적으로 처리하면서 충분한 신뢰도(Confidence)를 얻었을 경우 전체 입력을 모두 사용하지 않고 조기에 추론을 종료하는 것을 목표로 한다.

### 개발 목표

- Voice와 HandPD를 하나의 Shared PatchTST Encoder로 학습
- 모달리티별 Embedding, Shared Transformer 구조 설계
- Input-based Early Exit 적용
- 정확도를 유지하면서 평균 추론 시간 및 연산량 감소

---

# 📂 프로젝트 구조

```text
Shared-Encoder/
│
├── README.md
├── requirements.txt
├── train.py
├── test.py
│
├── configs/
│   ├── config.yaml
│   └── model.yaml
│
├── dataset/
│   ├── raw/
│   │   ├── voice/
│   │   └── handpd/
│   │
│   ├── processed/
│   │   ├── voice/
│   │   └── handpd/
│   │
│   └── split/
│       ├── holdout.csv
│       ├── fold1.csv
│       ├── fold2.csv
│       ├── fold3.csv
│       ├── fold4.csv
│       └── fold5.csv
│
├── preprocessing/
│   ├── voice/
│   │   └── preprocess_voice.py
│   │
│   └── handpd/
│       └── preprocess_hand.py
│
├── models/
│   ├── embedding.py
│   ├── patch_embedding.py
│   ├── shared_patchtst.py
│   └── classifier.py
│
├── early_exit/
│   └── input_based_exit.py
│
└── utils/
    ├── dataset.py
    └── metrics.py
```