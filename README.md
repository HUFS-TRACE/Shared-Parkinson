# Shared Encoder for Lightweight Parkinson's Disease Screening

## 목표

본 프로젝트는 음성 데이터와 NewHandPD 데이터를 하나의 Shared PatchTST Encoder로 학습하여 경량 파킨슨병 스크리닝 모델을 개발하는 것을 목표로 한다.

Input-based Early Exit을 적용하여 정확도는 유지하면서 평균 추론 시간과 연산량을 감소시키는 것을 목표로 한다.


1. Repository Structure

Shared-Encoder/

├── dataset/
│
├── preprocessing/
│
│   ├── voice/
│   └── handpd/
│
├── models/
│
├── early_exit/
│
├── train.py
├── test.py
└── README.md
