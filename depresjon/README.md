# Depresjon

Depresjon 데이터셋 기반 기분 삽화 예측 모델 구현

## 데이터
- 대상: 입원 수준 우울 환자 55명
- 센서: 손목 액티그래피 (1분 단위 raw activity count)
- 레이블: MADRS (1인 1회 측정)

## 피처
- 수면 피처 30개 + Circadian 피처 6개 (총 36개)
- Circadian phase: Forger ODE → CBTmin → DLMO = CBTmin − 7h

## 성능
- AUC: 0.845 (5-fold CV)

## 코드
추후 업로드 예정
