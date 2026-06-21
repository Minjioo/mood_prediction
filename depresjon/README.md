# Depresjon

Depresjon 데이터셋 기반 condition vs control 이진 분류 모델 구현 (1단계).
원 논문(Lim et al., npj Digital Medicine 2024)의 mood episode 예측 방법론을
공개 데이터셋으로 먼저 검증한 단계 — 전체 흐름은 [최상위 README](../README.md) 참고.

## 데이터
- 대상: 입원 수준 우울 환자 55명 (condition 23명 / control 32명)
- 센서: 손목 액티그래피 (1분 단위 raw activity count)
- 레이블: 없음 → condition vs control 이진 분류로 재정의 (원 논문은 MADRS 기반 mood episode)
- 평균 측정 기간: 12.6일 (5~20일)

## 전처리
- Cole-Kripke 알고리즘으로 raw activity count → sleep(0)/wake(1) 이진 변환
  (단순 threshold 적용 시 sleep 비율이 정상 범위를 벗어나 표준 알고리즘 채택)

## 피처
1차: Sleep Index 10개 + Circadian 2개 (DLMO, amplitude) = 12개
2차: 각 피처에 mean/SD/Z-score 적용 = 36개
- Circadian phase: Forger(1999) ODE → CBTmin 추정 → DLMO = CBTmin − 7h

## 성능
- AUC 0.73 (12개 피처, XGBoost, train 44명/test 11명)
- AUC 0.85 (36개 피처, mean/SD/Z-score 적용)

## 코드
추후 업로드 예정

