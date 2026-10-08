# Depresjon

Depresjon 데이터셋 기반 우울증(condition vs control) 분류 파이프라인.
논문(Lim et al., npj Digital Medicine 2024)의 수면·써카디안 피처 설계를 공개 액티그래피 데이터로 재현한 구현.

## 데이터
- 대상: condition 23명 + control 32명 (총 55명), 손목 액티그래피 (1분 단위 activity count)
- 레이블: condition / control (MADRS는 `scores.csv`)
- 원본: [Depresjon (Zenodo)](https://datasets.simula.no/depresjon/) — 용량 문제로 저장소에 포함하지 않음.
  `depresjon/data/depresjon/{condition,control}/*.csv` + `scores.csv` 위치에 두거나, 환경변수 `DEPRESJON_DATA`로 경로 지정.

## 이 브랜치의 버전: 임계값(threshold=40) 버전, 피처 36개
- 수면 판정: `activity < 40` → sleep, 10분 미만 wake gap은 sleep으로 병합, 225분(3.75h) 이상이면 Long / 미만이면 Short
- 수면 피처 30개: 수면 지표 10종 × (MN, SD, Z)
- 써카디안 피처 6개: Forger ODE로 CBT 시뮬레이션 → 일별 phase / amplitude × (MN, SD, Z)
- 모델: XGBoost + Bayesian Optimization(5 init + 25 iter), 5-fold Stratified CV

## 실행 순서
```
python src/01_load_data.py
python src/02_eda.py
python src/03_feature_engineering.py   # outputs/features_30.csv
python src/04_circadian.py             # outputs/features_36.csv
python src/05_modeling.py              # xgb_model.json, feature_cols.json
python src/06_shap.py
```
`outputs/features_30.csv`, `outputs/features_36.csv`는 위 코드로 재생성한 값과 일치함을 확인함(최대 오차 3e-11).

## 성능 (재실행 기준, 데이터/환경에 따라 ±수 %p 변동)
| 피처 | Bayesian 탐색 최고 CV AUC | 5-fold AUC |
|---|---|---|
| 36개, 임계값 (이 브랜치) | 0.853 | 0.832 |
| 36개, Cole-Kripke | 0.821 | 0.792 |

발표 자료의 AUC 0.845는 이 임계값 버전의 결과로 **추정**됨(정확히 재현되진 않음). 55명뿐이라 분산이 크므로 상한에 가까운 값으로 해석할 것.

## 알려진 한계
- `wt_long` / `wt_short`는 동일한 값(둘 다 하루 전체 wake 시간) → 36개 중 3개가 중복 피처
- `circ_phase`는 기록 시작 기준 24시간 구간의 CBTmin 위치이며 시계 시각이 아님. DLMO(= CBTmin − 7h)는 계산하지 않음
- Forger ODE는 파라미터를 직접 고정한 간이 구현
- 하이퍼파라미터 탐색과 최종 평가에 같은 CV를 사용하고 early stopping에 검증 fold를 사용 → AUC가 낙관적일 수 있음
- Cole-Kripke 버전(+log1p 스케일)은 별도 실험이었으며 이 브랜치에는 포함하지 않음
