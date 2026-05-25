# GLOBEM Depression Prediction

## 1. 프로젝트 개요

GLOBEM 데이터셋을 활용한 우울증 예측 파이프라인.

- Passive sensing (수면 + 걸음수) 기반 피처 추출
- Circadian 리듬 피처 설계 (IS, IV, RA, morning ratio)
- 개인별 중앙값 대체 + 결측 플래그로 결측 처리

## 2. 데이터셋

- 대상: GLOBEM 대학생 약 330명
- 기간: 2018~2021 (4개 연도, INS-W_1~4)
- 레이블: PHQ-9 기반 주간 우울증 이진 레이블 (person-week 단위)

## 3. 파이프라인 버전

| 버전 | 설명 | AUC |
|------|------|-----|
| v1 | baseline (global median imputation) | 0.5583 |
| v2 | person-level imputation + missing flags | 0.5593 |
| v3 | XGBoost + RA + morning ratio | 0.5642 |
| v4 | RandomForest + RA + morning ratio | **0.5656** ← 최고 |

## 4. 피처 구성

- 수면 피처: 80개 (5개 시간대 x 16개 RAPIDS 통계)
- 걸음수 피처: 45개 (5개 시간대 x 9개 RAPIDS 통계)
- Circadian 피처
  - IS (Interdaily Stability): 하루 단위 규칙성
  - IV (Intradaily Variability): 하루 내 변동성
  - RA (Relative Amplitude): (M10 - L5) / (M10 + L5)
  - morning_activity_ratio: 아침 걸음수 / (아침 + 저녁)

## 5. 실험 결과

| 버전 | AUC |
|------|-----|
| v1 | 0.5583 |
| v2 | 0.5593 |
| v3 | 0.5642 |
| v4 | 0.5656 |

## 6. SHAP 피처 중요도 (v4 기준)

| 피처 그룹 | SHAP 비중 |
|-----------|-----------|
| Sleep / Activity | 76.3% |
| Circadian (IS/IV/RA) | 16.3% |
| Morning ratio | 3.0% |
| Missing flag | 4.4% |

## 7. 시도했으나 성능 저하

- Linear Interpolation: person-level median보다 AUC 하락
- Within-person Z-score 정규화: 효과 없음
- 7일 평균 매칭 (experiments/globem_pipeline_exp_7day.py): AUC 0.5564 (설문 당일 매칭보다 낮음)
- LightGBM (experiments/globem_pipeline_exp_lgbm.py): AUC 0.5574 (XGBoost보다 낮음)

## 8. 설치 및 실행

```bash
pip install -r requirements.txt
python src/globem_pipeline_v4.py
```

## 9. 데이터 구조

```
data/raw/
├── INS-W_1/
│   ├── FeatureData/   # Fitbit RAPIDS 피처 (sleep.csv, steps.csv 등)
│   └── SurveyData/    # PHQ-9 레이블 (dep_weekly.csv)
├── INS-W_2/
├── INS-W_3/
└── INS-W_4/
```
