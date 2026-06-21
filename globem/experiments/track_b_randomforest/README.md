# Track B — RandomForest 트랙

핵심 가설: **결측을 잘 채우고, 새 써카디안 지표(RA)를 추가하자.**

| 파일 | 모델 | 핵심 변경 | AUC (5-fold CV) |
|---|---|---|---|
| `v1_baseline_xgboost.py` | XGBoost | 전체 중앙값 imputation (공통 출발점) | 0.5583 |
| `v2_person_imputation.py` | XGBoost | Person-level imputation + Missing flag | 0.5593 |
| `v3_ra_morning_ratio.py` | XGBoost | RA(Relative Amplitude), morning_activity_ratio 신규 써카디안 피처 추가 | 0.5642 |
| `v4_randomforest.py` | RandomForest | 모델 전환, class_weight="balanced" | **0.5656** (최고) |

## 신규 피처 (v3)

- **RA (Relative Amplitude)**: `(M10 - L5) / (M10 + L5)` — M10은 가장 활동적인 구간(morning+afternoon) 평균, L5는 가장 비활동적인 구간(night). 표준 actigraphy 비모수 써카디안 지표. RA가 낮을수록 활동-휴식 리듬의 진폭이 작다 → 우울 시 흔한 양상.
- **morning_activity_ratio**: 아침 걸음수 / (아침+저녁 걸음수) — 아침형/저녁형 성향 측정.

## SHAP 피처 그룹 기여도 (v4 기준)

| Sleep/Activity | Circadian (IS/IV/RA) | Missing Flag | Morning Ratio |
|---|---|---|---|
| 76.3% | 16.3% | 4.4% | 3.0% |

## 시도했으나 효과 없었던 것들

- **Linear interpolation imputation**: person-level median보다 AUC 낮음. 시간 순서 기반 보간이 결측 패턴 자체의 정보(신호)를 오히려 매끄럽게 지워버림.
- **Within-person Z-score 정규화**: 효과 없음. 트리 기반 모델은 스케일에 비교적 둔감.
- **7일 평균 매칭**: AUC 0.5564 (당일 매칭보다 낮음). 설문 시점과 가까운 1일 단위 매칭이 더 유효한 신호를 가짐.
- **LightGBM (Track B 버전)**: AUC 0.5574 (XGBoost보다 낮음). imputation을 유지한 채 모델만 LightGBM으로 바꿈 — Track A처럼 imputation을 제거하지 않아 효과가 제한적이었던 것으로 추정. (Track A의 LightGBM 0.5681과 정반대 결과 — imputation 유지 여부가 핵심 변수로 보임)

## 실행

```bash
pip install -r ../../requirements.txt
python v4_randomforest.py
```

결과 그래프는 `../../outputs/track_b/`에 저장된다.

상세 분석은 `docs/GLOBEM_Two_Tracks_Full_Story.pdf` (04장) 참고.
