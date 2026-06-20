# GLOBEM 기반 우울증 예측 모델

웨어러블(Fitbit) 데이터를 이용한 우울증 예측 파이프라인. GLOBEM 데이터셋(INS-W_1~4, 대학생 코호트)을 사용하며, Depresjon 데이터셋과 비교하여 모델 및 결측 처리 방식을 개선해왔다.

## 결과 요약

| 버전 | 모델 | 핵심 변경 | 피처 수 | AUC (5-fold CV) |
|---|---|---|---|---|
| v1 | XGBoost | 전체 중앙값 imputation | 127개 | 0.5583 |
| v2 | XGBoost | Person-level imputation + Missing flag | 224개 | 0.5574 |
| v3 | LightGBM | Imputation 제거, NaN 직접 처리 | 127개 | 0.5681 |

참고: Depresjon 데이터셋(임상 진단 기반, 결측률 ~5%) 동일 파이프라인 적용 시 AUC 0.845.

## SHAP 피처 그룹 기여도

| 버전 | Sleep/Activity | Circadian (IS/IV) | Missing Flag |
|---|---|---|---|
| v1 | 86.5% | 13.5% | - |
| v2 | 82.9% | 13.7% | 3.3% |
| v3 | 72.7% | 27.3% | - |

v3에서 써카디안(IS/IV) 피처의 SHAP 기여도가 약 2배로 증가. LightGBM이 NaN을 별도 트리 분기로 처리하면서 IS/IV의 고유 신호가 더 명확히 분리된 것으로 해석.

## 핵심 발견

1. 결측률(58.5%)이 핵심 장벽 - 단순 imputation 방식 변경(v1->v2)은 AUC에 거의 영향 없음 (-0.0009)
2. 모델 구조 변경이 더 효과적 - imputation을 제거하고 LightGBM으로 NaN을 직접 처리한 v3에서 첫 유의미한 개선 (+0.0098)
3. 써카디안 리듬이 핵심 신호 - IS/IV 기여도가 일관되게 유지/증가하며, 우울 예측에서 수면/활동량만큼 중요한 신호로 확인

## 파이프라인 구조

pipline/
- globem_pipeline.py      (v1 - XGBoost + 전체 중앙값)
- globem_pipeline_v2.py   (v2 - XGBoost + person-level imputation + missing flag)
- globem_pipeline_v3.py   (v3 - LightGBM + no imputation)

공통 사항:
- 피처: 수면(RAPIDS) 80개 + 걸음수 45개 + IS/IV 써카디안 프록시 2개
- IS/IV는 morning/afternoon/evening/night 4구간 걸음수로부터 직접 계산 (원본 파일에 미포함)
- 평가: StratifiedGroupKFold 5-fold CV (참여자 단위 분리, leakage 방지)
- 하이퍼파라미터: Bayesian Optimization

## 다음 단계 (v4 계획)

- Forger ODE 기반 CBTmin/DLMO 추정으로 써카디안 피처 정교화
- HR(심박수) 데이터 확보 시 통합 (현재 GLOBEM에는 HR 미포함)
- allday 단일 세그먼트 vs 시간대별 세그먼트 결측률 트레이드오프 추가 검증

## 발표 자료

전체 구현 과정 및 분석은 docs/GLOBEM_Full_Story.pdf 참고.
