# GLOBEM 기반 우울증 예측 모델

웨어러블(Fitbit) 데이터를 이용한 우울증 예측 파이프라인. GLOBEM 데이터셋(INS-W_1~4, 대학생 코호트)을 사용하며, Depresjon 데이터셋과 비교하여 모델 및 결측 처리 방식을 개선해왔다.

같은 v1에서 출발해 두 갈래(Track A, Track B)로 독립적으로 개선을 진행했다.

## 실험 분기 구조

v1 (공통 출발점, XGBoost + 전체 중앙값 imputation, AUC 0.5583)
    |
    +-- Track A (pipline/) : XGBoost -> LightGBM
    |     v2: Person-level imputation + Missing flag      AUC 0.5574
    |     v3: LightGBM 전환, imputation 완전 제거          AUC 0.5681  (최고)
    |
    +-- Track B (globem/)  : XGBoost -> RandomForest
          v2: Person-level imputation + Missing flag      AUC 0.5593
          v3: RA + Morning ratio 신규 써카디안 피처 추가     AUC 0.5642
          v4: RandomForest 전환                            AUC 0.5656

## 결과 요약

| 트랙 | 버전 | 모델 | 핵심 변경 | AUC (5-fold CV) |
|---|---|---|---|---|
| 공통 | v1 | XGBoost | 전체 중앙값 imputation | 0.5583 |
| Track A | v2 | XGBoost | Person-level imputation + Missing flag | 0.5574 |
| Track A | v3 | LightGBM | Imputation 완전 제거, NaN 직접 처리 | 0.5681 |
| Track B | v2 | XGBoost | Person-level imputation + Missing flag | 0.5593 |
| Track B | v3 | XGBoost | RA(Relative Amplitude) + Morning ratio 피처 추가 | 0.5642 |
| Track B | v4 | RandomForest | 모델 전환 | 0.5656 |

참고: Depresjon 데이터셋(임상 진단 기반, 결측률 ~5%) 동일 파이프라인 적용 시 AUC 0.845.

## SHAP 피처 그룹 기여도

| 구분 | Sleep/Activity | Circadian | Missing Flag | 기타 | 피처 수 |
|---|---|---|---|---|---|
| Track A v1 | 86.5% | 13.5% (IS/IV) | - | - | 127개 |
| Track A v2 | 82.9% | 13.7% (IS/IV) | 3.3% | - | 224개 |
| Track A v3 | 72.7% | 27.3% (IS/IV) | - | - | 127개 |
| Track B v4 | 76.3% | 16.3% (IS/IV/RA) | 4.4% | 3.0% (Morning ratio) | ~131개 |

Track A v3에서 써카디안(IS/IV) SHAP 기여도가 약 2배로 증가. LightGBM이 NaN을 별도 트리 분기로 처리하면서 IS/IV의 고유 신호가 더 명확히 분리된 것으로 해석.

## 핵심 발견

1. 결측률(58.5%)이 핵심 장벽 - 단순 imputation 방식 변경(v1->v2)은 두 트랙 모두 AUC에 거의 영향 없음
2. 모델 구조 변경이 피처 추가보다 효과적 - Track A(모델 변경, +0.0098)가 Track B(피처 추가, +0.0073)보다 큰 개선폭
3. 써카디안 리듬이 핵심 신호 - IS/IV, RA 등 모든 트랙에서 일관되게 유의미한 기여도 확인
4. 같은 실험도 구현 디테일에 따라 다른 결론 - 양 트랙 모두 LightGBM을 시도했으나 Track A(imputation 제거, AUC 0.5681)와 Track B(imputation 유지, AUC 0.5574)가 정반대 결과. imputation 제거 여부가 핵심 변수였던 것으로 추정.

## Track B 추가 실험 (효과 없었던 시도)

- Linear Interpolation: person-level median보다 AUC 하락
- Within-person Z-score 정규화: 트리 기반 모델에서 효과 없음
- 7일 평균 매칭: AUC 0.5564 (설문 당일 매칭보다 낮음)

## 파이프라인 구조

pipline/ (Track A)
- globem_pipeline.py : v1 - XGBoost + 전체 중앙값
- globem_pipeline_v2.py : v2 - XGBoost + person-level imputation + missing flag
- globem_pipeline_v3.py : v3 - LightGBM + no imputation

globem/src/ (Track B)
- globem_pipeline_v1.py : v1 - XGBoost + 전체 중앙값 (공통 출발점)
- globem_pipeline_v2.py : v2 - XGBoost + person-level imputation + missing flag
- globem_pipeline_v3.py : v3 - XGBoost + RA + morning ratio
- globem_pipeline_v4.py : v4 - RandomForest + RA + morning ratio

공통 사항:
- 피처: 수면(RAPIDS) 80개 + 걸음수 45개 기반
- 평가: StratifiedGroupKFold 5-fold CV (참여자 단위 분리, leakage 방지)
- 하이퍼파라미터: Bayesian Optimization

## 다음 단계 (v5 제안)

두 트랙에서 검증된 개선 요소를 결합:
- Track A에서: LightGBM + imputation 완전 제거
- Track B에서: RA(Relative Amplitude) + Morning ratio 써카디안 피처

추가 탐색 방향:
- HR 데이터 확보 시 Forger ODE 기반 CBTmin/DLMO 추정으로 써카디안 피처 고도화
- 7일 vs 당일 매칭을 LightGBM 기준으로 재검증
- LightGBM(Track A) + RandomForest(Track B) 앙상블 시도

## 발표 자료

전체 구현 과정 및 두 트랙 비교 분석은 docs/GLOBEM_Two_Tracks_Full_Story.pdf 참고 (21슬라이드).

이전 버전 자료: docs/GLOBEM_Full_Story.pdf (Track A 단독, v1~v3), docs/GLOBEM_v2_slides.pdf (v2 시점)
