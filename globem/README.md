# GLOBEM Depression Prediction

## 1. 프로젝트 개요

GLOBEM 데이터셋을 활용한 우울증 예측 파이프라인. Depresjon(입원 수준 환자 55명,
raw actigraphy)에서 시작한 연구를 일상생활이 가능한 대학생 코호트로 확장한 것 —
전체 연구 배경은 [최상위 README](../README.md) 참고.

- Passive sensing (수면 + 걸음수) 기반 피처 추출
- Circadian 리듬 피처 설계 (IS, IV, RA, morning ratio)
- 개인별 중앙값 대체 + 결측 플래그로 결측 처리

같은 v1(XGBoost + 전체 중앙값 imputation, AUC 0.5583)에서 출발해
**Track A(LightGBM)와 Track B(RandomForest)** 두 갈래로 독립적으로 개선을 진행했다.
아직 두 트랙을 통합한 정식 파이프라인(v5)은 확정되지 않았다.

## 2. 데이터셋

- 대상: GLOBEM 대학생 약 330명
- 기간: 2018~2021 (4개 연도, INS-W_1~4)
- 레이블: PHQ-9 기반 주간 우울증 이진 레이블 (person-week 단위)

## 3. 폴더 구조

```
globem/
├── data/raw/                         # 원본 데이터 배치 자리 (git 미포함) - data/README.md 참고
├── src/                              # 정식 채택 파이프라인 (v5 통합 후 채워질 예정, 현재 비어있음)
├── experiments/
│   ├── track_a_lightgbm/             # 결측을 채우지 말고 모델이 직접 처리 → AUC 0.5681
│   └── track_b_randomforest/         # 결측을 잘 채우고 새 써카디안 지표(RA) 추가 → AUC 0.5656
└── outputs/
    ├── track_a/                      # Track A 출력 (실행 시 자동 생성)
    ├── track_b/                      # Track B 출력
    └── unconfirmed_source/           # 출처 미확인 결과물 (정리 시점에 트랙 구분 불가)
```

각 트랙의 버전별 변경사항, AUC, 실행 방법은 해당 폴더의 README 참고.

## 4. 두 트랙 비교

| 트랙 | 버전 | 모델 | 핵심 변경 | AUC (5-fold CV) |
|---|---|---|---|---|
| 공통 | v1 | XGBoost | 전체 중앙값 imputation | 0.5583 |
| Track A | v2 | XGBoost | Person-level imputation + Missing flag | 0.5574 |
| Track A | v3 | LightGBM | Imputation 완전 제거, NaN 직접 처리 | **0.5681**(전체 최고) |
| Track B | v2 | XGBoost | Person-level imputation + Missing flag | 0.5593 |
| Track B | v3 | XGBoost | RA(Relative Amplitude) + Morning ratio 피처 추가 | 0.5642 |
| Track B | v4 | RandomForest | 모델 전환 | 0.5656 |

## 5. 피처 구성

- 수면 피처: 80개 (5개 시간대 x 16개 RAPIDS 통계)
- 걸음수 피처: 45개 (5개 시간대 x 9개 RAPIDS 통계)
- Circadian 피처
  - IS (Interdaily Stability): 하루 단위 규칙성
  - IV (Intradaily Variability): 하루 내 변동성
  - RA (Relative Amplitude, Track B v3+): (M10 - L5) / (M10 + L5)
  - morning_activity_ratio (Track B v3+): 아침 걸음수 / (아침 + 저녁)

## 6. SHAP 피처 그룹 기여도

| 구분 | Sleep/Activity | Circadian | Missing Flag | 기타 | 피처 수 |
|---|---|---|---|---|---|
| Track A v1 | 86.5% | 13.5% (IS/IV) | - | - | 127개 |
| Track A v3 (LightGBM) | 72.7% | 27.3% (IS/IV) | - | - | 127개 |
| Track B v4 (RandomForest) | 76.3% | 16.3% (IS/IV/RA) | 4.4% | 3.0% (Morning ratio) | ~131개 |

Track A v3에서 써카디안(IS/IV) SHAP 기여도가 약 2배로 증가. LightGBM이 NaN을 별도
트리 분기로 처리하면서 IS/IV의 고유 신호가 다른 피처와 섞이지 않고 더 명확히
분리된 것으로 해석.

## 7. 핵심 발견

1. 결측률(58.5%)이 핵심 장벽 — 단순 imputation 방식 변경(v1→v2)은 두 트랙 모두 AUC에 거의 영향 없음
2. 모델 구조 변경이 피처 추가보다 효과적 — Track A(모델 변경, +0.0098)가 Track B(피처 추가, +0.0073)보다 큰 개선폭
3. 써카디안 리듬이 핵심 신호 — IS/IV, RA 등 모든 트랙에서 일관되게 유의미한 기여도 확인
4. 같은 실험도 구현 디테일에 따라 다른 결론 — 양 트랙 모두 LightGBM을 시도했으나 Track A(imputation 제거, AUC 0.5681)와 Track B(imputation 유지, AUC 0.5574)가 정반대 결과. imputation 제거 여부가 핵심 변수였던 것으로 추정.

각 트랙에서 시도했으나 효과 없었던 실험은 `experiments/track_*/README.md` 참고.

## 8. 다음 단계 (v5 제안)

두 트랙에서 검증된 개선 요소를 결합:

```
v5 = LightGBM + RA + Morning Ratio + IS/IV (imputation 없음)
```

- Track A에서: LightGBM + imputation 완전 제거
- Track B에서: RA(Relative Amplitude) + Morning ratio 써카디안 피처

추가 탐색 방향:
- HR 데이터 확보 시 Forger ODE 기반 CBTmin/DLMO 추정으로 써카디안 피처 고도화
- 7일 vs 당일 매칭을 LightGBM 기준으로 재검증
- LightGBM(Track A) + RandomForest(Track B) 앙상블 시도

v5가 검증되면 `src/`에 정식 파이프라인으로 채택하고, 필요 시
`preprocessing.py` / `features.py` / `model.py` 등으로 모듈화한다.

## 9. 설치 및 실행

데이터를 먼저 `data/raw/`에 배치한다 (`data/README.md` 참고). 이후 `globem/` 디렉토리 기준:

```bash
pip install -r requirements.txt
python experiments/track_a_lightgbm/v3_lightgbm_no_imputation.py   # Track A 최고 성능
python experiments/track_b_randomforest/v4_randomforest.py         # Track B 최고 성능
```

모든 코드는 파일 위치 기준 상대경로로 `data/raw/`와 `outputs/`를 찾으므로,
로컬 환경에 맞춰 경로를 따로 수정할 필요가 없다.

## 10. 데이터 구조

```
data/raw/
├── INS-W_1/
│   ├── FeatureData/   # Fitbit RAPIDS 피처 (sleep.csv, steps.csv 등)
│   └── SurveyData/    # PHQ-9 레이블 (dep_weekly.csv)
├── INS-W_2/
├── INS-W_3/
└── INS-W_4/
```

데이터셋 자체는 git에 포함되어 있지 않다. 폴더 구조만 미리 만들어 두었으니
[GLOBEM](https://physionet.org/content/globem/)에서 다운로드해 채워 넣으면 된다.

## 11. 발표 자료

전체 구현 과정 및 두 트랙 비교 분석은 `../docs/GLOBEM_Two_Tracks_Full_Story.pdf` 참고 (21슬라이드).

이전 버전 자료: `../docs/GLOBEM_Full_Story.pdf` (Track A 단독, v1~v3),
`../docs/GLOBEM_v2_slides.pdf` (v2 시점)
