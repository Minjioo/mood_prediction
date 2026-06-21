# 웨어러블 기반 우울증 예측 연구

Actigraphy/Fitbit 등 웨어러블 센서 데이터의 수면·일주기(circadian) 리듬 피처로
우울증을 예측하는 연구. **Depresjon(입원 수준 환자) → GLOBEM(대학생 코호트)**
순서로 데이터셋을 확장하며 진행했다.

| 단계 | 데이터셋 | 대상 | AUC |
|---|---|---|---|
| 1 | Depresjon | 입원 수준 우울 환자 55명 | 0.73 (Sleep+Circadian 12개 피처) → **0.85** (mean/SD/Z-score 36개 피처) |
| 2 | GLOBEM | 일상생활 대학생 약 330명 | 0.5583 (baseline) → **0.5681** (Track A 최고) |

## 1. 연구 배경

[Lim et al., npj Digital Medicine 2024](https://doi.org/10.1038/s41746-024-01333-z)는
기분장애 환자 168명의 Fitbit 수면 데이터에서
Forger(1999) ODE 기반 circadian phase(DLMO)를 추정해 기분 삽화를 예측했다
(AUC 0.80/0.98/0.95, depressive/manic/hypomanic). 본 연구는 이 방법론을
공개 데이터셋에 재현·확장하는 것에서 출발했다.

## 2. Depresjon — 1단계 (소규모 검증)

원 논문의 MDCRC 코호트(168명, 587일 평균 추적)에 접근할 수 없어, 공개 데이터셋인
[Depresjon](https://datasets.simula.no/depresjon/)(condition 23명 / control 32명,
1분 단위 raw actigraphy, 평균 12.6일)으로 동일 파이프라인을 먼저 검증했다.
에피소드 레이블이 없어 mood episode 예측 대신 **condition vs control 이진 분류**로
목표를 재정의했다.

- Cole-Kripke 알고리즘으로 raw activity count → sleep/wake 이진 변환
- Sleep Window(Long/Short) 기반 Sleep Index 10개
- Forger ODE로 DLMO(circadian phase), Circadian amplitude 추정
- XGBoost 분류: 12개 피처(Sleep 10 + Circadian 2) AUC 0.73 → mean/SD/Z-score
  적용 36개 피처 AUC 0.85

상세 내용은 `depresjon/README.md` 참고.

## 3. GLOBEM — 2단계 (확장)

Depresjon에서 방법론을 검증한 뒤, 더 현실적인 환경(비임상, 일상생활, 대규모)으로
확장하기 위해 [GLOBEM](https://physionet.org/content/globem/) 데이터셋
(대학생 약 330명, Fitbit, 2018~2021, PHQ-9 주 1회 레이블)으로 이전했다.

**Depresjon과의 핵심 차이**: GLOBEM은 RAPIDS로 주 단위 집계된 피처만 제공되어
raw 1분 단위 activity count가 없다 → Forger ODE 기반 DLMO/CBTmin 계산이
불가능하고, IS(Interdaily Stability)/IV(Intradaily Variability) 등 활동량 기반
프록시로 circadian 리듬 불규칙성을 간접 추정해야 했다. 또한 경증 대학생 코호트라
우울군-정상군 신호 차이가 작고, 불규칙한 웨어러블 착용으로 결측률이
최대 58.5%까지 발생하는 것이 핵심 난제였다.

baseline(v1, XGBoost + 전체 중앙값 imputation, AUC 0.5583)에서 출발해
**두 갈래로 독립적으로 개선을 진행**했다:

- **Track A (LightGBM 트랙)**: 결측을 채우지 말고 모델이 직접 처리하게 하자 → AUC 0.5681 (최고)
- **Track B (RandomForest 트랙)**: 결측을 잘 채우고 새 circadian 지표(RA)를 추가하자 → AUC 0.5656

두 트랙 비교, 버전별 상세 변경 사항, SHAP 분석은 `globem/README.md` 및
`globem/experiments/track_a_lightgbm/`, `globem/experiments/track_b_randomforest/`
참고. 다음 단계는 두 트랙의 장점을 합친 v5 통합 — `globem/README.md` 8절 참고.

## 4. 폴더 구조

```
.
├── depresjon/              # 1단계: Depresjon 데이터셋 (condition vs control)
│   └── README.md
├── globem/                 # 2단계: GLOBEM 데이터셋 (Track A/B 분기)
│   ├── README.md
│   ├── data/                # 원본 데이터 배치 자리 (git 미포함, 직접 다운로드)
│   ├── src/                # 정식 채택 파이프라인 (v5 통합 후 채워질 예정)
│   ├── experiments/
│   │   ├── track_a_lightgbm/
│   │   └── track_b_randomforest/
│   └── outputs/
├── docs/                   # 발표 자료
│   ├── GLOBEM_Two_Tracks_Full_Story.pdf  # GLOBEM 두 트랙 종합 (최신)
│   ├── GLOBEM_Full_Story.pdf             # GLOBEM Track A 단독 (이전 버전)
│   └── GLOBEM_v2_slides.pdf              # GLOBEM v2 시점 (이전 버전)
└── README.md                # 이 문서
```

## 5. 핵심 발견 (전체 연구 관통)

1. **Circadian 리듬은 일관되게 유의미한 신호다.** Depresjon(raw data 기반 DLMO),
   GLOBEM(집계 데이터 기반 IS/IV/RA) 양쪽 모두에서 circadian 관련 피처가
   상위 importance에 랭크됨 — 원 논문의 핵심 발견(circadian phase가 가장 중요한
   예측 변수)과 일치.
2. **데이터 해상도가 모델 성능의 상한을 결정한다.** raw 데이터(Depresjon)에서는
   Forger ODE로 정밀한 DLMO 추정이 가능했지만, 집계 피처만 제공되는 GLOBEM에서는
   동일 방법론 적용이 불가능해 AUC가 구조적으로 낮게 형성됨(0.85 → 0.56대).
3. **결측 처리는 "방법"보다 "제거 여부"가 중요했다.** GLOBEM에서 person-level
   imputation으로 정교화해도(v1→v2) AUC는 거의 변화 없었으나, imputation 자체를
   없애고 모델(LightGBM)이 결측을 신호로 활용하게 하자 유의미하게 개선됨.

## 6. 다음 단계

- GLOBEM v5: Track A(LightGBM, imputation 제거) + Track B(RA, Morning ratio) 결합
- HR 데이터 확보 시 Forger ODE 기반 CBTmin/DLMO 재도입으로 circadian 피처 고도화
- 장기적으로 raw Fitbit 데이터 접근 시 LSTM/Transformer로 person-week 간
  시계열 패턴 학습 (steps.csv는 날짜별 시퀀스 구성 가능)

