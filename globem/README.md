# GLOBEM

GLOBEM 데이터셋 (INS-W_1, 2018–2021) 기반 우울증 예측 모델 구현

## 데이터
- 대상: 일상생활 가능 대학생 330명
- 센서: Fitbit (수면 + 걸음수 + HR)
- 레이블: PHQ-9 (주 1회, person-week)
- 규모: 2,360 person-weeks

## 피처 설계

### Step 1 — allday 집계 (44개)
- 수면 28개 + 걸음수 14개 + IS/IV 2개
- AUC: 0.555

### Step 2 — 시간대별 세그먼트 (127개)
- 수면 80개 + 걸음수 45개 + IS/IV 2개
- 세그먼트: allday / morning(06–12) / afternoon(12–18) / evening(18–24) / night(00–06)
- AUC: 0.558

## 결측 처리
- 행 결측률 70% 초과 제거 → 2,360 → 1,745 person-weeks 잔존
- 잔존 결측: 중앙값 대체 → StandardScaler 정규화

## 성능
| 모델 | AUC |
|---|---|
| allday | 0.555 |
| 시간대별 | 0.558 |

## 코드
추후 업로드 예정
