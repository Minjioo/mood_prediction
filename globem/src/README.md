# src/

현재 비어 있음.

GLOBEM 실험은 v1에서 Track A(LightGBM)와 Track B(RandomForest)로 분기되어 독립적으로
진행되었고, 아직 두 트랙을 통합한 단일 "정식 파이프라인"이 확정되지 않았다.
각 트랙의 코드는 `../experiments/track_a_lightgbm/`, `../experiments/track_b_randomforest/`에
있다.

## 다음 단계 (v5)

두 트랙에서 검증된 개선 요소를 결합한 v5가 여기에 들어올 예정:

- Track A에서: LightGBM + imputation 완전 제거
- Track B에서: RA(Relative Amplitude) + Morning ratio 써카디안 피처

```
v5 = LightGBM + RA + Morning Ratio + IS/IV (imputation 없음)
```

v5가 검증되면 이 파일을 지우고 `src/pipeline.py` (및 필요 시 `preprocessing.py`,
`features.py`, `model.py` 등으로 모듈화)를 추가한다.
