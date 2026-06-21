# Track A — LightGBM 트랙

핵심 가설: **결측을 채우지 말고 모델이 직접 처리하게 하자.**

| 파일 | 모델 | 핵심 변경 | AUC (5-fold CV) |
|---|---|---|---|
| `v1_baseline_xgboost.py` | XGBoost | 전체 중앙값 imputation (공통 출발점) | 0.5583 |
| `v2_person_imputation.py` | XGBoost | Person-level imputation + Missing flag (97개) | 0.5574 |
| `v3_lightgbm_no_imputation.py` | LightGBM | Imputation 완전 제거, NaN을 트리 분기에서 직접 처리 | **0.5681** (최고) |

## 핵심 발견

- Imputation 방식을 정교화(v1→v2)해도 AUC는 거의 바뀌지 않았다. Person-level이든 전체 중앙값이든 XGBoost 입장에서는 둘 다 "가짜로 채운 값"이라는 점은 동일하기 때문.
- LightGBM으로 전환해 imputation 자체를 제거하자(v3) AUC가 유의미하게 올랐다(+0.0098).
- SHAP 기준 써카디안(IS/IV) 피처 기여도가 v1 13.5% → v3 27.3%로 약 2배 증가. NaN을 별도 트리 경로로 분리 처리하면서 IS/IV의 순수 신호가 다른 피처와 섞이지 않고 더 명확하게 드러난 것으로 해석.

## 실행

```bash
pip install -r ../../requirements.txt lightgbm
python v3_lightgbm_no_imputation.py
```

결과 그래프는 `../../outputs/track_a/`에 저장된다.

상세 분석은 `docs/GLOBEM_Two_Tracks_Full_Story.pdf` (03장) 참고.
