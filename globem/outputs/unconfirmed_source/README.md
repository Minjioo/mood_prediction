# 출처 미확인

이 폴더의 이미지/CSV는 Track A, Track B 중 어느 실험에서 나온 결과물인지
2026년 정리 시점에 확인되지 않았다 (둘 다 v2 버전을 가지고 있어 파일명만으로
구분 불가, AUC 표기도 README의 공식 수치와 정확히 일치하지 않음 — 예:
v1.png는 Mean AUC=0.5559로 표시되나 공식 기록된 v1/v2 AUC와 다름).

- `v1.png`, `v2.png` — SHAP beeswarm + feature importance, Mean AUC=0.5559 표기
- `shap_v2.png`, `shap_v3.png`
- `globem_shap_results.csv` — feature, mean_shap, category 컬럼

추후 재실행하여 어느 트랙/버전 결과인지 확인되면 해당 트랙의
`experiments/track_*/` 또는 `outputs/track_*/`로 옮길 것.
