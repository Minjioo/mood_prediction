# data/

GLOBEM 원본 데이터셋을 두는 자리. **데이터셋 자체는 git에 포함되어 있지 않다**
(`.gitignore` 처리 — 용량, 라이선스 문제).

[GLOBEM](https://physionet.org/content/globem/)에서 INS-W_1~4를 다운로드한 뒤
아래 구조에 맞춰 배치한다:

```
data/raw/
├── INS-W_1/
│   ├── FeatureData/   # Fitbit RAPIDS 피처 (sleep.csv, steps.csv 등)
│   └── SurveyData/    # PHQ-9 레이블 (dep_weekly.csv)
├── INS-W_2/
│   ├── FeatureData/
│   └── SurveyData/
├── INS-W_3/
│   ├── FeatureData/
│   └── SurveyData/
└── INS-W_4/
    ├── FeatureData/
    └── SurveyData/
```

이 자리(폴더 구조)는 미리 만들어 두었으니, 다운로드한 csv 파일들을 해당 폴더에
그대로 채워 넣으면 된다. 모든 파이프라인 코드(`experiments/track_*/*.py`)는
파일 위치 기준 상대경로(`Path(__file__).parent.parent.parent / "data" / "raw"`)로
이 폴더를 찾으므로, 로컬 절대경로를 따로 수정할 필요가 없다.
