#데이터 다운로드
import pandas as pd
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # depresjon/

# ── 경로 설정 ──────────────────────────────────────
DATA_DIR = os.environ.get("DEPRESJON_DATA", os.path.join(ROOT, "data", "depresjon"))
CONDITION_DIR = os.path.join(DATA_DIR, "condition")
CONTROL_DIR   = os.path.join(DATA_DIR, "control")
SCORES_PATH   = os.path.join(DATA_DIR, "scores.csv")

# ── scores.csv 로드 ────────────────────────────────
scores = pd.read_csv(SCORES_PATH)
print("=== scores.csv ===")
print(scores.head())
print(f"shape: {scores.shape}\n")

# ── activity CSV 로드 함수 ─────────────────────────
def load_activity_files(folder, label):
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)
        df["subject_id"] = fname.replace(".csv", "")
        df["label"] = label  # 1=condition, 0=control
        records.append(df)
    return pd.concat(records, ignore_index=True)

# ── 로드 ──────────────────────────────────────────
condition_df = load_activity_files(CONDITION_DIR, label=1)
control_df   = load_activity_files(CONTROL_DIR,   label=0)
df = pd.concat([condition_df, control_df], ignore_index=True)

# ── 기본 확인 ─────────────────────────────────────
print("=== activity 데이터 ===")
print(df.head())
print(f"\nshape: {df.shape}")
print(f"\n컬럼: {df.columns.tolist()}")
print(f"\nsubject 수: {df['subject_id'].nunique()}")
print(f"\nlabel 분포:\n{df['label'].value_counts()}")
print(f"\n결측값:\n{df.isnull().sum()}")