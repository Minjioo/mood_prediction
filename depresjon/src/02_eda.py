#데이터 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # depresjon/

DATA_DIR = os.environ.get("DEPRESJON_DATA", os.path.join(ROOT, "data", "depresjon"))
CONDITION_DIR = os.path.join(DATA_DIR, "condition")
CONTROL_DIR   = os.path.join(DATA_DIR, "control")
SCORES_PATH   = os.path.join(DATA_DIR, "scores.csv")

def load_activity_files(folder, label):
    records = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(folder, fname))
        df["subject_id"] = fname.replace(".csv", "")
        df["label"] = label
        records.append(df)
    return pd.concat(records, ignore_index=True)

df = pd.concat([
    load_activity_files(CONDITION_DIR, 1),
    load_activity_files(CONTROL_DIR, 0)
], ignore_index=True)

scores = pd.read_csv(SCORES_PATH)

# ── 1. subject별 녹화 일수 ─────────────────────────
days_per_subject = df.groupby("subject_id")["date"].nunique()
print("=== subject별 녹화 일수 ===")
print(days_per_subject.describe())
print(f"최소: {days_per_subject.min()}일, 최대: {days_per_subject.max()}일\n")

# ── 2. activity 기술통계 ───────────────────────────
print("=== activity 기술통계 ===")
print(df["activity"].describe())
print()

# ── 3. MADRS 점수 분포 ────────────────────────────
print("=== MADRS 점수 (madrs1 기준) ===")
print(scores["madrs1"].describe())
print()

# ── 4. condition vs control activity 평균 비교 ─────
print("=== label별 activity 평균 ===")
print(df.groupby("label")["activity"].mean())
print()

# ── 5. 시각화 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# activity 분포
axes[0].hist(df["activity"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Activity distribution")
axes[0].set_xlabel("Activity count")
axes[0].set_ylabel("Frequency")

# subject별 녹화 일수
days_per_subject.hist(ax=axes[1], bins=15, color="mediumseagreen", edgecolor="white")
axes[1].set_title("Recording days per subject")
axes[1].set_xlabel("Days")

# MADRS 분포
scores["madrs1"].dropna().hist(ax=axes[2], bins=15, color="salmon", edgecolor="white")
axes[2].set_title("MADRS score distribution")
axes[2].set_xlabel("MADRS score")

plt.tight_layout()
output_path = os.path.join(os.environ.get("DEPRESJON_OUT", os.path.join(ROOT, "outputs")), "eda_plots.png")
plt.savefig(output_path, dpi=150)
print(f"그래프 저장됨: {output_path}")