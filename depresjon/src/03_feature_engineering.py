"""Depresjon: 수면 피처 30개 (임계값 버전, activity < 40 -> sleep).
[주의] 이 브랜치의 결과(AUC 0.845 PPT 수치)는 이 단순 임계값 버전 기준으로 추정됨.
"""
#feature 30개 계산(+ subject id, label)
import pandas as pd
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # depresjon/

DATA_DIR      = os.environ.get("DEPRESJON_DATA", os.path.join(ROOT, "data", "depresjon"))
CONDITION_DIR = os.path.join(DATA_DIR, "condition")
CONTROL_DIR   = os.path.join(DATA_DIR, "control")
OUTPUT_DIR    = os.environ.get("DEPRESJON_OUT", os.path.join(ROOT, "outputs"))

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

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

# ── Sleep window 정의 ──────────────────────────────
# activity < 40 → sleep (1), 이상 → wake (0)
SLEEP_THRESHOLD = 40

def compute_sleep_windows(subject_df):
    sdf = subject_df.copy().reset_index(drop=True)
    sdf["sleep"] = (sdf["activity"] < SLEEP_THRESHOLD).astype(int)

    # 10분 미만 gap 병합 (wake가 10분 미만이면 sleep으로 채움)
    in_wake = False
    wake_start = None
    for i in range(len(sdf)):
        if sdf.loc[i, "sleep"] == 0:
            if not in_wake:
                in_wake = True
                wake_start = i
        else:
            if in_wake:
                gap = i - wake_start
                if gap < 10:
                    sdf.loc[wake_start:i-1, "sleep"] = 1
                in_wake = False
                wake_start = None

    # bout 구분 (연속된 sleep/wake 구간)
    sdf["bout_id"] = (sdf["sleep"] != sdf["sleep"].shift()).cumsum()
    bouts = sdf.groupby("bout_id").agg(
        sleep=("sleep", "first"),
        start=("timestamp", "first"),
        end=("timestamp", "last"),
        length=("sleep", "count")
    ).reset_index()

    # Long(≥225분=3.75h) / Short 분류
    sleep_bouts = bouts[bouts["sleep"] == 1].copy()
    sleep_bouts["is_long"] = (sleep_bouts["length"] >= 225).astype(int)
    sleep_bouts["midpoint"] = sleep_bouts["start"] + (
        sleep_bouts["end"] - sleep_bouts["start"]) / 2
    sleep_bouts["date"] = sleep_bouts["midpoint"].dt.date

    return sleep_bouts

# ── 10가지 수면 index 계산 (day-level) ───────────────
def compute_daily_indices(sleep_bouts, total_minutes_per_day):
    if sleep_bouts.empty:
        return None
    daily = []
    for date, grp in sleep_bouts.groupby("date"):
        long_bouts  = grp[grp["is_long"] == 1]
        short_bouts = grp[grp["is_long"] == 0]
        wake_bouts  = total_minutes_per_day.get(str(date), 1440) - grp["length"].sum()

        num_long  = len(long_bouts)
        num_short = len(short_bouts)
        len_long  = long_bouts["length"].mean()  if num_long  > 0 else 0
        len_short = short_bouts["length"].mean() if num_short > 0 else 0
        st_long   = long_bouts["length"].sum()
        st_short  = short_bouts["length"].sum()
        wt_long   = wake_bouts
        wt_short  = wake_bouts
        sleep_pct = grp["length"].sum() / total_minutes_per_day.get(str(date), 1440)
        amp_vals  = grp["length"].values
        sleep_amp = np.std(amp_vals) / (np.mean(amp_vals) + 1e-8)

        daily.append({
            "date": date,
            "num_long": num_long, "num_short": num_short,
            "len_long": len_long, "len_short": len_short,
            "st_long": st_long,   "st_short": st_short,
            "wt_long": wt_long,   "wt_short": wt_short,
            "sleep_pct": sleep_pct, "sleep_amp": sleep_amp
        })
    return pd.DataFrame(daily)

INDEX_COLS = ["num_long","num_short","len_long","len_short",
              "st_long","st_short","wt_long","wt_short",
              "sleep_pct","sleep_amp"]

# ── subject별 피처 계산 → 30개 ──────────────────────
all_features = []
for subj, grp in df.groupby("subject_id"):
    label = grp["label"].iloc[0]
    total_min = grp.groupby("date")["activity"].count().to_dict()

    sleep_bouts = compute_sleep_windows(grp)
    daily_idx   = compute_daily_indices(sleep_bouts, total_min)

    if daily_idx is None or daily_idx.empty:
        continue

    row = {"subject_id": subj, "label": label}
    for col in INDEX_COLS:
        vals = daily_idx[col].dropna().values
        mn   = np.mean(vals)
        sd   = np.std(vals)
        row[f"{col}_MN"] = mn
        row[f"{col}_SD"] = sd
        row[f"{col}_Z"]  = mn / (sd + 1e-8)

    all_features.append(row)

features_df = pd.DataFrame(all_features)
print(f"피처 shape: {features_df.shape}")
print(f"컬럼 수: {len(features_df.columns)}")
print(features_df.head())

out_path = os.path.join(OUTPUT_DIR, "features_30.csv")
features_df.to_csv(out_path, index=False)
print(f"\n저장됨: {out_path}")
