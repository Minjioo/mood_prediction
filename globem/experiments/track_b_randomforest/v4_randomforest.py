
"""
globem_pipeline_v4.py
=====================
Depression prediction pipeline using the GLOBEM dataset.

Key changes from v3
-------------------
① XGBoost -> Random Forest (RandomForestClassifier)
② Bayesian Optimization pbounds updated for Random Forest
   (n_estimators, max_depth, min_samples_split,
    min_samples_leaf, max_features)
③ class_weight="balanced" replaces scale_pos_weight

Results
-------
v1 AUC : 0.5583
v2 AUC : 0.5593
v3 AUC : 0.5642
v4 AUC : 0.5656

Requirements
------------
    pip install pandas numpy scikit-learn bayesian-optimization shap matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# Configuration
# =============================================================================

BASE = Path(__file__).parent.parent.parent / "data" / "raw"

DATASETS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
SEGMENTS = ["allday", "morning", "afternoon", "evening", "night"]

SLEEP_BASE = [
    "fitbit_sleep_summary_rapids_avgdurationasleepmain",
    "fitbit_sleep_summary_rapids_sumdurationasleepmain",
    "fitbit_sleep_summary_rapids_avgdurationawakemain",
    "fitbit_sleep_summary_rapids_countepisodemain",
    "fitbit_sleep_summary_rapids_avgefficiencymain",
    "fitbit_sleep_summary_rapids_avgdurationinbedmain",
    "fitbit_sleep_summary_rapids_avgdurationtofallasleepmain",
    "fitbit_sleep_summary_rapids_firstbedtimemain",
    "fitbit_sleep_summary_rapids_lastbedtimemain",
    "fitbit_sleep_summary_rapids_firstwaketimemain",
    "fitbit_sleep_summary_rapids_lastwaketimemain",
    "fitbit_sleep_intraday_rapids_avgdurationasleepunifiedmain",
    "fitbit_sleep_intraday_rapids_avgdurationawakeunifiedmain",
    "fitbit_sleep_intraday_rapids_countepisodeasleepunifiedmain",
    "fitbit_sleep_intraday_rapids_stddurationasleepunifiedmain",
    "fitbit_sleep_intraday_rapids_ratiodurationasleepunifiedwithinmain",
]

STEPS_BASE = [
    "fitbit_steps_intraday_rapids_sumsteps",
    "fitbit_steps_intraday_rapids_avgsteps",
    "fitbit_steps_intraday_rapids_stdsteps",
    "fitbit_steps_intraday_rapids_countepisodesedentarybout",
    "fitbit_steps_intraday_rapids_avgdurationsedentarybout",
    "fitbit_steps_intraday_rapids_stddurationsedentarybout",
    "fitbit_steps_intraday_rapids_countepisodeactivebout",
    "fitbit_steps_intraday_rapids_avgdurationactivebout",
    "fitbit_steps_intraday_rapids_stddurationactivebout",
]

ROW_MISSING_THRESHOLD = 0.70
ADD_MISSING_FLAGS = True


# =============================================================================
# Circadian proxy: IS / IV
# =============================================================================

def compute_IS_IV(row):
    """
    Compute IS/IV circadian proxies from 4 time-segment step counts.

    IS (Interdaily Stability): consistency of daily rhythm (0-1, higher = more stable)
    IV (Intradaily Variability): fragmentation within a day (higher = more fragmented)
    """
    seg_cols = [
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:morning",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:night",
    ]
    segs = np.array([row.get(c, np.nan) for c in seg_cols], dtype=float)

    if np.isnan(segs).all():
        return np.nan, np.nan

    segs     = np.nan_to_num(segs, nan=0.0)
    mean_val = np.mean(segs)

    if mean_val == 0:
        return 0.0, 0.0

    IS    = 1.0 / (1.0 + np.std(segs) / (mean_val + 1e-8))
    diffs = np.diff(segs)
    IV    = np.mean(diffs ** 2) / (np.var(segs) + 1e-8)

    return IS, IV


# =============================================================================
# Feature engineering
# =============================================================================

def build_feature_cols():
    """Expand base names x segments into full RAPIDS column names."""
    sleep_cols = [f"f_slp:{base}:{seg}"   for base in SLEEP_BASE for seg in SEGMENTS]
    steps_cols = [f"f_steps:{base}:{seg}" for base in STEPS_BASE for seg in SEGMENTS]
    return sleep_cols, steps_cols


def add_missing_flags(df, feature_cols):
    """
    [v2] Create binary missing-indicator columns BEFORE imputation.
    Non-wear events may themselves signal low motivation or depressive behaviour.
    Indicators that are always 0 (feature never missing) are dropped.
    """
    flag_cols = []
    for col in feature_cols:
        flag_name = col + "__missing"
        df[flag_name] = df[col].isna().astype(np.int8)
        flag_cols.append(flag_name)

    useful  = [c for c in flag_cols if df[c].sum() > 0]
    useless = [c for c in flag_cols if c not in useful]
    df.drop(columns=useless, inplace=True)

    print(f"  Missing-indicator features: {len(useful)} "
          f"(out of {len(feature_cols)} had at least one NaN)")
    return df, useful


def person_level_impute(df, feature_cols):
    """
    [v2] Two-stage median imputation.

    Stage 1 - Personal median:
        Fill NaN with that participant's own historical median across their weeks.
    Stage 2 - Cohort median fallback:
        If a participant has no observed values at all, fall back to cohort median.
    """
    df = df.copy()
    person_medians = df.groupby("subject_id")[feature_cols].transform("median")
    df[feature_cols] = df[feature_cols].fillna(person_medians)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return df


def add_engineered_features(df):
    """
    [v3] Add morning activity ratio and RA (Relative Amplitude) features.

    morning_activity_ratio: proportion of (morning + evening) steps that occur
        in the morning -- proxy for circadian phase preference.

    circ_RA (Relative Amplitude): standard actigraphy non-parametric circadian
        measure. M10 = mean of most-active segments (morning + afternoon),
        L5 = least-active segment (night). RA = (M10 - L5) / (M10 + L5).
    """
    morning_col   = "f_steps:fitbit_steps_intraday_rapids_sumsteps:morning"
    afternoon_col = "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon"
    evening_col   = "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening"
    night_col     = "f_steps:fitbit_steps_intraday_rapids_sumsteps:night"

    new_cols = []

    # Morning activity ratio
    if morning_col in df.columns and evening_col in df.columns:
        total = df[morning_col] + df[evening_col]
        df["morning_activity_ratio"] = np.where(
            total > 0,
            df[morning_col] / total,
            np.nan,
        )
        new_cols.append("morning_activity_ratio")
        print("  morning_activity_ratio: added")
    else:
        print("  morning_activity_ratio: skipped (columns missing)")

    # RA (Relative Amplitude)
    m10_present = [c for c in [morning_col, afternoon_col] if c in df.columns]
    if m10_present and night_col in df.columns:
        M10   = df[m10_present].mean(axis=1)
        L5    = df[night_col]
        denom = M10 + L5
        df["circ_RA"] = np.where(denom > 0, (M10 - L5) / denom, np.nan)
        new_cols.append("circ_RA")
        print("  circ_RA (Relative Amplitude): added")
    else:
        print("  circ_RA: skipped (columns missing)")

    return df, new_cols


# =============================================================================
# Data loading
# =============================================================================

def load_dataset(name):
    feat_dir   = BASE / name / "FeatureData"
    label_path = BASE / name / "SurveyData" / "dep_weekly.csv"

    # Labels
    labels = pd.read_csv(label_path)
    labels.columns = labels.columns.str.strip()

    if "dep_weekly" in labels.columns:
        labels = labels.rename(columns={"pid": "subject_id", "dep_weekly": "label"})
    elif "dep" in labels.columns:
        labels = labels.rename(columns={"pid": "subject_id", "dep": "label"})
        labels["label"] = labels["label"].map({True: 1, False: 0, "True": 1, "False": 0})
    else:
        print(f"  {name}: label column not found")
        return None

    labels = labels[["subject_id", "label", "date"]].dropna(subset=["label"])
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["label"])
    labels["label"] = (labels["label"] >= 1).astype(int)

    print(f"\nLoading {name}...  labels: {len(labels):,}  dep=1: {labels['label'].sum()}")

    # Features
    sleep_cols, steps_cols = build_feature_cols()

    feat_dfs = []
    for fname in ["sleep.csv", "steps.csv"]:
        p = feat_dir / fname
        if p.exists():
            tmp = pd.read_csv(p, low_memory=False)
            tmp.columns = tmp.columns.str.strip()
            feat_dfs.append(tmp)

    if not feat_dfs:
        print(f"  {name}: no feature files found")
        return None

    feats = feat_dfs[0]
    for extra in feat_dfs[1:]:
        merge_on = [c for c in ["pid", "date"] if c in feats.columns and c in extra.columns]
        if merge_on:
            feats = feats.merge(extra, on=merge_on, how="outer")
    feats = feats.rename(columns={"pid": "subject_id"})

    # IS/IV circadian proxies (computed from raw segment step counts)
    seg_step_cols = [
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:morning",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:night",
    ]
    if all(c in feats.columns for c in seg_step_cols):
        feats[["circ_IS", "circ_IV"]] = feats.apply(
            lambda r: pd.Series(compute_IS_IV(r)), axis=1
        )
        circ_computed = ["circ_IS", "circ_IV"]
        print("  Circadian IS/IV: computed")
    else:
        circ_computed = []
        print("  Circadian IS/IV: skipped")

    avail_sleep = [c for c in sleep_cols if c in feats.columns]
    avail_steps = [c for c in steps_cols if c in feats.columns]
    avail_feat  = avail_sleep + avail_steps + circ_computed

    print(f"  Sleep features:    {len(avail_sleep)}/{len(sleep_cols)}")
    print(f"  Steps features:    {len(avail_steps)}/{len(steps_cols)}")
    print(f"  Circadian (IS/IV): {len(circ_computed)}")

    merge_keys = ["subject_id"] + (["date"] if "date" in feats.columns else [])
    df = labels.merge(
        feats[merge_keys + avail_feat].drop_duplicates(subset=merge_keys),
        on=merge_keys, how="left"
    )
    df["dataset"] = name

    row_miss     = df[avail_feat].isna().mean(axis=1)
    overall_miss = df[avail_feat].isna().mean().mean()
    print(f"  Overall missing rate: {overall_miss:.1%}")

    df = df[row_miss <= ROW_MISSING_THRESHOLD].copy()
    print(f"  Rows after filter: {len(df):,}  dep=1: {int(df['label'].sum())}")

    return df, avail_feat


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    print("=" * 62)
    print("GLOBEM Depression Prediction - Pipeline v6")
    print("RandomForest + RA + Morning Ratio")
    print("=" * 62)

    all_dfs, all_feat_cols = [], None
    for name in DATASETS:
        result = load_dataset(name)
        if result is None:
            continue
        df_ds, feat_cols = result
        all_dfs.append(df_ds)
        if all_feat_cols is None:
            all_feat_cols = feat_cols

    if not all_dfs:
        print("No data loaded - check BASE path.")
        return

    df        = pd.concat(all_dfs, ignore_index=True)
    feat_cols = all_feat_cols
    print(f"\nAll cohorts merged: {len(df):,} rows  dep=1: {int(df['label'].sum())}")

    # Step 1: missing flags (before imputation)
    if ADD_MISSING_FLAGS:
        print("\n[Step 1] Creating missing-indicator features ...")
        df, flag_cols = add_missing_flags(df, feat_cols)
    else:
        flag_cols = []

    # Step 2: person-level median imputation
    print("[Step 2] Person-level median imputation ...")
    before_nan = int(df[feat_cols].isna().sum().sum())
    df         = person_level_impute(df, feat_cols)
    after_nan  = int(df[feat_cols].isna().sum().sum())
    print(f"  NaNs: {before_nan:,} -> {after_nan:,}")

    # Step 3: engineered features (computed on imputed data)
    print("[Step 3] Adding engineered features ...")
    df, eng_cols = add_engineered_features(df)

    final_feat_cols = feat_cols + flag_cols + eng_cols
    print(f"\nFinal features: {len(final_feat_cols)} "
          f"(base {len(feat_cols)} + flags {len(flag_cols)} + engineered {len(eng_cols)})")

    X      = StandardScaler().fit_transform(df[final_feat_cols].values.astype(np.float32))
    y      = df["label"].values
    groups = df["subject_id"].values

    # Step 4: Bayesian optimisation
    print("\n[Step 4] Bayesian Optimisation (10 init + 20 iter) ...")

    def rf_cv(n_estimators, max_depth, min_samples_split,
              min_samples_leaf, max_features):
        params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        cv     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y, groups):
            m    = RandomForestClassifier(**params)
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[val])[:, 1]
            if len(np.unique(y[val])) == 2:
                scores.append(roc_auc_score(y[val], prob))
        return np.mean(scores) if scores else 0.5

    optimizer = BayesianOptimization(
        f=rf_cv,
        pbounds=dict(
            n_estimators=(50, 300),
            max_depth=(3, 20),
            min_samples_split=(2, 20),
            min_samples_leaf=(1, 20),
            max_features=(0.1, 1.0),
        ),
        random_state=42, verbose=0,
    )
    optimizer.maximize(init_points=10, n_iter=20)

    best = optimizer.max["params"]
    print(f"  Best CV AUC: {optimizer.max['target']:.4f}")

    # Step 5: final 5-fold CV
    print("\n[Step 5] Final 5-fold CV ...")
    final_params = dict(
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        min_samples_split=int(best["min_samples_split"]),
        min_samples_leaf=int(best["min_samples_leaf"]),
        max_features=best["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv        = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for fold, (tr, val) in enumerate(cv.split(X, y, groups)):
        m    = RandomForestClassifier(**final_params)
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[val])[:, 1]
        if len(np.unique(y[val])) == 2:
            auc = roc_auc_score(y[val], prob)
            fold_aucs.append(auc)
            print(f"  Fold {fold+1} AUC: {auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f"\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Step 6: SHAP
    print("\n[Step 6] SHAP analysis ...")
    model_full = RandomForestClassifier(**final_params)
    model_full.fit(X, y)
    explainer = shap.TreeExplainer(model_full)
    raw       = explainer.shap_values(X)
    if isinstance(raw, list):
        shap_vals = raw[1]           # list → pos class
    elif raw.ndim == 3:
        shap_vals = raw[:, :, 1]     # (samples, features, classes) → pos class
    else:
        shap_vals = raw              # already (samples, features)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Feature category indices
    CIRC_NAMES = {"circ_IS", "circ_IV", "circ_RA"}
    ENG_NAMES  = {"morning_activity_ratio"}

    flag_idx  = [i for i, c in enumerate(final_feat_cols) if "__missing" in c]
    circ_idx  = [i for i, c in enumerate(final_feat_cols)
                 if c in CIRC_NAMES or "bedtime" in c or "waketime" in c]
    eng_idx   = [i for i, c in enumerate(final_feat_cols) if c in ENG_NAMES]
    feat_idx  = [i for i, c in enumerate(final_feat_cols)
                 if i not in flag_idx and i not in circ_idx and i not in eng_idx]

    flag_shap  = mean_abs_shap[flag_idx].sum()
    circ_shap  = mean_abs_shap[circ_idx].sum()
    eng_shap   = mean_abs_shap[eng_idx].sum() if eng_idx else 0.0
    feat_shap  = mean_abs_shap[feat_idx].sum()
    total_shap = flag_shap + circ_shap + eng_shap + feat_shap

    print(f"\n  Sleep/Activity SHAP      : {feat_shap/total_shap:.1%}")
    print(f"  Circadian (IS/IV/RA) SHAP: {circ_shap/total_shap:.1%}")
    print(f"  Morning ratio SHAP       : {eng_shap/total_shap:.1%}")
    print(f"  Missing flag SHAP        : {flag_shap/total_shap:.1%}")

    # Top-20 bar chart
    top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
    top20_names = [final_feat_cols[i] for i in top20_idx]
    top20_vals  = mean_abs_shap[top20_idx]

    def get_color(name):
        if "__missing" in name: return "#E05A5A"  # red    = missing flag
        if name in CIRC_NAMES:  return "#F0A500"  # orange = circadian
        if name in ENG_NAMES:   return "#2CA02C"  # green  = engineered ratio
        return "#4C72B0"                           # blue   = sleep/activity

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    shap.summary_plot(shap_vals, X, feature_names=final_feat_cols,
                      show=False, max_display=20)
    plt.sca(axes[0])
    plt.title(f"SHAP Beeswarm  (AUC={mean_auc:.4f})", fontsize=13)

    bar_colors = [get_color(n) for n in top20_names]
    axes[1].barh(range(20), top20_vals[::-1], color=bar_colors[::-1])
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels(
        [n.split(":")[-1][:40] for n in top20_names[::-1]], fontsize=9
    )
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top 20 Feature Importance", fontsize=13)
    axes[1].legend(handles=[
        Patch(color="#4C72B0", label=f"Sleep/Activity ({feat_shap/total_shap:.0%})"),
        Patch(color="#F0A500", label=f"Circadian IS/IV/RA ({circ_shap/total_shap:.0%})"),
        Patch(color="#2CA02C", label=f"Morning ratio ({eng_shap/total_shap:.0%})"),
        Patch(color="#E05A5A", label=f"Missing flag ({flag_shap/total_shap:.0%})"),
    ], loc="lower right")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent.parent / "outputs" / "track_b" / "shap_v6.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  SHAP plot saved -> {out_path}")

    # Final comparison
    V1_AUC = 0.5583
    V2_AUC = 0.5593
    V3_AUC = 0.5642
    print("\n" + "=" * 62)
    print("AUC Comparison")
    print("-" * 62)
    print(f"  v1 (global median impute)              : {V1_AUC:.4f}")
    print(f"  v2 (person median + missing flags)     : {V2_AUC:.4f}  ({V2_AUC - V1_AUC:+.4f})")
    print(f"  v3 (person median + RA + morning ratio): {V3_AUC:.4f}  ({V3_AUC - V1_AUC:+.4f})")
    print(f"  v6 (RandomForest + RA + morning ratio) : {mean_auc:.4f}  ({mean_auc - V1_AUC:+.4f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
