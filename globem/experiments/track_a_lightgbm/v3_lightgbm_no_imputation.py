"""
globem_pipeline_v3.py
=====================
Depression prediction pipeline using the GLOBEM dataset.

Key changes from v2
-------------------
① XGBoost  →  LightGBM (handles NaN natively, no imputation needed)
② Imputation removed entirely — NaN passed directly to the model
③ Missing-indicator flags removed — LightGBM learns from NaN itself
④ Feature count reduced: 224 (v2) → 127 (original features only)

Results
-------
v1 AUC : 0.5583  (XGBoost + global median imputation)
v2 AUC : 0.5574  (XGBoost + person-level imputation + missing flags)
v3 AUC : TBD     (LightGBM + no imputation)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def compute_IS_IV(row):
    seg_cols = [
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:morning",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening",
        "f_steps:fitbit_steps_intraday_rapids_sumsteps:night",
    ]
    segs = np.array([row.get(c, np.nan) for c in seg_cols], dtype=float)
    if np.isnan(segs).all():
        return np.nan, np.nan
    segs = np.nan_to_num(segs, nan=0.0)
    mean_val = np.mean(segs)
    if mean_val == 0:
        return 0.0, 0.0
    IS = 1.0 / (1.0 + np.std(segs) / (mean_val + 1e-8))
    diffs = np.diff(segs)
    IV = np.mean(diffs ** 2) / (np.var(segs) + 1e-8)
    return IS, IV


def build_feature_cols():
    sleep_cols = [f"f_slp:{base}:{seg}"   for base in SLEEP_BASE for seg in SEGMENTS]
    steps_cols = [f"f_steps:{base}:{seg}" for base in STEPS_BASE for seg in SEGMENTS]
    return sleep_cols, steps_cols


def load_dataset(name):
    feat_dir   = BASE / name / "FeatureData"
    label_path = BASE / name / "SurveyData" / "dep_weekly.csv"

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
        print(f"  Circadian IS/IV: computed")
    else:
        circ_computed = []

    avail_sleep = [c for c in sleep_cols if c in feats.columns]
    avail_steps = [c for c in steps_cols if c in feats.columns]
    avail_feat  = avail_sleep + avail_steps + circ_computed

    print(f"  Sleep: {len(avail_sleep)}/{len(sleep_cols)}  Steps: {len(avail_steps)}/{len(steps_cols)}  Circ: {len(circ_computed)}")

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
    print(f"  Remaining NaNs (passed to LightGBM): {int(df[avail_feat].isna().sum().sum()):,}")

    return df, avail_feat


def main():
    print("=" * 62)
    print("GLOBEM Depression Prediction — Pipeline v3")
    print("LightGBM  +  No Imputation  +  NaN as Signal")
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
        print("No data loaded.")
        return

    df        = pd.concat(all_dfs, ignore_index=True)
    feat_cols = all_feat_cols
    print(f"\nAll cohorts merged: {len(df):,} rows  dep=1: {int(df['label'].sum())}")
    print(f"Total NaNs passed to model: {int(df[feat_cols].isna().sum().sum()):,}")
    print(f"Feature count: {len(feat_cols)}")

    X      = df[feat_cols].values.astype(np.float32)
    y      = df["label"].values
    groups = df["subject_id"].values

    print("\n[Step 1] Bayesian Optimisation (10 init + 20 iter) ...")

    def lgbm_cv(num_leaves, learning_rate, n_estimators,
                subsample, colsample_bytree, min_child_samples, reg_alpha, reg_lambda):
        params = dict(
            objective="binary", metric="auc",
            num_leaves=int(num_leaves),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=int(min_child_samples),
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=(y==0).sum() / max((y==1).sum(), 1),
            random_state=42,
            verbosity=-1,
        )
        cv     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y, groups):
            m = LGBMClassifier(**params)
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[val])[:, 1]
            if len(np.unique(y[val])) == 2:
                scores.append(roc_auc_score(y[val], prob))
        return np.mean(scores) if scores else 0.5

    optimizer = BayesianOptimization(
        f=lgbm_cv,
        pbounds=dict(
            num_leaves=(20, 150),
            learning_rate=(0.01, 0.3),
            n_estimators=(50, 500),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            min_child_samples=(5, 50),
            reg_alpha=(0, 2),
            reg_lambda=(0, 2),
        ),
        random_state=42, verbose=0,
    )
    optimizer.maximize(init_points=10, n_iter=20)
    best = optimizer.max["params"]
    print(f"  Best CV AUC: {optimizer.max['target']:.4f}")

    print("\n[Step 2] Final 5-fold CV ...")
    final_params = dict(
        objective="binary", metric="auc",
        num_leaves=int(best["num_leaves"]),
        learning_rate=best["learning_rate"],
        n_estimators=int(best["n_estimators"]),
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_samples=int(best["min_child_samples"]),
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
        scale_pos_weight=(y==0).sum() / max((y==1).sum(), 1),
        random_state=42, verbosity=-1,
    )

    cv        = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for fold, (tr, val) in enumerate(cv.split(X, y, groups)):
        m = LGBMClassifier(**final_params)
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[val])[:, 1]
        if len(np.unique(y[val])) == 2:
            auc = roc_auc_score(y[val], prob)
            fold_aucs.append(auc)
            print(f"  Fold {fold+1} AUC: {auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    print(f"\n  Mean AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  vs v1    : {mean_auc - 0.5583:+.4f}")
    print(f"  vs v2    : {mean_auc - 0.5574:+.4f}")

    print("\n[Step 3] SHAP analysis ...")
    model_full = LGBMClassifier(**final_params)
    model_full.fit(X, y)
    explainer     = shap.TreeExplainer(model_full)
    shap_vals     = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    circ_idx  = [i for i, c in enumerate(feat_cols)
                 if c in ("circ_IS", "circ_IV") or "bedtime" in c or "waketime" in c]
    feat_idx  = [i for i, c in enumerate(feat_cols) if i not in circ_idx]
    circ_shap  = mean_abs_shap[circ_idx].sum()
    feat_shap  = mean_abs_shap[feat_idx].sum()
    total_shap = circ_shap + feat_shap
    print(f"\n  Sleep/Activity SHAP   : {feat_shap/total_shap:.1%}")
    print(f"  Circadian (IS/IV) SHAP: {circ_shap/total_shap:.1%}")

    top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
    top20_names = [feat_cols[i] for i in top20_idx]
    top20_vals  = mean_abs_shap[top20_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    shap.summary_plot(shap_vals, X, feature_names=feat_cols, show=False, max_display=20)
    plt.sca(axes[0])
    plt.title(f"SHAP Beeswarm  (AUC={mean_auc:.4f})", fontsize=13)

    bar_colors = ["#F0A500" if n in ("circ_IS","circ_IV") else "#4C72B0" for n in top20_names]
    axes[1].barh(range(20), top20_vals[::-1], color=bar_colors[::-1])
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([n.split(":")[-1][:40] for n in top20_names[::-1]], fontsize=9)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top 20 Feature Importance", fontsize=13)
    axes[1].legend(handles=[
        Patch(color="#4C72B0", label=f"Sleep/Activity ({feat_shap/total_shap:.0%})"),
        Patch(color="#F0A500", label=f"Circadian IS/IV ({circ_shap/total_shap:.0%})"),
    ], loc="lower right")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent.parent / "outputs" / "track_a" / "shap_v3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  SHAP plot saved -> {out_path}")

    print("\n" + "=" * 62)
    print(f"v1 (XGBoost + global median)     : 0.5583")
    print(f"v2 (XGBoost + person-level imp.) : 0.5574")
    print(f"v3 (LightGBM + no imputation)    : {mean_auc:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
