import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

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

SLEEP_COLS = [f"f_slp:{b}:{s}" for b in SLEEP_BASE for s in SEGMENTS]
STEPS_COLS = [f"f_steps:{b}:{s}" for b in STEPS_BASE for s in SEGMENTS]

def compute_IS_IV(row):
    segs = []
    for seg in ["morning", "afternoon", "evening", "night"]:
        col = f"f_steps:fitbit_steps_intraday_rapids_sumsteps:{seg}"
        val = row.get(col, np.nan)
        segs.append(val)
    segs = np.array(segs, dtype=float)
    if np.isnan(segs).all():
        return np.nan, np.nan
    segs = np.nan_to_num(segs, nan=0.0)
    mean_val = np.mean(segs)
    if mean_val == 0:
        return 0.0, 0.0
    IS = 1.0 / (1.0 + np.std(segs) / (mean_val + 1e-8))
    diffs = np.diff(segs)
    IV = np.mean(diffs**2) / (np.var(segs) + 1e-8)
    return IS, IV

def load_dataset(ds_name):
    ds_path = BASE / ds_name
    print(f"\n{'='*50}\nLoading {ds_name}...")
    label_df = pd.read_csv(ds_path / "SurveyData" / "dep_weekly.csv")
    label_df = label_df[["pid", "date", "dep"]].copy()
    label_df["date"] = pd.to_datetime(label_df["date"])
    label_df["dep"] = label_df["dep"].map({True: 1, False: 0, "True": 1, "False": 0})
    label_df = label_df.dropna(subset=["dep"])
    print(f"  Labels: {len(label_df)} rows, dep=1: {label_df['dep'].sum()}")

    sleep_df = pd.read_csv(ds_path / "FeatureData" / "sleep.csv")
    sleep_df["date"] = pd.to_datetime(sleep_df["date"])
    avail_sleep = [c for c in SLEEP_COLS if c in sleep_df.columns]
    sleep_df = sleep_df[["pid", "date"] + avail_sleep]
    print(f"  Sleep features available: {len(avail_sleep)}/{len(SLEEP_COLS)}")

    steps_df = pd.read_csv(ds_path / "FeatureData" / "steps.csv")
    steps_df["date"] = pd.to_datetime(steps_df["date"])
    avail_steps = [c for c in STEPS_COLS if c in steps_df.columns]
    seg_cols = [c for c in ["f_steps:fitbit_steps_intraday_rapids_sumsteps:morning",
                             "f_steps:fitbit_steps_intraday_rapids_sumsteps:afternoon",
                             "f_steps:fitbit_steps_intraday_rapids_sumsteps:evening",
                             "f_steps:fitbit_steps_intraday_rapids_sumsteps:night"]
                if c in steps_df.columns]
    all_step_cols = list(set(avail_steps + seg_cols))
    steps_df = steps_df[["pid", "date"] + all_step_cols]
    print(f"  Steps features available: {len(avail_steps)}/{len(STEPS_COLS)}")

    steps_df["circ_IS"] = steps_df.apply(lambda r: compute_IS_IV(r)[0], axis=1)
    steps_df["circ_IV"] = steps_df.apply(lambda r: compute_IS_IV(r)[1], axis=1)

    merged = label_df.copy()
    merged = merged.merge(sleep_df, on=["pid", "date"], how="left")
    merged = merged.merge(steps_df[["pid", "date"] + avail_steps + ["circ_IS", "circ_IV"]],
                          on=["pid", "date"], how="left")

    feat_cols = avail_sleep + avail_steps + ["circ_IS", "circ_IV"]
    miss_rate = merged[feat_cols].isna().mean().mean()
    print(f"  Overall missing rate: {miss_rate:.1%}")

    row_miss = merged[feat_cols].isna().mean(axis=1)
    merged = merged[row_miss <= 0.7].copy()
    print(f"  Rows after missing filter: {len(merged)} (dep=1: {merged['dep'].sum()})")

    for c in feat_cols:
        merged[c] = merged[c].fillna(merged[c].median())

    merged["dataset"] = ds_name
    return merged, feat_cols

all_dfs = []
feat_cols = None

for ds in DATASETS:
    try:
        df, fc = load_dataset(ds)
        all_dfs.append(df)
        feat_cols = fc
    except Exception as e:
        print(f"  ERROR loading {ds}: {e}")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n{'='*50}")
print(f"Total rows: {len(full_df)}")
print(f"Total dep=1: {full_df['dep'].sum()} ({full_df['dep'].mean():.1%})")
print(f"Total features: {len(feat_cols)}")
print(f"Unique participants: {full_df['pid'].nunique()}")

X = full_df[feat_cols].values
y = full_df["dep"].values
groups = full_df["pid"].values

scale_pos = (y == 0).sum() / (y == 1).sum()
print(f"\nscale_pos_weight: {scale_pos:.2f}")

cv = StratifiedGroupKFold(n_splits=5)

def xgb_cv(max_depth, learning_rate, n_estimators, subsample,
           colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda):
    params = dict(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=int(min_child_weight),
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    aucs = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        clf = XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = clf.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            aucs.append(roc_auc_score(y_val, pred))
    return np.mean(aucs) if aucs else 0.5

pbounds = {
    "max_depth":        (3, 8),
    "learning_rate":    (0.01, 0.3),
    "n_estimators":     (50, 300),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma":            (0, 5),
    "reg_alpha":        (0, 2),
    "reg_lambda":       (0, 2),
}

print("\nRunning Bayesian Optimization (30 iterations)...")
optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=42, verbose=2)
optimizer.maximize(init_points=5, n_iter=25)

best_params = optimizer.max["params"]
best_params["max_depth"] = int(best_params["max_depth"])
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["min_child_weight"] = int(best_params["min_child_weight"])
best_params["scale_pos_weight"] = scale_pos
best_params["eval_metric"] = "auc"
best_params["use_label_encoder"] = False
best_params["random_state"] = 42
best_params["n_jobs"] = -1

print(f"\nBest AUC (CV): {optimizer.max['target']:.4f}")
print(f"Best params: {best_params}")

print("\nTraining final model for SHAP...")
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)
final_model = XGBClassifier(**best_params)
final_model.fit(X_scaled, y)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_scaled)

feat_names_short = []
for c in feat_cols:
    name = c.replace("f_slp:fitbit_sleep_summary_rapids_", "slp_")
    name = name.replace("f_slp:fitbit_sleep_intraday_rapids_", "slp_")
    name = name.replace("f_steps:fitbit_steps_intraday_rapids_", "stp_")
    name = name.replace("f_steps:fitbit_steps_summary_rapids_", "stp_")
    name = name.replace("main", "").replace("unified", "")
    feat_names_short.append(name)

mean_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": feat_names_short,
    "feature_full": feat_cols,
    "mean_shap": mean_shap
}).sort_values("mean_shap", ascending=False)

total_shap = shap_df["mean_shap"].sum()

def is_circadian(col):
    return col in ["circ_IS", "circ_IV"] or "bedtime" in col or "waketime" in col

shap_df["category"] = shap_df["feature_full"].apply(
    lambda c: "Circadian" if is_circadian(c) else "Sleep/Activity"
)

sleep_shap = shap_df[shap_df["category"] == "Sleep/Activity"]["mean_shap"].sum()
circ_shap  = shap_df[shap_df["category"] == "Circadian"]["mean_shap"].sum()

print(f"\n{'='*50}")
print("SHAP Feature Importance (Top 15)")
print(f"{'='*50}")
print(shap_df[["feature", "category", "mean_shap"]].head(15).to_string(index=False))
print(f"\nSleep/Activity SHAP: {sleep_shap/total_shap:.1%}")
print(f"Circadian SHAP:      {circ_shap/total_shap:.1%}")

print("\nFinal 5-fold CV AUC with best params:")
aucs_final = []
for train_idx, val_idx in cv.split(X, y, groups):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    clf = XGBClassifier(**best_params)
    clf.fit(X_tr, y_tr)
    pred = clf.predict_proba(X_val)[:, 1]
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, pred)
        aucs_final.append(auc)
        print(f"  Fold AUC: {auc:.4f}")

print(f"\nMean AUC: {np.mean(aucs_final):.4f} ± {np.std(aucs_final):.4f}")

shap_df.to_csv("globem_shap_results.csv", index=False)
print("\nDone! Results saved to globem_shap_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("GLOBEM Depression Prediction — SHAP Feature Importance", fontsize=14, fontweight='bold')

top20 = shap_df.head(20)
colors = ["#e07b54" if c == "Circadian" else "#5b8db8" for c in top20["category"]]
axes[0].barh(top20["feature"][::-1], top20["mean_shap"][::-1], color=colors[::-1])
axes[0].set_title("Top 20 Feature Importance (Mean |SHAP|)")
axes[0].set_xlabel("Mean |SHAP value|")
sleep_patch = mpatches.Patch(color="#5b8db8", label=f"Sleep/Activity ({sleep_shap/total_shap:.1%})")
circ_patch  = mpatches.Patch(color="#e07b54", label=f"Circadian ({circ_shap/total_shap:.1%})")
axes[0].legend(handles=[sleep_patch, circ_patch], loc="lower right")

plt.sca(axes[1])
shap.summary_plot(
    shap_values, X_scaled,
    feature_names=feat_names_short,
    max_display=20,
    show=False,
    plot_type="dot"
)
axes[1].set_title("SHAP Beeswarm Plot")

plt.tight_layout()
out_path = Path(__file__).parent.parent.parent / "outputs" / "track_b" / "shap_v2.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"그래프 저장됨: {out_path}")
plt.show()
