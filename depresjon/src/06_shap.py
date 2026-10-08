import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # depresjon/

OUTPUT_DIR = os.environ.get("DEPRESJON_OUT", os.path.join(ROOT, "outputs"))

# ── 데이터 & 모델 로드 ─────────────────────────────
df = pd.read_csv(os.path.join(OUTPUT_DIR, "features_36.csv"))

with open(os.path.join(OUTPUT_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

X = df[feature_cols].values
y = df["label"].values

model = xgb.Booster()
model.load_model(os.path.join(OUTPUT_DIR, "xgb_model.json"))

# ── SHAP 계산 ──────────────────────────────────────
dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(dmatrix)

print(f"SHAP values shape: {shap_values.shape}")

# ── 피처 중요도 (평균 |SHAP|) ──────────────────────
mean_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "mean_shap":  mean_shap
}).sort_values("mean_shap", ascending=False).reset_index(drop=True)

print("\n=== Top 15 중요 피처 ===")
print(importance_df.head(15).to_string())

# ── 시각화 1: Bar plot (Top 20) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top20 = importance_df.head(20)
colors = ["#e07b54" if "circ" in f else "#5b8db8" for f in top20["feature"]]
axes[0].barh(top20["feature"][::-1], top20["mean_shap"][::-1], color=colors[::-1])
axes[0].set_title("Top 20 feature importance (mean |SHAP|)")
axes[0].set_xlabel("Mean |SHAP value|")

# 범례
from matplotlib.patches import Patch
legend = [Patch(color="#e07b54", label="Circadian"),
          Patch(color="#5b8db8", label="Sleep")]
axes[0].legend(handles=legend, loc="lower right")

# ── 시각화 2: Beeswarm plot ────────────────────────
shap.summary_plot(shap_values, X,
                  feature_names=feature_cols,
                  max_display=20,
                  show=False,
                  plot_type="dot")
plt.sca(axes[1])
axes[1].set_title("SHAP beeswarm plot")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "shap_plots.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장됨: {out_path}")

# ── Circadian vs Sleep 피처 중요도 비교 ───────────
circ_imp  = importance_df[importance_df["feature"].str.contains("circ")]["mean_shap"].sum()
sleep_imp = importance_df[~importance_df["feature"].str.contains("circ")]["mean_shap"].sum()
total     = circ_imp + sleep_imp

print(f"\n=== 피처 그룹별 중요도 ===")
print(f"Circadian 피처: {circ_imp:.4f} ({circ_imp/total*100:.1f}%)")
print(f"Sleep 피처:     {sleep_imp:.4f} ({sleep_imp/total*100:.1f}%)")