#실제로 예측 AI모델을 만드는중
import pandas as pd
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix,
                             matthews_corrcoef)
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # depresjon/

OUTPUT_DIR = os.environ.get("DEPRESJON_OUT", os.path.join(ROOT, "outputs"))

# ── 데이터 로드 ────────────────────────────────────
df = pd.read_csv(os.path.join(OUTPUT_DIR, "features_36.csv"))
feature_cols = [c for c in df.columns if c not in ["subject_id", "label"]]

X = df[feature_cols].values
y = df["label"].values

print(f"피처 수: {len(feature_cols)}")
print(f"샘플 수: {len(y)}, condition: {y.sum()}, control: {(y==0).sum()}")

# ── Bayesian Optimization ──────────────────────────
def xgb_cv_auc(max_depth, learning_rate, subsample,
               colsample_bytree, min_child_weight, n_estimators):

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "max_depth":        int(max_depth),
        "learning_rate":    learning_rate,
        "subsample":        subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": int(min_child_weight),
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
        "seed":             42,
        "verbosity":        0,
    }
    n_est = int(n_estimators)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        d_tr  = xgb.DMatrix(X_tr, label=y_tr)
        d_val = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params, d_tr,
            num_boost_round=n_est,
            evals=[(d_val, "val")],
            verbose_eval=False,
            early_stopping_rounds=20
        )
        preds = model.predict(d_val)
        auc_scores.append(roc_auc_score(y_val, preds))

    return np.mean(auc_scores)

# 탐색 범위
pbounds = {
    "max_depth":        (3, 8),
    "learning_rate":    (0.01, 0.3),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "n_estimators":     (50, 300),
}

print("\nBayesian Optimization 시작 (30회)...")
optimizer = BayesianOptimization(
    f=xgb_cv_auc,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=5, n_iter=25)

best_params = optimizer.max["params"]
print(f"\n최적 파라미터: {best_params}")
print(f"최적 AUC: {optimizer.max['target']:.4f}")

# ── 최적 파라미터로 최종 모델 훈련 ────────────────
final_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        int(best_params["max_depth"]),
    "learning_rate":    best_params["learning_rate"],
    "subsample":        best_params["subsample"],
    "colsample_bytree": best_params["colsample_bytree"],
    "min_child_weight": int(best_params["min_child_weight"]),
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "seed":             42,
    "verbosity":        0,
}

# ── 5-fold CV 최종 평가 ────────────────────────────
print("\n=== 최종 5-fold CV 평가 ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds, all_true = [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    d_tr  = xgb.DMatrix(X_tr, label=y_tr)
    d_val = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        final_params, d_tr,
        num_boost_round=int(best_params["n_estimators"]),
        evals=[(d_val, "val")],
        verbose_eval=False,
        early_stopping_rounds=20
    )
    preds = model.predict(d_val)
    all_preds.extend(preds)
    all_true.extend(y_val)

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
binary_preds = (all_preds >= 0.5).astype(int)

print(f"AUC:      {roc_auc_score(all_true, all_preds):.4f}")
print(f"Accuracy: {accuracy_score(all_true, binary_preds):.4f}")
print(f"F1:       {f1_score(all_true, binary_preds):.4f}")
print(f"MCC:      {matthews_corrcoef(all_true, binary_preds):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(all_true, binary_preds)}")

# 모델 저장 (전체 데이터로 재훈련)
d_full = xgb.DMatrix(X, label=y)
final_model = xgb.train(
    final_params, d_full,
    num_boost_round=int(best_params["n_estimators"])
)
model_path = os.path.join(OUTPUT_DIR, "xgb_model.json")
final_model.save_model(model_path)
print(f"\n모델 저장됨: {model_path}")

# 피처 이름 저장 (SHAP용)
import json
with open(os.path.join(OUTPUT_DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)