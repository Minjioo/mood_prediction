"""
globem_pipeline_v2.py
─────────────────────
기존 대비 변경사항:
  ① 전체 중앙값 대체 → person-level imputation (개인별 과거 이력 기반)
  ② 결측 여부 이진 플래그 피처 추가 (착용 안 한 것도 신호로 활용)
"""

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
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────────
BASE = Path("/Users/hongseungmin/Desktop/Lab/Circadian/GlobeM구현본/globem-dataset-multi-year-datasets-for-longitudinal-human-behavior-modeling-generalization-1.1")
DATASETS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]

SEGMENTS = ["allday", "morning", "afternoon", "evening", "night"]

SLEEP_BASE = [
    "fitbit_sleep_summary_rapids_avgdurationasleepmain",
    "fitbit_sleep_summary_rapids_sumdurationasleepmain",
    "fitbit_sleep_summary_rapids_avgdurationawakemain",
    "fitbit_sleep_summary_rapids_sumdurationawakemain",
    "fitbit_sleep_summary_rapids_countepisodemain",
    "fitbit_sleep_summary_rapids_avgefficiencymain",
    "fitbit_sleep_summary_rapids_avgdurationinbedmain",
    "fitbit_sleep_summary_rapids_sumdurationinbedmain",
    "fitbit_sleep_summary_rapids_avgdurationtofallasleepmain",
    "fitbit_sleep_summary_rapids_sumdurationtofallasleepmain",
    "fitbit_sleep_summary_rapids_avgdurationafterwakeupmain",
    "fitbit_sleep_summary_rapids_sumdurationafterwakeupmain",
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
    "fitbit_steps_intraday_rapids_maxsteps",
    "fitbit_steps_intraday_rapids_minsteps",
    "fitbit_steps_summary_rapids_maxsumsteps",
    "fitbit_steps_summary_rapids_minsumsteps",
    "fitbit_steps_summary_rapids_avgsumsteps",
    "fitbit_steps_summary_rapids_stdsumsteps",
]

CIRCADIAN_COLS = [
    "f_cr:fitbit_steps_intraday_rapids_cosinormesor:allday",
    "f_cr:fitbit_steps_intraday_rapids_cosinoramplitude:allday",
    "f_cr:fitbit_steps_intraday_rapids_cosinorphaseshift:allday",
    "f_cr:fitbit_steps_intraday_rapids_IS:allday",
    "f_cr:fitbit_steps_intraday_rapids_IV:allday",
]

ROW_MISSING_THRESHOLD = 0.70   # 행 결측률 70% 초과 시 제거
ADD_MISSING_FLAGS = True        # ① 결측 플래그 피처 추가 여부


# ══════════════════════════════════════════════════════════════
# 핵심 변경 ① : person-level imputation
# ══════════════════════════════════════════════════════════════
def person_level_impute(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    각 person(subject_id)별로 자신의 이력 중앙값으로 결측 대체.
    이력이 없어 여전히 NaN이면 전체 중앙값으로 fallback.

    기존 방식: df[feature_cols].fillna(df[feature_cols].median())
    개선 방식: 아래 2단계
    """
    df = df.copy()

    # 단계 1: 개인별 중앙값으로 대체
    person_medians = (
        df.groupby("subject_id")[feature_cols]
        .transform("median")
    )
    df[feature_cols] = df[feature_cols].fillna(person_medians)

    # 단계 2: 여전히 NaN(개인 이력 전부 결측)이면 전체 중앙값으로 fallback
    global_medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(global_medians)

    return df


# ══════════════════════════════════════════════════════════════
# 핵심 변경 ② : 결측 여부 이진 플래그 피처 추가
# ══════════════════════════════════════════════════════════════
def add_missing_flags(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    imputation 전에 결측 여부를 이진 플래그(0/1)로 저장.
    착용하지 않은 것 자체가 우울 신호일 수 있음.

    반환: (flag가 추가된 df, 추가된 flag 컬럼 이름 리스트)
    """
    flag_cols = []
    for col in feature_cols:
        flag_name = col + "__missing"
        df[flag_name] = df[col].isna().astype(np.int8)
        flag_cols.append(flag_name)

    # 완전히 0인 플래그 컬럼(결측이 한 번도 없는 피처)은 제거 — 노이즈 방지
    useful_flags = [c for c in flag_cols if df[c].sum() > 0]
    useless_flags = [c for c in flag_cols if c not in useful_flags]
    df.drop(columns=useless_flags, inplace=True)

    print(f"  Missing flags 추가: {len(useful_flags)}개 "
          f"(전체 {len(feature_cols)}개 피처 중 결측 발생한 것만 유지)")
    return df, useful_flags


# ══════════════════════════════════════════════════════════════
# 데이터 로딩 (기존과 동일)
# ══════════════════════════════════════════════════════════════
def build_feature_cols():
    sleep_cols = [
        f"f_slp:{base}:{seg}"
        for base in SLEEP_BASE
        for seg in SEGMENTS
    ]
    steps_cols = [
        f"f_steps:{base}:{seg}"
        for base in STEPS_BASE
        for seg in SEGMENTS
    ]
    return sleep_cols, steps_cols


def load_dataset(name: str):
    feat_dir = BASE / name / "FeatureData"
    label_path = BASE / name / "SurveyData" / "dep_weekly.csv"

    labels = pd.read_csv(label_path)
    labels.columns = labels.columns.str.strip()
    # dep_weekly 또는 dep 컬럼명 모두 대응
    if "dep_weekly" in labels.columns:
        labels = labels.rename(columns={"pid": "subject_id", "dep_weekly": "label"})
    elif "dep" in labels.columns:
        labels = labels.rename(columns={"pid": "subject_id", "dep": "label"})
        labels["label"] = labels["label"].map({True: 1, False: 0, "True": 1, "False": 0})
    else:
        print(f"  {name}: label 컬럼 없음 — 컬럼 목록: {labels.columns.tolist()}")
        return None
    labels = labels[["subject_id", "label", "date"]].dropna(subset=["label"])
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["label"])
    labels["label"] = (labels["label"] >= 1).astype(int)

    print(f"\nLoading {name}... Labels: {len(labels)} rows, "
          f"dep=1: {labels['label'].sum()}")

    sleep_cols, steps_cols = build_feature_cols()
    all_feat_cols = sleep_cols + steps_cols + CIRCADIAN_COLS

    # 피처 파일 수집 — sleep.csv/steps.csv 우선, 없으면 폴더 전체 스캔
    sleep_path = feat_dir / "sleep.csv"
    steps_path = feat_dir / "steps.csv"

    feat_dfs = []
    if sleep_path.exists():
        tmp = pd.read_csv(sleep_path, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        feat_dfs.append(tmp)
    if steps_path.exists():
        tmp = pd.read_csv(steps_path, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        feat_dfs.append(tmp)

    if not feat_dfs:
        for csv_path in sorted(feat_dir.glob("*.csv")):
            try:
                tmp = pd.read_csv(csv_path, low_memory=False)
                tmp.columns = tmp.columns.str.strip()
                feat_dfs.append(tmp)
            except Exception:
                pass

    if not feat_dfs:
        print(f"  {name}: 피처 파일 없음, 스킵")
        return None

    # pid/date 기준으로 sleep + steps 병합
    feats = feat_dfs[0]
    for extra in feat_dfs[1:]:
        merge_on = [c for c in ["pid", "date"] if c in feats.columns and c in extra.columns]
        if merge_on:
            feats = feats.merge(extra, on=merge_on, how="outer")
    feats = feats.rename(columns={"pid": "subject_id"})

    # 존재하는 피처만 사용
    avail_sleep = [c for c in sleep_cols if c in feats.columns]
    avail_steps = [c for c in steps_cols if c in feats.columns]
    avail_cr    = [c for c in CIRCADIAN_COLS if c in feats.columns]
    avail_feat  = avail_sleep + avail_steps + avail_cr

    print(f"  Sleep features available: {len(avail_sleep)}/{len(sleep_cols)}")
    print(f"  Steps features available: {len(avail_steps)}/{len(steps_cols)}")
    print(f"  Circadian features available: {len(avail_cr)}/{len(CIRCADIAN_COLS)}")

    # label과 merge
    merge_keys = ["subject_id"] + (["date"] if "date" in feats.columns else [])
    df = labels.merge(feats[merge_keys + avail_feat].drop_duplicates(subset=merge_keys),
                      on=merge_keys, how="left")
    df["dataset"] = name

    # 행 결측률 필터
    row_miss = df[avail_feat].isna().mean(axis=1)
    overall_miss = df[avail_feat].isna().mean().mean()
    print(f"  Overall missing rate: {overall_miss:.1%}")

    df = df[row_miss <= ROW_MISSING_THRESHOLD].copy()
    print(f"  Rows after missing filter: {len(df)} (dep=1: {df['label'].sum()})")

    return df, avail_feat


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GLOBEM Pipeline v2 — Person-level Imputation + Missing Flags")
    print("=" * 60)

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
        print("데이터 없음 — BASE 경로 확인 필요")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    feat_cols = all_feat_cols  # 공통 피처 컬럼
    print(f"\n전체 병합: {len(df)} rows, dep=1: {df['label'].sum()}")

    # ── ② 결측 플래그 추가 (imputation 전에 반드시 실행) ──────
    if ADD_MISSING_FLAGS:
        print("\n[Step 1] Missing flag 피처 추가 중...")
        df, flag_cols = add_missing_flags(df, feat_cols)
    else:
        flag_cols = []

    # ── ① Person-level imputation ─────────────────────────────
    print("[Step 2] Person-level imputation 적용 중...")
    before_nan = df[feat_cols].isna().sum().sum()
    df = person_level_impute(df, feat_cols)
    after_nan  = df[feat_cols].isna().sum().sum()
    print(f"  결측 제거: {before_nan:,} → {after_nan:,} (잔여 결측: {after_nan}개)")

    # ── 피처 행렬 구성 ─────────────────────────────────────────
    final_feat_cols = feat_cols + flag_cols
    print(f"\n최종 피처 수: {len(final_feat_cols)} "
          f"(원본 {len(feat_cols)} + 플래그 {len(flag_cols)})")

    X = df[final_feat_cols].values.astype(np.float32)
    y = df["label"].values
    groups = df["subject_id"].values

    # StandardScaler (플래그 컬럼은 0/1이지만 스케일링 포함해도 무방)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Bayesian Optimization ─────────────────────────────────
    print("\n[Step 3] Bayesian Optimization (30 iter)...")

    def xgb_cv(max_depth, learning_rate, subsample,
                colsample_bytree, min_child_weight, n_estimators):
        params = dict(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=int(min_child_weight),
            n_estimators=int(n_estimators),
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
        )
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X, y, groups):
            m = XGBClassifier(**params)
            m.fit(X[tr], y[tr], verbose=False)
            prob = m.predict_proba(X[val])[:, 1]
            if len(np.unique(y[val])) == 2:
                scores.append(roc_auc_score(y[val], prob))
        return np.mean(scores) if scores else 0.5

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=dict(
            max_depth=(3, 8),
            learning_rate=(0.01, 0.3),
            subsample=(0.6, 1.0),
            colsample_bytree=(0.6, 1.0),
            min_child_weight=(1, 10),
            n_estimators=(50, 300),
        ),
        random_state=42,
        verbose=0,
    )
    optimizer.maximize(init_points=10, n_iter=20)

    best = optimizer.max["params"]
    print(f"  Best params: {best}")
    print(f"  Best CV AUC (Bayesian): {optimizer.max['target']:.4f}")

    # ── 최종 5-fold CV 평가 ───────────────────────────────────
    print("\n[Step 4] Final 5-fold CV AUC with best params:")
    final_params = dict(
        objective="binary:logistic",
        max_depth=int(best["max_depth"]),
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_weight=int(best["min_child_weight"]),
        n_estimators=int(best["n_estimators"]),
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    all_preds, all_true = [], []

    for fold, (tr, val) in enumerate(cv.split(X, y, groups)):
        m = XGBClassifier(**final_params)
        m.fit(X[tr], y[tr], verbose=False)
        prob = m.predict_proba(X[val])[:, 1]
        if len(np.unique(y[val])) == 2:
            auc = roc_auc_score(y[val], prob)
            fold_aucs.append(auc)
            print(f"  Fold AUC: {auc:.4f}")
        all_preds.extend(prob)
        all_true.extend(y[val])

    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    print(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"\n[v1 기준 AUC: 0.5583]  →  개선폭: {mean_auc - 0.5583:+.4f}")

    # ── SHAP 분석 ─────────────────────────────────────────────
    print("\n[Step 5] SHAP 분석 중...")
    m_full = XGBClassifier(**final_params)
    m_full.fit(X, y, verbose=False)

    explainer = shap.TreeExplainer(m_full)
    shap_vals  = explainer.shap_values(X)

    # 피처 중요도 상위 20개
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top20_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top20_names = [final_feat_cols[i] for i in top20_idx]
    top20_vals  = mean_abs_shap[top20_idx]

    # Flag vs. non-flag 기여도 분리
    flag_shap  = mean_abs_shap[[i for i, c in enumerate(final_feat_cols) if "__missing" in c]].sum()
    feat_shap  = mean_abs_shap[[i for i, c in enumerate(final_feat_cols) if "__missing" not in c]].sum()
    total_shap = flag_shap + feat_shap
    print(f"\n  원본 피처 SHAP 기여: {feat_shap/total_shap:.1%}")
    print(f"  Missing flag SHAP 기여: {flag_shap/total_shap:.1%}")

    # 그래프 저장
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 좌: beeswarm
    shap.summary_plot(shap_vals, X, feature_names=final_feat_cols,
                      show=False, max_display=20)
    plt.sca(axes[0])
    plt.title(f"SHAP Beeswarm (Mean AUC={mean_auc:.4f})", fontsize=13)

    # 우: bar (상위 20개, flag는 빨간색으로 구분)
    colors = ["#E05A5A" if "__missing" in n else "#4C72B0" for n in top20_names]
    axes[1].barh(range(20), top20_vals[::-1], color=colors[::-1])
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([n.split(":")[-1][:40] for n in top20_names[::-1]], fontsize=9)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top 20 Feature Importance\n(빨간색 = missing flag 피처)", fontsize=13)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color="#4C72B0", label=f"원본 피처 ({feat_shap/total_shap:.0%})"),
        Patch(color="#E05A5A", label=f"Missing flag ({flag_shap/total_shap:.0%})"),
    ], loc="lower right")

    plt.tight_layout()
    out_path = Path.home() / "Desktop/Lab/Circadian/GlobeM구현본/shap_v2.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSHAP 그래프 저장: {out_path}")

    print("\n" + "=" * 60)
    print(f"완료  |  v1 AUC 0.5583  →  v2 AUC {mean_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
