#Circadian feature 6개 추가 (총 36개가 됨)(+ subject id, label)
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
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

# ── Forger 1999 ODE 모델 ───────────────────────────
# x: pacemaker 상태변수 (CBT 일주기 리듬)
# xc: coupled oscillator
# light: 빛 입력 (lux)

def forger_ode(t, y, light_func):
    x, xc = y
    L = light_func(t)

    # Forger 1999 파라미터
    mu    = 0.13
    q     = 1/3
    kappa = 0.55
    tau   = 24.2         # 내인성 주기 (시간)
    omega = 2 * np.pi / tau

    # 빛 처리 함수
    B = 0.4 * (1 - 0.4 * x) * (1 - 0.4 * xc) * L / (L + 100)

    dx  = (np.pi / 12) * (xc + mu * (x / 3 + (4/3) * x**3 - (256/105) * x**7) + B)
    dxc = (np.pi / 12) * (q * B * xc - x * ((omega**2) / (np.pi / 12) + kappa * B))

    return [dx, dxc]

def compute_circadian_features(subject_df):
    sdf = subject_df.copy().reset_index(drop=True)

    # light profile: activity > 0 → 250 lux, 0 → 0 lux
    sdf["lux"] = np.where(sdf["activity"] > 0, 250.0, 0.0)

    # 시간 배열 (분 단위 → 시간 단위)
    t_minutes = np.arange(len(sdf))
    t_hours   = t_minutes / 60.0

    # light interpolation 함수
    from scipy.interpolate import interp1d
    lux_vals  = sdf["lux"].values
    light_func = interp1d(t_hours, lux_vals, kind="previous",
                          bounds_error=False, fill_value=0.0)

    # ODE 초기값 & 풀기
    y0 = [0.5, 0.1]
    t_span = (t_hours[0], t_hours[-1])
    t_eval = t_hours[::60]  # 1시간 간격으로 평가

    try:
        sol = solve_ivp(forger_ode, t_span, y0,
                        args=(light_func,),
                        t_eval=t_eval,
                        method="RK45",
                        max_step=0.5)
        x_vals = sol.y[0]  # CBT proxy
    except Exception:
        return None

    # 날짜별 CBTmin 추출
    dates = sdf["timestamp"].dt.date.unique()
    daily_phase = []
    daily_amp   = []

    for i, date in enumerate(dates):
        start_h = i * 24
        end_h   = start_h + 24
        mask    = (sol.t >= start_h) & (sol.t < end_h)
        if mask.sum() < 2:
            continue
        x_day = x_vals[mask]
        t_day = sol.t[mask]

        cbtmin_idx = np.argmin(x_day)
        phase = t_day[cbtmin_idx] % 24   # 하루 중 CBTmin 시각
        amp   = np.max(x_day) - np.min(x_day)

        daily_phase.append(phase)
        daily_amp.append(amp)

    if len(daily_phase) == 0:
        return None

    phase_arr = np.array(daily_phase)
    amp_arr   = np.array(daily_amp)

    return {
        "circ_phase_MN": np.mean(phase_arr),
        "circ_phase_SD": np.std(phase_arr),
        "circ_phase_Z":  np.mean(phase_arr) / (np.std(phase_arr) + 1e-8),
        "circ_amp_MN":   np.mean(amp_arr),
        "circ_amp_SD":   np.std(amp_arr),
        "circ_amp_Z":    np.mean(amp_arr) / (np.std(amp_arr) + 1e-8),
    }

# ── subject별 계산 ─────────────────────────────────
print("Circadian 피처 계산 중... (시간이 좀 걸려요)")
circ_rows = []
for i, (subj, grp) in enumerate(df.groupby("subject_id")):
    label = grp["label"].iloc[0]
    result = compute_circadian_features(grp)
    if result is None:
        continue
    result["subject_id"] = subj
    result["label"]      = label
    circ_rows.append(result)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/55 완료...")

circ_df = pd.DataFrame(circ_rows)
print(f"\nCircadian 피처 shape: {circ_df.shape}")
print(circ_df.head())

# ── 기존 30 피처와 병합 → 36 피처 ─────────────────
features_30 = pd.read_csv(os.path.join(OUTPUT_DIR, "features_30.csv"))
features_36 = features_30.merge(circ_df.drop(columns=["label"]),
                                 on="subject_id", how="inner")
print(f"\n최종 피처 shape: {features_36.shape}")

out_path = os.path.join(OUTPUT_DIR, "features_36.csv")
features_36.to_csv(out_path, index=False)
print(f"저장됨: {out_path}")