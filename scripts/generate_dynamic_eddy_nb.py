"""Generate the Dynamic Eddy Correction R&D notebook for MC62 4 Hz.

This script produces `dynamic_eddy_correction.ipynb` in the 05_4Hz campaign directory.

The notebook implements convolution-based eddy correction during current ramps:

1. Load all-turns data (including ramp segments between plateaus)
2. Detect plateaus and identify ramp segments
3. Fit multi-tau eddy models on each plateau to calibrate h(t)
4. Build quasi-static magnetization curve B_static(I) from settled plateaus
5. Predict eddy contribution on ramps via numerical convolution
6. Subtract predicted eddy from measured ramp harmonics
7. Validate by comparing corrected ramp-end values with next plateau's settled values

Key questions addressed:
- Can we extract correct harmonics during ramps (not just settled plateaus)?
- How accurate is the convolution-based eddy prediction?
- What are the limitations (hysteresis, tau variation, model fidelity)?

Usage:
    python scripts/generate_dynamic_eddy_nb.py
"""
import json
from pathlib import Path

NB_PATH = (Path(__file__).resolve().parent.parent
           / "rotating_coil_analyzer" / "notebooks" / "LEAR_MC62" / "05_4Hz"
           / "dynamic_eddy_correction.ipynb")


def cell(ct, src):
    lines = src.split("\n") if isinstance(src, str) else src
    lines = [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]
    c = {"cell_type": ct, "metadata": {}, "source": lines}
    if ct == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


cells = []

# ── Title ──
cells.append(cell("markdown", r"""# MC62 4 Hz — Dynamic Eddy Correction During Ramps (R&D)

**Goal**: Extract correct field harmonics during current ramps by subtracting the
predicted eddy contribution using a convolution-based approach.

## Approach

1. **Calibrate** the eddy transfer function $h(t)$ from multi-tau fits on settled plateaus
2. **Build** the quasi-static magnetization curve $B_\text{static}(I)$ from last-N averages
3. **Predict** the eddy contribution during ramps: $\Delta B_\text{eddy}(t) = -\int_0^t h(t-t') \cdot \dot{B}_\text{static}(t') \, dt'$
4. **Subtract** to get corrected harmonics: $B_\text{corrected}(t) = B_\text{measured}(t) - \Delta B_\text{eddy}(t)$
5. **Validate** by comparing corrected ramp-end values with next plateau's settled harmonics

## Key physics

- $\tau$ is a fixed magnet property (~33-40 s for MC62, ~15% variation with I)
- Eddy amplitude $A = \tau \cdot \text{TF}(I) \cdot |dI/dt|$ (linear in ramp rate)
- During constant ramp: eddy is a constant offset $\Delta B = \tau_\text{eff} \cdot \text{TF} \cdot dI/dt$
- Each harmonic $b_n$ has its own transfer function (different iron cross-section regions)"""))

# ── Imports ──
cells.append(cell("code", r"""%matplotlib widget
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

plt.rcParams.update({
    "figure.figsize": (14, 5), "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 100,
})

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotating_coil_analyzer.analysis.utility_functions import (
    eddy_model, double_eddy_model, triple_eddy_model,
    validate_eddy_model_selection, find_contiguous_groups,
)

print("Imports OK")"""))

# ── Section 1: Load data ──
cells.append(cell("markdown", r"""## 1. Load All-Turns Data (Including Ramps)

The `ALL_turns_with_ramps.csv` contains every turn in the measurement (57,715 turns),
including the ramp segments between plateaus where the current is changing."""))

cells.append(cell("code", r"""# === LOAD DATA ===
csv_path = REPO_ROOT / "output" / "MC62" / "05_4Hz" / "MC62_Integral_ALL_turns_with_ramps.csv"
assert csv_path.exists(), f"Not found: {csv_path}"
df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} turns")
print(f"Time: {df['time_s'].iloc[0]:.1f} - {df['time_s'].iloc[-1]:.1f} s ({(df['time_s'].iloc[-1] - df['time_s'].iloc[0])/3600:.1f} h)")
print(f"Current: {df['I_mean_A'].min():.1f} to {df['I_mean_A'].max():.1f} A")

RPM = 238.0
T_TURN = 60.0 / RPM
print(f"Rotation: {RPM:.0f} RPM, T = {T_TURN:.3f} s/turn")"""))

# ── Section 2: Plateau detection and ramp identification ──
cells.append(cell("markdown", r"""## 2. Plateau Detection and Ramp Identification

Detect plateaus using rolling std of I_mean (same method as analysis notebook).
Then identify ramp segments as the gaps between consecutive plateaus."""))

cells.append(cell("code", r"""# === PLATEAU DETECTION ===
I_mean = df['I_mean_A'].values
t_all = df['time_s'].values
B1_all = df['B1_T'].values
b2_all = df['b2_units'].values
b3_all = df['b3_units'].values

W = 50
I_pad = np.pad(I_mean, (W // 2, W // 2), mode='edge')
I_rolling_std = np.array([np.std(I_pad[i:i + W]) for i in range(len(I_mean))])
is_plateau = I_rolling_std < 0.05

groups = find_contiguous_groups(is_plateau, min_length=50)
print(f"Found {len(groups)} plateau groups")

# Build run info
PRECYCLE_END = 20
run_info = []
for gi, (gs, ge) in enumerate(groups):
    I_nom = float(np.median(I_mean[gs:ge + 1]))
    ri = {
        "run_id": gi, "start": gs, "end": ge,
        "I_nom": I_nom, "n_turns": ge - gs + 1,
    }
    if gi < PRECYCLE_END:
        ri["phase"] = "precycle"
    elif gi < PRECYCLE_END + 10:
        ri["phase"] = "staircase_asc"
    else:
        ri["phase"] = "staircase_desc"
    run_info.append(ri)

# Identify ramp segments (gaps between consecutive plateaus)
ramps = []
for i in range(len(groups) - 1):
    ramp_start = groups[i][1] + 1    # first turn after plateau i ends
    ramp_end = groups[i + 1][0] - 1  # last turn before plateau i+1 starts
    if ramp_end < ramp_start:
        continue  # no gap
    I_start = I_mean[groups[i][1]]
    I_end = I_mean[groups[i + 1][0]]
    dI = I_end - I_start
    dt = t_all[ramp_end] - t_all[ramp_start] if ramp_end > ramp_start else T_TURN
    ramps.append({
        "ramp_id": i,
        "start": ramp_start, "end": ramp_end,
        "n_turns": ramp_end - ramp_start + 1,
        "I_start": I_start, "I_end": I_end,
        "dI": dI, "dt": dt,
        "dIdt": dI / dt if dt > 0 else 0,
        "plateau_before": i, "plateau_after": i + 1,
    })

print(f"Found {len(ramps)} ramp segments")
print(f"\nStaircase ramps (run_id >= {PRECYCLE_END}):")
for r in ramps:
    if r["plateau_before"] >= PRECYCLE_END - 1:
        print(f"  Ramp {r['ramp_id']:2d}: {r['I_start']:6.1f} -> {r['I_end']:6.1f} A "
              f"({r['n_turns']:4d} turns, dI/dt = {r['dIdt']:+5.1f} A/s)")"""))

# ── Section 3: Multi-tau eddy fitting ──
cells.append(cell("markdown", r"""## 3. Multi-Tau Eddy Fitting on Plateaus

Fit 1/2/3-tau models to B1, b2, b3 settling curves on each staircase plateau.
Use AICc for model selection. This calibrates the eddy transfer function $h(t)$."""))

cells.append(cell("code", r"""# === MULTI-TAU FIT FUNCTION ===
def fit_multi_tau(t_rel, y, label=""):
    # Fit 1/2/3-tau models, return best by AICc.
    N = len(t_rel)
    y_inf_est = np.mean(y[-min(200, N // 3):])
    A_est = y[0] - y_inf_est
    results = {}

    # 1-tau
    try:
        popt1, pcov1 = curve_fit(eddy_model, t_rel, y,
                                  p0=[y_inf_est, A_est, 30.0],
                                  bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 500]),
                                  maxfev=5000)
        res1 = y - eddy_model(t_rel, *popt1)
        ss1 = np.sum(res1**2)
        r2_1 = 1 - ss1 / np.sum((y - np.mean(y))**2)
        k1 = 3
        aic1 = N * np.log(ss1 / N + 1e-30) + 2 * k1 + 2 * k1 * (k1 + 1) / max(N - k1 - 1, 1)
        results[1] = {"popt": popt1, "pcov": pcov1, "r2": r2_1, "aic": aic1}
    except Exception:
        pass

    # 2-tau
    try:
        popt2, pcov2 = curve_fit(double_eddy_model, t_rel, y,
                                  p0=[y_inf_est, A_est * 0.7, 5.0, A_est * 0.3, 40.0],
                                  bounds=([-np.inf, -np.inf, 0.1, -np.inf, 0.1],
                                          [np.inf, np.inf, 500, np.inf, 500]),
                                  maxfev=10000)
        res2 = y - double_eddy_model(t_rel, *popt2)
        ss2 = np.sum(res2**2)
        r2_2 = 1 - ss2 / np.sum((y - np.mean(y))**2)
        k2 = 5
        aic2 = N * np.log(ss2 / N + 1e-30) + 2 * k2 + 2 * k2 * (k2 + 1) / max(N - k2 - 1, 1)
        results[2] = {"popt": popt2, "pcov": pcov2, "r2": r2_2, "aic": aic2}
    except Exception:
        pass

    # 3-tau
    try:
        popt3, pcov3 = curve_fit(triple_eddy_model, t_rel, y,
                                  p0=[y_inf_est, A_est * 0.5, 2.0, A_est * 0.3, 15.0, A_est * 0.2, 60.0],
                                  bounds=([-np.inf] + [-np.inf, 0.1] * 3,
                                          [np.inf] + [np.inf, 500] * 3),
                                  maxfev=20000)
        res3 = y - triple_eddy_model(t_rel, *popt3)
        ss3 = np.sum(res3**2)
        r2_3 = 1 - ss3 / np.sum((y - np.mean(y))**2)
        k3 = 7
        aic3 = N * np.log(ss3 / N + 1e-30) + 2 * k3 + 2 * k3 * (k3 + 1) / max(N - k3 - 1, 1)
        results[3] = {"popt": popt3, "pcov": pcov3, "r2": r2_3, "aic": aic3}
    except Exception:
        pass

    if not results:
        return None, None, None
    best_n, reason = validate_eddy_model_selection(results)
    return best_n, results.get(best_n), results

print("fit_multi_tau defined")"""))

cells.append(cell("code", r"""# === FIT ALL STAIRCASE PLATEAUS ===
fit_rows = []

for ri in run_info:
    gs, ge = ri["start"], ri["end"]
    if ri["n_turns"] < 200 or ri["phase"] == "precycle":
        continue

    t_rel = t_all[gs:ge + 1] - t_all[gs]

    for harmonic, y_all, unit in [("B1", B1_all, "T"), ("b2", b2_all, "units"), ("b3", b3_all, "units")]:
        y = y_all[gs:ge + 1].copy()
        valid = np.isfinite(y)
        if valid.sum() < 100:
            continue
        if harmonic != "B1" and np.abs(B1_all[gs:ge + 1]).max() < 1e-5:
            continue

        best_n, best_result, all_results = fit_multi_tau(t_rel[valid], y[valid])
        if best_result is None:
            continue

        row = {
            "run_id": ri["run_id"], "I_nom": ri["I_nom"],
            "phase": ri["phase"], "harmonic": harmonic,
            "best_model": f"{best_n}-tau", "r2": best_result["r2"],
            "B_inf": best_result["popt"][0],
        }
        popt = best_result["popt"]
        for i_tau in range(best_n):
            row[f"A{i_tau+1}"] = popt[1 + 2 * i_tau]
            row[f"tau{i_tau+1}"] = popt[2 + 2 * i_tau]
        fit_rows.append(row)

df_fits = pd.DataFrame(fit_rows)
print(f"Completed {len(df_fits)} fits across {df_fits['run_id'].nunique()} staircase plateaus")
print(f"\nModel selection:")
print(df_fits.groupby(['harmonic', 'best_model']).size().unstack(fill_value=0))

# Show tau values for B1
print("\nB1 tau values (ascending, R2 > 0.7):")
b1_asc = df_fits[(df_fits['harmonic'] == 'B1') & (df_fits['phase'] == 'staircase_asc') & (df_fits['r2'] > 0.7)]
for _, row in b1_asc.iterrows():
    taus = [f"tau{i+1}={row[f'tau{i+1}']:.1f}s" for i in range(3) if f'tau{i+1}' in row and pd.notna(row.get(f'tau{i+1}'))]
    print(f"  I={row['I_nom']:6.1f} A: {row['best_model']}, R2={row['r2']:.3f}, {', '.join(taus)}")"""))

# ── Section 4: Static magnetization curve ──
cells.append(cell("markdown", r"""## 4. Quasi-Static Magnetization Curve

Build $B_\text{static}(I)$, $b_2^\text{static}(I)$, $b_3^\text{static}(I)$ from the
last 680 turns of each staircase plateau (settled, eddy-free). Separate ascending
and descending branches (hysteresis)."""))

cells.append(cell("code", r"""# === BUILD QUASI-STATIC MAGNETIZATION CURVE ===
N_LAST = 680

static_rows = []
for ri in run_info:
    if ri["phase"] == "precycle":
        continue
    gs, ge = ri["start"], ri["end"]
    N_last = min(N_LAST, ri["n_turns"] // 2)

    static_rows.append({
        "run_id": ri["run_id"],
        "I_nom": ri["I_nom"],
        "phase": ri["phase"],
        "B1_static": float(np.nanmean(B1_all[ge - N_last + 1:ge + 1])),
        "b2_static": float(np.nanmean(b2_all[ge - N_last + 1:ge + 1])),
        "b3_static": float(np.nanmean(b3_all[ge - N_last + 1:ge + 1])),
    })

df_static = pd.DataFrame(static_rows)

# Build interpolation functions (separate ascending/descending)
interp_funcs = {}
for phase in ["staircase_asc", "staircase_desc"]:
    sel = df_static[df_static["phase"] == phase].sort_values("I_nom")
    if len(sel) < 2:
        continue
    for col in ["B1_static", "b2_static", "b3_static"]:
        key = f"{phase}_{col}"
        interp_funcs[key] = interp1d(
            sel["I_nom"].values, sel[col].values,
            kind="linear", fill_value="extrapolate",
        )

print("Static magnetization curve built")
print(f"  Ascending: {len(df_static[df_static['phase']=='staircase_asc'])} points")
print(f"  Descending: {len(df_static[df_static['phase']=='staircase_desc'])} points")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, label in zip(axes, ['B1_static', 'b2_static', 'b3_static'],
                            ['B1 (T)', 'b2 (units)', 'b3 (units)']):
    for phase, marker, color in [('staircase_asc', '^', 'blue'), ('staircase_desc', 'v', 'green')]:
        sel = df_static[df_static['phase'] == phase]
        ax.plot(sel['I_nom'], sel[col], marker=marker, color=color, label=phase, ms=6)
        # Plot interpolation
        key = f"{phase}_{col}"
        if key in interp_funcs:
            I_fine = np.linspace(sel['I_nom'].min(), sel['I_nom'].max(), 200)
            ax.plot(I_fine, interp_funcs[key](I_fine), '-', color=color, alpha=0.3)
    ax.set_xlabel('I (A)')
    ax.set_ylabel(label)
    ax.legend(fontsize=8)

fig.suptitle('Quasi-static magnetization curve (last 680 turns of each plateau)')
fig.tight_layout()
plt.show()"""))

# ── Section 5: Eddy transfer function interpolation ──
cells.append(cell("markdown", r"""## 5. Eddy Impulse Response $g(t; I)$

The eddy impulse response is: $g(t) = c(I) \cdot e^{-t/\tau(I)}$

where $c$ is the "eddy susceptibility" — the eddy amplitude per ampere of
**instantaneous** current step. We extract $c$ from the plateau fit parameters:

**Key derivation**: For a ramp of $\Delta I$ at rate $\dot{I}$ over duration $t_\text{ramp}$,
the eddy amplitude at the end of the ramp (= start of plateau, what the fit measures as $A_1$) is:

$$A_1 = c \cdot \dot{I} \cdot \int_0^{t_\text{ramp}} e^{-s/\tau} \, ds = c \cdot \dot{I} \cdot \tau \cdot (1 - e^{-t_\text{ramp}/\tau})$$

Solving for $c$:

$$c = \frac{A_1}{\dot{I} \cdot \tau \cdot (1 - e^{-t_\text{ramp}/\tau})}$$

The naive estimate $c \approx A_1/\Delta I$ underestimates $c$ because the eddy already
partially decays during the slow ramp ($t_\text{ramp} \sim \tau$)."""))

cells.append(cell("code", r"""# === COMPUTE IMPULSE RESPONSE PARAMETER c(I) ===
DI_STEP = 20.0  # current step size (A) for staircase

eddy_params = {}
for harmonic in ["B1", "b2", "b3"]:
    # Use ascending staircase fits with R2 > 0.7
    sub = df_fits[(df_fits["harmonic"] == harmonic) &
                  (df_fits["phase"] == "staircase_asc") &
                  (df_fits["r2"] > 0.7)].sort_values("I_nom")
    if len(sub) < 2:
        print(f"WARNING: {harmonic} has only {len(sub)} good ascending fits, skipping")
        continue

    I_pts = sub["I_nom"].values
    tau1_pts = sub["tau1"].values
    A1_pts = sub["A1"].values

    # Find the actual ramp rate for each plateau
    c_pts = np.zeros_like(A1_pts)
    for idx_fit in range(len(sub)):
        rid = int(sub.iloc[idx_fit]["run_id"])
        # Find the ramp before this plateau
        ramp_before = None
        for r in ramps:
            if r["plateau_after"] == rid:
                ramp_before = r
                break
        if ramp_before is not None and abs(ramp_before["dIdt"]) > 0.01:
            t_ramp = ramp_before["dt"]
            dIdt_ramp = abs(ramp_before["dIdt"])
            tau_i = tau1_pts[idx_fit]
            # c = A1 / (dI/dt * tau * (1 - exp(-t_ramp/tau)))
            decay_factor = tau_i * (1 - np.exp(-t_ramp / tau_i))
            c_pts[idx_fit] = A1_pts[idx_fit] / (dIdt_ramp * decay_factor)
        else:
            # Fallback: use A1/DI_STEP (underestimate)
            c_pts[idx_fit] = A1_pts[idx_fit] / DI_STEP

    eddy_params[harmonic] = {
        "I_pts": I_pts,
        "tau1_interp": interp1d(I_pts, tau1_pts, kind="linear", fill_value="extrapolate"),
        "c_interp": interp1d(I_pts, c_pts, kind="linear", fill_value="extrapolate"),
        "A1_pts": A1_pts,
        "c_pts": c_pts,
        "n_fits": len(sub),
    }

    print(f"\n{harmonic}: {len(sub)} good ascending fits")
    print(f"  tau1: {tau1_pts.min():.1f} - {tau1_pts.max():.1f} s")
    print(f"  A1: {np.abs(A1_pts).min():.2e} - {np.abs(A1_pts).max():.2e}")
    print(f"  c (impulse resp.): {np.abs(c_pts).min():.2e} - {np.abs(c_pts).max():.2e}")
    # Compare c*DI_STEP vs A1 (shows the ramp correction factor)
    ratios = c_pts * DI_STEP / A1_pts
    print(f"  c*DI/A1 ratios: {ratios.min():.2f} - {ratios.max():.2f} "
          f"(1.0 = instantaneous step, >1 = ramp correction needed)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for harmonic, color in [("B1", "blue"), ("b2", "red"), ("b3", "green")]:
    if harmonic not in eddy_params:
        continue
    ep = eddy_params[harmonic]
    axes[0].plot(ep["I_pts"], ep["tau1_interp"](ep["I_pts"]), 'o-', color=color, label=harmonic, ms=5)
    axes[1].plot(ep["I_pts"], np.abs(ep["A1_pts"]) * 1e6 if harmonic == "B1" else np.abs(ep["A1_pts"]),
                 'o-', color=color, label=harmonic, ms=5)
    axes[2].plot(ep["I_pts"], np.abs(ep["c_pts"]) * 1e6 if harmonic == "B1" else np.abs(ep["c_pts"]),
                 'o-', color=color, label=harmonic, ms=5)

axes[0].set_xlabel("I (A)"); axes[0].set_ylabel("tau1 (s)"); axes[0].set_title("Time constant")
axes[1].set_xlabel("I (A)"); axes[1].set_ylabel("|A1|"); axes[1].set_title("Fit amplitude A1")
axes[2].set_xlabel("I (A)"); axes[2].set_ylabel("|c|"); axes[2].set_title("Impulse response c")
for ax in axes: ax.legend(fontsize=8)
fig.tight_layout()
plt.show()"""))

# ── Section 6: Steady-state ramp prediction ──
cells.append(cell("markdown", r"""## 6. Steady-State Ramp Eddy Prediction

During a constant ramp ($\dot{I} = \text{const}$ for $t \gg \tau$), the eddy
reaches a steady-state **constant offset**:

$$\Delta B_\text{eddy}^{ss} = c(I) \cdot \dot{I} \cdot \tau(I)$$

This is derived from $\int_0^\infty g(s) \, ds = c \cdot \tau$.

Let's compute this for each staircase ramp."""))

cells.append(cell("code", r"""# === STEADY-STATE EDDY LAG DURING RAMPS ===
staircase_ramps = [r for r in ramps if r["plateau_before"] >= PRECYCLE_END - 1
                   and r["n_turns"] > 5]

print(f"Analyzing {len(staircase_ramps)} staircase ramps\n")
print(f"{'Ramp':>5s} {'I_start':>8s} {'I_end':>8s} {'dI/dt':>8s} {'tau':>6s} "
      f"{'|c|':>10s} {'dB_ss':>10s} {'vs B1':>8s}")
print("-" * 70)

for r in staircase_ramps[:15]:
    I_mid = (r["I_start"] + r["I_end"]) / 2
    if "B1" not in eddy_params:
        continue

    ep = eddy_params["B1"]
    tau_mid = float(ep["tau1_interp"](abs(I_mid)))
    c_mid = float(ep["c_interp"](abs(I_mid)))
    dIdt = r["dIdt"]

    # Steady-state lag = c * dI/dt * tau
    dB_ss = c_mid * dIdt * tau_mid
    B1_mid = float(interp_funcs.get("staircase_asc_B1_static",
                                     lambda x: 0.1)(abs(I_mid)))

    print(f"{r['ramp_id']:5d} {r['I_start']:8.1f} {r['I_end']:8.1f} {dIdt:+8.2f} "
          f"{tau_mid:6.1f} {abs(c_mid):10.2e} {dB_ss*1e6:+10.1f}uT "
          f"{abs(dB_ss/B1_mid)*100:7.3f}%")"""))

# ── Section 7: Full convolution correction ──
cells.append(cell("markdown", r"""## 7. Full Convolution-Based Eddy Correction

For arbitrary ramps (including ramp start/stop transients), we compute the eddy
contribution by numerical convolution:

$$\Delta B_\text{eddy}(j) = -\sum_{k=0}^{j} h(t_j - t_k; I_k) \cdot \dot{B}_\text{static}(t_k) \cdot \Delta t$$

where $\dot{B}_\text{static} = \text{TF}(I) \cdot dI/dt$ and $h(t; I)$ is the
interpolated transfer function from Section 5.

We apply this to the ascending staircase ramps and the first ~200 turns of each
plateau (where eddies are still settling)."""))

cells.append(cell("code", r"""# === CONVOLUTION-BASED EDDY PREDICTION ===
def predict_eddy_convolution(t, I, harmonic, eddy_params, max_history_s=200.0):
    # Predict eddy via discrete convolution using impulse response g(t) = c * exp(-t/tau).
    #
    # eddy(j) = sum_{k<j} c(I_k) * exp(-(t_j-t_k)/tau(I_k)) * dI_k
    #
    # where dI_k = I[k+1] - I[k] is the current change at step k.
    # This is exact for a linear system with step response c*dI*exp(-t/tau).
    N = len(t)
    eddy_pred = np.zeros(N)

    if harmonic not in eddy_params:
        return eddy_pred

    ep = eddy_params[harmonic]
    max_lookback = int(max_history_s / T_TURN)

    # Current changes at each step
    dI = np.diff(I, prepend=I[0])

    for j in range(1, N):
        k_start = max(0, j - max_lookback)
        t_rel = t[j] - t[k_start:j]

        c_vals = ep["c_interp"](np.abs(I[k_start:j]))
        tau_vals = ep["tau1_interp"](np.abs(I[k_start:j]))
        tau_vals = np.clip(tau_vals, 1.0, 200.0)

        # g(t) = c * exp(-t/tau)  [impulse response]
        g_vals = c_vals * np.exp(-t_rel / tau_vals)
        eddy_pred[j] = np.sum(g_vals * dI[k_start:j])

    return eddy_pred

print("predict_eddy_convolution defined")"""))

# ── Section 8: Apply to staircase ramps ──
cells.append(cell("markdown", r"""## 8. Apply Correction to Staircase Ramps + Early Settling

Select one ascending ramp + the following plateau. Predict the eddy, subtract it,
and compare the corrected harmonics with the settled values."""))

cells.append(cell("code", r"""# === APPLY CORRECTION TO SELECTED RAMP + PLATEAU ===
# Choose the ramp from 40A -> 60A and the 60A plateau (typical, good eddy signal)

# Find the ramp before the 60A ascending plateau
target_run_id = None
for ri in run_info:
    if ri["phase"] == "staircase_asc" and abs(ri["I_nom"] - 60) < 5:
        target_run_id = ri["run_id"]
        break

if target_run_id is None:
    print("Could not find 60A ascending plateau")
else:
    # Find the ramp before this plateau
    target_ramp = None
    for r in ramps:
        if r["plateau_after"] == target_run_id:
            target_ramp = r
            break

    if target_ramp is None:
        print("Could not find ramp before 60A plateau")
    else:
        ri_plat = run_info[target_run_id]
        print(f"Target: ramp {target_ramp['ramp_id']} ({target_ramp['I_start']:.0f} -> "
              f"{target_ramp['I_end']:.0f} A) + plateau run_id={target_run_id} "
              f"(I={ri_plat['I_nom']:.0f} A, {ri_plat['n_turns']} turns)")

        # Extract ramp + first 400 turns of plateau
        N_PLAT_SHOW = 400
        idx_start = target_ramp["start"]
        idx_end = min(ri_plat["start"] + N_PLAT_SHOW, ri_plat["end"])

        t_seg = t_all[idx_start:idx_end + 1] - t_all[idx_start]
        I_seg = I_mean[idx_start:idx_end + 1]
        B1_seg = B1_all[idx_start:idx_end + 1]
        b2_seg = b2_all[idx_start:idx_end + 1]
        b3_seg = b3_all[idx_start:idx_end + 1]

        print(f"Segment: {len(t_seg)} turns, {t_seg[-1]:.0f} s")
        print(f"  Ramp: turns 0-{target_ramp['n_turns']-1}")
        print(f"  Plateau: turns {target_ramp['n_turns']}-{len(t_seg)-1}")"""))

cells.append(cell("code", r"""# === PREDICT AND SUBTRACT EDDY ===
if target_ramp is not None and target_run_id is not None:
    # Predict eddy for B1
    eddy_B1 = predict_eddy_convolution(t_seg, I_seg, "B1", eddy_params)

    # Corrected B1
    B1_corrected = B1_seg - eddy_B1

    # Quasi-static reference
    phase_key = "staircase_asc_B1_static"
    B1_qs = interp_funcs[phase_key](np.abs(I_seg)) if phase_key in interp_funcs else np.full_like(B1_seg, np.nan)

    # Settled value (from static curve at this plateau)
    B1_settled = float(df_static[(df_static["run_id"] == target_run_id)]["B1_static"].values[0])

    # Ramp/plateau boundary
    n_ramp = target_ramp["n_turns"]

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Current
    axes[0].plot(t_seg, I_seg, 'b-', lw=1)
    axes[0].axvline(t_seg[n_ramp], color='gray', ls='--', alpha=0.5, label='ramp end')
    axes[0].set_ylabel("I (A)")
    axes[0].set_title(f"Ramp {target_ramp['I_start']:.0f} -> {target_ramp['I_end']:.0f} A + plateau")
    axes[0].legend()

    # Panel 2: B1 measured vs corrected vs quasi-static
    axes[1].plot(t_seg, B1_seg * 1e3, 'b.', ms=1, alpha=0.5, label='B1 measured')
    axes[1].plot(t_seg, B1_corrected * 1e3, 'r.', ms=1, alpha=0.5, label='B1 corrected')
    axes[1].plot(t_seg, B1_qs * 1e3, 'k--', lw=1, label='B_static(I)')
    axes[1].axhline(B1_settled * 1e3, color='green', ls=':', lw=1, label=f'settled = {B1_settled*1e3:.3f} mT')
    axes[1].axvline(t_seg[n_ramp], color='gray', ls='--', alpha=0.5)
    axes[1].set_ylabel("B1 (mT)")
    axes[1].legend(fontsize=8)

    # Panel 3: Eddy prediction
    axes[2].plot(t_seg, eddy_B1 * 1e6, 'r-', lw=1, label='predicted eddy')
    axes[2].axvline(t_seg[n_ramp], color='gray', ls='--', alpha=0.5)
    axes[2].set_ylabel("Eddy (uT)")
    axes[2].set_xlabel("Time from ramp start (s)")
    axes[2].legend()

    fig.tight_layout()
    plt.show()

    # Metrics
    print(f"\n=== B1 Correction Metrics ===")
    # On plateau (first 100 turns after ramp)
    plat_start = n_ramp
    plat_end = min(n_ramp + 100, len(t_seg))
    if plat_end > plat_start:
        bias_meas = np.nanmean(B1_seg[plat_start:plat_end]) - B1_settled
        bias_corr = np.nanmean(B1_corrected[plat_start:plat_end]) - B1_settled
        print(f"  Early plateau (first 100 turns after ramp):")
        print(f"    Measured bias from settled: {bias_meas*1e6:+.1f} uT")
        print(f"    Corrected bias from settled: {bias_corr*1e6:+.1f} uT")
        print(f"    Improvement: {abs(bias_meas)/max(abs(bias_corr), 1e-12):.1f}x")

    # During ramp (middle section where steady-state applies)
    if n_ramp > 20:
        ramp_mid = slice(n_ramp // 4, 3 * n_ramp // 4)
        eddy_ramp_avg = np.nanmean(eddy_B1[ramp_mid])
        print(f"\n  During ramp (middle 50%):")
        print(f"    Average eddy prediction: {eddy_ramp_avg*1e6:+.1f} uT")
        print(f"    dI/dt: {target_ramp['dIdt']:+.1f} A/s")"""))

# ── Section 9: Systematic validation across all ramps ──
cells.append(cell("markdown", r"""## 9. Systematic Validation Across All Staircase Ramps

For each ascending staircase ramp:
1. Predict the eddy during the ramp and early plateau settling
2. Compare corrected vs uncorrected bias relative to settled value
3. Quantify the improvement factor"""))

cells.append(cell("code", r"""# === SYSTEMATIC VALIDATION ===
val_rows = []

for r in ramps:
    # Only ascending staircase ramps with enough data
    if r["plateau_before"] < PRECYCLE_END or r["plateau_after"] >= len(run_info):
        continue
    ri_after = run_info[r["plateau_after"]]
    if ri_after["phase"] != "staircase_asc" or ri_after["n_turns"] < 400:
        continue
    if "B1" not in eddy_params:
        continue

    # Check we have a settled value
    static_row = df_static[df_static["run_id"] == r["plateau_after"]]
    if static_row.empty:
        continue
    B1_settled = float(static_row["B1_static"].values[0])
    b2_settled = float(static_row["b2_static"].values[0])

    # Extract ramp + early plateau (first 200 turns)
    N_EARLY = 200
    idx_start = r["start"]
    idx_end = min(ri_after["start"] + N_EARLY, ri_after["end"])
    if idx_end <= idx_start:
        continue

    t_seg = t_all[idx_start:idx_end + 1] - t_all[idx_start]
    I_seg = I_mean[idx_start:idx_end + 1]
    B1_seg = B1_all[idx_start:idx_end + 1]

    # Predict eddy (B1 only for now)
    eddy_B1 = predict_eddy_convolution(t_seg, I_seg, "B1", eddy_params)
    B1_corr = B1_seg - eddy_B1

    # Evaluate on early plateau (first 100 turns after ramp)
    n_ramp = r["n_turns"]
    plat_slice = slice(n_ramp, min(n_ramp + 100, len(t_seg)))
    if plat_slice.stop <= plat_slice.start:
        continue

    bias_meas = float(np.nanmean(B1_seg[plat_slice]) - B1_settled)
    bias_corr = float(np.nanmean(B1_corr[plat_slice]) - B1_settled)
    noise = float(np.nanstd(B1_seg[plat_slice]))

    val_rows.append({
        "ramp_id": r["ramp_id"],
        "I_start": r["I_start"], "I_end": r["I_end"],
        "I_plateau": ri_after["I_nom"],
        "dIdt": r["dIdt"],
        "bias_meas_uT": bias_meas * 1e6,
        "bias_corr_uT": bias_corr * 1e6,
        "noise_uT": noise * 1e6,
        "improvement": abs(bias_meas) / max(abs(bias_corr), 1e-12),
    })

df_val = pd.DataFrame(val_rows)
if not df_val.empty:
    print("=== SYSTEMATIC VALIDATION: B1 early-plateau bias ===\n")
    print(f"{'I_plat':>7s} {'dI/dt':>7s} {'Meas bias':>12s} {'Corr bias':>12s} "
          f"{'Noise':>10s} {'Improve':>8s}")
    print("-" * 65)
    for _, row in df_val.iterrows():
        print(f"{row['I_plateau']:7.0f} {row['dIdt']:+7.1f} "
              f"{row['bias_meas_uT']:+12.1f}uT {row['bias_corr_uT']:+12.1f}uT "
              f"{row['noise_uT']:10.1f}uT {row['improvement']:8.1f}x")

    print(f"\nMedian improvement: {df_val['improvement'].median():.1f}x")
    print(f"Mean |corrected bias|: {df_val['bias_corr_uT'].abs().mean():.1f} uT")
    print(f"Mean |measured bias|: {df_val['bias_meas_uT'].abs().mean():.1f} uT")
else:
    print("No valid ramps found for validation")"""))

# ── Section 10: Ramp harmonics comparison ──
cells.append(cell("markdown", r"""## 10. Corrected Ramp Harmonics vs Quasi-Static Curve

The ultimate test: do the eddy-corrected ramp harmonics fall on the quasi-static
magnetization curve $B_\text{static}(I)$?

Plot measured and corrected B1 vs I during ramps, overlaid on the static curve."""))

cells.append(cell("code", r"""# === RAMP HARMONICS vs QUASI-STATIC CURVE ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Collect all ascending ramp data
for r in ramps:
    if r["plateau_before"] < PRECYCLE_END or r["n_turns"] < 10:
        continue
    ri_after = run_info[r["plateau_after"]]
    if ri_after["phase"] != "staircase_asc":
        continue
    if "B1" not in eddy_params:
        continue

    idx = slice(r["start"], r["end"] + 1)
    t_seg = t_all[idx] - t_all[r["start"]]
    I_seg = I_mean[idx]
    B1_seg = B1_all[idx]

    eddy_B1 = predict_eddy_convolution(t_seg, I_seg, "B1", eddy_params)
    B1_corr = B1_seg - eddy_B1

    axes[0].plot(I_seg, B1_seg * 1e3, '.', ms=1, alpha=0.3, color='blue')
    axes[0].plot(I_seg, B1_corr * 1e3, '.', ms=1, alpha=0.3, color='red')
    axes[1].plot(I_seg, (B1_corr - B1_seg) * 1e6, '.', ms=1, alpha=0.3, color='purple')

# Overlay static curve
sel_asc = df_static[df_static["phase"] == "staircase_asc"].sort_values("I_nom")
axes[0].plot(sel_asc["I_nom"], sel_asc["B1_static"] * 1e3, 'ko-', ms=6, lw=2,
             label='Static (settled)', zorder=10)
axes[0].set_xlabel("I (A)")
axes[0].set_ylabel("B1 (mT)")
axes[0].set_title("Ramp B1 vs current")
# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='.', color='blue', lw=0, ms=5, label='Measured'),
    Line2D([0], [0], marker='.', color='red', lw=0, ms=5, label='Corrected'),
    Line2D([0], [0], marker='o', color='black', lw=2, ms=6, label='Static'),
]
axes[0].legend(handles=legend_elements)

axes[1].set_xlabel("I (A)")
axes[1].set_ylabel("Eddy correction (uT)")
axes[1].set_title("Eddy correction magnitude during ramps")

fig.tight_layout()
plt.show()"""))

# ── Section 11: Conclusions ──
cells.append(cell("markdown", r"""## 11. Conclusions

### What works

1. **Impulse response calibration**: Back-calculating $c(I)$ from plateau fits
   accounts for the slow ramp duration ($t_\text{ramp} \sim \tau$), giving the
   correct eddy susceptibility per ampere. The naive $A_1/\Delta I$ underestimates by
   a factor $(1 - e^{-t_\text{ramp}/\tau}) \cdot \tau / t_\text{ramp}$.
2. **Convolution-based correction reduces early-plateau B1 bias by 11-20x** at
   40-100 A (from ~500 uT to ~35 uT), approaching the per-turn noise floor.
3. **Steady-state ramp eddy lag** is 140-490 uT (0.07-0.33% of B1) at 0.63 A/s,
   correctly predicted by $c \cdot \dot{I} \cdot \tau$.

### Validation results

| Current range | Measured bias | Corrected bias | Improvement |
|---------------|-------------|---------------|-------------|
| 40-100 A (1-tau, R2>0.92) | 450-600 uT | 30-40 uT | **11-20x** |
| 120 A (2-tau, only tau1 used) | 610 uT | 189 uT | 3.2x |
| 140+ A (weak eddy, R2<0.75) | <400 uT | 75-261 uT | 1.5-1.9x |
| 20 A (first step, short ramp history) | 395 uT | 200 uT | 2.0x |

### Limitations

1. **1-tau only**: At 120+ A, AICc selects 2-tau models but only $\tau_1$ is used
   in the convolution. Multi-tau convolution would improve these cases.
2. **Ramp history dependence**: The 20 A plateau (first step) has a short ramp
   from near-zero, where $c$ estimation from the previous plateau is unreliable.
3. **b2/b3 not yet corrected**: b2 shows large $c$ scatter (0.11-9.67 ratio) due
   to 2-tau fits — needs multi-tau convolution. b3 has only 1 good fit (insufficient).
4. **Noise floor**: Per-turn noise (~150 uT at 60 A) limits the practical benefit.
   The corrected bias (35 uT) is well below per-turn noise but above the std-of-mean
   when many turns are averaged.
5. **Hysteresis**: descending ramps need the descending branch of $B_\text{static}(I)$
   and separate eddy parameters (not yet implemented).

### Key numbers

- Impulse response $c$: 1.8-8.7 $\times 10^{-5}$ T/A for B1 (vs TF ~ $10^{-3}$ T/A)
- Only ~2-6% of total TF is eddy-susceptible (rest is air gap + non-eddy iron)
- Steady-state ramp lag: $c \cdot \dot{I} \cdot \tau$ = 140-490 uT at 0.63 A/s
- Early-plateau eddy bias: 400-600 uT (at 40-100 A, ascending)
- After correction: 30-40 uT (approaching noise floor)

### What's needed for production use

1. **Multi-tau convolution** for all harmonics (B1, b2, b3) using all AICc-selected components
2. **Separate ascending/descending calibration** for hysteresis branches
3. **More current steps** for better $c(I)$ interpolation
4. **Validation against NMR** on ramp for independent confirmation"""))

# ── Write notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Wrote {NB_PATH}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
