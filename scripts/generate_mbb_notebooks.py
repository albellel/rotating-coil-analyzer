"""SUPERSEDED by generate_notebooks.py -- kept for reference only.

Originally generated 3 MBB analysis notebooks for 2026-02-06 NCS supercycle
measurements. All functionality is now in the unified generate_notebooks.py.
"""

import json
from pathlib import Path

NOTEBOOK_DIR = Path("rotating_coil_analyzer/notebooks/SPS_MBB")


# ================================================================
# Notebook helpers
# ================================================================

def make_cell(cell_type, cell_id, source_lines):
    """Create an ipynb cell dict."""
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source_lines,
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def md(cell_id, text):
    """Create a markdown cell from multiline text."""
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("markdown", cell_id, source)


def code(cell_id, text):
    """Create a code cell from multiline text."""
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", cell_id, source)


def write_notebook(path, cells):
    """Write a notebook to disk."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.1"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Wrote {path} ({len(cells)} cells)")


# ================================================================
# Comprehensive analysis notebook (24 sections, NCS only)
# ================================================================

def build_analysis_cells(session, meas_subdir, energy_label, out_subdir):
    """Build cell list for a comprehensive 24-section NCS analysis notebook.

    Single segment (NCS), cross-session Kn from MBA Dec 2025.
    """
    cells = []

    # ==============================================================
    # Title & TOC
    # ==============================================================
    cells.append(md("title", f"""# SPS MBB Dipole -- Comprehensive NCS Analysis ({energy_label} MD1 Extended)

**Measurement session:** `{out_subdir} / {session.split('/')[-1]}`
**Segment:** NCS
**Magnet:** MBB (normal dipole, m=1)
**Kn calibration:** AC compensation (cross-session from MBA, Dec 2025)
**Supercycle:** LHC_pilot -> MD1 ({energy_label}) -> SFTPRO, x20 repetitions

### Part I: Setup & Data Quality
| # | Section |
|---|---------|
| 1 | Configuration & Imports |
| 2 | Kn Calibration |
| 3 | Data Loading & Channel Detection |
| 4 | Raw Signals Overview |
| 5 | cel/fed Safety Diagnostic |
| 6 | Plateau Detection & Turn Classification |
| 7 | FDI Stuck-Channel Diagnostic |

### Part II: Pipeline Processing
| # | Section |
|---|---------|
| 8 | Process Plateau Turns |
| 9 | All-Turn Harmonics vs Time |
| 10 | FFMM Golden Standard Validation |

### Part III: Harmonic Analysis
| # | Section |
|---|---------|
| 11 | Main Field (B1) |
| 12 | b2 (Quadrupole) |
| 13 | b3 (Sextupole) |
| 14 | Higher Harmonics Overview |
| 15 | Multipole Spectrum |

### Part IV: Transfer Function & Inductance
| # | Section |
|---|---------|
| 16 | Transfer Function B1/I |
| 17 | Apparent vs Differential Inductance |

### Part V: Eddy Current & Settling
| # | Section |
|---|---------|
| 18 | Raw Settling Curves |
| 19 | Exponential Fits |
| 20 | Settling Bias Analysis |
| 21 | N_LAST Sensitivity Study |

### Part VI: Summary
| # | Section |
|---|---------|
| 22 | Comprehensive Statistics Table |
| 23 | Analysis Choices Summary |
| 24 | CSV Export |"""))

    # ==============================================================
    # 1. Configuration & Imports
    # ==============================================================
    cells.append(md("s1-hdr", "---\n## 1. Configuration & Imports"))

    cells.append(code("s1-config", f"""# === CONFIGURATION ===
SEGMENT = "NCS"

SESSION = "{session}"
MEAS_SUBDIR = "{meas_subdir}"
KN_CROSS_SESSION = "MBB/2025-12-12/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt"

MAGNET_ORDER = 1          # dipole
R_REF = 0.02              # reference radius [m]
L_COIL = 0.47             # coil length [m]
SAMPLES_PER_TURN = 1024   # encoder samples per revolution

OPTIONS = ("dri", "rot", "cel", "fed")

MIN_B1_T = 1e-4           # minimum |B1| for normalization
PLATEAU_I_RANGE_MAX = 3.0 # block-averaged range threshold (A)
N_BLOCKS = 10             # blocks for range averaging

# Settling: last N turns per supercycle
N_LAST_TURNS_INJ = 18     # injection: keep last 18 of ~24
N_LAST_TURNS_HIGH = None   # flat-high: use all

# Outlier removal
N_SIGMA_CLIP = 5           # MAD sigma clipping

# Eddy current fit
MIN_INJECTION_TURNS = 5    # minimum turns for exponential fit

print(f"SPS MBB Dipole -- Comprehensive NCS Analysis ({energy_label} Extended)")
print("=" * 60)
print(f"  Session       : {{SESSION}}")
print(f"  Segment       : {{SEGMENT}}")
print(f"  Magnet order  : {{MAGNET_ORDER}} (dipole)")
print(f"  R_ref         : {{R_REF}} m")
print(f"  Samples/turn  : {{SAMPLES_PER_TURN}}")
print(f"  Options       : {{OPTIONS}}")
print(f"  Plateau thresh: {{PLATEAU_I_RANGE_MAX}} A")"""))

    cells.append(code("s1-imports", """import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

%matplotlib widget
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists():
        break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
    process_kn_pipeline,
    build_harmonic_rows,
    diagnose_cel_fed,
    diagnose_fdi_transitions,
    mad_sigma_clip,
    eddy_model,
)
from rotating_coil_analyzer.ingest.channel_detect import robust_range

SESSION_DIR = REPO_ROOT / "measurements" / SESSION
RUN_DIR = SESSION_DIR / MEAS_SUBDIR
KN_PATH = REPO_ROOT / "measurements" / KN_CROSS_SESSION
assert KN_PATH.exists(), f"Kn file not found: {KN_PATH}"

print(f"Repo root   : {REPO_ROOT}")
print(f"Session dir : {SESSION_DIR}")
print(f"Run dir     : {RUN_DIR}")
print(f"Kn file     : {KN_PATH}")
print("Imports ready.")"""))

    # ==============================================================
    # 2. Kn Calibration
    # ==============================================================
    cells.append(md("s2-hdr", "---\n## 2. Kn Calibration (AC Compensation)"))

    cells.append(code("s2-kn", """kn = load_segment_kn_txt(str(KN_PATH))
H = len(kn.orders)
Ns = SAMPLES_PER_TURN
m = MAGNET_ORDER

print(f"Kn: {H} harmonics from {KN_PATH.name}")
print(f"  Orders: {list(kn.orders)}")

kn_abs_n1 = abs(kn.kn_abs[0])
kn_cmp_n1 = abs(kn.kn_cmp[0])
ratio_n1 = kn_abs_n1 / max(kn_cmp_n1, 1e-30)

print(f"\\n  |Kn_abs(n=1)| = {kn_abs_n1:.6e}")
print(f"  |Kn_cmp(n=1)| = {kn_cmp_n1:.6e}")
print(f"  Abs/Cmp suppression ratio (n=1): {ratio_n1:.0f}x")"""))

    # ==============================================================
    # 3. Data Loading & Channel Detection
    # ==============================================================
    cells.append(md("s3-hdr", "---\n## 3. Data Loading & Channel Detection"))

    cells.append(code("s3-load", """FILE_PAT = re.compile(
    r"Run_(\\d+)_I_([\\d.]+)A_(N?CS)_raw_measurement_data\\.txt$"
)

ncs_files = []
for f in sorted(RUN_DIR.iterdir()):
    match = FILE_PAT.search(f.name)
    if match and match.group(3) == SEGMENT:
        ncs_files.append(f)
assert ncs_files, f"No {SEGMENT} raw files found in {RUN_DIR}"
raw_file = ncs_files[0]

raw = np.loadtxt(raw_file)
n_turns = raw.shape[0] // Ns
n_keep = n_turns * Ns
ncols = raw.shape[1]

print(f"Raw file: {raw_file.name}")
print(f"Shape: {raw.shape} -> {n_turns} turns, {ncols} columns")
print(f"Time span: {raw[-1,0] - raw[0,0]:.1f} s ({(raw[-1,0] - raw[0,0])/60:.1f} min)")

t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
flux_col1 = raw[:n_keep, 1].reshape(n_turns, Ns)
flux_col2 = raw[:n_keep, 2].reshape(n_turns, Ns)
I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

# Auto-detect channel swap
I_mean_quick = I_all.mean(axis=1)
best_turn = np.argmax(np.abs(I_mean_quick))
r1 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 1])
r2 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 2])
SWAP_FLUX = (r2 > r1)

if SWAP_FLUX:
    flux_abs_all = flux_col2
    flux_cmp_all = flux_col1
    print("  (flux columns swapped: col2=abs, col1=cmp)")
else:
    flux_abs_all = flux_col1
    flux_cmp_all = flux_col2

print(f"  Flux swap: {SWAP_FLUX}  (abs range={max(r1,r2):.4e}, cmp range={min(r1,r2):.4e})")"""))

    # ==============================================================
    # 4. Raw Signals Overview
    # ==============================================================
    cells.append(md("s4-hdr", "---\n## 4. Raw Signals Overview"))

    cells.append(code("s4-raw", """fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

x = np.arange(n_keep)
axes[0].plot(x, flux_abs_all.ravel(), linewidth=0.2, color="steelblue")
axes[0].set_ylabel("Flux abs (Wb)"); axes[0].set_title("Absolute flux channel")

axes[1].plot(x, flux_cmp_all.ravel(), linewidth=0.2, color="teal")
axes[1].set_ylabel("Flux cmp (Wb)"); axes[1].set_title("Compensated flux channel")

axes[2].plot(x, I_all.ravel(), linewidth=0.2, color="tab:orange")
axes[2].set_ylabel("Current (A)"); axes[2].set_title("Current channel")
axes[2].set_xlabel("Sample index")

fig.suptitle(f"Raw signals -- {SESSION} ({SEGMENT})", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 5. cel/fed Safety Diagnostic
    # ==============================================================
    cells.append(md("s5-hdr", "---\n## 5. cel/fed Safety Diagnostic"))

    cells.append(code("s5-celfed", """I_mean = I_all.mean(axis=1)
hi_mask = np.abs(I_mean) > 4000
if hi_mask.sum() < 5:
    hi_mask = np.abs(I_mean) > np.percentile(np.abs(I_mean), 90)

n_diag = min(100, int(hi_mask.sum()))
hi_idx = np.where(hi_mask)[0][:n_diag]

diag = diagnose_cel_fed(
    flux_abs_all[hi_idx], flux_cmp_all[hi_idx],
    t_all[hi_idx], I_all[hi_idx],
    kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER,
)
print(f"cel/fed diagnostic ({n_diag} high-I turns):")
print(f"  Recommendation: {diag.recommendation}")
print(f"  {diag.reason}")
Bd = np.max(np.abs(diag.B_main_with_fed - diag.B_main_without_fed))
print(f"  B_main max |diff|: {Bd:.4e} T")

if diag.recommendation == "UNSAFE":
    OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
    print(f"  -> cel/fed disabled, OPTIONS = {OPTIONS}")
else:
    print(f"  -> cel/fed safe, keeping OPTIONS = {OPTIONS}")"""))

    # ==============================================================
    # 6. Plateau Detection & Turn Classification
    # ==============================================================
    cells.append(md("s6-hdr", "---\n## 6. Plateau Detection & Turn Classification"))

    cells.append(code("s6-plateau", """I_mean = I_all.mean(axis=1)
t_mean = t_all.mean(axis=1)
I_range, I_blocks = compute_block_averaged_range(I_all, Ns, N_BLOCKS)

plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
is_plateau = plateau_info["is_plateau"]

turn_label = np.array(["ramp"] * n_turns, dtype=object)
for i in range(n_turns):
    if is_plateau[i]:
        turn_label[i] = classify_current(I_mean[i])

for lab in ["injection", "flat-mid", "flat-high"]:
    mask = turn_label == lab
    n = mask.sum()
    if n > 0:
        print(f"  {lab:12s}: {n:4d} turns, I = {I_mean[mask].mean():.1f} +/- {I_mean[mask].std():.1f} A")
print(f"  {'ramp':12s}: {(turn_label == 'ramp').sum():4d} turns")
print(f"\\nTotal: {n_turns} turns, {is_plateau.sum()} plateau")

inj_groups = find_contiguous_groups(turn_label == "injection", min_length=2)
fh_groups = find_contiguous_groups(turn_label == "flat-high", min_length=2)
print(f"\\nInjection groups (supercycles): {len(inj_groups)}")
print(f"Flat-high groups: {len(fh_groups)}")

# Current profile plot
label_colors = {"injection": "tab:green", "flat-mid": "tab:purple", "flat-high": "tab:blue"}
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(t_mean, I_mean, ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
for lab, col in label_colors.items():
    mask = turn_label == lab; idx = np.where(mask)[0]
    if len(idx) > 0:
        ax.scatter(t_mean[idx], I_mean[idx], s=6, color=col, zorder=2, label=lab)
ax.set_xlabel("Time (s)"); ax.set_ylabel("I (A)")
ax.set_title(f"Current profile with plateau classification -- {SEGMENT}")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 7. FDI Stuck-Channel Diagnostic
    # ==============================================================
    cells.append(md("s7-hdr", """---
## 7. FDI Stuck-Channel Diagnostic

Check whether the Fast Digital Integrator responds to current changes
between consecutive plateau groups."""))

    cells.append(code("s7-fdi", """# Build run_info from contiguous plateau groups
all_groups = []
for lab_name in ["injection", "flat-mid", "flat-high"]:
    groups = find_contiguous_groups(turn_label == lab_name, min_length=2)
    for gs, ge in groups:
        all_groups.append({"start": gs, "end": ge,
                           "I_nom": float(I_mean[gs:ge+1].mean())})
all_groups.sort(key=lambda x: x["start"])
for i, g in enumerate(all_groups):
    g["run_id"] = i

if len(all_groups) < 2:
    print("Fewer than 2 plateau groups, skipping FDI check")
else:
    flux_turns = flux_abs_all.mean(axis=1)
    checks = diagnose_fdi_transitions(
        flux_turns, I_mean, all_groups,
        stuck_threshold=0.3, partial_threshold=0.7, min_delta_I=5.0,
    )
    n_ok = sum(1 for c in checks if c.severity == "OK")
    n_partial = sum(1 for c in checks if c.severity == "PARTIAL")
    n_stuck = sum(1 for c in checks if c.severity == "STUCK")

    print(f"{len(checks)} transitions checked")
    print(f"  OK: {n_ok}, PARTIAL: {n_partial}, STUCK: {n_stuck}")
    for c in checks:
        if c.severity != "OK":
            print(f"  ! Run {c.run_before}->{c.run_after}: {c.severity} -- {c.reason}")

    if n_stuck > 0:
        print(f"  WARNING: {n_stuck} stuck transitions detected!")
    else:
        print("  All transitions OK.")"""))

    # ==============================================================
    # 8. Process Plateau Turns
    # ==============================================================
    cells.append(md("s8-hdr", """---
## 8. Process Plateau Turns

Process **plateau turns only** with the full OPTIONS, group by supercycle,
apply settling window and MAD sigma-clip."""))

    cells.append(code("s8-pipeline", """ANALYSIS_LABELS = {"injection", "flat-mid", "flat-high"}
is_analysis = np.array([l in ANALYSIS_LABELS for l in turn_label])
plateau_indices = np.where(is_analysis)[0]
print(f"Processing {len(plateau_indices)} plateau turns (OPTIONS={OPTIONS})")

result, C_merged, C_units, ok_main = process_kn_pipeline(
    flux_abs_turns=flux_abs_all[plateau_indices],
    flux_cmp_turns=flux_cmp_all[plateau_indices],
    t_turns=t_all[plateau_indices],
    I_turns=I_all[plateau_indices],
    kn=kn, r_ref=R_REF, magnet_order=m,
    options=OPTIONS, min_b1_T=MIN_B1_T,
)

extra = [
    {
        "global_turn": int(plateau_indices[t]),
        "label": str(turn_label[plateau_indices[t]]),
        "I_range_A": float(I_range[plateau_indices[t]]),
    }
    for t in range(len(plateau_indices))
]

rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
df = pd.DataFrame(rows)

# Group by supercycle
df["sc_idx"] = -1
settled_idx = []

for gi, (gs, ge) in enumerate(inj_groups):
    group_globals = set(range(gs, ge + 1))
    gmask = df["global_turn"].isin(group_globals) & (df["label"] == "injection")
    df.loc[gmask, "sc_idx"] = gi
    group_rows = df.index[gmask]
    if N_LAST_TURNS_INJ is not None and len(group_rows) > N_LAST_TURNS_INJ:
        settled_idx.extend(group_rows[-N_LAST_TURNS_INJ:])
    else:
        settled_idx.extend(group_rows)

for gi, (gs, ge) in enumerate(fh_groups):
    group_globals = set(range(gs, ge + 1))
    gmask = df["global_turn"].isin(group_globals) & (df["label"] == "flat-high")
    df.loc[gmask, "sc_idx"] = gi
    group_rows = df.index[gmask]
    if N_LAST_TURNS_HIGH is not None and len(group_rows) > N_LAST_TURNS_HIGH:
        settled_idx.extend(group_rows[-N_LAST_TURNS_HIGH:])
    else:
        settled_idx.extend(group_rows)

df_settled = df.loc[sorted(settled_idx)].copy()

n_before = len(df_settled)
df_settled, clip_info = mad_sigma_clip(df_settled, "B1_T", N_SIGMA_CLIP, label_col="label")
n_clipped = n_before - len(df_settled)
if n_clipped > 0:
    print(f"  Sigma clip ({N_SIGMA_CLIP} MAD sigma): removed {n_clipped} turns ({clip_info})")

# Compute transfer function
df["TF_TperkA"] = df["B1_T"] / (df["I_mean_A"] / 1000.0)
df_settled["TF_TperkA"] = df_settled["B1_T"] / (df_settled["I_mean_A"] / 1000.0)

print(f"\\nAll plateau turns: {len(df)}")
print(f"Settled turns: {len(df_settled)}")
for lab in ["injection", "flat-high"]:
    n_all = len(df[df["label"] == lab])
    n_set = len(df_settled[df_settled["label"] == lab])
    print(f"  {lab:12s}: {n_all} -> {n_set}")
print(f"ok_main: {df['ok_main'].sum()} / {len(df)}")
print(f"Harmonics: n=1..{H}")"""))

    # ==============================================================
    # 9. All-Turn Harmonics vs Time
    # ==============================================================
    cells.append(md("s9-hdr", """---
## 9. All-Turn Harmonics vs Time

Process **all turns** (including ramps) to show B1, b2, b3 evolution
across the full measurement window."""))

    cells.append(code("s9-allturn", """result_all, C_merged_all, C_units_all, ok_main_all = process_kn_pipeline(
    flux_abs_turns=flux_abs_all, flux_cmp_turns=flux_cmp_all,
    t_turns=t_all, I_turns=I_all,
    kn=kn, r_ref=R_REF, magnet_order=m,
    options=OPTIONS, min_b1_T=MIN_B1_T,
)

extra_all = [{"global_turn": int(i), "label": str(turn_label[i])} for i in range(n_turns)]
rows_all = build_harmonic_rows(result_all, C_merged_all, C_units_all, ok_main_all, m, extra_all)
df_all = pd.DataFrame(rows_all)
df_all["t_mean_s"] = t_mean
print(f"All-turn: {n_turns} turns processed, ok_main={ok_main_all.sum()}")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
ok_all = df_all["ok_main"]
for ax, (col, ylabel) in zip(axes, [("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
    ramp = df_all["label"] == "ramp"
    ax.scatter(df_all.loc[ok_all & ramp, "t_mean_s"], df_all.loc[ok_all & ramp, col],
               s=4, alpha=0.3, color="lightgrey", zorder=0, label="ramp")
    for lab, lc in label_colors.items():
        mask = ok_all & (df_all["label"] == lab)
        if mask.sum() > 0:
            ax.scatter(df_all.loc[mask, "t_mean_s"], df_all.loc[mask, col],
                       s=6, alpha=0.5, color=lc, zorder=2, label=lab)
    ax.set_ylabel(ylabel); ax.legend(fontsize=7, loc="upper right")
axes[0].set_title(f"All-turn evolution -- {SEGMENT}")
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"All-Turn Harmonics vs Time -- {SESSION}", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 10. FFMM Golden Standard (conditional)
    # ==============================================================
    cells.append(md("s10-hdr", """---
## 10. FFMM Golden Standard Validation

Compare our pipeline output against the FFMM per-turn results (if available).
Uses `legacy_rotate_excludes_last=True` for FFMM parity."""))

    cells.append(code("s10-ffmm", """RESULTS_PAT = re.compile(r"Run_\\d+_I_[\\d.]+A_(N?CS)_results\\.txt$")
ffmm_files = [f for f in sorted(RUN_DIR.iterdir())
              if RESULTS_PAT.search(f.name) and f"_{SEGMENT}_" in f.name]

if not ffmm_files:
    print("No FFMM per-turn results found -- skipping FFMM validation.")
else:
    OPTIONS_FFMM = ("dri", "rot")
    FFMM_ROTATE_EXCLUDES_LAST = True

    print("=" * 70)
    print("FFMM GOLDEN STANDARD COMPARISON")
    print(f"FFMM options: dri rot nor  ->  our pipeline: {OPTIONS_FFMM}")
    print("=" * 70)

    ffmm_df = pd.read_csv(ffmm_files[0], sep="\\t")
    print(f"FFMM per-turn: {ffmm_files[0].name}, {len(ffmm_df)} rows")

    result_cmp, C_merged_cmp, C_units_cmp, ok_main_cmp = process_kn_pipeline(
        flux_abs_turns=flux_abs_all, flux_cmp_turns=flux_cmp_all,
        t_turns=t_all, I_turns=I_all,
        kn=kn, r_ref=R_REF, magnet_order=m,
        options=OPTIONS_FFMM, min_b1_T=MIN_B1_T,
        legacy_rotate_excludes_last=FFMM_ROTATE_EXCLUDES_LAST,
    )
    extra_ffmm = [{"global_turn": int(i)} for i in range(n_turns)]
    rows_ffmm = build_harmonic_rows(result_cmp, C_merged_cmp, C_units_cmp, ok_main_cmp, m, extra_ffmm)
    our_df = pd.DataFrame(rows_ffmm)

    assert len(our_df) == len(ffmm_df), f"Turn count mismatch: {len(our_df)} vs {len(ffmm_df)}"

    ok_idx = our_df.index[our_df["ok_main"]].values
    print(f"Comparing {len(ok_idx)} / {n_turns} turns (ok_main=True)")

    our_bmain = our_df.loc[ok_idx, "B1_T"].values
    ffmm_bmain = ffmm_df.loc[ok_idx, "B_main(T)"].values
    rms_bmain = np.sqrt(np.mean((our_bmain - ffmm_bmain)**2))
    print(f"B_main: RMS diff = {rms_bmain:.4e} T")

    print(f"\\n  {'n':>3s}  {'RMS(bn)':>10s}  {'max(bn)':>10s}  {'RMS(an)':>10s}  {'max(an)':>10s}")
    print("  " + "-" * 50)
    for n in range(2, H + 1):
        bn_ours = our_df.loc[ok_idx, f"b{n}_units"].values
        bn_ffmm = ffmm_df.loc[ok_idx, f"b{n}(Units)"].values
        an_ours = our_df.loc[ok_idx, f"a{n}_units"].values
        an_ffmm = ffmm_df.loc[ok_idx, f"a{n}(Units)"].values
        print(f"  {n:3d}  {np.sqrt(np.mean((bn_ours-bn_ffmm)**2)):10.4f}  "
              f"{np.max(np.abs(bn_ours-bn_ffmm)):10.4f}  "
              f"{np.sqrt(np.mean((an_ours-an_ffmm)**2)):10.4f}  "
              f"{np.max(np.abs(an_ours-an_ffmm)):10.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ffmm_bmain, our_bmain, s=4, alpha=0.3, color="steelblue")
    lims = [min(ffmm_bmain.min(), our_bmain.min()), max(ffmm_bmain.max(), our_bmain.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("FFMM B_main (T)"); ax.set_ylabel("Our B1 (T)")
    ax.set_title(f"B_main parity -- {SEGMENT}"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 11. Main Field (B1)
    # ==============================================================
    cells.append(md("s11-hdr", "---\n## 11. Main Field (B1)"))

    cells.append(code("s11-b1", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ok = df["ok_main"]
axes[0, 0].scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "B1_T"], s=8, alpha=0.5, color="steelblue")
axes[0, 0].set_xlabel("I (A)"); axes[0, 0].set_ylabel("B1 (T)")
axes[0, 0].set_title(f"B1 vs current ({SEGMENT})")

axes[0, 1].plot(df.loc[ok, "time_s"].values, df.loc[ok, "B1_T"].values,
        ".-", markersize=2, linewidth=0.3, color="steelblue")
axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("B1 (T)")
axes[0, 1].set_title(f"B1 time series ({SEGMENT})")

for ax_idx, (lab, col, marker, title) in enumerate([
        ("injection", "tab:green", "o", "B1 per SC (injection)"),
        ("flat-high", "tab:blue", "s", "B1 per SC (flat-top)")]):
    ax = axes[1, ax_idx]
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        sc_avg = sub.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt=f"{marker}-", markersize=4, capsize=2, color=col)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)"); ax.set_title(title)

fig.suptitle(f"Main Field (B1) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\\nB1 per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'mean (T)':>12s} {'std (T)':>12s}")
print("-" * 45)
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        print(f"{lab:>12s} {len(sub):5d} {sub['B1_T'].mean():+12.6f} {sub['B1_T'].std():12.6f}")"""))

    # ==============================================================
    # 12. b2 (Quadrupole)
    # ==============================================================
    cells.append(md("s12-hdr", "---\n## 12. b2 (Quadrupole)\n\nFirst allowed harmonic error for a dipole."))

    cells.append(code("s12-b2", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "b2_units"], s=8, alpha=0.5, color="steelblue")
axes[0, 0].axhline(0, color="grey", linewidth=0.5)
axes[0, 0].set_xlabel("I (A)"); axes[0, 0].set_ylabel("b2 (units)"); axes[0, 0].set_title(f"b2 vs current")

axes[0, 1].plot(df.loc[ok, "time_s"].values, df.loc[ok, "b2_units"].values,
        ".-", markersize=2, linewidth=0.3, color="steelblue")
axes[0, 1].axhline(0, color="grey", linewidth=0.5)
axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("b2 (units)"); axes[0, 1].set_title("b2 time series")

ax = axes[1, 0]
for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        sc_avg = sub.groupby("sc_idx")["b2_units"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("SC index"); ax.set_ylabel("b2 (units)")
ax.set_title("b2 per supercycle (settled)"); ax.legend(fontsize=9)

ax = axes[1, 1]
for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        ax.hist(sub["b2_units"], bins=30, alpha=0.5, color=col, label=lab, edgecolor="black", linewidth=0.5)
ax.set_xlabel("b2 (units)"); ax.set_ylabel("Count")
ax.set_title("b2 distribution (settled)"); ax.legend(fontsize=9)

fig.suptitle(f"b2 (Quadrupole) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 13. b3 (Sextupole)
    # ==============================================================
    cells.append(md("s13-hdr", "---\n## 13. b3 (Sextupole)\n\nFirst non-allowed harmonic -- key quality indicator."))

    cells.append(code("s13-b3", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "b3_units"], s=8, alpha=0.5, color="steelblue")
axes[0, 0].axhline(0, color="grey", linewidth=0.5)
axes[0, 0].set_xlabel("I (A)"); axes[0, 0].set_ylabel("b3 (units)"); axes[0, 0].set_title("b3 vs current")

axes[0, 1].plot(df.loc[ok, "time_s"].values, df.loc[ok, "b3_units"].values,
        ".-", markersize=2, linewidth=0.3, color="steelblue")
axes[0, 1].axhline(0, color="grey", linewidth=0.5)
axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("b3 (units)"); axes[0, 1].set_title("b3 time series")

ax = axes[1, 0]
for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        sc_avg = sub.groupby("sc_idx")["b3_units"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("SC index"); ax.set_ylabel("b3 (units)")
ax.set_title("b3 per supercycle (settled)"); ax.legend(fontsize=9)

ax = axes[1, 1]
for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        ax.hist(sub["b3_units"], bins=30, alpha=0.5, color=col, label=lab, edgecolor="black", linewidth=0.5)
ax.set_xlabel("b3 (units)"); ax.set_ylabel("Count")
ax.set_title("b3 distribution (settled)"); ax.legend(fontsize=9)

fig.suptitle(f"b3 (Sextupole) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Per-supercycle evolution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col_name, ylabel) in zip(axes, [("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")[col_name].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, alpha=0.8, label=lab)
    ax.set_xlabel("SC index"); ax.set_ylabel(ylabel); ax.legend(fontsize=9)
axes[0].set_title("B1 per SC"); axes[1].set_title("b2 per SC"); axes[2].set_title("b3 per SC")
fig.suptitle(f"Per-Supercycle Evolution (settled) -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\\nStability across supercycles (settled turns):")
print(f"{'Quantity':>12s}  {'Label':>12s}  {'SC mean':>12s}  {'SC std':>12s}  {'SC p-p':>12s}")
print("-" * 65)
for col_name, label in [("B1_T", "B1"), ("b2_units", "b2"), ("b3_units", "b3")]:
    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_means = sub.groupby("sc_idx")[col_name].mean()
            print(f"{label:>12s}  {lab:>12s}  {sc_means.mean():+12.6f}  "
                  f"{sc_means.std():12.6f}  {sc_means.max()-sc_means.min():12.6f}")"""))

    # ==============================================================
    # 14. Higher Harmonics Overview
    # ==============================================================
    cells.append(md("s14-hdr", """---
## 14. Higher Harmonics Overview

Summary statistics for all harmonics b4..bH and a2..aH."""))

    cells.append(code("s14-higher", """ok_s = df_settled["ok_main"]

for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok_s]
    if len(sub) == 0: continue
    print(f"\\n=== {lab.upper()} (N={len(sub)} settled turns) ===")
    print(f"  {'n':>3s} {'bn mean':>10s} {'bn std':>10s} {'an mean':>10s} {'an std':>10s}")
    print("  " + "-" * 50)
    for n in range(2, H + 1):
        bn_col = f"b{n}_units"; an_col = f"a{n}_units"
        if bn_col in sub.columns:
            bn_m = sub[bn_col].mean(); bn_s = sub[bn_col].std()
            an_m = sub[an_col].mean(); an_s = sub[an_col].std()
            flag = " *" if abs(bn_m) > 2 * bn_s and abs(bn_m) > 0.5 else ""
            print(f"  {n:3d} {bn_m:+10.4f} {bn_s:10.4f} {an_m:+10.4f} {an_s:10.4f}{flag}")
    print("  (* = |mean| > 2*std and |mean| > 0.5 units)")"""))

    # ==============================================================
    # 15. Multipole Spectrum
    # ==============================================================
    cells.append(md("s15-hdr", "---\n## 15. Multipole Spectrum"))

    cells.append(code("s15-spectrum", """operating_points = {}
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok_s]
    if len(sub) > 0:
        operating_points[lab] = sub

n_ops = len(operating_points)
if n_ops > 0:
    fig, axes = plt.subplots(n_ops, 2, figsize=(16, 5 * n_ops))
    if n_ops == 1:
        axes = axes[np.newaxis, :]

    for i, (lab, sub) in enumerate(operating_points.items()):
        orders = list(range(2, H + 1))
        bn_means = [sub[f"b{n}_units"].mean() for n in orders]
        an_means = [sub[f"a{n}_units"].mean() for n in orders]
        bn_stds = [sub[f"b{n}_units"].std() for n in orders]
        an_stds = [sub[f"a{n}_units"].std() for n in orders]

        x = np.arange(len(orders)); w = 0.35
        axes[i, 0].bar(x - w/2, bn_means, w, yerr=bn_stds, label="bn",
                        color="steelblue", capsize=2, alpha=0.8)
        axes[i, 0].bar(x + w/2, an_means, w, yerr=an_stds, label="an",
                        color="tab:orange", capsize=2, alpha=0.8)
        axes[i, 0].axhline(0, color="grey", linewidth=0.5)
        axes[i, 0].set_xticks(x); axes[i, 0].set_xticklabels(orders)
        axes[i, 0].set_xlabel("n"); axes[i, 0].set_ylabel("Units")
        axes[i, 0].set_title(f"Spectrum -- {lab} (linear)"); axes[i, 0].legend(fontsize=8)

        axes[i, 1].bar(x - w/2, np.abs(bn_means), w, label="|bn|", color="steelblue", alpha=0.8)
        axes[i, 1].bar(x + w/2, np.abs(an_means), w, label="|an|", color="tab:orange", alpha=0.8)
        axes[i, 1].set_yscale("log")
        axes[i, 1].set_xticks(x); axes[i, 1].set_xticklabels(orders)
        axes[i, 1].set_xlabel("n"); axes[i, 1].set_ylabel("|Units|")
        axes[i, 1].set_title(f"Spectrum -- {lab} (log)"); axes[i, 1].legend(fontsize=8)

    fig.suptitle(f"Multipole Spectrum -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 16. Transfer Function B1/I
    # ==============================================================
    cells.append(md("s16-hdr", "---\n## 16. Transfer Function B1/I"))

    cells.append(code("s16-tf", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ok_tf = df_settled["ok_main"] & (df_settled["I_mean_A"].abs() > 10)
df_tf = df_settled[ok_tf]

axes[0].scatter(df_tf["I_mean_A"], df_tf["B1_T"], s=8, alpha=0.5, color="steelblue")
axes[0].set_xlabel("I (A)"); axes[0].set_ylabel("B1 (T)"); axes[0].set_title("B1 vs I")

for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_tf[df_tf["label"] == lab]
    if len(sub) > 0:
        axes[1].scatter(sub["I_mean_A"], sub["TF_TperkA"], s=10, alpha=0.5, color=col, label=lab)
axes[1].set_xlabel("I (A)"); axes[1].set_ylabel("TF (T/kA)"); axes[1].set_title("TF vs I"); axes[1].legend(fontsize=9)

tf_summary = {}
for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
    sub = df_tf[df_tf["label"] == lab]
    if len(sub) > 0:
        sc_avg = sub.groupby("sc_idx")["TF_TperkA"].agg(["mean", "std"]).reset_index()
        axes[2].errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
        tf_summary[lab] = sc_avg
axes[2].set_xlabel("SC index"); axes[2].set_ylabel("TF (T/kA)")
axes[2].set_title("TF per supercycle"); axes[2].legend(fontsize=9)

fig.suptitle(f"Transfer Function -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\\nTF per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'mean (T/kA)':>14s} {'std':>10s}")
print("-" * 45)
for lab in ["injection", "flat-high"]:
    sub = df_tf[df_tf["label"] == lab]
    if len(sub) > 0:
        print(f"{lab:>12s} {len(sub):5d} {sub['TF_TperkA'].mean():14.4f} {sub['TF_TperkA'].std():10.4f}")"""))

    # ==============================================================
    # 17. Apparent vs Differential Inductance
    # ==============================================================
    cells.append(md("s17-hdr", """---
## 17. Apparent vs Differential Inductance

**L_app** = B1/I, **L_d** = dB1/dI.  If L_d < L_app(FT), iron is saturated."""))

    cells.append(code("s17-inductance", """df_inj = df_settled[(df_settled["label"] == "injection") & df_settled["ok_main"]]
df_fh = df_settled[(df_settled["label"] == "flat-high") & df_settled["ok_main"]]

ld_df = pd.DataFrame()
if len(df_inj) > 0 and len(df_fh) > 0:
    inj_avg = df_inj.groupby("sc_idx").agg(
        B1_inj=("B1_T", "mean"), I_inj=("I_mean_A", "mean")).reset_index()
    fh_avg = df_fh.groupby("sc_idx").agg(
        B1_fh=("B1_T", "mean"), I_fh=("I_mean_A", "mean")).reset_index()
    merged = inj_avg.merge(fh_avg, on="sc_idx", how="inner")
    if len(merged) > 0:
        merged["Ld_TperkA"] = (merged["B1_fh"] - merged["B1_inj"]) / ((merged["I_fh"] - merged["I_inj"]) / 1000.0)
        ld_df = merged
        print(f"Ld: {len(merged)} SC pairs, mean = {merged['Ld_TperkA'].mean():.4f} +/- {merged['Ld_TperkA'].std():.4f} T/kA")

fig, ax = plt.subplots(figsize=(10, 5))
for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
    if lab in tf_summary:
        sc = tf_summary[lab]
        ax.errorbar(sc["sc_idx"], sc["mean"], yerr=sc["std"],
                    fmt=f"{marker}-", markersize=4, capsize=2, color=col, alpha=0.8, label=f"L_app ({lab})")
if len(ld_df) > 0:
    ax.plot(ld_df["sc_idx"], ld_df["Ld_TperkA"], "D-", markersize=4, color="tab:red", alpha=0.8, label="Ld")
ax.set_xlabel("SC index"); ax.set_ylabel("T/kA")
ax.set_title("L_app vs Ld -- Saturation Check"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

if len(ld_df) > 0 and "flat-high" in tf_summary:
    Ld_mean = ld_df["Ld_TperkA"].mean()
    Lapp_fh_mean = tf_summary["flat-high"]["mean"].mean()
    ratio = Ld_mean / Lapp_fh_mean
    verdict = "SATURATED" if ratio < 0.99 else "LINEAR"
    print(f"\\nSaturation: Ld={Ld_mean:.4f}, L_app(FT)={Lapp_fh_mean:.4f}, ratio={ratio:.4f} -> {verdict}")"""))

    # ==============================================================
    # 18. Raw Settling Curves
    # ==============================================================
    cells.append(md("s18-hdr", """---
## 18. Raw Settling Curves

B1 and b3 vs turn within each injection supercycle, overlaid."""))

    cells.append(code("s18-settling", """# Build per-supercycle injection data from plateau results
inj = df[df["label"] == "injection"].copy()
if len(inj) > 0:
    inj["t_mean_s"] = t_mean[inj["global_turn"].values]
    for sc_id in inj["sc_idx"].unique():
        if sc_id < 0: continue
        mask = inj["sc_idx"] == sc_id
        t0 = inj.loc[mask, "t_mean_s"].min()
        inj.loc[mask, "t_since_inj_start"] = inj.loc[mask, "t_mean_s"] - t0
        inj.loc[mask, "turn_in_group"] = np.arange(mask.sum())

    eddy_data = inj
    n_sc = len([s for s in inj["sc_idx"].unique() if s >= 0])
    print(f"{len(inj)} injection turns across {n_sc} supercycles")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc_ids = sorted([s for s in inj["sc_idx"].unique() if s >= 0])
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(sc_ids), 1)))
    for col_idx, (col, ylabel) in enumerate([("B1_T", "B1 (T)"), ("b3_units", "b3 (units)")]):
        ax = axes[col_idx]
        for k, sc_id in enumerate(sc_ids):
            sub = inj[inj["sc_idx"] == sc_id]
            ax.plot(sub["t_since_inj_start"], sub[col], ".-",
                    markersize=4, linewidth=0.8, alpha=0.7, color=cmap[k % len(cmap)])
        ax.set_xlabel("t - t_inj_start (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} settling -- {SEGMENT}")
    fig.suptitle("Injection Settling Curves -- Supercycle Overlay", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    eddy_data = pd.DataFrame()
    print("No injection data for settling analysis.")"""))

    # ==============================================================
    # 19. Exponential Fits
    # ==============================================================
    cells.append(md("s19-hdr", "---\n## 19. Exponential Fits\n\nFit b3(t) = b3_inf + A * exp(-t/tau) per supercycle."))

    cells.append(code("s19-fits", """def fit_supercycle(df_sc):
    t = df_sc["t_since_inj_start"].values
    b3 = df_sc["b3_units"].values
    ok = np.isfinite(b3) & np.isfinite(t)
    t, b3 = t[ok], b3[ok]
    if len(t) < MIN_INJECTION_TURNS:
        return None
    try:
        popt, pcov = curve_fit(
            eddy_model, t, b3,
            p0=[b3[-1], b3[0] - b3[-1], max(t[-1] / 3, 1.0)],
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 1000]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        b3_pred = eddy_model(t, *popt)
        ss_res = np.sum((b3 - b3_pred) ** 2)
        ss_tot = np.sum((b3 - b3.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"b3_inf": popt[0], "A": popt[1], "tau": popt[2],
                "b3_inf_err": perr[0], "A_err": perr[1], "tau_err": perr[2],
                "r2": r2, "n_turns": len(t)}
    except (RuntimeError, ValueError):
        return None

fits = []
df_eddy_fits = pd.DataFrame()

if len(eddy_data) > 0:
    for sc_id in sorted(eddy_data["sc_idx"].unique()):
        if sc_id < 0: continue
        result = fit_supercycle(eddy_data[eddy_data["sc_idx"] == sc_id])
        if result is not None:
            result["supercycle_id"] = sc_id
            fits.append(result)

    df_eddy_fits = pd.DataFrame(fits)
    print(f"Fitted {len(fits)} supercycles")

    if len(df_eddy_fits) > 0:
        for _, row in df_eddy_fits.iterrows():
            print(f"  SC {int(row['supercycle_id']):3d}: tau={row['tau']:.2f}s, "
                  f"A={row['A']:+.4f}, b3_inf={row['b3_inf']:+.4f}, R2={row['r2']:.3f}")

        # Fit overlay on representative supercycles
        n_show = min(3, len(df_eddy_fits))
        fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
        if n_show == 1:
            axes = [axes]
        sc_ids_fit = df_eddy_fits["supercycle_id"].values
        show_ids = sc_ids_fit[np.linspace(0, len(sc_ids_fit)-1, n_show, dtype=int)]
        for j, sc_id in enumerate(show_ids):
            ax = axes[j]
            sub = eddy_data[eddy_data["sc_idx"] == sc_id]
            fit_row = df_eddy_fits[df_eddy_fits["supercycle_id"] == sc_id].iloc[0]
            ax.scatter(sub["t_since_inj_start"], sub["b3_units"], s=15, alpha=0.7, color="tab:blue", label="data")
            t_fit = np.linspace(0, sub["t_since_inj_start"].max() * 1.05, 200)
            ax.plot(t_fit, eddy_model(t_fit, fit_row["b3_inf"], fit_row["A"], fit_row["tau"]),
                    "r-", linewidth=1.5, label="fit")
            ax.set_title(f"SC {int(sc_id)}: tau={fit_row['tau']:.1f}s, R2={fit_row['r2']:.3f}", fontsize=9)
            ax.set_xlabel("t (s)"); ax.legend(fontsize=7)
            if j == 0: ax.set_ylabel("b3 (units)")
        fig.suptitle("Exponential Fit -- Representative Supercycles", fontsize=13, y=1.02)
        plt.tight_layout(); plt.show()

        tau_v = df_eddy_fits["tau"].values
        print(f"\\nTau: mean={tau_v.mean():.2f} +/- {tau_v.std():.2f} s, "
              f"median={np.median(tau_v):.2f} s, R2 mean={df_eddy_fits['r2'].mean():.3f}")
else:
    print("No eddy data for fitting.")"""))

    # ==============================================================
    # 20. Settling Bias Analysis
    # ==============================================================
    cells.append(md("s20-hdr", "---\n## 20. Settling Bias Analysis"))

    cells.append(code("s20-bias", """if len(eddy_data) > 0 and "turn_in_group" in eddy_data.columns:
    sc_ids = sorted([s for s in eddy_data["sc_idx"].unique() if s >= 0])
    max_turns_per_sc = eddy_data.groupby("sc_idx").size().min()
    n_last_values = list(range(1, max_turns_per_sc + 1))

    bias_b3, bias_b2 = [], []
    for n_last in n_last_values:
        b3_means, b2_means = [], []
        for sc_id in sc_ids:
            sub = eddy_data[eddy_data["sc_idx"] == sc_id].sort_values("turn_in_group")
            tail = sub.tail(n_last)
            if len(tail) > 0 and tail["ok_main"].any():
                ok_tail = tail[tail["ok_main"]]
                b3_means.append(ok_tail["b3_units"].mean())
                b2_means.append(ok_tail["b2_units"].mean())
        bias_b3.append(np.mean(b3_means) if b3_means else np.nan)
        bias_b2.append(np.mean(b2_means) if b2_means else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(n_last_values, bias_b3, "o-", markersize=4, color="tab:blue")
    axes[0].axhline(bias_b3[-1], color="grey", linestyle="--", linewidth=0.8,
                     label=f"converged = {bias_b3[-1]:.3f}")
    axes[0].set_xlabel("N_LAST"); axes[0].set_ylabel("b3 mean (units)")
    axes[0].set_title("b3 bias vs averaging window"); axes[0].legend(fontsize=9)

    axes[1].plot(n_last_values, bias_b2, "o-", markersize=4, color="tab:orange")
    axes[1].axhline(bias_b2[-1], color="grey", linestyle="--", linewidth=0.8,
                     label=f"converged = {bias_b2[-1]:.3f}")
    axes[1].set_xlabel("N_LAST"); axes[1].set_ylabel("b2 mean (units)")
    axes[1].set_title("b2 bias vs averaging window"); axes[1].legend(fontsize=9)

    fig.suptitle("Settling Bias Analysis", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for bias analysis.")"""))

    # ==============================================================
    # 21. N_LAST Sensitivity Study
    # ==============================================================
    cells.append(md("s21-hdr", "---\n## 21. N_LAST Sensitivity Study"))

    cells.append(code("s21-nlast", """inj_all = df[df["label"] == "injection"].copy()

if len(inj_all) > 0:
    turns_per_sc = inj_all.groupby("sc_idx").size()
    max_n_last = int(turns_per_sc.min())
    n_last_scan = list(range(1, max_n_last + 1))

    scan_results = {"B1_T": [], "b2_units": [], "b3_units": []}
    for n_last in n_last_scan:
        settled_idx_scan = []
        for sc_id in inj_all["sc_idx"].unique():
            if sc_id < 0: continue
            group_rows = inj_all.index[inj_all["sc_idx"] == sc_id]
            if len(group_rows) > n_last:
                settled_idx_scan.extend(group_rows[-n_last:])
            else:
                settled_idx_scan.extend(group_rows)
        sub = inj_all.loc[sorted(settled_idx_scan)]
        sub_ok = sub[sub["ok_main"]]
        for col in ["B1_T", "b2_units", "b3_units"]:
            scan_results[col].append(sub_ok[col].mean() if len(sub_ok) > 0 else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, ylabel, color) in zip(axes, [
            ("B1_T", "B1 (T)", "steelblue"),
            ("b2_units", "b2 (units)", "tab:orange"),
            ("b3_units", "b3 (units)", "tab:green")]):
        ax.plot(n_last_scan, scan_results[col], "o-", markersize=3, color=color)
        ax.axvline(N_LAST_TURNS_INJ, color="red", linestyle="--", linewidth=1,
                    label=f"N_LAST={N_LAST_TURNS_INJ}")
        ax.set_xlabel("N_LAST"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} vs N_LAST (inj)"); ax.legend(fontsize=8)
    fig.suptitle("N_LAST Sensitivity -- Injection", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for N_LAST study.")"""))

    # ==============================================================
    # 22. Comprehensive Statistics Table
    # ==============================================================
    cells.append(md("s22-hdr", "---\n## 22. Comprehensive Statistics Table"))

    cells.append(code("s22-stats", f"""print("=" * 70)
print(f"SPS MBB DIPOLE -- COMPREHENSIVE NCS ANALYSIS ({energy_label} Extended)")
print("=" * 70)

print(f"\\nMeasurement  : {{SESSION}}")
print(f"Segment      : {{SEGMENT}}")
print(f"Kn file      : {{KN_PATH.name}} (cross-session)")
print(f"Options      : {{OPTIONS}}")
print(f"cel/fed      : {{diag.recommendation}}")

print(f"\\n--- Data Summary ---")
print(f"Total turns   : {{n_turns}}")
print(f"Plateau turns : {{is_plateau.sum()}}")
print(f"Injection     : {{(turn_label == 'injection').sum()}} turns, {{len(inj_groups)}} supercycles")
print(f"Flat-high     : {{(turn_label == 'flat-high').sum()}} turns, {{len(fh_groups)}} groups")

print(f"\\n--- Settled Turns ---")
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1e3)
        print(f"  {{lab:12s}}: N={{len(sub):4d}}, I={{sub['I_mean_A'].mean():.1f}} A, "
              f"B1={{sub['B1_T'].mean():+.6f}} T, "
              f"b2={{sub['b2_units'].mean():+.3f}}, b3={{sub['b3_units'].mean():+.3f}} units, "
              f"TF={{tf:.4f}} T/kA")

if len(df_eddy_fits) > 0:
    tau_v = df_eddy_fits["tau"].values
    print(f"\\nEddy tau  : {{tau_v.mean():.2f}} +/- {{tau_v.std():.2f}} s (N={{len(df_eddy_fits)}})")

if len(ld_df) > 0:
    print(f"Ld (diff) : {{ld_df['Ld_TperkA'].mean():.4f}} +/- {{ld_df['Ld_TperkA'].std():.4f}} T/kA")"""))

    # ==============================================================
    # 23. Analysis Choices Summary
    # ==============================================================
    cells.append(md("s23-hdr", "---\n## 23. Analysis Choices Summary"))

    cells.append(code("s23-choices", f"""import datetime

print("ANALYSIS CHOICES")
print("=" * 60)
print(f"Generated    : {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}}")
print(f"Session      : {{SESSION}}")
print(f"Segment      : {{SEGMENT}}")
print(f"Energy label : {energy_label}")
print(f"Magnet order : {{MAGNET_ORDER}} (dipole)")
print(f"R_ref        : {{R_REF}} m")
print(f"Samples/turn : {{SAMPLES_PER_TURN}}")
print(f"Kn file      : {{KN_PATH.name}} (cross-session)")
print(f"OPTIONS      : {{OPTIONS}}")
print(f"cel/fed diag : {{diag.recommendation}}")
print(f"MIN_B1_T     : {{MIN_B1_T}}")
print(f"PLATEAU_I_RANGE_MAX : {{PLATEAU_I_RANGE_MAX}} A")
print(f"N_BLOCKS     : {{N_BLOCKS}}")
print(f"N_LAST_TURNS_INJ    : {{N_LAST_TURNS_INJ}}")
print(f"N_LAST_TURNS_HIGH   : {{N_LAST_TURNS_HIGH}}")
print(f"N_SIGMA_CLIP        : {{N_SIGMA_CLIP}}")
print(f"MIN_INJECTION_TURNS : {{MIN_INJECTION_TURNS}}")"""))

    # ==============================================================
    # 24. CSV Export
    # ==============================================================
    cells.append(md("s24-hdr", "---\n## 24. CSV Export"))

    cells.append(code("s24-export", f"""out_dir = REPO_ROOT / "output" / "MBB/2026-02-06_supercycle" / "{out_subdir}"
out_dir.mkdir(parents=True, exist_ok=True)

fname = f"MBB_{{SEGMENT}}_streaming_plateau.csv"
df.to_csv(out_dir / fname, index=False)
print(f"Wrote {{out_dir / fname}}  ({{len(df)}} rows)")

fname_s = f"MBB_{{SEGMENT}}_streaming_settled.csv"
df_settled.to_csv(out_dir / fname_s, index=False)
print(f"Wrote {{out_dir / fname_s}}  ({{len(df_settled)}} rows)")

# Eddy current CSVs
if len(eddy_data) > 0:
    eddy_data.to_csv(out_dir / "b3_injection_NCS.csv", index=False)
    print(f"Wrote b3_injection_NCS.csv  ({{len(eddy_data)}} rows)")

if len(df_eddy_fits) > 0:
    df_eddy_fits.to_csv(out_dir / "b3_fits_NCS.csv", index=False)
    print(f"Wrote b3_fits_NCS.csv  ({{len(df_eddy_fits)}} rows)")

# Inductance summary
ind_rows = []
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        ind_rows.append({{
            "level": lab, "N": len(sub),
            "I_mean_A": sub["I_mean_A"].mean(),
            "B1_T": sub["B1_T"].mean(), "B1_std": sub["B1_T"].std(),
            "TF_TperkA": sub["TF_TperkA"].mean(), "TF_std": sub["TF_TperkA"].std(),
        }})
if len(ld_df) > 0:
    ind_rows.append({{
        "level": "Ld_differential", "N": len(ld_df),
        "I_mean_A": np.nan, "B1_T": np.nan, "B1_std": np.nan,
        "TF_TperkA": ld_df["Ld_TperkA"].mean(), "TF_std": ld_df["Ld_TperkA"].std(),
    }})
if ind_rows:
    df_ind = pd.DataFrame(ind_rows)
    df_ind.to_csv(out_dir / "inductance_summary.csv", index=False)
    print(f"Wrote inductance_summary.csv  ({{len(df_ind)}} rows)")

print("\\nDone.")"""))

    return cells


# ================================================================
# Thin comparison notebook
# ================================================================

def build_comparison_cells():
    """Build thin comparison notebook (loads CSVs from both analyses)."""
    cells = []

    cells.append(md("title", """# B1, b2, b3 Comparison: 200 GeV vs 26 GeV MD1 (NCS Supercycle)

## Objective

Compare harmonics at injection (~301 A) and flat-top (~4815 A) for two
MD1 cycle energies.  Loads pre-computed CSVs from the comprehensive notebooks.

| Dataset | MD1 energy | Session |
|---------|-----------|---------|
| **200 GeV** | 200 GeV Extended | `01_200_extended` |
| **26 GeV** | 26 GeV Extended | `03_26_extended` |

| # | Section |
|---|---------|
| 1 | Configuration & Imports |
| 2 | Load Settled CSVs |
| 3 | B1 Comparison |
| 4 | b2, b3 Comparison |
| 5 | Multipole Spectrum Comparison |
| 6 | Statistical Significance |
| 7 | Summary |"""))

    # 1. Config
    cells.append(md("c1-hdr", "---\n## 1. Configuration & Imports"))

    cells.append(code("c1-config", """N_LAST_TURNS_INJ = 18
SEGMENT = "NCS"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib widget
plt.rcParams.update({"figure.figsize": (14, 5), "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists(): break
    REPO_ROOT = REPO_ROOT.parent

OUT_200 = REPO_ROOT / "output" / "MBB/2026-02-06_supercycle" / "01_200_extended"
OUT_26  = REPO_ROOT / "output" / "MBB/2026-02-06_supercycle" / "03_26_extended"
assert OUT_200.exists(), f"200 GeV output not found: {OUT_200}"
assert OUT_26.exists(), f"26 GeV output not found: {OUT_26}"

print("Comparison: 200 GeV vs 26 GeV (NCS Supercycle)")
print(f"  200 GeV CSVs: {OUT_200}")
print(f"  26 GeV  CSVs: {OUT_26}")"""))

    # 2. Load
    cells.append(md("c2-hdr", "---\n## 2. Load Settled CSVs"))

    cells.append(code("c2-load", """ds = {}
for name, out_dir in [("200 GeV", OUT_200), ("26 GeV", OUT_26)]:
    fname = f"MBB_{SEGMENT}_streaming_settled.csv"
    fpath = out_dir / fname
    assert fpath.exists(), f"Missing: {fpath}"
    df = pd.read_csv(fpath)
    ds[name] = df
    print(f"  {name}: {len(df)} settled turns")

# Load eddy fits if available
eddy_fits = {}
for name, out_dir in [("200 GeV", OUT_200), ("26 GeV", OUT_26)]:
    fpath = out_dir / "b3_fits_NCS.csv"
    if fpath.exists():
        eddy_fits[name] = pd.read_csv(fpath)
        print(f"  {name} eddy fits: {len(eddy_fits[name])} rows")
    else:
        eddy_fits[name] = pd.DataFrame()"""))

    # 3. B1
    cells.append(md("c3-hdr", "---\n## 3. B1 Comparison"))

    cells.append(code("c3-b1", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
    ax = axes[j]
    for ds_name, col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
        dfs = ds[ds_name]
        sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
        if len(sub) == 0: continue
        sc_avg = sub.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
    ax.set_xlabel("SC index"); ax.set_ylabel("B1 (T)")
    ax.set_title(f"B1 -- {title_suffix}"); ax.legend(fontsize=9)

fig.suptitle(f"B1 per Supercycle (settled)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 4. b2, b3
    cells.append(md("c4-hdr", "---\n## 4. b2, b3 Comparison"))

    cells.append(code("c4-harmonics", """for harm_name, harm_col, ylabel in [("b2", "b2_units", "b2 (units)"), ("b3", "b3_units", "b3 (units)")]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
        ax = axes[j]
        for ds_name, col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
            dfs = ds[ds_name]
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            sc_avg = sub.groupby("sc_idx")[harm_col].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("SC index"); ax.set_ylabel(ylabel)
        ax.set_title(f"{harm_name} -- {title_suffix}"); ax.legend(fontsize=9)
    fig.suptitle(f"{harm_name} per Supercycle (settled)", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()

# Box plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax_idx, (col_name, ylabel, title) in enumerate([
        ("B1_T", "B1 (T)", "B1"), ("b2_units", "b2 (units)", "b2"), ("b3_units", "b3 (units)", "b3")]):
    ax = axes[ax_idx]
    box_data, box_labels, box_colors = [], [], []
    for ds_name, base_col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
        for lab, short in [("injection", "Inj"), ("flat-high", "FT")]:
            dfs = ds[ds_name]
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            box_data.append(sub[col_name].values)
            box_labels.append(f"{ds_name}\\n{short}\\n(N={len(sub)})")
            box_colors.append(base_col)
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, col in zip(bp["boxes"], box_colors): patch.set_facecolor(col); patch.set_alpha(0.5)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", labelsize=7)
fig.suptitle("Distribution Comparison (settled turns)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 5. Spectrum
    cells.append(md("c5-hdr", "---\n## 5. Multipole Spectrum Comparison"))

    cells.append(code("c5-spectrum", """lab = "injection"
bn_cols = [c for c in ds["200 GeV"].columns if c.startswith("b") and c.endswith("_units")]
orders = sorted([int(c.replace("b", "").replace("_units", "")) for c in bn_cols])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(orders)); w = 0.35

for ax_idx, (title, yscale) in enumerate([("Linear", "linear"), ("Log", "log")]):
    ax = axes[ax_idx]
    for ds_name, offset, color in [("200 GeV", -w/2, "tab:blue"), ("26 GeV", w/2, "tab:orange")]:
        dfs = ds[ds_name]
        sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
        if len(sub) == 0: continue
        means = [sub[f"b{n}_units"].mean() for n in orders]
        if yscale == "log":
            means = [abs(v) for v in means]
        ax.bar(x + offset, means, w, label=ds_name, color=color, alpha=0.8)
    if yscale == "linear":
        ax.axhline(0, color="grey", linewidth=0.5)
    else:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(orders)
    ax.set_xlabel("n"); ax.set_ylabel("bn (units)" if yscale == "linear" else "|bn|")
    ax.set_title(f"Spectrum -- {lab} ({title})"); ax.legend(fontsize=9)

fig.suptitle("Multipole Spectrum Comparison -- NCS Injection", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 6. Stats
    cells.append(md("c6-hdr", "---\n## 6. Statistical Significance"))

    cells.append(code("c6-stats", """print(f"Difference: (200 GeV) - (26 GeV)  [settled]")
print("=" * 90)

for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
    s200 = ds["200 GeV"]; s200 = s200[(s200["label"] == lab) & s200["ok_main"]]
    s26 = ds["26 GeV"]; s26 = s26[(s26["label"] == lab) & s26["ok_main"]]
    if len(s200) == 0 or len(s26) == 0: continue

    print(f"\\n  {desc} (N: {len(s200)} vs {len(s26)} turns)")
    for name, col, unit in [("B1", "B1_T", "T"), ("b2", "b2_units", "units"), ("b3", "b3_units", "units")]:
        diff = s200[col].mean() - s26[col].mean()
        err = np.sqrt((s200[col].std()**2/len(s200)) + (s26[col].std()**2/len(s26)))
        sig = abs(diff) / err if err > 0 else 0
        verdict = "REAL (>3 sigma)" if sig > 3 else ("suggestive" if sig >= 2 else "no evidence")
        diff_str = f"{diff*1e6:+.1f} uT" if unit == "T" else f"{diff:+.4f} {unit}"
        print(f"    {name:>3s}: {diff_str:>16s}  ({sig:.1f} sigma) -> {verdict}")"""))

    # 7. Summary
    cells.append(md("c7-hdr", "---\n## 7. Summary"))

    cells.append(code("c7-summary", """summary_rows = []
for ds_name in ["200 GeV", "26 GeV"]:
    dfs = ds[ds_name]
    for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
        sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
        if len(sub) == 0: continue
        tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1000.0) if "I_mean_A" in sub.columns else np.nan
        summary_rows.append({
            "Dataset": ds_name, "Op. point": desc, "N turns": len(sub),
            "I mean (A)": f"{sub['I_mean_A'].mean():.1f}" if "I_mean_A" in sub.columns else "-",
            "B1 mean (T)": f"{sub['B1_T'].mean():.6f}",
            "b2 mean": f"{sub['b2_units'].mean():+.4f}",
            "b3 mean": f"{sub['b3_units'].mean():+.4f}",
            "TF (T/kA)": f"{tf:.4f}" if not np.isnan(tf) else "-",
        })
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

out_dir = REPO_ROOT / "output" / "MBB/2026-02-06_supercycle" / "compare_200_vs_26"
out_dir.mkdir(parents=True, exist_ok=True)
df_summary.to_csv(out_dir / "summary_comparison_settled.csv", index=False)
print(f"\\nWrote {out_dir / 'summary_comparison_settled.csv'}")
print("\\nDone.")"""))

    return cells


# ================================================================
# Generate all 3 notebooks
# ================================================================

if __name__ == "__main__":
    print("Generating MBB notebooks...")

    # 1. 200 GeV comprehensive analysis
    cells_200 = build_analysis_cells(
        session="MBB/2026-02-06_supercycle/01_200_extended/20260206_144537_SPS_MBB",
        meas_subdir="20260206_144559_MBB",
        energy_label="200 GeV",
        out_subdir="01_200_extended",
    )
    write_notebook(NOTEBOOK_DIR / "analysis" / "2026-02-06_01_200_extended_NCS.ipynb", cells_200)

    # 2. 26 GeV comprehensive analysis
    cells_26 = build_analysis_cells(
        session="MBB/2026-02-06_supercycle/03_26_extended/20260206_151808_SPS_MBB",
        meas_subdir="20260206_151827_MBB",
        energy_label="26 GeV",
        out_subdir="03_26_extended",
    )
    write_notebook(NOTEBOOK_DIR / "analysis" / "2026-02-06_03_26_extended_NCS.ipynb", cells_26)

    # 3. Thin comparison
    cells_comp = build_comparison_cells()
    write_notebook(NOTEBOOK_DIR / "comparison" / "2026-02-06_200GeV_vs_26GeV.ipynb", cells_comp)

    print("\nAll 3 notebooks generated successfully.")
