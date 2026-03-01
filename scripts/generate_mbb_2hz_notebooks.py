"""SUPERSEDED by generate_notebooks.py -- kept for reference only.

Originally generated 3 MBB 2 Hz analysis notebooks for 2026-02-25 measurements.
All functionality is now in the unified generate_notebooks.py.
"""

import json
from pathlib import Path

NOTEBOOK_DIR = Path("rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-25_2Hz")


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
# Comprehensive analysis notebook (24 sections)
# ================================================================

def build_analysis_cells(session, meas_subdir, energy_label, out_subdir):
    """Build cell list for a comprehensive 24-section analysis notebook.

    Merges harmonic analysis, eddy-current settling, and inductance
    into a single notebook.  Processes **both** CS and NCS segments.
    """
    cells = []

    # ==============================================================
    # Title & TOC
    # ==============================================================
    cells.append(md("title", f"""# SPS MBB Dipole -- Comprehensive Analysis ({energy_label} MD1, 2 Hz)

**Measurement session:** `{session}`
**Segments:** NCS (non-connection side) + CS (connection side, fringe field)
**Magnet:** MBB (normal dipole, m=1)
**Rotation speed:** 2 Hz
**Supercycle:** LHC_pilot -> MD1 ({energy_label}) -> SFTPRO
**Kn calibration:** AC compensation (Uppsala reference)

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
SEGMENTS = ["NCS", "CS"]          # CS is fringe-field (end of magnet)

SESSION = "{session}"
MEAS_SUBDIR = "{meas_subdir}"
KN_PATH_REL = "MBB/2026-02-25_2Hz/MBB/Kn Uppsala/Kn_values_Seg_Main_A_AC.txt"

MAGNET_ORDER = 1          # dipole
R_REF = 0.02              # reference radius [m]
L_COIL = 0.47             # coil length [m]
SAMPLES_PER_TURN = 1024   # encoder samples per revolution

OPTIONS = ("dri", "rot", "cel", "fed")

MIN_B1_T = 1e-4           # minimum |B1| for normalization
PLATEAU_I_RANGE_MAX = 2.5 # block-averaged range threshold (A)
N_BLOCKS = 10             # blocks for range averaging

# Settling: last N turns per supercycle
N_LAST_TURNS_INJ = 18
N_LAST_TURNS_HIGH = None   # flat-high: use all

# Outlier removal
N_SIGMA_CLIP = 5           # MAD sigma clipping

# Eddy current fit
MIN_INJECTION_TURNS = 5    # minimum turns for exponential fit

print(f"SPS MBB Dipole -- Comprehensive Analysis ({energy_label}, 2 Hz)")
print("=" * 60)
print(f"  Session       : {{SESSION}}")
print(f"  Segments      : {{SEGMENTS}}")
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
KN_PATH = REPO_ROOT / "measurements" / KN_PATH_REL
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
    cells.append(md("s3-hdr", "---\n## 3. Data Loading & Channel Detection\n\n"
                     "Load raw measurement data for **both** CS and NCS segments."))

    cells.append(code("s3-load", """FILE_PAT = re.compile(
    r"Run_(\\d+)_I_([\\d.]+)A_(N?CS)_raw_measurement_data\\.txt$"
)

data = {}
for seg in SEGMENTS:
    seg_files = [
        f for f in sorted(RUN_DIR.iterdir())
        if FILE_PAT.search(f.name) and FILE_PAT.search(f.name).group(3) == seg
    ]
    assert seg_files, f"No {seg} raw files in {RUN_DIR}"
    raw_file = seg_files[0]

    raw = np.loadtxt(raw_file)
    n_turns = raw.shape[0] // Ns
    n_keep = n_turns * Ns
    ncols = raw.shape[1]

    t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
    flux_col1 = raw[:n_keep, 1].reshape(n_turns, Ns)
    flux_col2 = raw[:n_keep, 2].reshape(n_turns, Ns)
    I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

    # Auto-detect channel swap
    I_mean_quick = I_all.mean(axis=1)
    best_turn = np.argmax(np.abs(I_mean_quick))
    r1 = robust_range(raw[best_turn * Ns:(best_turn + 1) * Ns, 1])
    r2 = robust_range(raw[best_turn * Ns:(best_turn + 1) * Ns, 2])
    swap = r2 > r1

    if swap:
        flux_abs_all, flux_cmp_all = flux_col2, flux_col1
    else:
        flux_abs_all, flux_cmp_all = flux_col1, flux_col2

    data[seg] = {
        "raw_file": raw_file, "n_turns": n_turns,
        "t_all": t_all, "flux_abs": flux_abs_all, "flux_cmp": flux_cmp_all,
        "I_all": I_all, "swap": swap, "r1": r1, "r2": r2,
    }
    fringe_tag = " [FRINGE FIELD]" if seg == "CS" else ""
    print(f"\\n{seg}{fringe_tag}: {raw_file.name}")
    print(f"  Shape: {raw.shape} -> {n_turns} turns, {ncols} columns")
    print(f"  Time span: {raw[-1,0] - raw[0,0]:.1f} s ({(raw[-1,0] - raw[0,0])/60:.1f} min)")
    print(f"  Flux swap: {swap}  (abs range={max(r1,r2):.4e}, cmp range={min(r1,r2):.4e})")"""))

    # ==============================================================
    # 4. Raw Signals Overview
    # ==============================================================
    cells.append(md("s4-hdr", "---\n## 4. Raw Signals Overview"))

    cells.append(code("s4-raw", """fig, axes = plt.subplots(len(SEGMENTS), 3, figsize=(18, 5 * len(SEGMENTS)), sharex="col")
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    d = data[seg]
    n_keep = d["n_turns"] * Ns
    x = np.arange(n_keep)
    axes[i, 0].plot(x, d["flux_abs"].ravel(), linewidth=0.2, color="steelblue")
    axes[i, 0].set_ylabel(f"Flux abs ({seg})")
    axes[i, 0].set_title(f"Absolute flux -- {seg}")
    axes[i, 1].plot(x, d["flux_cmp"].ravel(), linewidth=0.2, color="teal")
    axes[i, 1].set_ylabel(f"Flux cmp ({seg})")
    axes[i, 1].set_title(f"Compensated flux -- {seg}")
    axes[i, 2].plot(x, d["I_all"].ravel(), linewidth=0.2, color="tab:orange")
    axes[i, 2].set_ylabel(f"Current ({seg})")
    axes[i, 2].set_title(f"Current -- {seg}")

axes[-1, 0].set_xlabel("Sample index")
axes[-1, 1].set_xlabel("Sample index")
axes[-1, 2].set_xlabel("Sample index")
fig.suptitle(f"Raw signals -- {SESSION}", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

    # ==============================================================
    # 5. cel/fed Safety Diagnostic
    # ==============================================================
    cells.append(md("s5-hdr", "---\n## 5. cel/fed Safety Diagnostic\n\n"
                     "Run `diagnose_cel_fed()` on NCS high-current turns."))

    cells.append(code("s5-celfed", """# Diagnostic on NCS (main segment, not fringe field)
d = data["NCS"]
I_mean = d["I_all"].mean(axis=1)
hi_mask = np.abs(I_mean) > 4000
if hi_mask.sum() < 5:
    hi_mask = np.abs(I_mean) > np.percentile(np.abs(I_mean), 90)

n_diag = min(100, int(hi_mask.sum()))
hi_idx = np.where(hi_mask)[0][:n_diag]

diag = diagnose_cel_fed(
    d["flux_abs"][hi_idx], d["flux_cmp"][hi_idx],
    d["t_all"][hi_idx], d["I_all"][hi_idx],
    kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER,
)
print(f"cel/fed diagnostic ({n_diag} NCS high-I turns):")
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

    cells.append(code("s6-plateau", """label_colors = {"injection": "tab:green", "flat-mid": "tab:purple", "flat-high": "tab:blue"}

for seg in SEGMENTS:
    d = data[seg]
    I_mean = d["I_all"].mean(axis=1)
    t_mean = d["t_all"].mean(axis=1)
    I_range, I_blocks = compute_block_averaged_range(d["I_all"], Ns, N_BLOCKS)

    plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
    is_plateau = plateau_info["is_plateau"]

    turn_label = np.array(["ramp"] * d["n_turns"], dtype=object)
    for j in range(d["n_turns"]):
        if is_plateau[j]:
            turn_label[j] = classify_current(I_mean[j])

    inj_groups = find_contiguous_groups(turn_label == "injection", min_length=2)
    fh_groups = find_contiguous_groups(turn_label == "flat-high", min_length=2)

    d.update({
        "I_mean": I_mean, "t_mean": t_mean, "I_range": I_range,
        "is_plateau": is_plateau, "turn_label": turn_label,
        "inj_groups": inj_groups, "fh_groups": fh_groups,
    })

    fringe = " [FRINGE FIELD]" if seg == "CS" else ""
    print(f"\\n{seg}{fringe}: {is_plateau.sum()} plateau, "
          f"{len(inj_groups)} inj groups, {len(fh_groups)} flat-high groups")
    for lab in ["injection", "flat-mid", "flat-high"]:
        mask = turn_label == lab
        if mask.sum() > 0:
            print(f"  {lab:12s}: {mask.sum():4d} turns, I = {I_mean[mask].mean():.1f} +/- {I_mean[mask].std():.1f} A")
    print(f"  {'ramp':12s}: {(turn_label == 'ramp').sum():4d} turns")

# Plot
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5), sharey=True)
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    d = data[seg]
    ax.plot(d["t_mean"], d["I_mean"], ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
    for lab, col in label_colors.items():
        mask = d["turn_label"] == lab
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.scatter(d["t_mean"][idx], d["I_mean"][idx], s=6, color=col, zorder=2, label=lab)
    fringe = " [fringe]" if seg == "CS" else ""
    ax.set_xlabel("Time (s)"); ax.set_ylabel("I (A)")
    ax.set_title(f"Current profile -- {seg}{fringe}"); ax.legend(fontsize=9)
fig.suptitle(f"Plateau Detection -- {SESSION}", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 7. FDI Stuck-Channel Diagnostic
    # ==============================================================
    cells.append(md("s7-hdr", """---
## 7. FDI Stuck-Channel Diagnostic

Check whether the Fast Digital Integrator responds to current changes
between consecutive plateau groups.  A stuck FDI produces flat flux
regardless of current ramps."""))

    cells.append(code("s7-fdi", """for seg in SEGMENTS:
    d = data[seg]
    fringe = " [FRINGE]" if seg == "CS" else ""

    # Build run_info from contiguous plateau groups
    all_groups = []
    for lab_name in ["injection", "flat-mid", "flat-high"]:
        groups = find_contiguous_groups(d["turn_label"] == lab_name, min_length=2)
        for gs, ge in groups:
            all_groups.append({"start": gs, "end": ge,
                               "I_nom": float(d["I_mean"][gs:ge+1].mean())})
    all_groups.sort(key=lambda x: x["start"])
    for i, g in enumerate(all_groups):
        g["run_id"] = i

    if len(all_groups) < 2:
        print(f"{seg}{fringe}: fewer than 2 plateau groups, skipping FDI check")
        continue

    # Per-turn flux amplitude (mean of absolute channel)
    flux_turns = d["flux_abs"].mean(axis=1)

    checks = diagnose_fdi_transitions(
        flux_turns, d["I_mean"], all_groups,
        stuck_threshold=0.3, partial_threshold=0.7, min_delta_I=5.0,
    )
    n_ok = sum(1 for c in checks if c.severity == "OK")
    n_partial = sum(1 for c in checks if c.severity == "PARTIAL")
    n_stuck = sum(1 for c in checks if c.severity == "STUCK")

    print(f"\\n{seg}{fringe}: {len(checks)} transitions checked")
    print(f"  OK: {n_ok}, PARTIAL: {n_partial}, STUCK: {n_stuck}")
    for c in checks:
        if c.severity != "OK":
            print(f"  ! Run {c.run_before}->{c.run_after}: {c.severity} -- {c.reason}")

    if n_stuck > 0:
        print(f"  WARNING: {n_stuck} stuck transitions detected!")
    else:
        print(f"  All transitions OK.")"""))

    # ==============================================================
    # 8. Process Plateau Turns
    # ==============================================================
    cells.append(md("s8-hdr", """---
## 8. Process Plateau Turns

Re-process **plateau turns only** with the full OPTIONS (including cel/fed
if safe). Group by supercycle, apply settling window and MAD sigma-clip."""))

    cells.append(code("s8-pipeline", """ANALYSIS_LABELS = {"injection", "flat-mid", "flat-high"}

results = {}  # results[seg] = {"df": ..., "df_settled": ...}

for seg in SEGMENTS:
    d = data[seg]
    turn_label = d["turn_label"]
    n_turns = d["n_turns"]

    is_analysis = np.array([l in ANALYSIS_LABELS for l in turn_label])
    plateau_indices = np.where(is_analysis)[0]
    print(f"\\n{seg}: processing {len(plateau_indices)} plateau turns (OPTIONS={OPTIONS})")

    result, C_merged, C_units, ok_main = process_kn_pipeline(
        flux_abs_turns=d["flux_abs"][plateau_indices],
        flux_cmp_turns=d["flux_cmp"][plateau_indices],
        t_turns=d["t_all"][plateau_indices],
        I_turns=d["I_all"][plateau_indices],
        kn=kn, r_ref=R_REF, magnet_order=m,
        options=OPTIONS, min_b1_T=MIN_B1_T,
    )

    extra = [
        {
            "global_turn": int(plateau_indices[t]),
            "label": str(turn_label[plateau_indices[t]]),
            "I_range_A": float(d["I_range"][plateau_indices[t]]),
            "segment": seg,
        }
        for t in range(len(plateau_indices))
    ]

    rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
    df = pd.DataFrame(rows)

    # Group by supercycle
    df["sc_idx"] = -1
    settled_idx = []

    for gi, (gs, ge) in enumerate(d["inj_groups"]):
        group_globals = set(range(gs, ge + 1))
        gmask = df["global_turn"].isin(group_globals) & (df["label"] == "injection")
        df.loc[gmask, "sc_idx"] = gi
        group_rows = df.index[gmask]
        if N_LAST_TURNS_INJ is not None and len(group_rows) > N_LAST_TURNS_INJ:
            settled_idx.extend(group_rows[-N_LAST_TURNS_INJ:])
        else:
            settled_idx.extend(group_rows)

    for gi, (gs, ge) in enumerate(d["fh_groups"]):
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

    results[seg] = {"df": df, "df_settled": df_settled}

    fringe = " [FRINGE FIELD]" if seg == "CS" else ""
    print(f"\\n  {seg}{fringe}:")
    print(f"    All plateau turns : {len(df)}")
    print(f"    Settled turns     : {len(df_settled)}")
    for lab in ["injection", "flat-high"]:
        n_all = len(df[df["label"] == lab])
        n_set = len(df_settled[df_settled["label"] == lab])
        print(f"    {lab:12s}: {n_all} -> {n_set}")
    print(f"    ok_main: {df['ok_main'].sum()} / {len(df)}")
    print(f"    Harmonics: n=1..{H}")"""))

    # ==============================================================
    # 9. All-Turn Harmonics vs Time
    # ==============================================================
    cells.append(md("s9-hdr", """---
## 9. All-Turn Harmonics vs Time

Process **all turns** (including ramps) to show B1, b2, b3 evolution
across the full measurement window.  Ramp turns are plotted in grey."""))

    cells.append(code("s9-allturn", """all_turn_dfs = {}

for seg in SEGMENTS:
    d = data[seg]
    n_turns = d["n_turns"]

    result_all, C_merged_all, C_units_all, ok_main_all = process_kn_pipeline(
        flux_abs_turns=d["flux_abs"], flux_cmp_turns=d["flux_cmp"],
        t_turns=d["t_all"], I_turns=d["I_all"],
        kn=kn, r_ref=R_REF, magnet_order=m,
        options=OPTIONS, min_b1_T=MIN_B1_T,
    )

    extra_all = [{"global_turn": int(i), "label": str(d["turn_label"][i]),
                   "segment": seg} for i in range(n_turns)]
    rows_all = build_harmonic_rows(result_all, C_merged_all, C_units_all, ok_main_all, m, extra_all)
    df_all = pd.DataFrame(rows_all)
    df_all["t_mean_s"] = d["t_mean"]
    all_turn_dfs[seg] = df_all
    print(f"{seg}: {n_turns} all-turns processed, ok_main={ok_main_all.sum()}")

# Plot B1, b2, b3 vs time for NCS
fig, axes = plt.subplots(3, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 12))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df_all = all_turn_dfs[seg]
    ok = df_all["ok_main"]
    fringe = " [fringe]" if seg == "CS" else ""
    for ax_idx, (col, ylabel) in enumerate([("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
        ax = axes[ax_idx, j]
        # Ramp turns in grey
        ramp = df_all["label"] == "ramp"
        ax.scatter(df_all.loc[ok & ramp, "t_mean_s"], df_all.loc[ok & ramp, col],
                   s=4, alpha=0.3, color="lightgrey", zorder=0, label="ramp")
        for lab, lc in label_colors.items():
            mask = ok & (df_all["label"] == lab)
            if mask.sum() > 0:
                ax.scatter(df_all.loc[mask, "t_mean_s"], df_all.loc[mask, col],
                           s=6, alpha=0.5, color=lc, zorder=2, label=lab)
        ax.set_ylabel(ylabel)
        if ax_idx == 0:
            ax.set_title(f"All-turn evolution -- {seg}{fringe}")
        if ax_idx == 2:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7, loc="upper right")

fig.suptitle(f"All-Turn Harmonics vs Time -- {SESSION}", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 10. FFMM Golden-Standard Comparison
    # ==============================================================
    cells.append(md("s10-hdr", """---
## 10. FFMM Golden-Standard Comparison

Compare our pipeline output against the FFMM per-turn results (`dri rot nor`)
and FFMM average results.  Uses `legacy_rotate_excludes_last=True` for FFMM
parity (their C++ rotation loop excludes the last harmonic n=H)."""))

    cells.append(code("s10-ffmm", """OPTIONS_FFMM = ("dri", "rot")
FFMM_ROTATE_EXCLUDES_LAST = True
RESULTS_PAT = re.compile(r"Run_\\d+_I_[\\d.]+A_(N?CS)_results\\.txt$")

print("=" * 70)
print("FFMM GOLDEN STANDARD COMPARISON")
print(f"FFMM options: dri rot nor  ->  our pipeline: {OPTIONS_FFMM}")
print(f"legacy_rotate_excludes_last = {FFMM_ROTATE_EXCLUDES_LAST}")
print("=" * 70)

ffmm_data = {}

for seg in SEGMENTS:
    print(f"\\n{'='*55}")
    fringe_tag = " [FRINGE FIELD]" if seg == "CS" else ""
    print(f"  Segment: {seg}{fringe_tag}")
    print(f"{'='*55}")

    d = data[seg]
    ffmm_files = [
        f for f in sorted(RUN_DIR.iterdir())
        if RESULTS_PAT.search(f.name) and f"_{seg}_" in f.name
    ]
    assert ffmm_files, f"No FFMM per-turn results for {seg}"
    ffmm_df = pd.read_csv(ffmm_files[0], sep="\\t")
    print(f"  FFMM per-turn: {ffmm_files[0].name}, {len(ffmm_df)} rows")

    avg_file = SESSION_DIR / f"MBB_{seg}_Average_results.txt"
    assert avg_file.exists(), f"FFMM average not found: {avg_file}"
    ffmm_avg = pd.read_csv(avg_file, sep="\\t")
    print(f"  FFMM average : {avg_file.name}")

    result_cmp, C_merged_cmp, C_units_cmp, ok_main_cmp = process_kn_pipeline(
        flux_abs_turns=d["flux_abs"], flux_cmp_turns=d["flux_cmp"],
        t_turns=d["t_all"], I_turns=d["I_all"],
        kn=kn, r_ref=R_REF, magnet_order=m,
        options=OPTIONS_FFMM, min_b1_T=MIN_B1_T,
        legacy_rotate_excludes_last=FFMM_ROTATE_EXCLUDES_LAST,
    )

    extra = [{"global_turn": int(i)} for i in range(d["n_turns"])]
    rows = build_harmonic_rows(result_cmp, C_merged_cmp, C_units_cmp, ok_main_cmp, m, extra)
    our_df = pd.DataFrame(rows)

    assert len(our_df) == len(ffmm_df), (
        f"Turn count mismatch: ours={len(our_df)}, FFMM={len(ffmm_df)}"
    )

    ok_idx = our_df.index[our_df["ok_main"]].values
    n_compare = len(ok_idx)
    print(f"\\n  Comparing {n_compare} / {d['n_turns']} turns (ok_main=True)")

    our_bmain = our_df.loc[ok_idx, "B1_T"].values
    ffmm_bmain = ffmm_df.loc[ok_idx, "B_main(T)"].values
    rms_bmain = np.sqrt(np.mean((our_bmain - ffmm_bmain)**2))
    max_bmain = np.max(np.abs(our_bmain - ffmm_bmain))
    print(f"  B_main: RMS diff = {rms_bmain:.4e} T, max |diff| = {max_bmain:.4e} T")

    print(f"\\n  {'n':>3s}  {'RMS(bn)':>10s}  {'max(bn)':>10s}  {'RMS(an)':>10s}  {'max(an)':>10s}")
    print("  " + "-" * 50)
    for n in range(2, H + 1):
        bn_ours = our_df.loc[ok_idx, f"b{n}_units"].values
        bn_ffmm = ffmm_df.loc[ok_idx, f"b{n}(Units)"].values
        an_ours = our_df.loc[ok_idx, f"a{n}_units"].values
        an_ffmm = ffmm_df.loc[ok_idx, f"a{n}(Units)"].values
        rms_bn = np.sqrt(np.mean((bn_ours - bn_ffmm)**2))
        max_bn = np.max(np.abs(bn_ours - bn_ffmm))
        rms_an = np.sqrt(np.mean((an_ours - an_ffmm)**2))
        max_an = np.max(np.abs(an_ours - an_ffmm))
        print(f"  {n:3d}  {rms_bn:10.4f}  {max_bn:10.4f}  {rms_an:10.4f}  {max_an:10.4f}")

    ffmm_data[seg] = {"ffmm_df": ffmm_df, "ffmm_avg": ffmm_avg, "our_df": our_df}

# Validation plot
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(7 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    ok_idx = ffmm_data[seg]["our_df"].index[ffmm_data[seg]["our_df"]["ok_main"]].values
    our_b = ffmm_data[seg]["our_df"].loc[ok_idx, "B1_T"].values
    ffmm_b = ffmm_data[seg]["ffmm_df"].loc[ok_idx, "B_main(T)"].values
    ax.scatter(ffmm_b, our_b, s=4, alpha=0.3, color="steelblue")
    lims = [min(ffmm_b.min(), our_b.min()), max(ffmm_b.max(), our_b.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("FFMM B_main (T)"); ax.set_ylabel("Our B1 (T)")
    fringe = " [fringe]" if seg == "CS" else ""
    ax.set_title(f"B_main validation -- {seg}{fringe}"); ax.legend(fontsize=9)
fig.suptitle("FFMM Golden Standard: B_main parity", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 11. Main Field (B1) Analysis
    # ==============================================================
    cells.append(md("s11-hdr", "---\n## 11. Main Field (B1)"))

    cells.append(code("s11-b1", """fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]
    ok = df["ok_main"]
    fringe = " [fringe]" if seg == "CS" else ""

    ax = axes[0, j]
    ax.scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "B1_T"], s=8, alpha=0.5, color="steelblue")
    ax.set_xlabel("I (A)"); ax.set_ylabel("B1 (T)")
    ax.set_title(f"B1 vs current -- {seg}{fringe}")

    ax = axes[1, j]
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)")
    ax.set_title(f"B1 per supercycle (settled) -- {seg}{fringe}"); ax.legend(fontsize=9)

fig.suptitle(f"Main Field (B1) -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Statistics table
print("\\nB1 per operating point (settled turns):")
print(f"{'Segment':>8s} {'Label':>12s} {'N':>5s} {'mean (T)':>12s} {'std (T)':>12s}")
print("-" * 55)
for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            fringe = "*" if seg == "CS" else " "
            print(f"{seg:>7s}{fringe} {lab:>12s} {len(sub):5d} "
                  f"{sub['B1_T'].mean():+12.6f} {sub['B1_T'].std():12.6f}")
print("  * = fringe-field segment")"""))

    # ==============================================================
    # 12. b2 (Quadrupole)
    # ==============================================================
    cells.append(md("s12-hdr", """---
## 12. b2 (Quadrupole)

First **allowed** harmonic error for a dipole."""))

    cells.append(code("s12-b2", """fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]
    ok = df["ok_main"]
    fringe = " [fringe]" if seg == "CS" else ""

    ax = axes[0, j]
    ax.scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "b2_units"], s=8, alpha=0.5, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("I (A)"); ax.set_ylabel("b2 (units)")
    ax.set_title(f"b2 vs current -- {seg}{fringe}")

    ax = axes[1, j]
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["b2_units"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("b2 (units)")
    ax.set_title(f"b2 per supercycle (settled) -- {seg}{fringe}"); ax.legend(fontsize=9)

fig.suptitle(f"b2 (Quadrupole) -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 13. b3 (Sextupole)
    # ==============================================================
    cells.append(md("s13-hdr", """---
## 13. b3 (Sextupole)

First **non-allowed** harmonic -- key quality indicator."""))

    cells.append(code("s13-b3", """fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]
    ok = df["ok_main"]
    fringe = " [fringe]" if seg == "CS" else ""

    ax = axes[0, j]
    ax.scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "b3_units"], s=8, alpha=0.5, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("I (A)"); ax.set_ylabel("b3 (units)")
    ax.set_title(f"b3 vs current -- {seg}{fringe}")

    ax = axes[1, j]
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["b3_units"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("b3 (units)")
    ax.set_title(f"b3 per supercycle (settled) -- {seg}{fringe}"); ax.legend(fontsize=9)

fig.suptitle(f"b3 (Sextupole) -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Per-SC evolution panel (B1, b2, b3)
fig, axes = plt.subplots(len(SEGMENTS), 3, figsize=(16, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    df_settled = results[seg]["df_settled"]
    fringe = " [fringe]" if seg == "CS" else ""
    for ax_idx, (col_name, ylabel) in enumerate([
            ("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
        ax = axes[i, ax_idx]
        for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
            sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
            if len(sub) > 0:
                sc_avg = sub.groupby("sc_idx")[col_name].agg(["mean", "std"]).reset_index()
                ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                            fmt=f"{marker}-", markersize=4, capsize=2, color=col, alpha=0.8, label=lab)
        ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel); ax.legend(fontsize=9)
        ax.set_title(f"{ylabel.split()[0]} per SC -- {seg}{fringe}")

fig.suptitle(f"Per-Supercycle Evolution (settled) -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Stability table
print("\\nStability across supercycles (settled turns):")
print(f"{'Segment':>8s} {'Qty':>4s} {'Label':>12s} {'SC mean':>12s} {'SC std':>12s} {'SC p-p':>12s}")
print("-" * 65)
for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    for col_name, label in [("B1_T", "B1"), ("b2_units", "b2"), ("b3_units", "b3")]:
        for lab in ["injection", "flat-high"]:
            sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
            if len(sub) > 0:
                sc_means = sub.groupby("sc_idx")[col_name].mean()
                print(f"{seg:>8s} {label:>4s} {lab:>12s} {sc_means.mean():+12.6f} "
                      f"{sc_means.std():12.6f} {sc_means.max()-sc_means.min():12.6f}")"""))

    # ==============================================================
    # 14. Higher Harmonics Overview
    # ==============================================================
    cells.append(md("s14-hdr", """---
## 14. Higher Harmonics Overview

Summary statistics for all harmonics b4..bH and a2..aH at each
operating point (settled turns, NCS only)."""))

    cells.append(code("s14-higher", """seg = "NCS"  # focus on main segment
df_settled = results[seg]["df_settled"]
ok = df_settled["ok_main"]

for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok]
    if len(sub) == 0:
        continue
    print(f"\\n=== {lab.upper()} (N={len(sub)} settled turns, NCS) ===")
    print(f"  {'n':>3s} {'bn mean':>10s} {'bn std':>10s} {'an mean':>10s} {'an std':>10s}")
    print("  " + "-" * 50)
    for n in range(2, H + 1):
        bn_col = f"b{n}_units"
        an_col = f"a{n}_units"
        if bn_col in sub.columns:
            bn_m = sub[bn_col].mean()
            bn_s = sub[bn_col].std()
            an_m = sub[an_col].mean()
            an_s = sub[an_col].std()
            flag = " *" if abs(bn_m) > 2 * bn_s and abs(bn_m) > 0.5 else ""
            print(f"  {n:3d} {bn_m:+10.4f} {bn_s:10.4f} {an_m:+10.4f} {an_s:10.4f}{flag}")
    print("  (* = |mean| > 2*std and |mean| > 0.5 units)")"""))

    # ==============================================================
    # 15. Multipole Spectrum
    # ==============================================================
    cells.append(md("s15-hdr", """---
## 15. Multipole Spectrum

Bar charts of normal (bn) and skew (an) harmonics at key operating points.
Both linear and log scale."""))

    cells.append(code("s15-spectrum", """seg = "NCS"
df_settled = results[seg]["df_settled"]
ok = df_settled["ok_main"]

operating_points = {}
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok]
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

        # Linear scale
        ax = axes[i, 0]
        x = np.arange(len(orders))
        w = 0.35
        ax.bar(x - w/2, bn_means, w, yerr=bn_stds, label="bn (normal)",
               color="steelblue", capsize=2, alpha=0.8)
        ax.bar(x + w/2, an_means, w, yerr=an_stds, label="an (skew)",
               color="tab:orange", capsize=2, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(orders)
        ax.set_xlabel("Harmonic order n"); ax.set_ylabel("Units")
        ax.set_title(f"Multipole spectrum -- {lab} (NCS, linear)")
        ax.legend(fontsize=8)

        # Log scale
        ax = axes[i, 1]
        ax.bar(x - w/2, np.abs(bn_means), w, label="|bn|",
               color="steelblue", alpha=0.8)
        ax.bar(x + w/2, np.abs(an_means), w, label="|an|",
               color="tab:orange", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(orders)
        ax.set_xlabel("Harmonic order n"); ax.set_ylabel("|Units|")
        ax.set_title(f"Multipole spectrum -- {lab} (NCS, log)")
        ax.legend(fontsize=8)

    fig.suptitle(f"Multipole Spectrum -- {SESSION} (NCS)", fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No operating points with data for spectrum plot.")"""))

    # ==============================================================
    # 16. Transfer Function B1/I
    # ==============================================================
    cells.append(md("s16-hdr", """---
## 16. Transfer Function B1/I

TF = B1 / I (units: T/kA). Scatter and per-supercycle plots."""))

    cells.append(code("s16-tf", """fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

tf_summary = {}
for j, seg in enumerate(SEGMENTS):
    df_settled = results[seg]["df_settled"]
    ok = df_settled["ok_main"]
    fringe = " [fringe]" if seg == "CS" else ""

    ax = axes[0, j]
    sub_ok = df_settled[ok]
    ax.scatter(sub_ok["I_mean_A"], sub_ok["TF_TperkA"], s=8, alpha=0.5, color="steelblue")
    ax.set_xlabel("I (A)"); ax.set_ylabel("TF = B1/I (T/kA)")
    ax.set_title(f"TF vs current -- {seg}{fringe}")

    ax = axes[1, j]
    ds_tf = {}
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & ok]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["TF_TperkA"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
            ds_tf[lab] = sc_avg
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("TF = B1/I (T/kA)")
    ax.set_title(f"TF per supercycle -- {seg}{fringe}"); ax.legend(fontsize=9)
    tf_summary[seg] = ds_tf

fig.suptitle(f"Transfer Function B1/I -- {SESSION}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\\nTF summary (settled turns):")
print(f"  {'Segment':>8s} {'Level':>12s} {'N SC':>5s} {'mean (T/kA)':>14s} {'std':>10s}")
print(f"  {'-'*55}")
for seg in SEGMENTS:
    for lab in ["injection", "flat-high"]:
        if lab in tf_summary.get(seg, {}):
            sc = tf_summary[seg][lab]
            print(f"  {seg:>8s} {lab:>12s} {len(sc):5d} {sc['mean'].mean():14.4f} {sc['mean'].std():10.4f}")"""))

    # ==============================================================
    # 17. Apparent vs Differential Inductance
    # ==============================================================
    cells.append(md("s17-hdr", """---
## 17. Apparent vs Differential Inductance

**L_app** = B1/I (apparent inductance, = transfer function)
**L_d** = dB1/dI (differential inductance, from paired supercycle levels)

If L_d < L_app(flat-high), the iron is in **saturation**."""))

    cells.append(code("s17-inductance", """ld_results = {}

for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    ok = df_settled["ok_main"]
    df_inj = df_settled[(df_settled["label"] == "injection") & ok]
    df_fh = df_settled[(df_settled["label"] == "flat-high") & ok]

    if len(df_inj) == 0 or len(df_fh) == 0:
        ld_results[seg] = pd.DataFrame()
        continue

    inj_avg = df_inj.groupby("sc_idx").agg(
        B1_inj=("B1_T", "mean"), I_inj=("I_mean_A", "mean"), n_inj=("B1_T", "count")).reset_index()
    fh_avg = df_fh.groupby("sc_idx").agg(
        B1_fh=("B1_T", "mean"), I_fh=("I_mean_A", "mean"), n_fh=("B1_T", "count")).reset_index()

    merged = inj_avg.merge(fh_avg, on="sc_idx", how="inner")
    if len(merged) == 0:
        ld_results[seg] = pd.DataFrame()
        continue

    merged["Ld_TperkA"] = (merged["B1_fh"] - merged["B1_inj"]) / ((merged["I_fh"] - merged["I_inj"]) / 1000.0)
    ld_results[seg] = merged
    print(f"{seg}: {len(merged)} SC pairs, "
          f"Ld = {merged['Ld_TperkA'].mean():.4f} +/- {merged['Ld_TperkA'].std():.4f} T/kA")

# Plot L_app vs Ld
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(7 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    fringe = " [fringe]" if seg == "CS" else ""
    if "injection" in tf_summary.get(seg, {}):
        sc = tf_summary[seg]["injection"]
        ax.errorbar(sc["sc_idx"], sc["mean"], yerr=sc["std"],
                    fmt="o-", markersize=4, capsize=2, color="tab:green", alpha=0.8, label="L_app (inj)")
    if "flat-high" in tf_summary.get(seg, {}):
        sc = tf_summary[seg]["flat-high"]
        ax.errorbar(sc["sc_idx"], sc["mean"], yerr=sc["std"],
                    fmt="s-", markersize=4, capsize=2, color="tab:blue", alpha=0.8, label="L_app (FT)")
    m_df = ld_results.get(seg, pd.DataFrame())
    if len(m_df) > 0:
        ax.plot(m_df["sc_idx"], m_df["Ld_TperkA"], "D-", markersize=4, color="tab:red", alpha=0.8, label="Ld")
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("T/kA")
    ax.set_title(f"L_app vs Ld -- {seg}{fringe}"); ax.legend(fontsize=8)
fig.suptitle("Apparent vs Differential Inductance", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()

print("\\nSaturation check (Ld < L_app(FT) => saturated):")
for seg in SEGMENTS:
    m_df = ld_results.get(seg, pd.DataFrame())
    if len(m_df) == 0:
        continue
    Ld_mean = m_df["Ld_TperkA"].mean()
    Lapp_fh = tf_summary.get(seg, {}).get("flat-high")
    if Lapp_fh is not None and len(Lapp_fh) > 0:
        Lapp_fh_mean = Lapp_fh["mean"].mean()
        ratio = Ld_mean / Lapp_fh_mean
        verdict = "SATURATED" if ratio < 0.99 else "LINEAR"
        print(f"  {seg}: Ld={Ld_mean:.4f}, L_app(FT)={Lapp_fh_mean:.4f}, ratio={ratio:.4f} -> {verdict}")"""))

    # ==============================================================
    # 18. Raw Settling Curves
    # ==============================================================
    cells.append(md("s18-hdr", """---
## 18. Raw Settling Curves

B1 and b3 vs turn within each injection supercycle, all supercycles
overlaid.  Eddy currents cause exponential decay after the ramp."""))

    cells.append(code("s18-settling", """# Build per-supercycle injection data using the full pipeline results
eddy_data = {}

for seg in SEGMENTS:
    d = data[seg]
    df = results[seg]["df"]
    inj = df[df["label"] == "injection"].copy()
    if len(inj) == 0:
        eddy_data[seg] = pd.DataFrame()
        continue

    # Add time since injection start per supercycle
    inj["t_mean_s"] = d["t_mean"][inj["global_turn"].values]
    for sc_id in inj["sc_idx"].unique():
        if sc_id < 0:
            continue
        mask = inj["sc_idx"] == sc_id
        t0 = inj.loc[mask, "t_mean_s"].min()
        inj.loc[mask, "t_since_inj_start"] = inj.loc[mask, "t_mean_s"] - t0

    # Turn index within group
    for sc_id in inj["sc_idx"].unique():
        if sc_id < 0:
            continue
        mask = inj["sc_idx"] == sc_id
        inj.loc[mask, "turn_in_group"] = np.arange(mask.sum())

    eddy_data[seg] = inj
    n_sc = inj["sc_idx"].nunique()
    print(f"{seg}: {len(inj)} injection turns across {n_sc} supercycles")

# Plot b3 overlay
fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    inj = eddy_data[seg]
    fringe = " [fringe]" if seg == "CS" else ""
    for col_idx, (col, ylabel) in enumerate([("B1_T", "B1 (T)"), ("b3_units", "b3 (units)")]):
        ax = axes[i, col_idx]
        if len(inj) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{seg}{fringe}"); continue
        sc_ids = sorted(inj["sc_idx"].unique())
        sc_ids = [s for s in sc_ids if s >= 0]
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(sc_ids), 1)))
        for k, sc_id in enumerate(sc_ids):
            sub = inj[inj["sc_idx"] == sc_id]
            ax.plot(sub["t_since_inj_start"], sub[col], ".-",
                    markersize=4, linewidth=0.8, alpha=0.7, color=cmap[k % len(cmap)])
        ax.set_xlabel("t - t_inj_start (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} settling -- {seg}{fringe}")

fig.suptitle("Injection Settling Curves -- Supercycle Overlay", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # ==============================================================
    # 19. Exponential Fits
    # ==============================================================
    cells.append(md("s19-hdr", """---
## 19. Exponential Fits

Fit b3(t) = b3_inf + A * exp(-t/tau) per supercycle.
Tau is the eddy-current settling time constant."""))

    cells.append(code("s19-fits", """def fit_supercycle(df_sc):
    \"\"\"Fit single-exponential eddy model to one injection supercycle.\"\"\"
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

fit_results = {}
df_fits = {}
for seg in SEGMENTS:
    inj = eddy_data[seg]
    fits = []
    if len(inj) == 0:
        fit_results[seg] = fits
        df_fits[seg] = pd.DataFrame()
        continue
    for sc_id in sorted(inj["sc_idx"].unique()):
        if sc_id < 0:
            continue
        result = fit_supercycle(inj[inj["sc_idx"] == sc_id])
        if result is not None:
            result["supercycle_id"] = sc_id
            fits.append(result)
    fit_results[seg] = fits
    df_fits[seg] = pd.DataFrame(fits)
    print(f"{seg}: {len(fits)} / {len([s for s in inj['sc_idx'].unique() if s >= 0])} supercycles fitted")

for seg, df_f in df_fits.items():
    if len(df_f) == 0:
        continue
    print(f"\\n{seg}:")
    print(f"  {'SC':>3s} {'tau (s)':>10s} {'A (units)':>12s} {'b3_inf':>10s} {'R2':>6s}")
    print(f"  {'-'*45}")
    for _, row in df_f.iterrows():
        print(f"  {int(row['supercycle_id']):3d} {row['tau']:10.2f} {row['A']:+12.4f} "
              f"{row['b3_inf']:+10.4f} {row['r2']:6.3f}")

# Fit overlay on representative supercycles
n_show = 3
fig, axes = plt.subplots(len(SEGMENTS), n_show, figsize=(14, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for row_idx, seg in enumerate(SEGMENTS):
    inj = eddy_data[seg]
    df_f = df_fits[seg]
    fringe = " [fringe]" if seg == "CS" else ""
    if len(df_f) == 0:
        for j in range(n_show):
            axes[row_idx, j].text(0.5, 0.5, "No fits", ha="center", va="center",
                                  transform=axes[row_idx, j].transAxes)
            axes[row_idx, j].set_title(f"{seg}{fringe}")
        continue
    sc_ids = df_f["supercycle_id"].values
    show_ids = sc_ids[np.linspace(0, len(sc_ids)-1, min(n_show, len(sc_ids)), dtype=int)]
    for j, sc_id in enumerate(show_ids):
        ax = axes[row_idx, j]
        sub = inj[inj["sc_idx"] == sc_id]
        fit_row = df_f[df_f["supercycle_id"] == sc_id].iloc[0]
        ax.scatter(sub["t_since_inj_start"], sub["b3_units"], s=15, alpha=0.7, color="tab:blue", label="data")
        t_fit = np.linspace(0, sub["t_since_inj_start"].max() * 1.05, 200)
        ax.plot(t_fit, eddy_model(t_fit, fit_row["b3_inf"], fit_row["A"], fit_row["tau"]),
                "r-", linewidth=1.5, label="fit")
        ax.set_title(f"{seg}{fringe} SC {int(sc_id)}\\ntau={fit_row['tau']:.1f}s, R2={fit_row['r2']:.3f}", fontsize=9)
        ax.set_xlabel("t (s)")
        if j == 0:
            ax.set_ylabel("b3 (units)")
        ax.legend(fontsize=7)
    for j in range(len(show_ids), n_show):
        axes[row_idx, j].set_visible(False)

fig.suptitle("Exponential Fit -- Representative Supercycles", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()

# Tau summary
print("\\nTau statistics:")
print(f"  {'Segment':>8s} {'N':>4s} {'mean (s)':>10s} {'std (s)':>10s} {'median':>10s} {'R2 mean':>10s}")
print(f"  {'-'*55}")
for seg in SEGMENTS:
    df_f = df_fits[seg]
    if len(df_f) == 0:
        print(f"  {seg:>8s} {'--':>4s}"); continue
    tau_v = df_f["tau"].values; r2_v = df_f["r2"].values
    print(f"  {seg:>8s} {len(df_f):4d} {tau_v.mean():10.2f} {tau_v.std():10.2f} "
          f"{np.median(tau_v):10.2f} {r2_v.mean():10.3f}")"""))

    # ==============================================================
    # 20. Settling Bias Analysis
    # ==============================================================
    cells.append(md("s20-hdr", """---
## 20. Settling Bias Analysis

How do the per-supercycle b2 and b3 averages change when you vary the
averaging window (last N turns)?  Shows the bias from including
early (unsettled) turns."""))

    cells.append(code("s20-bias", """seg = "NCS"  # focus on main segment
inj = eddy_data[seg]
if len(inj) > 0 and "turn_in_group" in inj.columns:
    sc_ids = sorted([s for s in inj["sc_idx"].unique() if s >= 0])
    max_turns_per_sc = inj.groupby("sc_idx").size().min()
    n_last_values = list(range(1, max_turns_per_sc + 1))

    bias_b3 = []
    bias_b2 = []
    for n_last in n_last_values:
        b3_means = []
        b2_means = []
        for sc_id in sc_ids:
            sub = inj[inj["sc_idx"] == sc_id].sort_values("turn_in_group")
            tail = sub.tail(n_last)
            if len(tail) > 0 and tail["ok_main"].any():
                ok_tail = tail[tail["ok_main"]]
                b3_means.append(ok_tail["b3_units"].mean())
                b2_means.append(ok_tail["b2_units"].mean())
        if b3_means:
            bias_b3.append(np.mean(b3_means))
            bias_b2.append(np.mean(b2_means))
        else:
            bias_b3.append(np.nan)
            bias_b2.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(n_last_values, bias_b3, "o-", markersize=4, color="tab:blue")
    axes[0].axhline(bias_b3[-1], color="grey", linestyle="--", linewidth=0.8,
                     label=f"converged = {bias_b3[-1]:.3f}")
    axes[0].set_xlabel("N_LAST (turns from end)"); axes[0].set_ylabel("b3 mean (units)")
    axes[0].set_title("b3 bias vs averaging window (NCS injection)"); axes[0].legend(fontsize=9)

    axes[1].plot(n_last_values, bias_b2, "o-", markersize=4, color="tab:orange")
    axes[1].axhline(bias_b2[-1], color="grey", linestyle="--", linewidth=0.8,
                     label=f"converged = {bias_b2[-1]:.3f}")
    axes[1].set_xlabel("N_LAST (turns from end)"); axes[1].set_ylabel("b2 mean (units)")
    axes[1].set_title("b2 bias vs averaging window (NCS injection)"); axes[1].legend(fontsize=9)

    fig.suptitle("Settling Bias Analysis", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for bias analysis.")"""))

    # ==============================================================
    # 21. N_LAST Sensitivity Study
    # ==============================================================
    cells.append(md("s21-hdr", """---
## 21. N_LAST Sensitivity Study

Scan N_LAST_TURNS_INJ from 1 to max and show how the global
average B1, b2, b3 converges (NCS, settled injection turns)."""))

    cells.append(code("s21-nlast", """seg = "NCS"
d = data[seg]
df = results[seg]["df"]
inj_all = df[df["label"] == "injection"].copy()

if len(inj_all) > 0:
    # Determine turns per supercycle
    turns_per_sc = inj_all.groupby("sc_idx").size()
    max_n_last = int(turns_per_sc.min())
    n_last_scan = list(range(1, max_n_last + 1))

    scan_results = {"B1_T": [], "b2_units": [], "b3_units": []}
    for n_last in n_last_scan:
        settled_idx = []
        for sc_id in inj_all["sc_idx"].unique():
            if sc_id < 0:
                continue
            group_rows = inj_all.index[inj_all["sc_idx"] == sc_id]
            if len(group_rows) > n_last:
                settled_idx.extend(group_rows[-n_last:])
            else:
                settled_idx.extend(group_rows)
        sub = inj_all.loc[sorted(settled_idx)]
        sub_ok = sub[sub["ok_main"]]
        for col in ["B1_T", "b2_units", "b3_units"]:
            scan_results[col].append(sub_ok[col].mean() if len(sub_ok) > 0 else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, ylabel, color) in zip(axes, [
            ("B1_T", "B1 (T)", "steelblue"),
            ("b2_units", "b2 (units)", "tab:orange"),
            ("b3_units", "b3 (units)", "tab:green")]):
        vals = scan_results[col]
        ax.plot(n_last_scan, vals, "o-", markersize=3, color=color)
        ax.axvline(N_LAST_TURNS_INJ, color="red", linestyle="--", linewidth=1,
                    label=f"N_LAST={N_LAST_TURNS_INJ}")
        ax.set_xlabel("N_LAST"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} vs N_LAST (NCS inj)")
        ax.legend(fontsize=8)

    fig.suptitle("N_LAST Sensitivity -- NCS Injection", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for N_LAST sensitivity study.")"""))

    # ==============================================================
    # 22. Comprehensive Statistics Table
    # ==============================================================
    cells.append(md("s22-hdr", "---\n## 22. Comprehensive Statistics Table"))

    cells.append(code("s22-stats", f"""print("=" * 70)
print(f"SPS MBB DIPOLE -- COMPREHENSIVE ANALYSIS ({energy_label}, 2 Hz)")
print("=" * 70)

print(f"\\nMeasurement  : {{SESSION}}")
print(f"Segments     : {{SEGMENTS}}")
print(f"Kn file      : {{KN_PATH.name}} (Uppsala)")
print(f"Options      : {{OPTIONS}}")
print(f"cel/fed      : {{diag.recommendation}}")

for seg in SEGMENTS:
    d = data[seg]
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]
    fringe = " [FRINGE]" if seg == "CS" else ""

    print(f"\\n--- {{seg}}{{fringe}} ---")
    print(f"  Total turns   : {{d['n_turns']}}")
    print(f"  Plateau turns : {{d['is_plateau'].sum()}}")
    print(f"  Injection     : {{(d['turn_label'] == 'injection').sum()}} turns, {{len(d['inj_groups'])}} supercycles")
    print(f"  Flat-high     : {{(d['turn_label'] == 'flat-high').sum()}} turns, {{len(d['fh_groups'])}} groups")

    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1e3)
            print(f"  {{lab:12s}}: N={{len(sub):4d}}, I={{sub['I_mean_A'].mean():.1f}} A, "
                  f"B1={{sub['B1_T'].mean():+.6f}} T, "
                  f"b2={{sub['b2_units'].mean():+.3f}}, b3={{sub['b3_units'].mean():+.3f}} units, "
                  f"TF={{tf:.4f}} T/kA")

    # Eddy current tau
    df_f = df_fits.get(seg, pd.DataFrame())
    if len(df_f) > 0:
        tau_v = df_f["tau"].values
        print(f"  Eddy tau   : {{tau_v.mean():.2f}} +/- {{tau_v.std():.2f}} s (N={{len(df_f)}})")

    # Inductance
    m_df = ld_results.get(seg, pd.DataFrame())
    if len(m_df) > 0:
        print(f"  Ld (diff)  : {{m_df['Ld_TperkA'].mean():.4f}} +/- {{m_df['Ld_TperkA'].std():.4f}} T/kA")"""))

    # ==============================================================
    # 23. Analysis Choices Summary
    # ==============================================================
    cells.append(md("s23-hdr", """---
## 23. Analysis Choices Summary

Document all analysis parameters for reproducibility."""))

    cells.append(code("s23-choices", f"""import datetime

print("ANALYSIS CHOICES")
print("=" * 60)
print(f"Generated    : {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}}")
print(f"Session      : {{SESSION}}")
print(f"Segments     : {{SEGMENTS}}")
print(f"Energy label : {energy_label}")
print(f"Magnet order : {{MAGNET_ORDER}} (dipole)")
print(f"R_ref        : {{R_REF}} m")
print(f"L_coil       : {{L_COIL}} m")
print(f"Samples/turn : {{SAMPLES_PER_TURN}}")
print(f"Kn file      : {{KN_PATH.name}}")
print(f"OPTIONS      : {{OPTIONS}}")
print(f"cel/fed diag : {{diag.recommendation}}")
print(f"MIN_B1_T     : {{MIN_B1_T}}")
print(f"PLATEAU_I_RANGE_MAX : {{PLATEAU_I_RANGE_MAX}} A")
print(f"N_BLOCKS     : {{N_BLOCKS}}")
print(f"N_LAST_TURNS_INJ    : {{N_LAST_TURNS_INJ}}")
print(f"N_LAST_TURNS_HIGH   : {{N_LAST_TURNS_HIGH}}")
print(f"N_SIGMA_CLIP        : {{N_SIGMA_CLIP}}")
print(f"MIN_INJECTION_TURNS : {{MIN_INJECTION_TURNS}}")
print(f"FFMM rotate excl.  : {{FFMM_ROTATE_EXCLUDES_LAST}}")"""))

    # ==============================================================
    # 24. CSV Export
    # ==============================================================
    cells.append(md("s24-hdr", "---\n## 24. CSV Export"))

    cells.append(code("s24-export", f"""out_dir = REPO_ROOT / "output" / "MBB/2026-02-25_2Hz" / "{out_subdir}"
out_dir.mkdir(parents=True, exist_ok=True)

for seg in SEGMENTS:
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]

    fname = f"MBB_{{seg}}_streaming_plateau.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"Wrote {{out_dir / fname}}  ({{len(df)}} rows)")

    fname_s = f"MBB_{{seg}}_streaming_settled.csv"
    df_settled.to_csv(out_dir / fname_s, index=False)
    print(f"Wrote {{out_dir / fname_s}}  ({{len(df_settled)}} rows)")

# Eddy current CSVs
for seg in SEGMENTS:
    inj = eddy_data.get(seg, pd.DataFrame())
    if len(inj) > 0:
        fname = f"b3_injection_{{seg}}.csv"
        inj.to_csv(out_dir / fname, index=False)
        print(f"Wrote {{out_dir / fname}}  ({{len(inj)}} rows)")

    df_f = df_fits.get(seg, pd.DataFrame())
    if len(df_f) > 0:
        fname = f"b3_fits_{{seg}}.csv"
        df_f.to_csv(out_dir / fname, index=False)
        print(f"Wrote {{out_dir / fname}}  ({{len(df_f)}} rows)")

# Inductance summary
ind_rows = []
for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    ok = df_settled["ok_main"]
    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & ok]
        if len(sub) > 0:
            ind_rows.append({{
                "segment": seg, "level": lab, "N": len(sub),
                "I_mean_A": sub["I_mean_A"].mean(),
                "B1_T": sub["B1_T"].mean(), "B1_std": sub["B1_T"].std(),
                "TF_TperkA": sub["TF_TperkA"].mean(), "TF_std": sub["TF_TperkA"].std(),
            }})
    m_df = ld_results.get(seg, pd.DataFrame())
    if len(m_df) > 0:
        ind_rows.append({{
            "segment": seg, "level": "Ld_differential", "N": len(m_df),
            "I_mean_A": np.nan,
            "B1_T": np.nan, "B1_std": np.nan,
            "TF_TperkA": m_df["Ld_TperkA"].mean(), "TF_std": m_df["Ld_TperkA"].std(),
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
    """Build thin comparison notebook that loads CSVs from both analyses."""
    cells = []

    cells.append(md("title", """# B1, b2, b3 Comparison: 200 GeV MD1 vs 26 GeV MD1 (2 Hz Rotation)

## Objective

Compare harmonics at two operating points (injection ~301 A, flat-top ~4815 A)
for two **MD1 cycle energies** on the same SPS MBB dipole at **2 Hz rotation speed**.

This notebook loads **pre-computed CSVs** from the comprehensive analysis notebooks.

| Dataset | MD1 energy | Session | Date |
|---------|-----------|---------|------|
| **200 GeV** | 200 GeV | `20260225_183154_SPS_MBB` | 2026-02-25 |
| **26 GeV** | 26 GeV | `20260225_181040_SPS_MBB` | 2026-02-25 |

| # | Section |
|---|---------|
| 1 | Configuration & Imports |
| 2 | Load Settled CSVs |
| 3 | B1 Comparison |
| 4 | b2, b3 Comparison |
| 5 | Multipole Spectrum Comparison |
| 6 | Statistical Significance |
| 7 | Summary |"""))

    # 1. Config & Imports
    cells.append(md("c1-hdr", "---\n## 1. Configuration & Imports"))

    cells.append(code("c1-config", """SEGMENTS = ["NCS", "CS"]
N_LAST_TURNS_INJ = 18

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

OUT_200 = REPO_ROOT / "output" / "MBB/2026-02-25_2Hz" / "200GeV"
OUT_26  = REPO_ROOT / "output" / "MBB/2026-02-25_2Hz" / "26GeV"
assert OUT_200.exists(), f"200 GeV output not found: {OUT_200}"
assert OUT_26.exists(), f"26 GeV output not found: {OUT_26}"

print("Comparison: 200 GeV vs 26 GeV (2 Hz)")
print(f"  200 GeV CSVs: {OUT_200}")
print(f"  26 GeV  CSVs: {OUT_26}")"""))

    # 2. Load Settled CSVs
    cells.append(md("c2-hdr", "---\n## 2. Load Settled CSVs"))

    cells.append(code("c2-load", """ds = {}
for name, out_dir in [("200 GeV", OUT_200), ("26 GeV", OUT_26)]:
    ds[name] = {}
    for seg in SEGMENTS:
        fname = f"MBB_{seg}_streaming_settled.csv"
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing: {fpath}"
        df = pd.read_csv(fpath)
        ds[name][seg] = df
        print(f"  {name} {seg}: {len(df)} settled turns")

# Also load eddy fit results if available
eddy_fits = {}
for name, out_dir in [("200 GeV", OUT_200), ("26 GeV", OUT_26)]:
    eddy_fits[name] = {}
    for seg in SEGMENTS:
        fpath = out_dir / f"b3_fits_{seg}.csv"
        if fpath.exists():
            eddy_fits[name][seg] = pd.read_csv(fpath)
            print(f"  {name} {seg} eddy fits: {len(eddy_fits[name][seg])} rows")
        else:
            eddy_fits[name][seg] = pd.DataFrame()"""))

    # 3. B1 Comparison
    cells.append(md("c3-hdr", "---\n## 3. B1 Comparison\n\nPer-supercycle B1 at injection and flat-top."))

    cells.append(code("c3-b1", """fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    fringe = " [fringe]" if seg == "CS" else ""
    for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
        ax = axes[i, j]
        for ds_name, col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
            dfs = ds[ds_name][seg]
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            sc_avg = sub.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
        ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)")
        ax.set_title(f"B1 {title_suffix} -- {seg}{fringe}"); ax.legend(fontsize=9)

fig.suptitle(f"B1 per Supercycle (settled, last {N_LAST_TURNS_INJ}/SC at injection)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 4. b2, b3 Comparison
    cells.append(md("c4-hdr", "---\n## 4. b2, b3 Comparison"))

    cells.append(code("c4-harmonics", """for harm_name, harm_col, ylabel in [("b2", "b2_units", "b2 (units)"), ("b3", "b3_units", "b3 (units)")]:
    fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
    if len(SEGMENTS) == 1:
        axes = axes[np.newaxis, :]

    for i, seg in enumerate(SEGMENTS):
        fringe = " [fringe]" if seg == "CS" else ""
        for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
            ax = axes[i, j]
            for ds_name, col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
                dfs = ds[ds_name][seg]
                sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
                if len(sub) == 0: continue
                sc_avg = sub.groupby("sc_idx")[harm_col].agg(["mean", "std"]).reset_index()
                ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                            fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel)
            ax.set_title(f"{harm_name} {title_suffix} -- {seg}{fringe}"); ax.legend(fontsize=9)

    fig.suptitle(f"{harm_name} per Supercycle (settled)", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()

# Box plots
fig, axes = plt.subplots(len(SEGMENTS), 3, figsize=(16, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    fringe = " [fringe]" if seg == "CS" else ""
    for ax_idx, (col_name, ylabel, title) in enumerate([
            ("B1_T", "B1 (T)", "B1"), ("b2_units", "b2 (units)", "b2"), ("b3_units", "b3 (units)", "b3")]):
        ax = axes[i, ax_idx]
        box_data, box_labels, box_colors = [], [], []
        for ds_name, base_col in [("200 GeV", "tab:blue"), ("26 GeV", "tab:orange")]:
            for lab, short in [("injection", "Inj"), ("flat-high", "FT")]:
                dfs = ds[ds_name][seg]
                sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
                if len(sub) == 0: continue
                box_data.append(sub[col_name].values)
                box_labels.append(f"{ds_name}\\n{short}\\n(N={len(sub)})")
                box_colors.append(base_col)
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            for patch, col in zip(bp["boxes"], box_colors): patch.set_facecolor(col); patch.set_alpha(0.5)
        ax.set_ylabel(ylabel); ax.set_title(f"{title} -- {seg}{fringe}")
        ax.tick_params(axis="x", labelsize=7)

fig.suptitle("Distribution Comparison (settled turns)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 5. Multipole Spectrum Comparison
    cells.append(md("c5-hdr", """---
## 5. Multipole Spectrum Comparison

Overlay normal harmonic spectra at injection for both energies (NCS only)."""))

    cells.append(code("c5-spectrum", """seg = "NCS"
lab = "injection"

# Determine number of harmonics from column names
bn_cols = [c for c in ds["200 GeV"][seg].columns if c.startswith("b") and c.endswith("_units")]
orders = sorted([int(c.replace("b", "").replace("_units", "")) for c in bn_cols])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(orders))
w = 0.35

for ax_idx, (title, yscale) in enumerate([("Linear", "linear"), ("Log", "log")]):
    ax = axes[ax_idx]
    for ds_name, offset, color in [("200 GeV", -w/2, "tab:blue"), ("26 GeV", w/2, "tab:orange")]:
        dfs = ds[ds_name][seg]
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
    ax.set_xlabel("Harmonic order n"); ax.set_ylabel("bn (units)" if yscale == "linear" else "|bn| (units)")
    ax.set_title(f"Multipole spectrum -- {lab} ({title})")
    ax.legend(fontsize=9)

fig.suptitle("Multipole Spectrum Comparison -- NCS Injection", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # 6. Statistical Significance
    cells.append(md("c6-hdr", "---\n## 6. Statistical Significance\n\n"
                     "Sigma = |diff| / sqrt(std1^2/N1 + std2^2/N2). > 3 sigma = real difference."))

    cells.append(code("c6-stats", """print(f"Difference: (200 GeV) - (26 GeV)  [settled, last {N_LAST_TURNS_INJ}/SC at injection]")
print("=" * 110)
all_results = []

for seg in SEGMENTS:
    fringe = " [FRINGE]" if seg == "CS" else ""
    print(f"\\n--- {seg}{fringe} ---")

    for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
        s200 = ds["200 GeV"][seg]
        s200 = s200[(s200["label"] == lab) & s200["ok_main"]]
        s26 = ds["26 GeV"][seg]
        s26 = s26[(s26["label"] == lab) & s26["ok_main"]]
        if len(s200) == 0 or len(s26) == 0: continue

        results_row = {"seg": seg, "label": lab, "desc": desc, "N_200": len(s200), "N_26": len(s26)}
        for name, col in [("B1", "B1_T"), ("b2", "b2_units"), ("b3", "b3_units")]:
            diff = s200[col].mean() - s26[col].mean()
            err = np.sqrt((s200[col].std()**2/len(s200)) + (s26[col].std()**2/len(s26)))
            sig = abs(diff) / err if err > 0 else 0
            results_row[f"d{name}"] = diff
            results_row[f"sig_{name}"] = sig
        all_results.append(results_row)

        print(f"  {desc:>12s}  dB1={results_row['dB1']:+.6f}  db2={results_row['db2']:+.4f}  "
              f"db3={results_row['db3']:+.4f}")
        print(f"  {'(sigma)':>12s}  {results_row['sig_B1']:>12.1f}  {results_row['sig_b2']:>14.1f}  "
              f"{results_row['sig_b3']:>14.1f}")

print("\\nINTERPRETATION")
print("-" * 70)
for r in all_results:
    fringe = " [fringe]" if r["seg"] == "CS" else ""
    print(f"\\n  {r['seg']}{fringe} -- {r['desc']}  (N: {r['N_200']} vs {r['N_26']} turns)")
    for name, unit in [("B1", "T"), ("b2", "units"), ("b3", "units")]:
        diff = r[f"d{name}"]
        sig = r[f"sig_{name}"]
        verdict = "REAL (>3 sigma)" if sig > 3 else ("suggestive (2-3 sigma)" if sig >= 2 else "no evidence (<2 sigma)")
        diff_str = f"{diff*1e6:+.1f} uT" if unit == "T" else f"{diff:+.4f} {unit}"
        print(f"    {name:>3s}: {diff_str:>16s}  ({sig:.1f} sigma) -> {verdict}")"""))

    # 7. Summary
    cells.append(md("c7-hdr", "---\n## 7. Summary"))

    cells.append(code("c7-summary", """summary_rows = []
for ds_name in ["200 GeV", "26 GeV"]:
    for seg in SEGMENTS:
        dfs = ds[ds_name][seg]
        for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1000.0) if "I_mean_A" in sub.columns else np.nan
            fringe = "*" if seg == "CS" else ""
            summary_rows.append({
                "Dataset": ds_name, "Segment": f"{seg}{fringe}", "Op. point": desc,
                "N turns": len(sub),
                "I mean (A)": f"{sub['I_mean_A'].mean():.1f}" if "I_mean_A" in sub.columns else "-",
                "B1 mean (T)": f"{sub['B1_T'].mean():.6f}", "B1 std (T)": f"{sub['B1_T'].std():.6f}",
                "b2 mean": f"{sub['b2_units'].mean():+.4f}", "b3 mean": f"{sub['b3_units'].mean():+.4f}",
                "TF (T/kA)": f"{tf:.4f}" if not np.isnan(tf) else "-",
            })
df_summary = pd.DataFrame(summary_rows)
print(f"[Settled data: last {N_LAST_TURNS_INJ} injection turns per supercycle]")
print(f"[* = fringe-field segment]\\n")
print(df_summary.to_string(index=False))

# Export comparison summary
out_dir = REPO_ROOT / "output" / "MBB/2026-02-25_2Hz" / "compare_200_vs_26"
out_dir.mkdir(parents=True, exist_ok=True)
df_summary.to_csv(out_dir / "summary_comparison_settled.csv", index=False)
print(f"\\nWrote {out_dir / 'summary_comparison_settled.csv'}")
print("\\nDone.")"""))

    return cells


# ================================================================
# Generate all 3 notebooks
# ================================================================

if __name__ == "__main__":
    print("Generating MBB 2 Hz notebooks (2026-02-25)...")

    # 1. 200 GeV comprehensive analysis
    cells_200 = build_analysis_cells(
        session="MBB/2026-02-25_2Hz/MBB/200 GeV/20260225_183154_SPS_MBB",
        meas_subdir="20260225_183213_MBB",
        energy_label="200 GeV",
        out_subdir="200GeV",
    )
    write_notebook(NOTEBOOK_DIR / "200GeV_analysis.ipynb", cells_200)

    # 2. 26 GeV comprehensive analysis
    cells_26 = build_analysis_cells(
        session="MBB/2026-02-25_2Hz/MBB/26 GeV/20260225_181040_SPS_MBB",
        meas_subdir="20260225_181058_MBB",
        energy_label="26 GeV",
        out_subdir="26GeV",
    )
    write_notebook(NOTEBOOK_DIR / "26GeV_analysis.ipynb", cells_26)

    # 3. Thin comparison
    cells_comp = build_comparison_cells()
    write_notebook(NOTEBOOK_DIR / "comparison.ipynb", cells_comp)

    print("\nAll 3 notebooks generated successfully.")
