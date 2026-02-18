"""Generate the 4 MBB analysis notebooks for 200 GeV / 26 GeV measurements."""

import json
from pathlib import Path

NOTEBOOK_DIR = Path("rotating_coil_analyzer/notebooks/SPS_MBB")


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


# ============================================================
# 1 & 2. Analysis notebooks (200 GeV and 26 GeV)
# ============================================================

def build_analysis_cells(session, meas_subdir, energy_label, out_subdir):
    """Build cell list for a streaming supercycle analysis notebook."""
    cells = []

    cells.append(md("title", f"""# SPS MBB Dipole -- NCS Streaming Supercycle Analysis ({energy_label} Extended)

**Measurement session:** `{out_subdir} / {session.split('/')[-1]}`
**Segment:** NCS
**Magnet:** MBB (normal dipole, m=1)
**Kn calibration:** AC compensation (cross-session from MBA, Dec 2025)
**Supercycle:** LHC_pilot -> MD1 ({energy_label}) -> SFTPRO, x20 repetitions

| Section | Content |
|---------|----------|
| 1 | Configuration & imports |
| 2 | Kn calibration |
| 3 | Data loading & channel detection |
| 4 | Raw signals overview |
| 5 | cel/fed safety diagnostic |
| 6 | Plateau detection & current classification |
| 7 | Pipeline processing |
| 8 | Main field (B1) analysis |
| 9 | Transfer function B/I vs I |
| 10 | b2 (quadrupole) analysis |
| 11 | b3 (sextupole) analysis |
| 12 | Per-supercycle evolution |
| 13 | Summary & export |"""))

    cells.append(md("sec1-hdr", "---\n## 1. Configuration & Imports"))

    cells.append(code("config", f"""# === CONFIGURATION ===
SEGMENT = "NCS"

SESSION = "{session}"
MEAS_SUBDIR = "{meas_subdir}"
KN_CROSS_SESSION = "20251212_171026_SPS_MBA/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt"

MAGNET_ORDER = 1          # dipole
R_REF = 0.02              # reference radius [m]
L_COIL = 0.47             # coil length [m]
SAMPLES_PER_TURN = 1024   # encoder samples per revolution

OPTIONS = ("dri", "rot", "cel", "fed")

MIN_B1_T = 1e-4           # minimum |B1| for normalization
PLATEAU_I_RANGE_MAX = 2.5 # block-averaged range threshold (A)
N_BLOCKS = 10             # blocks for range averaging

# Settling: last N turns per supercycle
N_LAST_TURNS_INJ = 18     # injection: keep last 18 of ~24
N_LAST_TURNS_HIGH = None   # flat-high: use all

# Outlier removal
N_SIGMA_CLIP = 5           # MAD sigma clipping

print(f"SPS MBB Dipole -- {{SEGMENT}} Streaming Supercycle Analysis ({energy_label} Extended)")
print("=" * 60)
print(f"  Session       : {{SESSION}}")
print(f"  Segment       : {{SEGMENT}}")
print(f"  Magnet order  : {{MAGNET_ORDER}} (dipole)")
print(f"  R_ref         : {{R_REF}} m")
print(f"  Samples/turn  : {{SAMPLES_PER_TURN}}")
print(f"  Options       : {{OPTIONS}}")
print(f"  Plateau thresh: {{PLATEAU_I_RANGE_MAX}} A")"""))

    cells.append(code("imports", """import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

%matplotlib widget
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})

repo_root = Path("../..").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
    process_kn_pipeline,
    build_harmonic_rows,
    diagnose_cel_fed,
    mad_sigma_clip,
)
from rotating_coil_analyzer.ingest.channel_detect import robust_range

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists():
        break
    REPO_ROOT = REPO_ROOT.parent

SESSION_DIR = REPO_ROOT / "measurements" / SESSION
RUN_DIR = SESSION_DIR / MEAS_SUBDIR
KN_PATH = REPO_ROOT / "measurements" / KN_CROSS_SESSION
assert KN_PATH.exists(), f"Kn file not found: {KN_PATH}"

print(f"Repo root   : {REPO_ROOT}")
print(f"Session dir : {SESSION_DIR}")
print(f"Run dir     : {RUN_DIR}")
print(f"Kn file     : {KN_PATH}")
print("Imports ready.")"""))

    cells.append(md("sec2-hdr", "---\n## 2. Kn Calibration (AC Compensation)"))

    cells.append(code("kn-load", """kn = load_segment_kn_txt(str(KN_PATH))
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

    cells.append(md("sec3-hdr", "---\n## 3. Data Loading & Channel Detection"))

    cells.append(code("data-load", """FILE_PAT = re.compile(
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

# Reshape
t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
flux_col1 = raw[:n_keep, 1].reshape(n_turns, Ns)
flux_col2 = raw[:n_keep, 2].reshape(n_turns, Ns)
I_all = raw[:n_keep, 3].reshape(n_turns, Ns)
has_voltage = ncols > 4
if has_voltage:
    V_all = raw[:n_keep, 4].reshape(n_turns, Ns)

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

print(f"\\nColumn assignment:")
print(f"  col 0  -> time (s)")
print(f"  col {'2' if SWAP_FLUX else '1'}  -> flux abs (range={max(r1,r2):.4e})")
print(f"  col {'1' if SWAP_FLUX else '2'}  -> flux cmp (range={min(r1,r2):.4e})")
print(f"  col 3  -> current")
if has_voltage:
    print(f"  col 4  -> voltage")"""))

    cells.append(md("sec4-hdr", "---\n## 4. Raw Signals Overview"))

    cells.append(code("raw-signals", """fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

x = np.arange(n_keep)
ax = axes[0]
ax.plot(x, raw[:n_keep, 1 if not SWAP_FLUX else 2], linewidth=0.2, color="steelblue")
ax.set_ylabel("Flux abs (Wb)")
ax.set_title("Absolute flux channel")

ax = axes[1]
ax.plot(x, raw[:n_keep, 2 if not SWAP_FLUX else 1], linewidth=0.2, color="teal")
ax.set_ylabel("Flux cmp (Wb)")
ax.set_title("Compensated flux channel")

ax = axes[2]
ax.plot(x, raw[:n_keep, 3], linewidth=0.2, color="tab:orange")
ax.set_ylabel("Current (A)")
ax.set_title("Current channel")

axes[-1].set_xlabel("Sample index")
fig.suptitle(f"Raw signals -- {SESSION} ({SEGMENT})", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

    cells.append(md("sec5-hdr", "---\n## 5. cel/fed Safety Diagnostic\n\nRun `diagnose_cel_fed()` on high-current turns to check whether the\ncentre-location + feeddown correction is safe for this data."))

    cells.append(code("cel-fed-diag", """I_mean = I_all.mean(axis=1)
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

    cells.append(md("sec6-hdr", "---\n## 6. Plateau Detection & Current Classification"))

    cells.append(code("plateau-detect", """I_mean = I_all.mean(axis=1)
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
fig, ax = plt.subplots(figsize=(16, 5))
label_colors = {"injection": "tab:green", "flat-mid": "tab:purple", "flat-high": "tab:blue"}
ax.plot(t_mean, I_mean, ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
for lab, col in label_colors.items():
    mask = turn_label == lab
    idx = np.where(mask)[0]
    if len(idx) > 0:
        ax.scatter(t_mean[idx], I_mean[idx], s=6, color=col, zorder=2, label=lab)
ax.set_xlabel("Time (s)")
ax.set_ylabel("I (A)")
ax.set_title(f"Current profile with plateau classification -- {SEGMENT}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()"""))

    cells.append(md("sec7-hdr", "---\n## 7. Pipeline Processing"))

    cells.append(code("pipeline", """ANALYSIS_LABELS = {"injection", "flat-mid", "flat-high"}
is_analysis = np.array([l in ANALYSIS_LABELS for l in turn_label])
plateau_indices = np.where(is_analysis)[0]
print(f"Processing {len(plateau_indices)} plateau turns through Kn pipeline...")

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

# Group injection turns by supercycle
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

print(f"\\nAll plateau turns: {len(df)}")
print(f"Settled turns (after sigma clip): {len(df_settled)}")
for lab in ["injection", "flat-high"]:
    n_all = len(df[df["label"] == lab])
    n_set = len(df_settled[df_settled["label"] == lab])
    print(f"  {lab:12s}: {n_all} -> {n_set}")
print(f"\\nok_main: {df['ok_main'].sum()} / {len(df)}")
print(f"Harmonics: n=1..{H}")"""))

    cells.append(md("sec8-hdr", "---\n## 8. Main Field (B1) Analysis"))

    cells.append(code("b1-analysis", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ok = df["ok_main"]
ax = axes[0, 0]
ax.scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "B1_T"], s=8, alpha=0.5, color="steelblue")
ax.set_xlabel("I (A)"); ax.set_ylabel("B1 (T)")
ax.set_title(f"B1 vs current ({SEGMENT})")

ax = axes[0, 1]
ax.plot(df.loc[ok, "time_s"].values, df.loc[ok, "B1_T"].values,
        ".-", markersize=2, linewidth=0.3, color="steelblue")
ax.set_xlabel("Time (s)"); ax.set_ylabel("B1 (T)")
ax.set_title(f"B1 time series ({SEGMENT})")

ax = axes[1, 0]
inj_set = df_settled[(df_settled["label"] == "injection") & df_settled["ok_main"]]
if len(inj_set) > 0:
    sc_b1 = inj_set.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b1["sc_idx"], sc_b1["mean"], yerr=sc_b1["std"],
                fmt="o-", markersize=4, capsize=2, color="tab:green")
ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)")
ax.set_title("B1 per supercycle (injection, settled)")

ax = axes[1, 1]
fh_set = df_settled[(df_settled["label"] == "flat-high") & df_settled["ok_main"]]
if len(fh_set) > 0:
    sc_b1_fh = fh_set.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b1_fh["sc_idx"], sc_b1_fh["mean"], yerr=sc_b1_fh["std"],
                fmt="s-", markersize=4, capsize=2, color="tab:blue")
ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)")
ax.set_title("B1 per supercycle (SFTPRO flat-top)")

fig.suptitle(f"Main Field (B1) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\\nB1 per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'mean (T)':>12s} {'std (T)':>12s}")
print("-" * 45)
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        print(f"{lab:>12s} {len(sub):5d} {sub['B1_T'].mean():+12.6f} {sub['B1_T'].std():12.6f}")"""))

    cells.append(md("sec9-hdr", "---\n## 9. Transfer Function B/I vs I"))

    cells.append(code("tf-analysis", """ok_tf = df_settled["ok_main"] & (df_settled["I_mean_A"].abs() > 10)
df_tf = df_settled[ok_tf].copy()
df_tf["TF_mTpA"] = df_tf["B1_T"] / df_tf["I_mean_A"] * 1e3

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(df_tf["I_mean_A"], df_tf["B1_T"], s=8, alpha=0.5, color="steelblue")
ax.set_xlabel("I (A)"); ax.set_ylabel("B1 (T)"); ax.set_title(f"B1 vs I ({SEGMENT})")

ax = axes[1]
for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_tf[df_tf["label"] == lab]
    if len(sub) > 0:
        ax.scatter(sub["I_mean_A"], sub["TF_mTpA"], s=10, alpha=0.5, color=col, label=lab)
ax.set_xlabel("I (A)"); ax.set_ylabel("TF = B1/I (mT/A)")
ax.set_title(f"Transfer function vs current ({SEGMENT})"); ax.legend(fontsize=9)

ax = axes[2]
inj_tf = df_tf[df_tf["label"] == "injection"]
if len(inj_tf) > 0:
    sc_tf = inj_tf.groupby("sc_idx")["TF_mTpA"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_tf["sc_idx"], sc_tf["mean"], yerr=sc_tf["std"],
                fmt="o-", markersize=4, capsize=2, color="tab:green")
ax.set_xlabel("Supercycle index"); ax.set_ylabel("TF (mT/A)")
ax.set_title("TF per supercycle (injection)")

fig.suptitle(f"Transfer Function -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\\nTransfer function per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'TF mean (mT/A)':>16s} {'TF std (mT/A)':>16s}")
print("-" * 55)
for lab in ["injection", "flat-high"]:
    sub = df_tf[df_tf["label"] == lab]
    if len(sub) > 0:
        print(f"{lab:>12s} {len(sub):5d} {sub['TF_mTpA'].mean():16.6f} {sub['TF_mTpA'].std():16.6f}")"""))

    # b2 analysis
    cells.append(md("sec10-hdr", "---\n## 10. b2 (Quadrupole) Analysis\n\nb2 is the first allowed harmonic error for a dipole."))

    cells.append(code("b2-analysis", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ok_nz = df["ok_main"]
b2_all = df.loc[ok_nz, "b2_units"].values

ax = axes[0, 0]
ax.scatter(df.loc[ok_nz, "I_mean_A"], b2_all, s=8, alpha=0.5, color="steelblue")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("I (A)"); ax.set_ylabel("b2 (units)")
ax.set_title(f"b2 vs current ({SEGMENT})")

ax = axes[0, 1]
ax.plot(df.loc[ok_nz, "time_s"].values, b2_all, ".-", markersize=2, linewidth=0.3, color="steelblue")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Time (s)"); ax.set_ylabel("b2 (units)")
ax.set_title(f"b2 time series ({SEGMENT})")

ax = axes[1, 0]
inj_set = df_settled[(df_settled["label"] == "injection") & df_settled["ok_main"]]
if len(inj_set) > 0:
    sc_b2 = inj_set.groupby("sc_idx")["b2_units"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b2["sc_idx"], sc_b2["mean"], yerr=sc_b2["std"],
                fmt="o-", markersize=4, capsize=2, color="tab:green", label="injection")
fh_set = df_settled[(df_settled["label"] == "flat-high") & df_settled["ok_main"]]
if len(fh_set) > 0:
    sc_b2_fh = fh_set.groupby("sc_idx")["b2_units"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b2_fh["sc_idx"], sc_b2_fh["mean"], yerr=sc_b2_fh["std"],
                fmt="s-", markersize=4, capsize=2, color="tab:blue", label="flat-high")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Supercycle index"); ax.set_ylabel("b2 (units)")
ax.set_title("b2 per supercycle (settled)"); ax.legend(fontsize=9)

ax = axes[1, 1]
for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        ax.hist(sub["b2_units"], bins=30, alpha=0.5, color=col, label=lab, edgecolor="black", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("b2 (units)"); ax.set_ylabel("Count")
ax.set_title("b2 distribution (settled)"); ax.legend(fontsize=9)

fig.suptitle(f"b2 (Quadrupole) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\\nb2 per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'mean':>10s} {'std':>10s} {'median':>10s}")
print("-" * 50)
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        v = sub["b2_units"].values
        print(f"{lab:>12s} {len(sub):5d} {v.mean():+10.4f} {v.std():10.4f} {np.median(v):+10.4f}")"""))

    # b3 analysis
    cells.append(md("sec11-hdr", "---\n## 11. b3 (Sextupole) Analysis\n\nb3 is the first non-allowed harmonic for an ideal dipole and a key quality indicator."))

    cells.append(code("b3-analysis", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ok_nz = df["ok_main"]
b3_all = df.loc[ok_nz, "b3_units"].values

ax = axes[0, 0]
ax.scatter(df.loc[ok_nz, "I_mean_A"], b3_all, s=8, alpha=0.5, color="steelblue")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("I (A)"); ax.set_ylabel("b3 (units)")
ax.set_title(f"b3 vs current ({SEGMENT})")

ax = axes[0, 1]
ax.plot(df.loc[ok_nz, "time_s"].values, b3_all, ".-", markersize=2, linewidth=0.3, color="steelblue")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Time (s)"); ax.set_ylabel("b3 (units)")
ax.set_title(f"b3 time series ({SEGMENT})")

ax = axes[1, 0]
inj_set = df_settled[(df_settled["label"] == "injection") & df_settled["ok_main"]]
if len(inj_set) > 0:
    sc_b3 = inj_set.groupby("sc_idx")["b3_units"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b3["sc_idx"], sc_b3["mean"], yerr=sc_b3["std"],
                fmt="o-", markersize=4, capsize=2, color="tab:green", label="injection")
fh_set = df_settled[(df_settled["label"] == "flat-high") & df_settled["ok_main"]]
if len(fh_set) > 0:
    sc_b3_fh = fh_set.groupby("sc_idx")["b3_units"].agg(["mean", "std"]).reset_index()
    ax.errorbar(sc_b3_fh["sc_idx"], sc_b3_fh["mean"], yerr=sc_b3_fh["std"],
                fmt="s-", markersize=4, capsize=2, color="tab:blue", label="flat-high")
ax.axhline(0, color="grey", linewidth=0.5)
ax.set_xlabel("Supercycle index"); ax.set_ylabel("b3 (units)")
ax.set_title("b3 per supercycle (settled)"); ax.legend(fontsize=9)

ax = axes[1, 1]
for lab, col in [("injection", "tab:green"), ("flat-high", "tab:blue")]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        ax.hist(sub["b3_units"], bins=30, alpha=0.5, color=col, label=lab, edgecolor="black", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("b3 (units)"); ax.set_ylabel("Count")
ax.set_title("b3 distribution (settled)"); ax.legend(fontsize=9)

fig.suptitle(f"b3 (Sextupole) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\\nb3 per operating point (settled turns):")
print(f"{'Label':>12s} {'N':>5s} {'mean':>10s} {'std':>10s} {'median':>10s}")
print("-" * 50)
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        v = sub["b3_units"].values
        print(f"{lab:>12s} {len(sub):5d} {v.mean():+10.4f} {v.std():10.4f} {np.median(v):+10.4f}")"""))

    # Per-supercycle evolution
    cells.append(md("sec12-hdr", "---\n## 12. Per-Supercycle Evolution\n\nTrack how B1, b2, b3 evolve across the ~20 supercycles at each operating point."))

    cells.append(code("per-sc", """fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (col_name, ylabel) in zip(axes, [("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")[col_name].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, alpha=0.8, label=lab)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel); ax.legend(fontsize=9)

axes[0].set_title("B1 per supercycle")
axes[1].set_title("b2 per supercycle")
axes[2].set_title("b3 per supercycle")

fig.suptitle(f"Per-Supercycle Evolution (settled turns) -- {SESSION} ({SEGMENT})", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\\nStability across supercycles (settled turns):")
print(f"{'Quantity':>12s}  {'Label':>12s}  {'SC mean':>12s}  {'SC std':>12s}  {'SC p-p':>12s}")
print("-" * 65)
for col_name, label, unit in [("B1_T", "B1", "T"), ("b2_units", "b2", "units"), ("b3_units", "b3", "units")]:
    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_means = sub.groupby("sc_idx")[col_name].mean()
            print(f"{label:>12s}  {lab:>12s}  {sc_means.mean():+12.6f}  "
                  f"{sc_means.std():12.6f}  {sc_means.max()-sc_means.min():12.6f}")"""))

    # Summary & Export
    cells.append(md("sec13-hdr", "---\n## 13. Summary & Export"))

    cells.append(code("summary", f"""print("=" * 70)
print(f"SPS MBB DIPOLE -- {{SEGMENT}} STREAMING SUPERCYCLE ANALYSIS ({energy_label} Extended)")
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

print(f"\\n--- Settled Turns Summary ---")
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
    if len(sub) > 0:
        tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1e3)
        print(f"  {{lab:12s}}: N={{len(sub):4d}}, I={{sub['I_mean_A'].mean():.1f}} A, "
              f"B1={{sub['B1_T'].mean():+.6f}} T, "
              f"b2={{sub['b2_units'].mean():+.3f}}, b3={{sub['b3_units'].mean():+.3f}} units, "
              f"TF={{tf:.4f}} T/kA")"""))

    cells.append(code("export", f"""out_dir = REPO_ROOT / "output" / "2026_02_06" / "{out_subdir}"
out_dir.mkdir(parents=True, exist_ok=True)

fname = f"MBB_{{SEGMENT}}_streaming_plateau.csv"
df.to_csv(out_dir / fname, index=False)
print(f"Wrote {{out_dir / fname}}  ({{len(df)}} rows)")

fname_s = f"MBB_{{SEGMENT}}_streaming_settled.csv"
df_settled.to_csv(out_dir / fname_s, index=False)
print(f"Wrote {{out_dir / fname_s}}  ({{len(df_settled)}} rows)")

print("\\nDone.")"""))

    return cells


# ============================================================
# 3. Eddy current notebook
# ============================================================

def build_eddy_current_cells():
    """Build cell list for the eddy current notebook."""
    cells = []

    cells.append(md("title", """# Eddy Current b3 Settling Time -- SPS MBB 200 GeV & 26 GeV Cycles

## Physics Motivation

The SPS supercycle follows **LHC_pilot -> MD1 -> SFTPRO**. The LHC pilot
excites the magnet to ~5785 A. During the subsequent MD1 injection plateau
(~301 A), eddy currents cause b3 to drift exponentially:

$$b_3(t) = b_{3,\\infty} + A \\cdot \\exp\\!\\left(-\\frac{t - t_0}{\\tau}\\right)$$

## Datasets

| Label | Session | Description |
|-------|---------|-------------|
| **200 GeV Extended** | `01_200_extended` | Extended MD1 at 200 GeV, ~20 supercycles |
| **200 GeV Original** | `02_200_original` | Original MD1 at 200 GeV |
| **26 GeV Extended** | `03_26_extended` | Extended MD1 at 26 GeV, ~20 supercycles |
| **26 GeV Original** | `04_26_original` | Original MD1 at 26 GeV |"""))

    cells.append(md("sec1-hdr", "---\n## 1. Configuration"))

    cells.append(code("config", """# --- 200 GeV ---
EXT_200_SESSION = "2026_02_06/01_200_extended/20260206_144537_SPS_MBB"
EXT_200_SUBDIR  = "20260206_144559_MBB"
ORIG_200_SESSION = "2026_02_06/02_200_original/20260206_150502_SPS_MBB"
ORIG_200_SUBDIR  = "20260206_150529_MBB"

# --- 26 GeV ---
EXT_26_SESSION = "2026_02_06/03_26_extended/20260206_151808_SPS_MBB"
EXT_26_SUBDIR  = "20260206_151827_MBB"
ORIG_26_SESSION = "2026_02_06/04_26_original/20260206_153712_SPS_MBB"
ORIG_26_SUBDIR  = "20260206_153730_MBB"

# --- Common ---
SEGMENT = "NCS"
KN_CROSS_SESSION = "20251212_171026_SPS_MBA/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt"
MAGNET_ORDER = 1
R_REF = 0.02
SAMPLES_PER_TURN = 1024
OPTIONS = ("dri", "rot", "cel", "fed")
PLATEAU_I_RANGE_MAX = 3.0
MIN_B1_T = 1e-4
MIN_INJECTION_TURNS = 5

print("Eddy Current b3 Settling -- SPS MBB (200 GeV & 26 GeV)")
print("=" * 60)"""))

    cells.append(md("sec2-hdr", "---\n## 2. Imports"))

    cells.append(code("imports", """import sys
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

repo_root = Path("../..").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
    process_kn_pipeline,
    build_harmonic_rows,
    diagnose_cel_fed,
)
from rotating_coil_analyzer.ingest.channel_detect import robust_range

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists():
        break
    REPO_ROOT = REPO_ROOT.parent

KN_PATH = REPO_ROOT / "measurements" / KN_CROSS_SESSION
assert KN_PATH.exists(), f"Kn file not found: {KN_PATH}"
kn = load_segment_kn_txt(str(KN_PATH))
print(f"Kn loaded: {len(kn.orders)} harmonics from {KN_PATH.name}")
print("Imports ready.")"""))

    cells.append(md("sec3-hdr", "---\n## 3. cel/fed Safety Diagnostic\n\nRun `diagnose_cel_fed()` on high-current turns from the 200 GeV extended dataset."))

    cells.append(code("cel-fed", """Ns = SAMPLES_PER_TURN
m = MAGNET_ORDER
FILE_PAT = re.compile(r"Run_(\\d+)_I_([\\d.]+)A_(N?CS)_raw_measurement_data\\.txt$")

_session_dir = REPO_ROOT / "measurements" / EXT_200_SESSION
_run_dir = _session_dir / EXT_200_SUBDIR
_ncs_files = [f for f in sorted(_run_dir.iterdir()) if FILE_PAT.search(f.name) and SEGMENT in f.name]
assert _ncs_files, "No NCS raw file found"
_raw = np.loadtxt(_ncs_files[0])
_n_turns = _raw.shape[0] // Ns
_n_keep = _n_turns * Ns

_t_all = _raw[:_n_keep, 0].reshape(_n_turns, Ns)
_flux_abs = _raw[:_n_keep, 1].reshape(_n_turns, Ns)
_flux_cmp = _raw[:_n_keep, 2].reshape(_n_turns, Ns)
_I_all = _raw[:_n_keep, 3].reshape(_n_turns, Ns)

_I_mean_quick = _I_all.mean(axis=1)
_best_turn = np.argmax(np.abs(_I_mean_quick))
_r1 = robust_range(_raw[_best_turn * Ns:(_best_turn + 1) * Ns, 1])
_r2 = robust_range(_raw[_best_turn * Ns:(_best_turn + 1) * Ns, 2])
if _r2 > _r1:
    _flux_abs = _raw[:_n_keep, 2].reshape(_n_turns, Ns)
    _flux_cmp = _raw[:_n_keep, 1].reshape(_n_turns, Ns)

_I_mean = _I_all.mean(axis=1)
_hi_mask = np.abs(_I_mean) > 4000
if _hi_mask.sum() < 5:
    _hi_mask = np.abs(_I_mean) > np.percentile(np.abs(_I_mean), 90)
_n_diag = min(100, int(_hi_mask.sum()))
_hi_idx = np.where(_hi_mask)[0][:_n_diag]

diag = diagnose_cel_fed(
    _flux_abs[_hi_idx], _flux_cmp[_hi_idx],
    _t_all[_hi_idx], _I_all[_hi_idx],
    kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER,
)
print(f"cel/fed diagnostic ({_n_diag} high-I turns from 200 GeV extended):")
print(f"  {diag.recommendation}")
print(f"  {diag.reason}")
_Bd = np.max(np.abs(diag.B_main_with_fed - diag.B_main_without_fed))
print(f"  B_main max |diff|: {_Bd:.4e} T")
del _session_dir, _run_dir, _ncs_files, _raw, _n_turns, _n_keep
del _t_all, _flux_abs, _flux_cmp, _I_all, _I_mean_quick, _best_turn
del _r1, _r2, _I_mean, _hi_mask, _n_diag, _hi_idx, _Bd

if diag.recommendation == "UNSAFE":
    OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
    print(f"  -> cel/fed disabled, OPTIONS = {OPTIONS}")
else:
    print(f"  -> cel/fed safe, keeping OPTIONS = {OPTIONS}")"""))

    cells.append(md("sec4-hdr", "---\n## 4. Helper: Load & Process One Dataset"))

    cells.append(code("helper", """N_BLOCKS = 10

def load_and_process(session, meas_subdir, dataset_label=""):
    session_dir = REPO_ROOT / "measurements" / session
    run_dir = session_dir / meas_subdir
    ncs_files = []
    for f in sorted(run_dir.iterdir()):
        match = FILE_PAT.search(f.name)
        if match and match.group(3) == SEGMENT:
            ncs_files.append(f)
    assert ncs_files, f"No {SEGMENT} raw files found in {run_dir}"
    raw_file = ncs_files[0]
    print(f"\\n{'='*60}")
    print(f"  Dataset: {dataset_label or session}")
    print(f"  Raw file: {raw_file.name}")

    raw = np.loadtxt(raw_file)
    n_turns = raw.shape[0] // Ns
    n_keep = n_turns * Ns
    print(f"  Shape: {raw.shape} -> {n_turns} turns")
    print(f"  Time span: {raw[-1,0] - raw[0,0]:.1f} s ({(raw[-1,0] - raw[0,0])/60:.1f} min)")

    t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
    flux_abs_all = raw[:n_keep, 1].reshape(n_turns, Ns)
    flux_cmp_all = raw[:n_keep, 2].reshape(n_turns, Ns)
    I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

    I_mean_quick = I_all.mean(axis=1)
    best_turn = np.argmax(np.abs(I_mean_quick))
    r1 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 1])
    r2 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 2])
    if r2 > r1:
        flux_abs_all = raw[:n_keep, 2].reshape(n_turns, Ns)
        flux_cmp_all = raw[:n_keep, 1].reshape(n_turns, Ns)
        print("  (flux columns swapped)")

    I_mean = I_all.mean(axis=1)
    t_mean = t_all.mean(axis=1)
    I_range, I_blocks = compute_block_averaged_range(I_all, Ns, N_BLOCKS)
    plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
    is_plateau = plateau_info["is_plateau"]

    turn_label = np.array(["ramp"] * n_turns, dtype=object)
    for i in range(n_turns):
        if is_plateau[i]:
            turn_label[i] = classify_current(I_mean[i])

    inj_mask = turn_label == "injection"
    inj_groups = find_contiguous_groups(inj_mask, min_length=2)
    print(f"  Plateau turns: {is_plateau.sum()}, Injection groups: {len(inj_groups)}")

    all_rows = []
    for sc_id, (gs, ge) in enumerate(inj_groups):
        idx = np.arange(gs, ge + 1)
        if len(idx) == 0:
            continue
        result, C_merged, C_units, ok_main = process_kn_pipeline(
            flux_abs_turns=flux_abs_all[idx], flux_cmp_turns=flux_cmp_all[idx],
            t_turns=t_all[idx], I_turns=I_all[idx],
            kn=kn, r_ref=R_REF, magnet_order=m,
            options=OPTIONS, min_b1_T=MIN_B1_T,
        )
        t_inj_start = t_mean[gs]
        extra = [
            {"global_turn": int(idx[t]), "label": str(turn_label[idx[t]]),
             "supercycle_id": sc_id,
             "t_since_inj_start": float(t_mean[idx[t]] - t_inj_start)}
            for t in range(len(idx))
        ]
        rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"  Injection turns processed: {len(df)}")
    if len(df) > 0:
        print(f"  Supercycles: {df['supercycle_id'].nunique()}")
        print(f"  b3 range: {df['b3_units'].min():.2f} .. {df['b3_units'].max():.2f} units")
    return df, t_mean, I_mean, turn_label, is_plateau, inj_groups"""))

    cells.append(md("sec5-hdr", "---\n## 5. Load All Four Datasets"))

    cells.append(code("load-all", """df_ext200, t_ext200, I_ext200, lbl_ext200, plat_ext200, inj_ext200 = load_and_process(
    EXT_200_SESSION, EXT_200_SUBDIR, "200 GeV Extended")
df_orig200, t_orig200, I_orig200, lbl_orig200, plat_orig200, inj_orig200 = load_and_process(
    ORIG_200_SESSION, ORIG_200_SUBDIR, "200 GeV Original")
df_ext26, t_ext26, I_ext26, lbl_ext26, plat_ext26, inj_ext26 = load_and_process(
    EXT_26_SESSION, EXT_26_SUBDIR, "26 GeV Extended")
df_orig26, t_orig26, I_orig26, lbl_orig26, plat_orig26, inj_orig26 = load_and_process(
    ORIG_26_SESSION, ORIG_26_SUBDIR, "26 GeV Original")"""))

    cells.append(md("sec6-hdr", "---\n## 6. Current Profile Overview"))

    cells.append(code("current-profiles", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))
datasets_plot = [
    (axes[0,0], t_ext200, I_ext200, lbl_ext200, plat_ext200, inj_ext200, "200 GeV Extended"),
    (axes[0,1], t_orig200, I_orig200, lbl_orig200, plat_orig200, inj_orig200, "200 GeV Original"),
    (axes[1,0], t_ext26, I_ext26, lbl_ext26, plat_ext26, inj_ext26, "26 GeV Extended"),
    (axes[1,1], t_orig26, I_orig26, lbl_orig26, plat_orig26, inj_orig26, "26 GeV Original"),
]
for ax, t_m, I_m, lbl, plat, inj_grps, title in datasets_plot:
    ax.plot(t_m, I_m, ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
    for gs, ge in inj_grps:
        ax.plot(t_m[gs:ge+1], I_m[gs:ge+1], ".-", markersize=3, linewidth=1, color="tab:green", zorder=2)
    non_inj_plat = (plat) & (lbl != "injection") & (lbl != "ramp")
    idx_nip = np.where(non_inj_plat)[0]
    if len(idx_nip) > 0:
        ax.scatter(t_m[idx_nip], I_m[idx_nip], s=4, color="tab:blue", zorder=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("I (A)"); ax.set_title(title)
    ax.legend([Patch(color="tab:green"), Patch(color="tab:blue")],
             ["injection", "flat-top"], fontsize=8, loc="upper right")
fig.suptitle("Current Profiles -- All Datasets", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()"""))

    cells.append(md("sec7-hdr", "---\n## 7. b3 vs Time Since Injection Start -- Supercycle Overlay"))

    cells.append(code("b3-overlay", """fig, axes = plt.subplots(2, 2, figsize=(16, 10))
all_datasets = [
    (axes[0,0], df_ext200, "200 GeV Extended"), (axes[0,1], df_orig200, "200 GeV Original"),
    (axes[1,0], df_ext26, "26 GeV Extended"), (axes[1,1], df_orig26, "26 GeV Original"),
]
for ax, df, title in all_datasets:
    if len(df) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title); continue
    sc_ids = sorted(df["supercycle_id"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(sc_ids), 1)))
    for i, sc_id in enumerate(sc_ids):
        sub = df[df["supercycle_id"] == sc_id]
        ax.plot(sub["t_since_inj_start"], sub["b3_units"], ".-",
                markersize=4, linewidth=0.8, alpha=0.7, color=cmap[i % len(cmap)])
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("t - t_inj_start (s)"); ax.set_ylabel("b3 (units)")
    ax.set_title(f"{title} -- supercycles overlaid")
fig.suptitle("b3 vs Time Since Injection Start", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()"""))

    cells.append(md("sec8-hdr", "---\n## 8. Exponential Fit per Supercycle"))

    cells.append(code("exp-fit", """def eddy_model(t, b3_inf, A, tau):
    return b3_inf + A * np.exp(-t / tau)

def fit_supercycle(df_sc):
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

all_dfs = [
    ("200 GeV Ext", df_ext200), ("200 GeV Orig", df_orig200),
    ("26 GeV Ext", df_ext26), ("26 GeV Orig", df_orig26),
]
fit_results = {}
for ds_name, df in all_dfs:
    fits = []
    if len(df) == 0:
        fit_results[ds_name] = fits; continue
    for sc_id in sorted(df["supercycle_id"].unique()):
        result = fit_supercycle(df[df["supercycle_id"] == sc_id])
        if result is not None:
            result["supercycle_id"] = sc_id
            fits.append(result)
    fit_results[ds_name] = fits
    print(f"{ds_name}: {len(fits)} / {df['supercycle_id'].nunique()} supercycles fitted")

df_fits = {name: pd.DataFrame(fits) for name, fits in fit_results.items()}

for name, df_f in df_fits.items():
    if len(df_f) == 0: continue
    print(f"\\n{name}:")
    print(f"  {'SC':>3s} {'tau (s)':>10s} {'A (units)':>12s} {'b3_inf':>10s} {'R2':>6s}")
    print(f"  {'-'*45}")
    for _, row in df_f.iterrows():
        print(f"  {int(row['supercycle_id']):3d} {row['tau']:10.2f} {row['A']:+12.4f} "
              f"{row['b3_inf']:+10.4f} {row['r2']:6.3f}")"""))

    cells.append(md("sec9-hdr", "---\n## 9. Fit Overlay on Representative Supercycles"))

    cells.append(code("fit-overlay", """n_show = 3
fig, axes = plt.subplots(4, n_show, figsize=(14, 16))
for row_idx, (ds_name, df, df_f) in enumerate([
    ("200 GeV Ext", df_ext200, df_fits["200 GeV Ext"]),
    ("200 GeV Orig", df_orig200, df_fits["200 GeV Orig"]),
    ("26 GeV Ext", df_ext26, df_fits["26 GeV Ext"]),
    ("26 GeV Orig", df_orig26, df_fits["26 GeV Orig"]),
]):
    if len(df_f) == 0:
        for j in range(n_show):
            axes[row_idx, j].text(0.5, 0.5, "No fits", ha="center", va="center",
                                  transform=axes[row_idx, j].transAxes)
            axes[row_idx, j].set_title(ds_name)
        continue
    sc_ids = df_f["supercycle_id"].values
    show_ids = sc_ids[np.linspace(0, len(sc_ids)-1, min(n_show, len(sc_ids)), dtype=int)]
    for j, sc_id in enumerate(show_ids):
        ax = axes[row_idx, j]
        sub = df[df["supercycle_id"] == sc_id]
        fit_row = df_f[df_f["supercycle_id"] == sc_id].iloc[0]
        ax.scatter(sub["t_since_inj_start"], sub["b3_units"], s=15, alpha=0.7, color="tab:blue", label="data")
        t_fit = np.linspace(0, sub["t_since_inj_start"].max() * 1.05, 200)
        ax.plot(t_fit, eddy_model(t_fit, fit_row["b3_inf"], fit_row["A"], fit_row["tau"]),
                "r-", linewidth=1.5, label="fit")
        ax.set_title(f"{ds_name} SC {int(sc_id)}\\ntau={fit_row['tau']:.1f}s, R2={fit_row['r2']:.3f}", fontsize=9)
        ax.set_xlabel("t (s)")
        if j == 0: ax.set_ylabel("b3 (units)")
        ax.legend(fontsize=7)
    for j in range(len(show_ids), n_show):
        axes[row_idx, j].set_visible(False)
fig.suptitle("Exponential Fit -- Representative Supercycles", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()"""))

    cells.append(md("sec10-hdr", "---\n## 10. Tau Comparison Across All Datasets"))

    cells.append(code("tau-comparison", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
box_data, box_labels = [], []
colors_bp = ["tab:green", "tab:olive", "tab:orange", "tab:red"]
for (name, df_f), col in zip(df_fits.items(), colors_bp):
    if len(df_f) > 0:
        box_data.append(df_f["tau"].values)
        box_labels.append(f"{name}\\n(N={len(df_f)})")
if box_data:
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, col in zip(bp["boxes"], colors_bp[:len(box_data)]):
        patch.set_facecolor(col); patch.set_alpha(0.5)
ax.set_ylabel("tau (s)"); ax.set_title("Settling Time Constant Distribution")

ax = axes[1]
markers = ["o", "s", "^", "D"]
for (name, df_f), col, marker in zip(df_fits.items(), colors_bp, markers):
    if len(df_f) > 0:
        ax.errorbar(df_f["supercycle_id"], df_f["tau"], yerr=df_f["tau_err"],
                    fmt=marker, color=col, markersize=5, capsize=3, alpha=0.7, label=name)
ax.set_xlabel("Supercycle ID"); ax.set_ylabel("tau (s)")
ax.set_title("tau per Supercycle"); ax.legend(fontsize=8)
fig.suptitle("tau Comparison -- All Datasets", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print("\\nTau statistics:")
print(f"  {'Dataset':>16s} {'N':>4s} {'mean (s)':>10s} {'std (s)':>10s} {'median':>10s} {'R2 mean':>10s}")
print(f"  {'-'*65}")
for name, df_f in df_fits.items():
    if len(df_f) == 0:
        print(f"  {name:>16s} {'--':>4s}"); continue
    tau_v = df_f["tau"].values; r2_v = df_f["r2"].values
    print(f"  {name:>16s} {len(df_f):4d} {tau_v.mean():10.2f} {tau_v.std():10.2f} "
          f"{np.median(tau_v):10.2f} {r2_v.mean():10.3f}")"""))

    cells.append(md("sec11-hdr", "---\n## 11. Global Fit (All Supercycles Stacked)"))

    cells.append(code("global-fit", """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
global_fits = {}
for ax, (ds_name, df) in zip(axes.ravel(), all_dfs):
    if len(df) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(ds_name); continue
    t_stack = df["t_since_inj_start"].values
    b3_stack = df["b3_units"].values
    ok = np.isfinite(t_stack) & np.isfinite(b3_stack)
    t_stack, b3_stack = t_stack[ok], b3_stack[ok]
    ax.scatter(t_stack, b3_stack, s=6, alpha=0.3, color="tab:blue", label="data")
    b3_inf_0 = np.median(b3_stack[t_stack > np.percentile(t_stack, 80)])
    A_0 = np.median(b3_stack[t_stack < np.percentile(t_stack, 20)]) - b3_inf_0
    try:
        popt, pcov = curve_fit(eddy_model, t_stack, b3_stack,
            p0=[b3_inf_0, A_0, max(t_stack.max()/3, 1.0)],
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 1000]), maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        b3_pred = eddy_model(t_stack, *popt)
        ss_res = np.sum((b3_stack - b3_pred) ** 2)
        ss_tot = np.sum((b3_stack - b3_stack.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        global_fits[ds_name] = {"tau": popt[2], "tau_err": perr[2], "A": popt[1], "A_err": perr[1],
                                 "b3_inf": popt[0], "b3_inf_err": perr[0], "r2": r2, "n_points": len(t_stack)}
        t_fit = np.linspace(0, t_stack.max() * 1.05, 300)
        ax.plot(t_fit, eddy_model(t_fit, *popt), "r-", linewidth=2,
                label=f"fit: tau={popt[2]:.1f}+/-{perr[2]:.1f}s\\nR2={r2:.4f}")
        ax.legend(fontsize=8)
    except (RuntimeError, ValueError) as e:
        ax.text(0.5, 0.05, f"Fit failed: {e}", ha="center", transform=ax.transAxes, fontsize=9, color="red")
    ax.set_xlabel("t since injection start (s)"); ax.set_ylabel("b3 (units)")
    ax.set_title(f"{ds_name} -- global fit")
fig.suptitle("Global Exponential Fit -- All Supercycles Stacked", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print("\\nGlobal fit results:")
print(f"  {'Dataset':>16s} {'tau (s)':>14s} {'A (units)':>16s} {'b3_inf':>14s} {'R2':>8s} {'N':>6s}")
print(f"  {'-'*80}")
for name in [n for n, _ in all_dfs]:
    if name not in global_fits:
        print(f"  {name:>16s} -- fit failed --"); continue
    g = global_fits[name]
    print(f"  {name:>16s} {g['tau']:7.2f}+/-{g['tau_err']:.2f} "
          f"{g['A']:+10.4f}+/-{g['A_err']:.4f} "
          f"{g['b3_inf']:+8.4f}+/-{g['b3_inf_err']:.4f} "
          f"{g['r2']:8.4f} {g['n_points']:6d}")"""))

    cells.append(md("sec12-hdr", "---\n## 12. Summary & Export"))

    cells.append(code("summary-export", """print("=" * 70)
print("EDDY CURRENT b3 SETTLING TIME -- SUMMARY")
print("=" * 70)
print(f"\\nModel: b3(t) = b3_inf + A * exp(-t/tau)")
print(f"Segment: {SEGMENT}")
print(f"cel/fed: {diag.recommendation}")
print(f"Options: {OPTIONS}")

print(f"\\n--- Per-Supercycle Fits ---")
for name, df_f in df_fits.items():
    if len(df_f) == 0:
        print(f"  {name}: no fits"); continue
    tau_v = df_f["tau"].values
    print(f"  {name}: tau = {tau_v.mean():.2f} +/- {tau_v.std():.2f} s "
          f"(N={len(df_f)}, median={np.median(tau_v):.2f} s)")

print(f"\\n--- Global Fits ---")
for name in [n for n, _ in all_dfs]:
    if name in global_fits:
        g = global_fits[name]
        print(f"  {name}: tau = {g['tau']:.2f} +/- {g['tau_err']:.2f} s (R2={g['r2']:.4f})")
    else:
        print(f"  {name}: fit failed")

out_dir = REPO_ROOT / "output" / "2026_02_06" / "eddy_current_b3_settling"
out_dir.mkdir(parents=True, exist_ok=True)
for short, df in [("ext200", df_ext200), ("orig200", df_orig200),
                   ("ext26", df_ext26), ("orig26", df_orig26)]:
    if len(df) > 0:
        fname = f"b3_injection_{short}.csv"
        df.to_csv(out_dir / fname, index=False)
        print(f"Wrote {out_dir / fname}  ({len(df)} rows)")
for short, df_f in [("ext200", df_fits["200 GeV Ext"]), ("orig200", df_fits["200 GeV Orig"]),
                      ("ext26", df_fits["26 GeV Ext"]), ("orig26", df_fits["26 GeV Orig"])]:
    if len(df_f) > 0:
        fname = f"b3_fits_{short}.csv"
        df_f.to_csv(out_dir / fname, index=False)
        print(f"Wrote {out_dir / fname}  ({len(df_f)} rows)")
if global_fits:
    df_global = pd.DataFrame([{"dataset": k, **v} for k, v in global_fits.items()])
    df_global.to_csv(out_dir / "b3_global_fit_summary.csv", index=False)
    print(f"Wrote b3_global_fit_summary.csv  ({len(df_global)} rows)")
print("\\nDone.")"""))

    return cells


# ============================================================
# 4. Comparison notebook
# ============================================================

def build_comparison_cells():
    """Build cell list for the 200 GeV vs 26 GeV comparison notebook."""
    cells = []

    cells.append(md("title", """# B1, b2, b3 Comparison: 200 GeV Extended vs 26 GeV Extended

## Objective

Compare the main field **B1** (T), normal quadrupole **b2** (units), and normal
sextupole **b3** (units) at two operating points:

- **Injection before MD1** (~301 A, 26 GeV) -- the MD1 injection plateau
- **Top of SFTPRO** (~4815 A, 400 GeV) -- the SFTPRO flat-top

for two measurement campaigns on the same SPS MBB dipole:

| Dataset | Path | Description |
|---------|------|-------------|
| **200 GeV extended** | `01_200_extended` | Extended measurement, 200 GeV cycle |
| **26 GeV extended** | `03_26_extended` | Extended measurement, 26 GeV cycle |

Both share the supercycle **LHC_pilot -> MD1 -> SFTPRO** (~20 repetitions)
but were measured in separate sessions (~30 min apart).

### Settling Correction

At injection (~301 A), each supercycle has ~24 turns. The first few are
contaminated by eddy-current settling. This notebook keeps only the
**last N_LAST_TURNS_INJ** turns per supercycle and removes outliers via
MAD-based sigma clipping on B1."""))

    cells.append(md("sec1-hdr", "---\n## 1. Configuration"))

    cells.append(code("config", """SESSION_200 = "2026_02_06/01_200_extended/20260206_144537_SPS_MBB"
SUBDIR_200  = "20260206_144559_MBB"
SESSION_26 = "2026_02_06/03_26_extended/20260206_151808_SPS_MBB"
SUBDIR_26  = "20260206_151827_MBB"

SEGMENT = "NCS"
KN_CROSS_SESSION = "20251212_171026_SPS_MBA/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt"
MAGNET_ORDER = 1
R_REF = 0.02
SAMPLES_PER_TURN = 1024
OPTIONS = ("dri", "rot", "cel", "fed")
PLATEAU_I_RANGE_MAX = 2.5
MIN_B1_T = 1e-4
N_LAST_TURNS_INJ = 18
N_LAST_TURNS_HIGH = None
N_SIGMA_CLIP = 5
FLIP_FIELD_SIGN = False

print("B1 / b2 / b3 Comparison: 200 GeV vs 26 GeV Extended")
print("=" * 55)
print(f"  200 GeV : {SESSION_200}")
print(f"  26 GeV  : {SESSION_26}")
print(f"  Segment : {SEGMENT}")
print(f"  N_LAST_TURNS_INJ  = {N_LAST_TURNS_INJ}")
print(f"  N_SIGMA_CLIP      = {N_SIGMA_CLIP}")"""))

    cells.append(md("sec2-hdr", "---\n## 2. Imports"))

    cells.append(code("imports", """import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

%matplotlib widget
plt.rcParams.update({"figure.figsize": (14, 5), "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists(): break
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range, detect_plateau_turns, classify_current,
    find_contiguous_groups, process_kn_pipeline, build_harmonic_rows,
    mad_sigma_clip, diagnose_cel_fed,
)
from rotating_coil_analyzer.ingest.channel_detect import robust_range

KN_PATH = REPO_ROOT / "measurements" / KN_CROSS_SESSION
assert KN_PATH.exists(), f"Kn file not found: {KN_PATH}"
kn = load_segment_kn_txt(str(KN_PATH))
print(f"Kn loaded: {len(kn.orders)} harmonics from {KN_PATH.name}")
print("Imports ready.")"""))

    cells.append(md("sec3-hdr", "---\n## 3. cel/fed Safety Diagnostic"))

    cells.append(code("cel-fed", """Ns = SAMPLES_PER_TURN
m = MAGNET_ORDER
FILE_PAT = re.compile(r"Run_(\\d+)_I_([\\d.]+)A_(N?CS)_raw_measurement_data\\.txt$")

_session_dir = REPO_ROOT / "measurements" / SESSION_200
_run_dir = _session_dir / SUBDIR_200
_ncs_files = [f for f in sorted(_run_dir.iterdir()) if FILE_PAT.search(f.name) and SEGMENT in f.name]
assert _ncs_files, "No NCS raw file found"
_raw = np.loadtxt(_ncs_files[0])
_n_turns = _raw.shape[0] // Ns; _n_keep = _n_turns * Ns

_t_all = _raw[:_n_keep, 0].reshape(_n_turns, Ns)
_flux_abs = _raw[:_n_keep, 1].reshape(_n_turns, Ns)
_flux_cmp = _raw[:_n_keep, 2].reshape(_n_turns, Ns)
_I_all = _raw[:_n_keep, 3].reshape(_n_turns, Ns)

_I_mean_quick = _I_all.mean(axis=1)
_best_turn = np.argmax(np.abs(_I_mean_quick))
_r1 = robust_range(_raw[_best_turn*Ns:(_best_turn+1)*Ns, 1])
_r2 = robust_range(_raw[_best_turn*Ns:(_best_turn+1)*Ns, 2])
if _r2 > _r1:
    _flux_abs = _raw[:_n_keep, 2].reshape(_n_turns, Ns)
    _flux_cmp = _raw[:_n_keep, 1].reshape(_n_turns, Ns)

_I_mean = _I_all.mean(axis=1)
_hi_mask = np.abs(_I_mean) > 4000
if _hi_mask.sum() < 5: _hi_mask = np.abs(_I_mean) > np.percentile(np.abs(_I_mean), 90)
_n_diag = min(100, int(_hi_mask.sum()))
_hi_idx = np.where(_hi_mask)[0][:_n_diag]

diag = diagnose_cel_fed(_flux_abs[_hi_idx], _flux_cmp[_hi_idx],
    _t_all[_hi_idx], _I_all[_hi_idx], kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER)
print(f"cel/fed diagnostic ({_n_diag} high-I turns from 200 GeV):")
print(f"  {diag.recommendation}")
print(f"  {diag.reason}")
_Bd = np.max(np.abs(diag.B_main_with_fed - diag.B_main_without_fed))
print(f"  B_main max |diff|: {_Bd:.4e} T")
del _session_dir, _run_dir, _ncs_files, _raw, _n_turns, _n_keep
del _t_all, _flux_abs, _flux_cmp, _I_all, _I_mean_quick, _best_turn
del _r1, _r2, _I_mean, _hi_mask, _n_diag, _hi_idx, _Bd

if diag.recommendation == "UNSAFE":
    OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
    print(f"  -> cel/fed disabled, OPTIONS = {OPTIONS}")
else:
    print(f"  -> cel/fed safe, keeping OPTIONS = {OPTIONS}")"""))

    cells.append(md("sec4-hdr", "---\n## 4. Helper: Load & Process One Dataset"))

    cells.append(code("helper", """N_BLOCKS = 10
ANALYSIS_LABELS = {"injection", "flat-mid", "flat-high"}

def load_and_process(session, meas_subdir, dataset_label=""):
    session_dir = REPO_ROOT / "measurements" / session
    run_dir = session_dir / meas_subdir
    ncs_files = [f for f in sorted(run_dir.iterdir()) if FILE_PAT.search(f.name) and SEGMENT in f.name]
    assert ncs_files, f"No {SEGMENT} raw files in {run_dir}"
    raw_file = ncs_files[0]
    print(f"\\n{'='*60}")
    print(f"  Dataset: {dataset_label or session}")
    print(f"  Raw file: {raw_file.name}")

    raw = np.loadtxt(raw_file)
    n_turns = raw.shape[0] // Ns; n_keep = n_turns * Ns
    print(f"  Shape: {raw.shape} -> {n_turns} turns")

    t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
    flux_abs_all = raw[:n_keep, 1].reshape(n_turns, Ns)
    flux_cmp_all = raw[:n_keep, 2].reshape(n_turns, Ns)
    I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

    I_mean_quick = I_all.mean(axis=1)
    best_turn = np.argmax(np.abs(I_mean_quick))
    r1 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 1])
    r2 = robust_range(raw[best_turn*Ns:(best_turn+1)*Ns, 2])
    if r2 > r1:
        flux_abs_all = raw[:n_keep, 2].reshape(n_turns, Ns)
        flux_cmp_all = raw[:n_keep, 1].reshape(n_turns, Ns)
        print("  (flux columns swapped)")

    I_mean = I_all.mean(axis=1); t_mean = t_all.mean(axis=1)
    I_range, I_blocks = compute_block_averaged_range(I_all, Ns, N_BLOCKS)
    plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
    is_plateau = plateau_info["is_plateau"]

    turn_label = np.array(["ramp"] * n_turns, dtype=object)
    for i in range(n_turns):
        if is_plateau[i]: turn_label[i] = classify_current(I_mean[i])

    for lab in ["injection", "flat-mid", "flat-high"]:
        mask = turn_label == lab; n = mask.sum()
        if n > 0:
            print(f"  {lab:12s}: {n:4d} turns, I = {I_mean[mask].mean():.1f} A")

    is_analysis = np.array([l in ANALYSIS_LABELS for l in turn_label])
    plateau_indices = np.where(is_analysis)[0]
    if len(plateau_indices) == 0:
        empty = pd.DataFrame()
        return empty, empty, [], t_mean, I_mean, turn_label, is_plateau

    result, C_merged, C_units, ok_main = process_kn_pipeline(
        flux_abs_turns=flux_abs_all[plateau_indices], flux_cmp_turns=flux_cmp_all[plateau_indices],
        t_turns=t_all[plateau_indices], I_turns=I_all[plateau_indices],
        kn=kn, r_ref=R_REF, magnet_order=m, options=OPTIONS, min_b1_T=MIN_B1_T)

    extra = [{"global_turn": int(plateau_indices[t]), "label": str(turn_label[plateau_indices[t]]),
              "I_range_A": float(I_range[plateau_indices[t]])} for t in range(len(plateau_indices))]
    rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
    df = pd.DataFrame(rows)

    if FLIP_FIELD_SIGN:
        t_cols = [c for c in df.columns if c.endswith("_T")]
        df[t_cols] *= -1

    # Group by supercycle
    inj_mask_global = (turn_label == "injection")
    sc_groups_inj = find_contiguous_groups(inj_mask_global, min_length=2)
    df["sc_idx"] = -1; settled_idx = []

    for gi, (gs, ge) in enumerate(sc_groups_inj):
        group_globals = set(range(gs, ge + 1))
        gmask = df["global_turn"].isin(group_globals) & (df["label"] == "injection")
        df.loc[gmask, "sc_idx"] = gi
        group_rows = df.index[gmask]
        if N_LAST_TURNS_INJ is not None and len(group_rows) > N_LAST_TURNS_INJ:
            settled_idx.extend(group_rows[-N_LAST_TURNS_INJ:])
        else:
            settled_idx.extend(group_rows)

    fh_mask_global = (turn_label == "flat-high")
    sc_groups_fh = find_contiguous_groups(fh_mask_global, min_length=2)
    for gi, (gs, ge) in enumerate(sc_groups_fh):
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
    df_settled, clip_removed = mad_sigma_clip(df_settled, "B1_T", N_SIGMA_CLIP, label_col="label")
    n_clipped = n_before - len(df_settled)
    if n_clipped > 0:
        print(f"  Sigma clip: removed {n_clipped} turns ({clip_removed})")

    n_inj_set = len(df_settled[df_settled["label"] == "injection"])
    n_fh_set = len(df_settled[df_settled["label"] == "flat-high"])
    print(f"  Final settled: injection {n_inj_set}, flat-high {n_fh_set}")

    return df, df_settled, sc_groups_inj, t_mean, I_mean, turn_label, is_plateau"""))

    cells.append(md("sec5-hdr", "---\n## 5. Load Both Datasets"))

    cells.append(code("load-both", """df_200, dfs_200, sc_200, t_200, I_200, lbl_200, plat_200 = load_and_process(
    SESSION_200, SUBDIR_200, "200 GeV Extended")
df_26, dfs_26, sc_26, t_26, I_26, lbl_26, plat_26 = load_and_process(
    SESSION_26, SUBDIR_26, "26 GeV Extended")"""))

    cells.append(md("sec6-hdr", "---\n## 6. Current Profile Overview"))

    cells.append(code("current-profiles", """label_colors = {"injection": "tab:green", "flat-mid": "tab:purple", "flat-high": "tab:blue"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, t_m, I_m, lbl, plat, title in [
    (axes[0], t_200, I_200, lbl_200, plat_200, "200 GeV Extended"),
    (axes[1], t_26, I_26, lbl_26, plat_26, "26 GeV Extended")]:
    ax.plot(t_m, I_m, ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
    for lab, col in label_colors.items():
        mask = lbl == lab; idx = np.where(mask)[0]
        if len(idx) > 0: ax.scatter(t_m[idx], I_m[idx], s=6, color=col, zorder=2, label=lab)
    ax.set_xlabel("Time (s)"); ax.set_title(title); ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("I (A)")
fig.suptitle("Current Profile -- 200 GeV vs 26 GeV Extended", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    cells.append(md("sec7-hdr", "---\n## 7. Per-Supercycle Injection Harmonics"))

    cells.append(code("per-sc-inj", """fig, axes = plt.subplots(1, 3, figsize=(16, 5))
harmonics = [("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]
for ax, (col_name, ylabel) in zip(axes, harmonics):
    for ds_name, dfs, col in [("200 GeV", dfs_200, "tab:blue"), ("26 GeV", dfs_26, "tab:orange")]:
        inj = dfs[(dfs["label"] == "injection") & dfs["ok_main"]]
        if len(inj) == 0: continue
        sc_avg = inj.groupby("sc_idx")[col_name].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel); ax.legend(fontsize=9)
axes[0].set_title("B1"); axes[1].set_title("b2"); axes[2].set_title("b3")
fig.suptitle(f"Per-Supercycle Injection Harmonics (last {N_LAST_TURNS_INJ} settled turns)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    cells.append(md("sec8-hdr", "---\n## 8. SFTPRO Flat-Top: Effect of Preceding MD1"))

    cells.append(code("sftpro-per-sc", """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (col_name, ylabel) in zip(axes.ravel(),
    [("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)"), ("TF_TperkA", "TF (T/kA)")]):
    for ds_name, dfs, col in [("After 200 GeV MD1", dfs_200, "tab:blue"), ("After 26 GeV MD1", dfs_26, "tab:orange")]:
        fh = dfs[(dfs["label"] == "flat-high") & dfs["ok_main"]].copy()
        if len(fh) == 0: continue
        if col_name == "TF_TperkA": fh["TF_TperkA"] = fh["B1_T"] / (fh["I_mean_A"] / 1000.0)
        sc_avg = fh.groupby("sc_idx")[col_name].agg(["mean", "std"]).reset_index()
        ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                    fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
    ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel); ax.legend(fontsize=9)
axes[0,0].set_title("B1"); axes[0,1].set_title("b2"); axes[1,0].set_title("b3"); axes[1,1].set_title("TF")
fig.suptitle("SFTPRO Flat-Top: After 200 GeV MD1 vs After 26 GeV MD1", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    cells.append(md("sec9-hdr", "---\n## 9. Summary Statistics"))

    cells.append(code("summary-stats", """summary_rows = []
for ds_name, dfs in [("200 GeV", dfs_200), ("26 GeV", dfs_26)]:
    for lab, desc in [("injection", "Injection (MD1)"), ("flat-high", "Top of SFTPRO")]:
        sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
        if len(sub) == 0: continue
        tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1000.0)
        summary_rows.append({
            "Dataset": ds_name, "Operating point": desc, "N turns": len(sub),
            "I mean (A)": f"{sub['I_mean_A'].mean():.1f}",
            "B1 mean (T)": f"{sub['B1_T'].mean():.6f}", "B1 std (T)": f"{sub['B1_T'].std():.6f}",
            "b2 mean": f"{sub['b2_units'].mean():+.4f}", "b2 std": f"{sub['b2_units'].std():.4f}",
            "b3 mean": f"{sub['b3_units'].mean():+.4f}", "b3 std": f"{sub['b3_units'].std():.4f}",
            "TF (T/kA)": f"{tf:.4f}",
        })
df_summary = pd.DataFrame(summary_rows)
print(f"[Settled data: last {N_LAST_TURNS_INJ} injection turns per supercycle]\\n")
print(df_summary.to_string(index=False))"""))

    cells.append(md("sec10-hdr", "---\n## 10. B1, b2, b3 Comparison Plots"))

    cells.append(code("comparison-plots", """for harm_name, harm_col, ylabel in [("B1", "B1_T", "B1 (T)"), ("b2", "b2_units", "b2 (units)"), ("b3", "b3_units", "b3 (units)")]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lab, title in [
        (axes[0], "injection", f"Injection (~301 A, settled, last {N_LAST_TURNS_INJ}/SC)"),
        (axes[1], "flat-high", "Top of SFTPRO (~4815 A)")]:
        for ds_name, dfs, col in [("200 GeV", dfs_200, "tab:blue"), ("26 GeV", dfs_26, "tab:orange")]:
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            ax.plot(sub["time_s"].values, sub[harm_col].values, ".", markersize=3, alpha=0.6, color=col, label=ds_name)
        if harm_col != "B1_T": ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    fig.suptitle(f"{harm_name} Comparison (settled turns)", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()"""))

    cells.append(md("sec11-hdr", "---\n## 11. Box Plots"))

    cells.append(code("boxplots", """fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (col_name, ylabel, title) in zip(axes.ravel(),
    [("B1_T", "B1 (T)", "B1"), ("b2_units", "b2 (units)", "b2"),
     ("b3_units", "b3 (units)", "b3"), ("TF_TperkA", "TF (T/kA)", "TF")]):
    box_data, box_labels, box_colors = [], [], []
    for ds_name, dfs, base_col in [("200 GeV", dfs_200, "tab:blue"), ("26 GeV", dfs_26, "tab:orange")]:
        for lab, short in [("injection", "Inj"), ("flat-high", "SFTPRO")]:
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]].copy()
            if len(sub) == 0: continue
            vals = (sub["B1_T"] / (sub["I_mean_A"] / 1000.0)).values if col_name == "TF_TperkA" else sub[col_name].values
            box_data.append(vals); box_labels.append(f"{ds_name}\\n{short}\\n(N={len(sub)})"); box_colors.append(base_col)
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, col in zip(bp["boxes"], box_colors): patch.set_facecolor(col); patch.set_alpha(0.5)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Distribution Comparison (settled turns)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    cells.append(md("sec12-hdr", "---\n## 12. Statistical Significance\n\nSigma = |diff| / sqrt(std1^2/N1 + std2^2/N2). > 3 sigma = real difference."))

    cells.append(code("diff-table", """print(f"Difference: (200 GeV) - (26 GeV)  [settled, last {N_LAST_TURNS_INJ}/SC at injection]")
print("=" * 100)
results = []
for lab, desc in [("injection", "Injection (~301 A)"), ("flat-high", "SFTPRO (~4815 A)")]:
    s200 = dfs_200[(dfs_200["label"] == lab) & dfs_200["ok_main"]]
    s26 = dfs_26[(dfs_26["label"] == lab) & dfs_26["ok_main"]]
    if len(s200) == 0 or len(s26) == 0: continue

    dB1 = s200["B1_T"].mean() - s26["B1_T"].mean()
    db2 = s200["b2_units"].mean() - s26["b2_units"].mean()
    db3 = s200["b3_units"].mean() - s26["b3_units"].mean()
    tf200 = s200["B1_T"].mean() / (s200["I_mean_A"].mean() / 1000.0)
    tf26 = s26["B1_T"].mean() / (s26["I_mean_A"].mean() / 1000.0)
    dTF = tf200 - tf26

    dB1_err = np.sqrt((s200["B1_T"].std()**2/len(s200)) + (s26["B1_T"].std()**2/len(s26)))
    db2_err = np.sqrt((s200["b2_units"].std()**2/len(s200)) + (s26["b2_units"].std()**2/len(s26)))
    db3_err = np.sqrt((s200["b3_units"].std()**2/len(s200)) + (s26["b3_units"].std()**2/len(s26)))
    tf200_vals = s200["B1_T"] / (s200["I_mean_A"] / 1000.0)
    tf26_vals = s26["B1_T"] / (s26["I_mean_A"] / 1000.0)
    dTF_err = np.sqrt((tf200_vals.std()**2/len(s200)) + (tf26_vals.std()**2/len(s26)))

    sig_B1 = abs(dB1)/dB1_err if dB1_err > 0 else 0
    sig_b2 = abs(db2)/db2_err if db2_err > 0 else 0
    sig_b3 = abs(db3)/db3_err if db3_err > 0 else 0
    sig_TF = abs(dTF)/dTF_err if dTF_err > 0 else 0

    print(f"{desc:>30s}  dB1={dB1:+.6f}+/-{dB1_err:.6f}  db2={db2:+.4f}+/-{db2_err:.4f}  "
          f"db3={db3:+.4f}+/-{db3_err:.4f}  dTF={dTF:+.4f}+/-{dTF_err:.4f}")
    print(f"{'(sigma)':>30s}  {sig_B1:>12.1f}  {sig_b2:>14.1f}  {sig_b3:>14.1f}  {sig_TF:>14.1f}")
    results.append((lab, desc, dB1, sig_B1, db2, sig_b2, db3, sig_b3, dTF, sig_TF, len(s200), len(s26)))

print("\\nINTERPRETATION")
print("-" * 70)
for lab, desc, dB1, sB1, db2, sb2, db3, sb3, dTF, sTF, n200, n26 in results:
    print(f"\\n  {desc}  (N: {n200} vs {n26} turns)")
    for name, diff, sig, unit in [("B1", dB1, sB1, "T"), ("b2", db2, sb2, "units"),
                                   ("b3", db3, sb3, "units"), ("TF", dTF, sTF, "T/kA")]:
        verdict = "REAL (>3 sigma)" if sig > 3 else ("suggestive (2-3 sigma)" if sig >= 2 else "no evidence (<2 sigma)")
        diff_str = f"{diff*1e6:+.1f} uT" if "T" in unit and "units" not in unit and "kA" not in unit else f"{diff:+.4f} {unit}"
        print(f"    {name:>3s}: {diff_str:>16s}  ({sig:.1f} sigma) -> {verdict}")

print("\\nNOTE: high sigma = reliably detected, NOT necessarily large.")
print("Datasets measured ~30 min apart with different cycle histories.")"""))

    cells.append(md("sec13-hdr", "---\n## 13. Export"))

    cells.append(code("export", """out_dir = REPO_ROOT / "output" / "2026_02_06" / "compare_200_vs_26"
out_dir.mkdir(parents=True, exist_ok=True)

for name, df in [("200GeV", df_200), ("26GeV", df_26)]:
    if len(df) > 0:
        fname = f"plateau_harmonics_{name}.csv"
        df.to_csv(out_dir / fname, index=False)
        print(f"Wrote {out_dir / fname}  ({len(df)} rows)")

for name, dfs in [("200GeV", dfs_200), ("26GeV", dfs_26)]:
    if len(dfs) > 0:
        fname = f"plateau_harmonics_{name}_settled.csv"
        dfs.to_csv(out_dir / fname, index=False)
        print(f"Wrote {out_dir / fname}  ({len(dfs)} rows)")

for name, dfs in [("200GeV", dfs_200), ("26GeV", dfs_26)]:
    inj = dfs[(dfs["label"] == "injection") & dfs["ok_main"]].copy()
    if len(inj) > 0:
        inj["TF_TperkA"] = inj["B1_T"] / (inj["I_mean_A"] / 1000.0)
        sc_summary = inj.groupby("sc_idx").agg(
            B1_mean=("B1_T", "mean"), B1_std=("B1_T", "std"),
            b2_mean=("b2_units", "mean"), b2_std=("b2_units", "std"),
            b3_mean=("b3_units", "mean"), b3_std=("b3_units", "std"),
            TF_mean=("TF_TperkA", "mean"), TF_std=("TF_TperkA", "std"),
            I_mean=("I_mean_A", "mean"), n_turns=("B1_T", "count"),
        ).reset_index()
        fname = f"per_supercycle_injection_{name}.csv"
        sc_summary.to_csv(out_dir / fname, index=False)
        print(f"Wrote {out_dir / fname}  ({len(sc_summary)} rows)")

if len(df_summary) > 0:
    df_summary.to_csv(out_dir / "summary_comparison_settled.csv", index=False)
    print(f"Wrote summary_comparison_settled.csv  ({len(df_summary)} rows)")
print("\\nDone.")"""))

    return cells


# ============================================================
# Generate all 4 notebooks
# ============================================================

if __name__ == "__main__":
    print("Generating MBB notebooks...")

    # 1. 200 GeV analysis
    cells_200 = build_analysis_cells(
        session="2026_02_06/01_200_extended/20260206_144537_SPS_MBB",
        meas_subdir="20260206_144559_MBB",
        energy_label="200 GeV",
        out_subdir="01_200_extended",
    )
    write_notebook(NOTEBOOK_DIR / "analysis" / "2026-02-06_01_200_extended_NCS.ipynb", cells_200)

    # 2. 26 GeV analysis
    cells_26 = build_analysis_cells(
        session="2026_02_06/03_26_extended/20260206_151808_SPS_MBB",
        meas_subdir="20260206_151827_MBB",
        energy_label="26 GeV",
        out_subdir="03_26_extended",
    )
    write_notebook(NOTEBOOK_DIR / "analysis" / "2026-02-06_03_26_extended_NCS.ipynb", cells_26)

    # 3. Eddy current (combined)
    cells_eddy = build_eddy_current_cells()
    write_notebook(NOTEBOOK_DIR / "eddy_current" / "2026-02-06_b3_settling.ipynb", cells_eddy)

    # 4. Comparison (regenerated)
    cells_comp = build_comparison_cells()
    write_notebook(NOTEBOOK_DIR / "comparison" / "2026-02-06_200GeV_vs_26GeV.ipynb", cells_comp)

    print("\nAll 4 notebooks generated successfully.")
