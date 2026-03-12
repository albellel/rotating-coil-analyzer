"""Generate notebook for integral coil flux integration.

Two H5 files from 2026-03-10 session:
  - 20260310_224015HolecStabililty.h5  (200 GeV cycle)
  - 20260310_225837HolecStabililty.h5  (26 GeV cycle)

Each file has 4 columns at 100 kHz:
  col 0: Time (s)
  col 1: DCCT voltage (x 650 = current in A)
  col 2: coil_2 voltage (rightmost coil, actually coil 8)
  col 3: coil_5 voltage (central coil)

Pipeline:
  1. Downsample DCCT current to 100 Hz for plateau/window detection only.
  2. Identify integration windows (MD1 / SFTPRO / LHC) via library functions.
  3. Integrate coil voltages at full 100 kHz resolution, drift-corrected
     using the baseline from the idle plateau before each ramp.
  4. Plot cumulative flux vs time and vs current for each window.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5] / "scripts"))
from nb_helpers import code, md, write_notebook, safe_print
from pathlib import Path

safe_print()

HERE = Path(__file__).resolve().parent

cells = []

# ── Title ────────────────────────────────────────────────────────────────
cells.append(md("title", r"""
# Integral Coil — Flux Integration

**2026-03-10 Max Speed Idle session**

| Type | Integration window | Expected result |
|------|-------------------|----------------|
| **MD1** | Full round-trip: idle -> peak -> idle (+ settling) | delta-flux ~ 0 (closed loop) |
| **SFTPRO** | One-way up: idle -> SFTPRO top plateau end | flux(4816 A) - flux(idle) |
| **LHC** | One-way up: idle -> LHC top plateau end (through injection) | flux(5781 A) - flux(idle) |

**Drift correction:** for each window, the mean coil voltage over the last
`BASELINE_S` seconds of the preceding idle plateau (where dB/dt = 0, so
V should be zero) is subtracted before integration.

**Resolution:** window detection uses DCCT current downsampled to 100 Hz
(for speed).  The actual flux integration runs at the **full 100 kHz**
acquisition rate, reading slices directly from the H5 file.

- **Coil 5** (col 3): central coil — calibration 0.101 T-m/V-s
- **Coil 2** (col 2, physically coil 8): rightmost coil — calibration 0.10123 T-m/V-s
- DCCT voltage-to-current: 650 A/V
- Acquisition rate: 100 kHz
"""))

# ── Imports + constants ──────────────────────────────────────────────────
cells.append(code("imports", r'''
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from rotating_coil_analyzer.analysis.utility_functions import (
    downsample_block_avg,
    detect_plateaus_continuous,
    find_integration_windows,
)

%matplotlib widget

# ── Acquisition ──
DCCT_FACTOR   = 650.0      # A/V
ACQ_RATE      = 100_000    # Hz
DT            = 1.0 / ACQ_RATE

# ── Detection (downsampled to 100 Hz — for window-finding only) ──
DS_BLOCK      = 1000       # 100 kHz -> 100 Hz
DIDT_THRESH   = 10.0       # A/s
MIN_PLAT_S    = 0.3        # s
I_IDLE_LO     = 100.0      # A
I_IDLE_HI     = 250.0      # A
MIN_SETTLED_S = 5.0        # s  (~3-5 tau, eddies < 5%)
SETTLE_AFTER_S = 5.0       # s  (MD1: settle into next idle)
BASELINE_S    = 2.0        # s  (drift baseline window)

# ── Coil calibrations ──
CAL_COIL5 = 0.101          # T-m / (V-s)
CAL_COIL2 = 0.10123        # T-m / (V-s)

# ── Plotting ──
DS_PLOT = 1000              # downsample flux traces for plotting only
FIG_W   = 12                # figure width (inches)
'''))

# ── Helper functions ─────────────────────────────────────────────────────
cells.append(code("helpers", r'''
# =======================================================================
#  Window detection (operates on downsampled current)
# =======================================================================

def detect_windows(h5_name):
    """Downsample DCCT current, detect integration windows.

    Returns dict with keys: t, I, all_plats, idle_plats, windows, didt.
    (Coil voltages are NOT loaded here — integration reads full-res slices.)
    """
    with h5py.File(Path(h5_name), "r") as f:
        raw = f["RawData/DAQAcquisition"]
        n = raw.shape[0]
        print(f"Reading {h5_name}: {n:,} samples ({n/ACQ_RATE:.1f} s)")
        data = raw[:]

    t_ds = downsample_block_avg(data[:, 0], DS_BLOCK)
    I_ds = downsample_block_avg(data[:, 1], DS_BLOCK) * DCCT_FACTOR
    del data

    print(f"  Downsampled to {len(t_ds):,} pts at {ACQ_RATE/DS_BLOCK:.0f} Hz")
    print(f"  Current range: {I_ds.min():.0f} to {I_ds.max():.0f} A")

    all_plats, didt = detect_plateaus_continuous(
        t_ds, I_ds, didt_thresh=DIDT_THRESH, min_dur_s=MIN_PLAT_S)
    windows, idle_plats = find_integration_windows(
        t_ds, I_ds, all_plats,
        I_idle_lo=I_IDLE_LO, I_idle_hi=I_IDLE_HI,
        min_settled_s=MIN_SETTLED_S,
        settle_after_s=SETTLE_AFTER_S,
        baseline_s=BASELINE_S,
        ds_block=DS_BLOCK)

    counts = Counter(w.cycle_type for w in windows)
    for tp in ("MD1_200GeV", "MD1_26GeV", "SFTPRO", "LHC"):
        if counts[tp]:
            print(f"  {tp}: {counts[tp]}")

    return dict(t=t_ds, I=I_ds, all_plats=all_plats,
                idle_plats=idle_plats, windows=windows, didt=didt)


# =======================================================================
#  Full-resolution flux integration (reads H5 slices directly)
# =======================================================================

def integrate_all_windows(h5_name, windows):
    """Integrate coil voltages at full 100 kHz for each window.

    For each window:
      1. Read baseline slice -> mean voltage offset per coil.
      2. Read integration slice -> subtract baseline -> cumsum * dt * cal.
      3. Keep full-res delta-flux; downsample trace for plotting only.

    Returns list of result dicts.
    """
    results = []

    with h5py.File(Path(h5_name), "r") as f:
        raw = f["RawData/DAQAcquisition"]

        for w in windows:
            # ── baseline (full-res) ──
            bl = raw[w.baseline_full_start:w.baseline_full_end, :]
            bl5 = float(np.mean(bl[:, 3]))
            bl2 = float(np.mean(bl[:, 2]))
            del bl

            # ── integration window (full-res) ──
            chunk = raw[w.idx_full_start:w.idx_full_end, :]
            V5 = chunk[:, 3].astype(np.float64) - bl5
            V2 = chunk[:, 2].astype(np.float64) - bl2
            I_full = chunk[:, 1].astype(np.float64) * DCCT_FACTOR
            t_rel = chunk[:, 0].astype(np.float64)
            t_rel = t_rel - t_rel[0]
            del chunk

            # ── cumulative flux at full resolution ──
            flux5 = np.cumsum(V5) * DT * CAL_COIL5
            flux2 = np.cumsum(V2) * DT * CAL_COIL2
            del V5, V2

            # ── downsample for plotting only ──
            n = len(flux5)
            idx = np.arange(0, n, DS_PLOT)
            if idx[-1] != n - 1:
                idx = np.append(idx, n - 1)

            results.append({
                "window": w,
                "t_rel": t_rel[idx],
                "I": I_full[idx],
                "flux_coil5": flux5[idx],
                "flux_coil2": flux2[idx],
                "delta_flux_coil5": float(flux5[-1]),
                "delta_flux_coil2": float(flux2[-1]),
                "baseline_coil5_V": bl5,
                "baseline_coil2_V": bl2,
            })
            del flux5, flux2, t_rel, I_full

    print(f"Integrated {len(results)} windows from {h5_name}")
    return results


# =======================================================================
#  Plotting
# =======================================================================

_WIN_COLORS = {
    "MD1_26GeV": "C0", "MD1_200GeV": "C0",
    "SFTPRO": "C1", "LHC": "C3",
}


def plot_overview(d, title):
    """Full waveform with shaded integration windows."""
    t, I = d["t"], d["I"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13)

    ax1.plot(t, I, lw=0.4, color="0.5")
    for i, (s, e) in enumerate(d["idle_plats"]):
        ax1.axvspan(t[s], t[min(e - 1, len(t) - 1)],
                    alpha=0.12, color="green", zorder=0,
                    label="idle plateau" if i == 0 else None)
    seen = set()
    for w in d["windows"]:
        c = _WIN_COLORS.get(w.cycle_type, "C4")
        lbl = w.cycle_type if w.cycle_type not in seen else None
        seen.add(w.cycle_type)
        ax1.axvspan(w.t_start, w.t_end, alpha=0.18, color=c, zorder=0,
                    label=lbl)
        mid_t = (w.t_start + w.t_end) / 2
        ax1.annotate(w.label, xy=(mid_t, w.I_peak * 0.5),
                     fontsize=5, ha="center", va="center",
                     color="k", fontweight="bold", rotation=90)
    ax1.set_ylabel("Current (A)")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, d["didt"], lw=0.3, color="C1")
    ax2.axhline(+DIDT_THRESH, ls="--", color="gray", lw=0.8)
    ax2.axhline(-DIDT_THRESH, ls="--", color="gray", lw=0.8)
    ax2.set_ylabel("dI/dt (A/s)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_meas_zoom(d, title):
    """Zoomed view of SFTPRO and LHC windows with start/end markers."""
    t, I = d["t"], d["I"]
    meas = [w for w in d["windows"] if w.cycle_type in ("SFTPRO", "LHC")]
    if not meas:
        print("No SFTPRO/LHC windows found.")
        return
    t_lo = meas[0].t_start - 40
    t_hi = meas[-1].t_end + 40
    mask = (t >= t_lo) & (t <= t_hi)

    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    fig.suptitle(title, fontsize=13)
    ax.plot(t[mask], I[mask], lw=0.8, color="C0")
    for w in meas:
        c = _WIN_COLORS.get(w.cycle_type, "C4")
        ax.axvspan(w.t_start, w.t_end, alpha=0.15, color=c)
        ax.axvline(w.t_start, color="green", ls="--", lw=1.2)
        ax.axvline(w.t_end, color="red", ls="--", lw=1.2)
        ax.annotate(
            w.label + "\n" + f"{w.duration_s:.1f} s",
            xy=((w.t_start + w.t_end) / 2, w.I_peak),
            fontsize=9, ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Current (A)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# =======================================================================
#  Summary tables
# =======================================================================

def window_summary_df(windows):
    """Summary DataFrame from IntegrationWindow list."""
    rows = [{
        "id": w.id, "type": w.cycle_type, "label": w.label,
        "t_start": w.t_start, "t_end": w.t_end,
        "duration_s": w.duration_s,
        "I_start": w.I_start, "I_peak": w.I_peak, "I_end": w.I_end,
    } for w in windows]
    return pd.DataFrame(rows)


def flux_summary_df(flux_results):
    """Summary DataFrame with delta-flux, baselines, and drift impact.

    Columns
    -------
    baseline coil 5 / 2 (uV)
        Mean voltage subtracted as drift correction.
    drift bias coil 5 / 2 (units)
        Relative impact of the drift on the integral, in units (1e-4):
        (baseline_V * duration * calibration) / delta_flux * 1e4.
        Shows how many units the integral would be biased without
        correction.  NaN for MD1 (delta-flux ~ 0, ratio meaningless).
    delta flux coil 5 / 2 (T*m)
        Final drift-corrected delta-flux.
    """
    rows = []
    for r in flux_results:
        w = r["window"]
        dur = w.duration_s
        bl5 = r["baseline_coil5_V"]
        bl2 = r["baseline_coil2_V"]
        dF5 = r["delta_flux_coil5"]
        dF2 = r["delta_flux_coil2"]
        drift5 = bl5 * dur * CAL_COIL5
        drift2 = bl2 * dur * CAL_COIL2
        # relative bias in units (1e-4) — only meaningful for one-way ramps
        if abs(dF5) > 1e-8:
            bias5 = drift5 / dF5 * 1e4
        else:
            bias5 = np.nan
        if abs(dF2) > 1e-8:
            bias2 = drift2 / dF2 * 1e4
        else:
            bias2 = np.nan
        rows.append({
            "id": w.id,
            "type": w.cycle_type,
            "label": w.label,
            "duration (s)": dur,
            "baseline coil 5 (uV)": bl5 * 1e6,
            "baseline coil 2 (uV)": bl2 * 1e6,
            "drift bias coil 5 (units)": bias5,
            "drift bias coil 2 (units)": bias2,
            "delta flux coil 5 (T*m)": dF5,
            "delta flux coil 2 (T*m)": dF2,
        })
    return pd.DataFrame(rows)
'''))

# ── 200 GeV cycle ───────────────────────────────────────────────────────
cells.append(md("md_200", "## 200 GeV cycle"))

cells.append(code("detect_200", r'''
d_200 = detect_windows("20260310_224015HolecStabililty.h5")
df_w = window_summary_df(d_200["windows"])
with pd.option_context("display.max_rows", None):
    display(df_w.round(1))
'''))

cells.append(code("plot_overview_200", r'''
plot_overview(d_200, "200 GeV cycle — integration windows")
'''))

cells.append(code("plot_zoom_200", r'''
plot_meas_zoom(d_200, "200 GeV cycle — SFTPRO / LHC (zoom)")
'''))

cells.append(md("md_integrate_200",
                "### Full-resolution flux integration (200 GeV)"))

cells.append(code("integrate_200", r'''
flux_200 = integrate_all_windows(
    "20260310_224015HolecStabililty.h5", d_200["windows"])
'''))

cells.append(code("plot_md1_flux_200", r'''
md1 = [r for r in flux_200 if r["window"].cycle_type.startswith("MD1")]
if md1:
    fig, (ax5, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 7), sharex=True)
    fig.suptitle("200 GeV — MD1 round-trip flux (should return to ~0)",
                 fontsize=13)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(md1)))
    for i, r in enumerate(md1):
        ax5.plot(r["t_rel"], r["flux_coil5"], lw=0.5, alpha=0.6,
                 color=cmap[i])
        ax2.plot(r["t_rel"], r["flux_coil2"], lw=0.5, alpha=0.6,
                 color=cmap[i])
    ax5.axhline(0, color="k", ls="--", lw=0.5)
    ax5.set_ylabel("Coil 5 flux (T*m)")
    ax5.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_ylabel("Coil 2 flux (T*m)")
    ax2.set_xlabel("Relative time (s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    dF5 = [r["delta_flux_coil5"] for r in md1]
    dF2 = [r["delta_flux_coil2"] for r in md1]
    print(f"MD1 round-trip residual (coil 5): "
          f"mean = {np.mean(dF5):.5f} T*m, std = {np.std(dF5):.5f} T*m")
    print(f"MD1 round-trip residual (coil 2): "
          f"mean = {np.mean(dF2):.5f} T*m, std = {np.std(dF2):.5f} T*m")
'''))

cells.append(code("plot_meas_flux_200", r'''
meas = [r for r in flux_200 if r["window"].cycle_type in ("SFTPRO", "LHC")]
for r in meas:
    fig, (ax_t, ax_I) = plt.subplots(1, 2, figsize=(FIG_W, 5))
    fig.suptitle(f'200 GeV — {r["window"].label} flux integration',
                 fontsize=13)

    ax_t.plot(r["t_rel"], r["flux_coil5"], lw=0.8, label="Coil 5")
    ax_t.plot(r["t_rel"], r["flux_coil2"], lw=0.8, label="Coil 2")
    ax_t.set_xlabel("Relative time (s)")
    ax_t.set_ylabel("Cumulative flux (T*m)")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_I.plot(r["I"], r["flux_coil5"], lw=0.8, label="Coil 5")
    ax_I.plot(r["I"], r["flux_coil2"], lw=0.8, label="Coil 2")
    ax_I.set_xlabel("Current (A)")
    ax_I.set_ylabel("Cumulative flux (T*m)")
    ax_I.legend()
    ax_I.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    print(f'{r["window"].label}:  '
          f'dF_coil5 = {r["delta_flux_coil5"]:.5f} T*m,  '
          f'dF_coil2 = {r["delta_flux_coil2"]:.5f} T*m')
'''))

cells.append(code("flux_table_200", r'''
print("=== 200 GeV — delta-flux & drift correction summary ===")
df_f = flux_summary_df(flux_200)
with pd.option_context("display.max_rows", None, "display.float_format",
                       "{:.5f}".format):
    display(df_f)
'''))

# ── 26 GeV cycle ────────────────────────────────────────────────────────
cells.append(md("md_26", "## 26 GeV cycle"))

cells.append(code("detect_26", r'''
d_26 = detect_windows("20260310_225837HolecStabililty.h5")
df_w = window_summary_df(d_26["windows"])
with pd.option_context("display.max_rows", None):
    display(df_w.round(1))
'''))

cells.append(code("plot_overview_26", r'''
plot_overview(d_26, "26 GeV cycle — integration windows")
'''))

cells.append(code("plot_zoom_26", r'''
plot_meas_zoom(d_26, "26 GeV cycle — SFTPRO / LHC (zoom)")
'''))

cells.append(md("md_integrate_26",
                "### Full-resolution flux integration (26 GeV)"))

cells.append(code("integrate_26", r'''
flux_26 = integrate_all_windows(
    "20260310_225837HolecStabililty.h5", d_26["windows"])
'''))

cells.append(code("plot_md1_flux_26", r'''
md1 = [r for r in flux_26 if r["window"].cycle_type.startswith("MD1")]
if md1:
    fig, (ax5, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 7), sharex=True)
    fig.suptitle("26 GeV — MD1 round-trip flux (should return to ~0)",
                 fontsize=13)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(md1)))
    for i, r in enumerate(md1):
        ax5.plot(r["t_rel"], r["flux_coil5"], lw=0.5, alpha=0.6,
                 color=cmap[i])
        ax2.plot(r["t_rel"], r["flux_coil2"], lw=0.5, alpha=0.6,
                 color=cmap[i])
    ax5.axhline(0, color="k", ls="--", lw=0.5)
    ax5.set_ylabel("Coil 5 flux (T*m)")
    ax5.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_ylabel("Coil 2 flux (T*m)")
    ax2.set_xlabel("Relative time (s)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    dF5 = [r["delta_flux_coil5"] for r in md1]
    dF2 = [r["delta_flux_coil2"] for r in md1]
    print(f"MD1 round-trip residual (coil 5): "
          f"mean = {np.mean(dF5):.5f} T*m, std = {np.std(dF5):.5f} T*m")
    print(f"MD1 round-trip residual (coil 2): "
          f"mean = {np.mean(dF2):.5f} T*m, std = {np.std(dF2):.5f} T*m")
'''))

cells.append(code("plot_meas_flux_26", r'''
meas = [r for r in flux_26 if r["window"].cycle_type in ("SFTPRO", "LHC")]
for r in meas:
    fig, (ax_t, ax_I) = plt.subplots(1, 2, figsize=(FIG_W, 5))
    fig.suptitle(f'26 GeV — {r["window"].label} flux integration',
                 fontsize=13)

    ax_t.plot(r["t_rel"], r["flux_coil5"], lw=0.8, label="Coil 5")
    ax_t.plot(r["t_rel"], r["flux_coil2"], lw=0.8, label="Coil 2")
    ax_t.set_xlabel("Relative time (s)")
    ax_t.set_ylabel("Cumulative flux (T*m)")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_I.plot(r["I"], r["flux_coil5"], lw=0.8, label="Coil 5")
    ax_I.plot(r["I"], r["flux_coil2"], lw=0.8, label="Coil 2")
    ax_I.set_xlabel("Current (A)")
    ax_I.set_ylabel("Cumulative flux (T*m)")
    ax_I.legend()
    ax_I.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    print(f'{r["window"].label}:  '
          f'dF_coil5 = {r["delta_flux_coil5"]:.5f} T*m,  '
          f'dF_coil2 = {r["delta_flux_coil2"]:.5f} T*m')
'''))

cells.append(code("flux_table_26", r'''
print("=== 26 GeV — delta-flux & drift correction summary ===")
df_f = flux_summary_df(flux_26)
with pd.option_context("display.max_rows", None, "display.float_format",
                       "{:.5f}".format):
    display(df_f)
'''))

# ── Write ────────────────────────────────────────────────────────────────
out = HERE / "integral_coil_visualization.ipynb"
write_notebook(out, cells)
safe_print(f"Done: {out}")
