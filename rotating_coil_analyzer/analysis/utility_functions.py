"""Reusable utility functions for streaming rotating-coil analysis.

These functions extract common logic from the Jupyter analysis notebooks
and the GUI so that notebook code stays concise and multiple notebooks /
the GUI can share the same tested implementations.  They are designed for
CERN accelerator-magnet rotating-coil measurements across all machine
complexes (LHC, SPS, PS, PSB, transfer lines, test benches such as SM18).

Functions
---------
compute_block_averaged_range
    Block-averaged peak-to-peak current range per turn (noise-robust).
detect_plateau_turns
    Three-rule plateau detection (range + start boundary + end boundary).
classify_current
    Classify a current value into a cycle-type label.  Default thresholds
    are tuned for SPS; fully customisable for other machines.
find_contiguous_groups
    Find contiguous runs of True values in a boolean mask.
process_kn_pipeline
    Full Kn pipeline wrapper: dit -> drift -> FFT -> kn -> merge -> normalise.
build_harmonic_rows
    Build a list of dicts (one per turn) from pipeline results, ready for
    ``pd.DataFrame()``.
build_run_averages
    Per-run mean b3 with run ordering (for hysteresis / ramp analysis).
ba_table_from_C
    Convert complex coefficients to legacy B/A DataFrame (all Tesla).
mixed_format_table
    Bottura Section 3.7 mixed-format DataFrame (Tesla for n<=m, units for n>m).
mad_sigma_clip
    MAD-based outlier removal per operating point.
discover_runs
    Parse Run_XX_I_YYA filenames from a measurement directory.
plateau_summary
    Per-run/per-level mean+std of B1, TF, and all harmonics.
plot_hysteresis
    Hysteresis loop with run-order gradient coloring.
eddy_model
    Exponential eddy-current settling model for curve_fit.
compute_level_stats
    Mean/std of I, B1, b2, b3, TF for a given operating point.
diff_sigma
    Difference, propagated error, and sigma significance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .kn_pipeline import (
    compute_legacy_kn_per_turn,
    merge_coefficients,
    safe_normalize_to_units,
)


# =====================================================================
#  Plateau detection helpers
# =====================================================================

def compute_block_averaged_range(
    I_all: np.ndarray,
    samples_per_turn: int,
    n_blocks: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a noise-robust current range for each turn.

    Each turn's samples are split into *n_blocks* blocks.  Each block is
    averaged to a single value, then the range (max - min) of these block
    means is returned.  This filters out sample-level ADC noise while
    capturing real current drift or ramp contamination.

    Parameters
    ----------
    I_all : ndarray, shape (n_turns, samples_per_turn)
        Current samples reshaped into turns.
    samples_per_turn : int
        Number of samples per turn (only used for block-size calculation).
    n_blocks : int, optional
        Number of blocks to split each turn into (default 10).

    Returns
    -------
    I_range : ndarray, shape (n_turns,)
        Block-averaged peak-to-peak range per turn.
    I_blocks : ndarray, shape (n_turns, n_blocks)
        Block means per turn (useful for boundary checks).
    """
    n_turns = I_all.shape[0]
    block_sz = samples_per_turn // n_blocks
    I_blocks = (
        I_all[:, : n_blocks * block_sz]
        .reshape(n_turns, n_blocks, block_sz)
        .mean(axis=2)
    )
    I_range = I_blocks.max(axis=1) - I_blocks.min(axis=1)
    return I_range, I_blocks


def detect_plateau_turns(
    I_blocks: np.ndarray,
    I_mean: np.ndarray,
    I_range: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Three-rule plateau detection.

    A turn is accepted as "on a plateau" only if **all three** rules pass:

    * **(a)** block-averaged I range < *threshold*
    * **(b)** |first-block mean - turn mean| < *threshold*  (starts on plateau)
    * **(c)** |last-block  mean - turn mean| < *threshold*  (ends on plateau)

    Parameters
    ----------
    I_blocks : ndarray, shape (n_turns, n_blocks)
        Block means per turn (from :func:`compute_block_averaged_range`).
    I_mean : ndarray, shape (n_turns,)
        Mean current per turn.
    I_range : ndarray, shape (n_turns,)
        Block-averaged range per turn.
    threshold : float
        Maximum allowed current variation (A).

    Returns
    -------
    dict with keys:
        ``is_plateau``            – bool mask, True for turns passing all 3 rules
        ``is_boundary_rejected``  – bool mask, True for turns passing (a) but
                                    failing (b) or (c)
        ``range_ok``              – bool mask, rule (a)
        ``start_ok``              – bool mask, rule (b)
        ``end_ok``                – bool mask, rule (c)
    """
    range_ok = I_range < threshold
    start_ok = np.abs(I_blocks[:, 0] - I_mean) < threshold
    end_ok = np.abs(I_blocks[:, -1] - I_mean) < threshold
    is_plateau = range_ok & start_ok & end_ok
    is_boundary_rejected = range_ok & ~is_plateau
    return {
        "is_plateau": is_plateau,
        "is_boundary_rejected": is_boundary_rejected,
        "range_ok": range_ok,
        "start_ok": start_ok,
        "end_ok": end_ok,
    }


# =====================================================================
#  Current-level classification
# =====================================================================

#: Default current-level thresholds (A), tuned for SPS cycle structure.
#: Override with a custom dict for other machines (PS, PSB, LHC, ...).
SPS_CURRENT_THRESHOLDS = {
    "zero": 50,
    "pre-ramp": 200,
    "injection": 500,
    "flat-low": 2000,
    "flat-mid": 4000,
    # anything above -> "flat-high"
}


def classify_current(
    I: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Classify a current value into a machine cycle-type label.

    The function walks through the *thresholds* dict in insertion order
    and returns the first label whose upper bound exceeds *I*.  If *I*
    is above all bounds, the fallback label ``"flat-high"`` is returned.

    The default thresholds are tuned for SPS cycle structure.  For other
    CERN machines, pass a custom dictionary::

        psb_thresholds = {"zero": 10, "injection": 100, "flat-top": 500}
        label = classify_current(I_value, thresholds=psb_thresholds)

    Parameters
    ----------
    I : float
        Current value (A).
    thresholds : dict, optional
        Ordered mapping ``{label: upper_bound_A}``.  If *None*, uses
        :data:`SPS_CURRENT_THRESHOLDS`.

    Returns
    -------
    str
        The cycle-type label for the given current value.
    """
    if thresholds is None:
        thresholds = SPS_CURRENT_THRESHOLDS
    for label, upper in thresholds.items():
        if I < upper:
            return label
    return "flat-high"


# =====================================================================
#  Contiguous group finder
# =====================================================================

def find_contiguous_groups(
    mask: np.ndarray,
    min_length: int = 2,
) -> list[tuple[int, int]]:
    """Find contiguous runs of True in a boolean array.

    Parameters
    ----------
    mask : ndarray of bool
        Boolean array to scan.
    min_length : int, optional
        Only return groups with at least this many consecutive True values.

    Returns
    -------
    list of (start, end) tuples
        Each tuple gives the inclusive start and end indices of a group.
    """
    groups: list[tuple[int, int]] = []
    in_group = False
    start = 0
    for i, val in enumerate(mask):
        if val:
            if not in_group:
                start = i
                in_group = True
        else:
            if in_group:
                groups.append((start, i - 1))
                in_group = False
    if in_group:
        groups.append((start, len(mask) - 1))
    return [(s, e) for s, e in groups if (e - s + 1) >= min_length]


# =====================================================================
#  Kn pipeline wrapper
# =====================================================================

def process_kn_pipeline(
    flux_abs_turns: np.ndarray,
    flux_cmp_turns: np.ndarray,
    t_turns: np.ndarray,
    I_turns: np.ndarray,
    kn,
    r_ref: float,
    magnet_order: int,
    options: tuple[str, ...] = ("dri", "rot", "cel", "fed"),
    drift_mode: str = "legacy",
    min_b1_T: float = 1e-4,
    max_zr: float = 0.01,
    merge_mode: str = "abs_upto_m_cmp_above",
):
    """Run the full Kn pipeline on selected turns.

    Wraps :func:`compute_legacy_kn_per_turn`, :func:`merge_coefficients`,
    and :func:`safe_normalize_to_units` into a single call.

    Parameters
    ----------
    flux_abs_turns, flux_cmp_turns : ndarray, shape (n_turns, Ns)
        Absolute and compensated flux per turn.
    t_turns, I_turns : ndarray, shape (n_turns, Ns)
        Time and current per turn.
    kn : SegmentKn
        Calibration coefficients.
    r_ref : float
        Reference radius (m).
    magnet_order : int
        Main harmonic order (1 for dipole).
    options : tuple of str
        Pipeline steps to enable.
    drift_mode : str
        ``"legacy"`` or ``"weighted"``.
    min_b1_T : float
        Minimum |B1| for normalisation.
    max_zr : float
        Maximum z-rotation for CEL step.
    merge_mode : str
        Channel merge strategy.

    Returns
    -------
    result : LegacyKnPerTurn
        Full per-turn pipeline results.
    C_merged : ndarray, shape (n_turns, n_orders)
        Merged complex coefficients.
    C_units : ndarray, shape (n_turns, n_orders)
        Normalised coefficients in units.
    ok_main : ndarray of bool, shape (n_turns,)
        True where |B_main| > *min_b1_T*.
    """
    result = compute_legacy_kn_per_turn(
        df_abs_turns=flux_abs_turns,
        df_cmp_turns=flux_cmp_turns,
        t_turns=t_turns,
        I_turns=I_turns,
        kn=kn,
        Rref_m=r_ref,
        magnet_order=magnet_order,
        options=options,
        drift_mode=drift_mode,
        legacy_rotate_excludes_last=False,
        max_zR=max_zr,
    )

    C_merged, _ = merge_coefficients(
        C_abs=result.C_abs,
        C_cmp=result.C_cmp,
        magnet_order=magnet_order,
        mode=merge_mode,
    )

    C_units, ok_main = safe_normalize_to_units(
        C_merged,
        magnet_order=magnet_order,
        min_main_field=min_b1_T,
    )

    return result, C_merged, C_units, ok_main


def build_harmonic_rows(
    result,
    C_merged: np.ndarray,
    C_units: np.ndarray,
    ok_main: np.ndarray,
    magnet_order: int,
    extra_columns: list[dict] | None = None,
) -> list[dict]:
    """Build a list of row-dicts from pipeline results.

    Each row contains per-turn scalars (time, current, position, phi)
    plus Bn/An (T) for orders <= *magnet_order* and bn/an (units) for
    higher orders.

    Parameters
    ----------
    result : LegacyKnPerTurn
        Pipeline results.
    C_merged : ndarray, shape (n_turns, n_orders)
        Merged complex coefficients.
    C_units : ndarray, shape (n_turns, n_orders)
        Normalised coefficients.
    ok_main : ndarray of bool
        Normalisation flag per turn.
    magnet_order : int
        Main harmonic order.
    extra_columns : list of dict, optional
        One dict per turn with additional columns to include in rows.
        Must have the same length as the number of turns.

    Returns
    -------
    list of dict
        One dict per turn, suitable for ``pd.DataFrame(rows)``.
    """
    n_turns = C_merged.shape[0]
    rows: list[dict] = []
    for t in range(n_turns):
        row = {
            "time_s": result.time_median_s[t],
            "I_mean_A": result.I_mean_A[t],
            "ok_main": bool(ok_main[t]),
            "phi_rad": result.phi_out_rad[t],
            "x_mm": result.x_m[t] * 1000,
            "y_mm": result.y_m[t] * 1000,
        }
        for i, n_ord in enumerate(result.orders):
            C = C_merged[t, i]
            if n_ord <= magnet_order:
                row[f"B{n_ord}_T"] = C.real
                row[f"A{n_ord}_T"] = C.imag
            else:
                row[f"b{n_ord}_units"] = C_units[t, i].real
                row[f"a{n_ord}_units"] = C_units[t, i].imag
        if extra_columns is not None:
            row.update(extra_columns[t])
        rows.append(row)
    return rows


# =====================================================================
#  Run-level aggregation
# =====================================================================

def build_run_averages(df_in: pd.DataFrame) -> pd.DataFrame:
    """Build per-run mean b3 with run ordering.

    Parameters
    ----------
    df_in : DataFrame
        Must contain columns ``run``, ``I_mean_A``, ``I_nom_A``,
        ``b3_units``, and ``turn_in_run``.

    Returns
    -------
    DataFrame
        One row per run with columns: ``run``, ``I_mean``, ``I_nom``,
        ``b3_mean``, ``b3_std``, ``n_turns``.  Sorted by ``run``.
    """
    avgs = df_in.groupby("run").agg(
        I_mean=("I_mean_A", "mean"),
        I_nom=("I_nom_A", "first"),
        b3_mean=("b3_units", "mean"),
        b3_std=("b3_units", "std"),
        n_turns=("turn_in_run", "count"),
    ).reset_index().sort_values("run")
    return avgs


# =====================================================================
#  DataFrame export helpers (shared by GUI and notebooks)
# =====================================================================

def ba_table_from_C(
    C: np.ndarray,
    orders: np.ndarray,
    *,
    prefix: str = "",
) -> pd.DataFrame:
    """Convert complex coefficients to legacy B/A tables per turn.

    Convention: B_n = Re(C_n), A_n = Im(C_n).
    The pipeline ``C_n`` already includes the 2/N FFT fold factor.

    Parameters
    ----------
    C : ndarray, shape (n_turns, H)
        Complex harmonic coefficients.
    orders : ndarray, shape (H,)
        Harmonic orders (1-based).
    prefix : str
        Column name prefix (e.g. ``"abs_"``, ``"cmp_"``).

    Returns
    -------
    DataFrame
        Columns ``{prefix}normal_B{n}`` and ``{prefix}skew_A{n}``.
    """
    out: Dict[str, np.ndarray] = {}
    for j, n in enumerate([int(x) for x in orders]):
        out[f"{prefix}normal_B{n}"] = np.real(C[:, j])
        out[f"{prefix}skew_A{n}"] = np.imag(C[:, j])
    return pd.DataFrame(out)


def mixed_format_table(
    C_merged: np.ndarray,
    C_units: np.ndarray,
    orders: np.ndarray,
    magnet_order: int,
    *,
    nor_was_checked: bool = False,
    prefix: str = "mrg_",
) -> pd.DataFrame:
    """Build a Bottura Section 3.7 mixed-format table.

    * ``n <= m``: columns ``B{n}_T`` / ``A{n}_T`` from *C_merged* (Tesla).
    * ``n > m``: columns ``b{n}_units`` / ``a{n}_units`` from *C_units*.

    When *nor_was_checked* is True (legacy SM18 workflow where normalization
    happened inside ``compute_legacy_kn_per_turn``), ALL harmonics are
    exported as units (``b{n}_units`` / ``a{n}_units``).

    Parameters
    ----------
    C_merged : ndarray, shape (n_turns, H)
        Merged complex coefficients (Tesla when nor not checked, units when
        nor checked).
    C_units : ndarray, shape (n_turns, H)
        Normalised coefficients in units.
    orders : ndarray, shape (H,)
        Harmonic orders (1-based).
    magnet_order : int
        Main harmonic order m.
    nor_was_checked : bool
        True if the ``"nor"`` option was active in the pipeline.
    prefix : str
        Column name prefix.

    Returns
    -------
    DataFrame
    """
    out: Dict[str, np.ndarray] = {}
    m = int(magnet_order)
    for j, n in enumerate([int(x) for x in orders]):
        if nor_was_checked:
            out[f"{prefix}b{n}_units"] = np.real(C_merged[:, j])
            out[f"{prefix}a{n}_units"] = np.imag(C_merged[:, j])
        elif n <= m:
            out[f"{prefix}B{n}_T"] = np.real(C_merged[:, j])
            out[f"{prefix}A{n}_T"] = np.imag(C_merged[:, j])
        else:
            out[f"{prefix}b{n}_units"] = np.real(C_units[:, j])
            out[f"{prefix}a{n}_units"] = np.imag(C_units[:, j])
    return pd.DataFrame(out)


# =====================================================================
#  Outlier removal (MAD sigma clip)
# =====================================================================

def mad_sigma_clip(
    df: pd.DataFrame,
    col: str,
    n_sigma: float = 5,
    label_col: str = "label",
) -> tuple[pd.DataFrame, dict]:
    """Remove outliers per operating-point label using MAD.

    For each unique value in *label_col*, computes the median and MAD
    (median absolute deviation) of *col*, then flags rows more than
    *n_sigma* scaled-MAD from the median as outliers.

    Parameters
    ----------
    df : DataFrame
        Input data.
    col : str
        Column to test for outliers.
    n_sigma : float
        Number of MAD-scaled sigmas for the clipping threshold.
    label_col : str
        Column containing operating-point labels.

    Returns
    -------
    df_clean : DataFrame
        Copy of *df* with outliers removed.
    removed : dict
        ``{label: count}`` of removed rows per operating point.
    """
    keep = pd.Series(True, index=df.index)
    removed: dict = {}
    for lab in df[label_col].unique():
        mask = df[label_col] == lab
        vals = df.loc[mask, col]
        if len(vals) < 5:
            continue
        med = vals.median()
        mad = np.median(np.abs(vals - med))
        sigma = 1.4826 * mad
        if sigma < 1e-15:
            continue
        outlier = np.abs(vals - med) > n_sigma * sigma
        n_out = outlier.sum()
        if n_out > 0:
            keep.loc[vals.index[outlier]] = False
            removed[lab] = int(n_out)
    return df[keep].copy(), removed


# =====================================================================
#  Run discovery
# =====================================================================

def discover_runs(
    run_dir: str | Path,
    pcb_label: str,
    file_pattern: str | None = None,
) -> list[dict]:
    """Discover measurement runs by parsing filenames.

    Scans *run_dir* for files matching
    ``*_{pcb_label}_raw_measurement_data.txt`` and extracts run ID and
    nominal current from the ``Run_XX_I_YYA`` portion of the filename.

    Parameters
    ----------
    run_dir : path-like
        Directory containing raw measurement files.
    pcb_label : str
        PCB segment label, e.g. ``"Integral"`` or ``"Central"``.
    file_pattern : str, optional
        Override glob pattern.  Default derives from *pcb_label*.

    Returns
    -------
    list of dict
        Each dict has keys ``run_id`` (int), ``I_nom`` (float), ``file``
        (Path).
    """
    run_dir = Path(run_dir)
    if file_pattern is None:
        file_pattern = f"*_{pcb_label}_raw_measurement_data.txt"
    files = sorted(run_dir.glob(file_pattern))
    runs: list[dict] = []
    for f in files:
        m = re.search(r'Run_(\d+)_I_([-\d.]+)A', f.name)
        if m:
            runs.append({
                "run_id": int(m.group(1)),
                "I_nom": float(m.group(2)),
                "file": f,
            })
    return runs


# =====================================================================
#  Plateau summary
# =====================================================================

def plateau_summary(
    df: pd.DataFrame,
    n_last: int,
    harmonics_range=range(2, 16),
) -> pd.DataFrame:
    """Per-run mean and std of B1, TF, and all harmonics.

    For each run, selects the last *n_last* turns, keeps only those with
    ``ok_main == True``, and computes mean/std of B1 and every harmonic
    column found in the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain ``run_id``, ``turn_in_run``, ``ok_main``, ``I_nom``,
        ``branch``, ``B1_T``, and harmonic columns ``b{n}_units`` /
        ``a{n}_units``.
    n_last : int
        Number of last turns per run to average.
    harmonics_range : range
        Harmonic orders to include (default ``range(2, 16)``).

    Returns
    -------
    DataFrame
        One row per run with mean/std columns plus quality flag.
    """
    records: list[dict] = []
    for run_id in sorted(df["run_id"].unique()):
        rdf = df[df["run_id"] == run_id].sort_values("turn_in_run")
        sel = rdf.tail(n_last)
        ok = sel["ok_main"].astype(bool)
        rec: dict = {
            "run_id": run_id,
            "I_nom": sel["I_nom"].iloc[0],
            "branch": sel["branch"].iloc[0],
            "n_total": len(rdf),
            "n_selected": len(sel),
            "n_ok": int(ok.sum()),
        }
        rec["B1_mean"] = sel.loc[ok, "B1_T"].mean() if ok.any() else np.nan
        rec["B1_std"] = (
            (sel.loc[ok, "B1_T"].std() if ok.sum() > 1 else 0.0)
            if ok.any() else np.nan
        )
        for h in harmonics_range:
            for prefix in ["b", "a"]:
                col = f"{prefix}{h}_units"
                if col in sel.columns and ok.any():
                    rec[f"{col}_mean"] = sel.loc[ok, col].mean()
                    rec[f"{col}_std"] = (
                        sel.loc[ok, col].std() if ok.sum() > 1 else 0.0
                    )
                else:
                    rec[f"{col}_mean"] = np.nan
                    rec[f"{col}_std"] = np.nan
        rec["TF"] = (
            rec["B1_mean"] / (rec["I_nom"] / 1000.0)
            if ok.any() and abs(rec["I_nom"]) > 1.0
            else np.nan
        )
        rec["quality"] = "good" if rec["n_ok"] >= max(1, n_last // 2) else "bad"
        records.append(rec)
    return pd.DataFrame(records)


# =====================================================================
#  Hysteresis plotting
# =====================================================================

def plot_hysteresis(
    ax,
    summ: pd.DataFrame,
    xcol: str,
    ycol: str,
    yerr_col: str | None = None,
    branch_col: str = "branch",
    branch_colors: dict | None = None,
):
    """Plot a hysteresis loop with run-order gradient coloring.

    Parameters
    ----------
    ax : matplotlib Axes
    summ : DataFrame
        Summary table (one row per run), must contain *xcol*, *ycol*,
        *branch_col*, and ``"quality"`` and ``"run_id"`` columns.
    xcol, ycol : str
        Column names for x and y data.
    yerr_col : str, optional
        Column name for y error bars.
    branch_col : str
        Column identifying ascending / descending branch.
    branch_colors : dict, optional
        ``{branch_label: color}``.  Defaults to
        ``{"ascending": "tab:blue", "descending": "tab:red"}``.
    """
    if branch_colors is None:
        branch_colors = {"ascending": "tab:blue", "descending": "tab:red"}

    s = summ.sort_values("run_id")
    valid = (s["quality"] == "good") & s[ycol].notna()
    if valid.sum() > 1:
        sv = s[valid].reset_index(drop=True)
        xg, yg = sv[xcol].values, sv[ycol].values
        n = len(xg)
        for i in range(n - 1):
            frac = i / max(n - 2, 1)
            ax.plot(
                [xg[i], xg[i + 1]], [yg[i], yg[i + 1]], "-",
                color=branch_colors.get(sv[branch_col].iloc[i + 1], "grey"),
                lw=1.0 + 2.5 * frac,
                alpha=0.15 + 0.75 * frac,
                solid_capstyle="round",
                zorder=2,
            )
    for br, col in branch_colors.items():
        good = (
            (s[branch_col] == br)
            & (s["quality"] == "good")
            & s[ycol].notna()
        )
        if good.any():
            kw = dict(yerr=s.loc[good, yerr_col]) if yerr_col else {}
            ax.errorbar(
                s.loc[good, xcol], s.loc[good, ycol],
                fmt="o", color=col, ms=4, capsize=2,
                label=br, zorder=4, **kw,
            )


# =====================================================================
#  Eddy-current model
# =====================================================================

def eddy_model(t, B_inf, A, tau):
    r"""Exponential eddy-current settling model.

    .. math:: B(t) = B_\infty + A \, e^{-t/\tau}

    Intended for use with :func:`scipy.optimize.curve_fit`.
    """
    return B_inf + A * np.exp(-t / tau)


# =====================================================================
#  Statistical comparison helpers
# =====================================================================

def compute_level_stats(
    df: pd.DataFrame,
    label: str,
    ok_col: str = "ok_main",
    label_col: str = "label",
) -> dict:
    """Mean/std of I, B1, b2, b3, TF for a given operating point.

    Parameters
    ----------
    df : DataFrame
        Settled / cleaned data with columns ``label``, ``ok_main``,
        ``I_mean_A``, ``B1_T``, ``b2_units``, ``b3_units``.
    label : str
        Operating-point label to filter on.
    ok_col : str
        Boolean column for quality gate.
    label_col : str
        Column containing operating-point labels.

    Returns
    -------
    dict
        Keys: ``N``, ``I_mean``, ``B1_mean``, ``B1_std``,
        ``b2_mean``, ``b2_std``, ``b3_mean``, ``b3_std``,
        ``TF_mean``, ``TF_std``.  Empty dict if no data.
    """
    sub = df[(df[label_col] == label) & df[ok_col]].copy()
    if len(sub) == 0:
        return {}
    tf = sub["B1_T"] / (sub["I_mean_A"] / 1000.0)
    return {
        "N": len(sub),
        "I_mean": sub["I_mean_A"].mean(),
        "B1_mean": sub["B1_T"].mean(),
        "B1_std": sub["B1_T"].std(),
        "b2_mean": sub["b2_units"].mean(),
        "b2_std": sub["b2_units"].std(),
        "b3_mean": sub["b3_units"].mean(),
        "b3_std": sub["b3_units"].std(),
        "TF_mean": tf.mean(),
        "TF_std": tf.std(),
    }


def diff_sigma(
    stats1: dict,
    stats2: dict,
    key: str,
) -> tuple[float, float, float]:
    """Compute difference, propagated error, and sigma significance.

    Parameters
    ----------
    stats1, stats2 : dict
        Output of :func:`compute_level_stats`.
    key : str
        Base key (e.g. ``"B1"``).  The dicts must contain
        ``{key}_mean``, ``{key}_std``, and ``N``.

    Returns
    -------
    diff : float
        ``stats1[key_mean] - stats2[key_mean]``
    error : float
        Propagated standard error of the difference.
    sigma : float
        ``|diff| / error`` (0 if error is zero).
    """
    d = stats1[f"{key}_mean"] - stats2[f"{key}_mean"]
    err = np.sqrt(
        (stats1[f"{key}_std"] ** 2 / stats1["N"])
        + (stats2[f"{key}_std"] ** 2 / stats2["N"])
    )
    sig = abs(d) / err if err > 0 else 0.0
    return d, err, sig
