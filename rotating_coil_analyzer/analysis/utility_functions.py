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
double_eddy_model
    Two-exponential eddy-current model (2 time constants).
triple_eddy_model
    Three-exponential eddy-current model (3 time constants).
validate_eddy_model_selection
    AICc-based model selection across 1/2/3-tau fits.
EddyFitResult
    Dataclass result container for eddy fits (B_inf, A, tau, R2, quality).
fit_eddy_per_run
    Fit single-exponential eddy model with 2-pass MAD clipping.
CelFedDiagnostic
    Dataclass result container for cel/fed safety diagnostic.
diagnose_cel_fed
    Run pipeline with/without cel+fed, return SAFE/UNSAFE/MIXED recommendation.
FdiTransitionCheck
    Dataclass result container for FDI stuck-channel detection.
diagnose_fdi_transitions
    Detect FDI stuck-channel issues between consecutive plateau runs.
compute_level_stats
    Mean/std of I, B1, b2, b3, TF for a given operating point.
diff_sigma
    Difference, propagated error, and sigma significance.
SPS_CURRENT_THRESHOLDS
    Default current thresholds dict for SPS cycle classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..ingest.channel_detect import robust_range
from .kn_pipeline import (
    LegacyKnPerTurn,
    SegmentKn,
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
#  FDI stuck-channel diagnostic
# =====================================================================

@dataclass(frozen=True)
class FdiTransitionCheck:
    """Diagnostic result for one transition between consecutive plateau runs.

    Detects Fast Digital Integrator (FDI) stuck-channel issues where the
    flux signal fails to respond during current ramps, causing the field
    at the start of the next plateau to remain at the previous level's
    value.  This creates fake settling transients that corrupt
    eddy-current fits.
    """

    run_before: int         #: run_id of the plateau before the gap
    run_after: int          #: run_id of the plateau after the gap
    I_before: float         #: I_nom of plateau before gap
    I_after: float          #: I_nom of plateau after gap
    delta_I: float          #: signed current change
    n_gap_turns: int        #: turns in the ramp gap between plateaus
    flux_rng_before: float  #: avg flux range at end of previous plateau
    flux_rng_gap_mid: float #: avg flux range in middle of gap (NaN if gap < 5)
    flux_rng_after_start: float   #: avg flux range at start of next plateau
    flux_rng_after_settled: float #: avg flux range at settled part of next plateau
    response_ratio: float   #: fraction of expected flux change seen at plateau start
    is_stuck: bool           #: True if FDI appears stuck at this transition
    severity: str            #: "OK", "PARTIAL", or "STUCK"
    reason: str              #: human-readable explanation


def diagnose_fdi_transitions(
    flux_turns: np.ndarray,
    I_mean: np.ndarray,
    run_info: list[dict],
    *,
    n_boundary: int = 10,
    stuck_threshold: float = 0.3,
    partial_threshold: float = 0.7,
    min_delta_I: float = 5.0,
) -> list[FdiTransitionCheck]:
    """Detect FDI stuck-channel issues between consecutive plateau runs.

    For each consecutive pair of runs, compares the per-turn flux range
    (``robust_range``) at the end of the previous run, the start of the
    next run, and the settled portion of the next run.  If the flux
    range at the start of a plateau hasn't changed as expected (given
    the settled-value change), the FDI may have been stuck during the
    ramp.

    Parameters
    ----------
    flux_turns : ndarray, shape (n_total_turns, samples_per_turn)
        One flux channel (absolute), reshaped into turns.
    I_mean : ndarray, shape (n_total_turns,)
        Per-turn mean current.
    run_info : list of dict
        Each dict must have keys ``run_id``, ``start``, ``end``,
        ``I_nom``.  Runs must be in chronological order.
    n_boundary : int
        Number of turns to average at plateau boundaries (default 10).
    stuck_threshold : float
        ``|response_ratio|`` below this classifies as STUCK (default 0.3).
    partial_threshold : float
        ``|response_ratio|`` below this classifies as PARTIAL (default 0.7).
    min_delta_I : float
        Skip transitions with ``|delta_I| < min_delta_I`` (default 5 A)
        since flux change would be too small to measure reliably.

    Returns
    -------
    list of FdiTransitionCheck
        One entry per consecutive run pair (skipping pairs with small
        ``|delta_I|``).
    """
    n_total = flux_turns.shape[0]
    # Pre-compute per-turn flux range for all turns
    flux_rng = np.array([robust_range(flux_turns[t]) for t in range(n_total)])

    checks: list[FdiTransitionCheck] = []

    for i in range(len(run_info) - 1):
        r_prev = run_info[i]
        r_next = run_info[i + 1]

        delta_I = r_next["I_nom"] - r_prev["I_nom"]
        if abs(delta_I) < min_delta_I:
            continue

        s_prev, e_prev = r_prev["start"], r_prev["end"]
        s_next, e_next = r_next["start"], r_next["end"]
        n_gap = s_next - e_prev - 1

        # Flux range at end of previous plateau
        tail_prev = slice(max(s_prev, e_prev - n_boundary + 1), e_prev + 1)
        flux_rng_before = float(np.mean(flux_rng[tail_prev]))

        # Flux range at start of next plateau
        head_next = slice(s_next, min(s_next + n_boundary, e_next + 1))
        flux_rng_after_start = float(np.mean(flux_rng[head_next]))

        # Flux range at settled portion of next plateau
        tail_next = slice(max(s_next, e_next - n_boundary + 1), e_next + 1)
        flux_rng_after_settled = float(np.mean(flux_rng[tail_next]))

        # Flux range in middle of gap (if gap large enough)
        if n_gap >= 5:
            gap_start = e_prev + 1
            gap_end = s_next
            gap_mid_s = gap_start + (gap_end - gap_start) // 4
            gap_mid_e = gap_end - (gap_end - gap_start) // 4
            flux_rng_gap_mid = float(np.mean(flux_rng[gap_mid_s:gap_mid_e]))
        else:
            flux_rng_gap_mid = float("nan")

        # Compute response ratio
        expected_change = flux_rng_after_settled - flux_rng_before
        actual_change = flux_rng_after_start - flux_rng_before

        # Need a minimum expected change to avoid division by noise
        noise_floor = 0.01 * max(abs(flux_rng_before), abs(flux_rng_after_settled), 1e-12)
        if abs(expected_change) > noise_floor:
            response_ratio = actual_change / expected_change
        else:
            # No significant expected change — cannot diagnose, assume OK
            response_ratio = 1.0

        # Classify
        abs_rr = abs(response_ratio)
        if abs_rr >= partial_threshold:
            severity = "OK"
            is_stuck = False
            reason = f"response_ratio={response_ratio:.3f} >= {partial_threshold}"
        elif abs_rr >= stuck_threshold:
            severity = "PARTIAL"
            is_stuck = False
            reason = (
                f"response_ratio={response_ratio:.3f} "
                f"({stuck_threshold} <= |rr| < {partial_threshold})"
            )
        else:
            severity = "STUCK"
            is_stuck = True
            reason = (
                f"response_ratio={response_ratio:.3f} < {stuck_threshold} -- "
                f"flux did not respond during ramp"
            )

        checks.append(FdiTransitionCheck(
            run_before=r_prev["run_id"],
            run_after=r_next["run_id"],
            I_before=r_prev["I_nom"],
            I_after=r_next["I_nom"],
            delta_I=delta_I,
            n_gap_turns=n_gap,
            flux_rng_before=flux_rng_before,
            flux_rng_gap_mid=flux_rng_gap_mid,
            flux_rng_after_start=flux_rng_after_start,
            flux_rng_after_settled=flux_rng_after_settled,
            response_ratio=response_ratio,
            is_stuck=is_stuck,
            severity=severity,
            reason=reason,
        ))

    return checks


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
    merge_mode: str = "abs_upto_m_cmp_above",
    dit_signed: bool = False,
    max_zR: float | None = None,
    encoder_offset_rad: float = 0.0,
    flip_signal_polarity: bool = False,
    legacy_rotate_excludes_last: bool = False,
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
    merge_mode : str
        Channel merge strategy.
    dit_signed : bool
        If True, use signed thresholds for the dit correction matching
        the FFMM C++ native path.  Default False uses absolute-value
        thresholds.
    max_zR : float or None
        If not None, clamp |zR| after cel and before fed.  Turns with
        |zR| > max_zR have zR set to 0 and are flagged in
        ``result.zR_clamped``.
    encoder_offset_rad : float
        Known encoder trigger offset in radians.  Pre-rotates harmonics
        before the rotation step.  Default 0.0 (no pre-rotation).
    flip_signal_polarity : bool
        If True, negate all harmonics after kn application (before
        rotation/cel/fed/normalization).  Use when B1 is negative at
        positive current due to inverted coil/cable polarity.
    legacy_rotate_excludes_last : bool
        If True, exclude the last harmonic (k=H) from rotation.
        Matches the SM18 C++ off-by-one behaviour.  Default False
        rotates all harmonics k=1..H (Bottura AIV.6, standard FFMM).

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
        legacy_rotate_excludes_last=legacy_rotate_excludes_last,
        dit_signed=dit_signed,
        max_zR=max_zR,
        encoder_offset_rad=encoder_offset_rad,
        flip_signal_polarity=flip_signal_polarity,
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
#  cel/fed diagnostic
# =====================================================================

@dataclass(frozen=True)
class CelFedDiagnostic:
    """Diagnostic result from :func:`diagnose_cel_fed`.

    Compares the pipeline with and without cel/fed to help the user
    decide whether the centre-location / feeddown correction is safe
    for their data.
    """

    zR_abs: np.ndarray  # (n_turns,) — |zR| per turn (from cel run)
    n_suspect: int  # turns with |zR| > max_zR_threshold
    n_total: int
    max_zR_threshold: float
    recommendation: str  # "SAFE", "UNSAFE", or "MIXED"
    reason: str  # human-readable explanation
    # comparison data (for plotting / inspection)
    B_main_with_fed: np.ndarray  # (n_turns,) — B_main from cel+fed pipeline
    B_main_without_fed: np.ndarray  # (n_turns,) — B_main from dri+rot pipeline
    result_with_fed: LegacyKnPerTurn
    result_without_fed: LegacyKnPerTurn


def diagnose_cel_fed(
    flux_abs_turns: np.ndarray,
    flux_cmp_turns: np.ndarray,
    t_turns: np.ndarray,
    I_turns: np.ndarray,
    *,
    kn: SegmentKn,
    r_ref: float,
    magnet_order: int,
    max_zR: float = 0.01,
    dit_signed: bool = False,
    drift_mode: str = "legacy",
) -> CelFedDiagnostic:
    """Diagnose whether cel/fed is safe for the given data.

    Runs the pipeline twice — once with ``("dri", "rot", "cel", "fed")``
    and once with ``("dri", "rot")`` — then compares the results and
    produces a recommendation.

    Parameters
    ----------
    flux_abs_turns, flux_cmp_turns : ndarray, shape (n_turns, Ns)
        Incremental absolute and compensated signals.
    t_turns, I_turns : ndarray, shape (n_turns, Ns)
        Time and current per turn.
    kn : SegmentKn
        Calibration coefficients.
    r_ref : float
        Reference radius (m).
    magnet_order : int
        Main harmonic order (1 for dipole, 2 for quadrupole, ...).
    max_zR : float
        Threshold for suspect |zR| values (dimensionless).  Default
        0.01 (~0.33 mm for a 33 mm coil).
    dit_signed : bool
        Passed through to the pipeline.
    drift_mode : str
        Passed through to the pipeline.

    Returns
    -------
    CelFedDiagnostic
        Structured diagnostic with recommendation and comparison data.
    """
    # Run with cel+fed (no clamping — we want the raw zR values)
    result_with = compute_legacy_kn_per_turn(
        df_abs_turns=flux_abs_turns,
        df_cmp_turns=flux_cmp_turns,
        t_turns=t_turns,
        I_turns=I_turns,
        kn=kn,
        Rref_m=r_ref,
        magnet_order=magnet_order,
        options=("dri", "rot", "cel", "fed"),
        drift_mode=drift_mode,
        legacy_rotate_excludes_last=False,
        dit_signed=dit_signed,
    )

    # Run without cel/fed
    result_without = compute_legacy_kn_per_turn(
        df_abs_turns=flux_abs_turns,
        df_cmp_turns=flux_cmp_turns,
        t_turns=t_turns,
        I_turns=I_turns,
        kn=kn,
        Rref_m=r_ref,
        magnet_order=magnet_order,
        options=("dri", "rot"),
        drift_mode=drift_mode,
        legacy_rotate_excludes_last=False,
        dit_signed=dit_signed,
    )

    # Analyse zR from the cel run
    zR_abs = np.abs(result_with.zR)
    n_total = len(zR_abs)
    suspect = zR_abs > float(max_zR)
    n_suspect = int(np.sum(suspect))

    # B_main comparison
    m = int(magnet_order)
    B_main_with = np.real(result_with.C_abs[:, m - 1])
    B_main_without = np.real(result_without.C_abs[:, m - 1])

    # Recommendation logic
    frac_suspect = n_suspect / n_total if n_total > 0 else 0.0
    median_zR = float(np.median(zR_abs))
    max_zR_val = float(np.max(zR_abs)) if n_total > 0 else 0.0

    if frac_suspect == 0:
        recommendation = "SAFE"
        reason = (
            f"All {n_total} turns have |zR| <= {max_zR:.4f} "
            f"(median {median_zR:.4f}, max {max_zR_val:.4f}). "
            f"cel/fed is reliable — apply it."
        )
    elif frac_suspect > 0.5:
        recommendation = "UNSAFE"
        reason = (
            f"{n_suspect}/{n_total} turns ({100*frac_suspect:.0f}%) have "
            f"|zR| > {max_zR:.4f} (median {median_zR:.4f}, max {max_zR_val:.4f}). "
            f"cel/fed produces unreliable offsets — skip it and use "
            f"OPTIONS = ('dri', 'rot')."
        )
    else:
        recommendation = "MIXED"
        reason = (
            f"{n_suspect}/{n_total} turns ({100*frac_suspect:.0f}%) have "
            f"|zR| > {max_zR:.4f} (median {median_zR:.4f}, max {max_zR_val:.4f}). "
            f"Some turns are reliable, some are not — use max_zR={max_zR} "
            f"to clamp suspect turns."
        )

    return CelFedDiagnostic(
        zR_abs=zR_abs,
        n_suspect=n_suspect,
        n_total=n_total,
        max_zR_threshold=float(max_zR),
        recommendation=recommendation,
        reason=reason,
        B_main_with_fed=B_main_with,
        B_main_without_fed=B_main_without,
        result_with_fed=result_with,
        result_without_fed=result_without,
    )


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
    n_skip_end: int = 0,
) -> pd.DataFrame:
    """Per-run mean and std of B1, TF, and all harmonics.

    For each run, drops the last *n_skip_end* turns (to avoid ramp-start
    contamination), then selects the last *n_last* of the remaining turns,
    keeps only those with ``ok_main == True``, and computes mean/std of
    B1 and every harmonic column found in the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain ``run_id``, ``turn_in_run``, ``ok_main``, ``I_nom``,
        ``branch``, ``B1_T``, and harmonic columns ``b{n}_units`` /
        ``a{n}_units``.
    n_last : int
        Number of turns per run to average (after dropping the tail).
    harmonics_range : range
        Harmonic orders to include (default ``range(2, 16)``).
    n_skip_end : int
        Number of turns to drop from the end of each run (default 0).

    Returns
    -------
    DataFrame
        One row per run with mean/std columns plus quality flag.
    """
    records: list[dict] = []
    for run_id in sorted(df["run_id"].unique()):
        rdf = df[df["run_id"] == run_id].sort_values("turn_in_run")
        if n_skip_end > 0 and len(rdf) > n_skip_end:
            rdf = rdf.iloc[:-n_skip_end]
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
    """Plot a hysteresis loop with lines connecting adjacent current levels.

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

    for br, col in branch_colors.items():
        good = (
            (summ[branch_col] == br)
            & (summ["quality"] == "good")
            & summ[ycol].notna()
        )
        if not good.any():
            continue
        seg = summ.loc[good].sort_values(xcol)
        # Split into sub-segments at I=0 to avoid connecting lines
        # across the centre where ascending/descending branches
        # from different parts of the cycle would create spurious jumps.
        x_vals = seg[xcol].values
        sub_masks = []
        if (x_vals > 0).any():
            sub_masks.append(seg[xcol] > 0)
        if (x_vals < 0).any():
            sub_masks.append(seg[xcol] < 0)
        if (x_vals == 0).any():
            sub_masks.append(seg[xcol] == 0)
        first = True
        for mask in sub_masks:
            sub = seg.loc[mask].sort_values(xcol)
            if sub.empty:
                continue
            kw = dict(yerr=sub[yerr_col]) if yerr_col else {}
            ax.errorbar(
                sub[xcol], sub[ycol],
                fmt="o-", color=col, ms=4, capsize=2,
                label=br if first else "_nolegend_",
                zorder=4, **kw,
            )
            first = False
        # Plot I=0 points as standalone markers (no connecting line)
        zero = seg.loc[seg[xcol] == 0]
        if not zero.empty and len(sub_masks) > 1:
            kw = dict(yerr=zero[yerr_col]) if yerr_col else {}
            ax.errorbar(
                zero[xcol], zero[ycol],
                fmt="o", color=col, ms=4, capsize=2,
                label="_nolegend_", zorder=4, **kw,
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


def double_eddy_model(t, B_inf, A1, tau1, A2, tau2):
    r"""Two-exponential eddy-current settling model.

    .. math:: B(t) = B_\infty + A_1 \, e^{-t/\tau_1} + A_2 \, e^{-t/\tau_2}

    Parameters are ordered so that ``tau1 < tau2`` (fast + slow component).
    Intended for use with :func:`scipy.optimize.curve_fit`.
    """
    return B_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def triple_eddy_model(t, B_inf, A1, tau1, A2, tau2, A3, tau3):
    r"""Three-exponential eddy-current settling model.

    .. math:: B(t) = B_\infty + A_1 \, e^{-t/\tau_1} + A_2 \, e^{-t/\tau_2}
              + A_3 \, e^{-t/\tau_3}

    Parameters are ordered so that ``tau1 < tau2 < tau3``.
    Intended for use with :func:`scipy.optimize.curve_fit`.
    """
    return (B_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)
            + A3 * np.exp(-t / tau3))


def validate_eddy_model_selection(fit_results, *, max_tau=50.0, min_tau_ratio=1.5):
    """Choose the best eddy-current model while guarding against overfitting.

    Takes a dict ``{1: {...}, 2: {...}, 3: {...}}`` where each value has
    keys ``"popt"`` (parameter array or None), ``"r2"``, ``"aic"``.

    Overfitting indicators (any triggers a downgrade to simpler model):

    * **Unphysical tau**: any tau > *max_tau* (default 50 s for laminated
      iron — eddy time constants above this are not physical for typical
      accelerator-magnet yoke laminations).
    * **Redundant taus**: ratio tau_{i+1}/tau_i < *min_tau_ratio* (default
      1.5) — the extra exponential is just duplicating the other.

    Model complexity is otherwise governed by AICc (which already penalises
    extra parameters).  No additional R² threshold is applied because AICc
    is a more principled criterion that accounts for sample size.

    Returns ``(best_n_tau, reason)`` where *reason* is ``"OK"`` or a short
    explanation of the downgrade.
    """
    # Gather valid fits (converged, positive R²)
    valid = {k: v for k, v in fit_results.items()
             if isinstance(k, int) and v.get("popt") is not None and v.get("r2", 0) > 0}
    if not valid:
        return (1, "NO_FIT")

    def _extract_taus(popt, n_tau):
        if n_tau == 1:
            return [popt[2]]
        elif n_tau == 2:
            return sorted([popt[2], popt[4]])
        else:  # 3
            return sorted([popt[2], popt[4], popt[6]])

    def _is_physical(n_tau, v):
        taus = _extract_taus(v["popt"], n_tau)
        # Check max tau
        if any(t > max_tau for t in taus):
            return False, f"tau={max(taus):.1f}s>{max_tau}s"
        # Check tau separation (only for multi-tau)
        if len(taus) >= 2:
            for i in range(len(taus) - 1):
                if taus[i] > 0 and taus[i + 1] / taus[i] < min_tau_ratio:
                    return False, f"taus too close ({taus[i]:.2f}/{taus[i+1]:.2f})"
        return True, "OK"

    # Pick AICc-best among all valid, then validate downward
    aic_best = min(valid, key=lambda k: valid[k].get("aic", float("inf")))

    for n_tau in range(aic_best, 0, -1):
        if n_tau not in valid:
            continue
        ok, msg = _is_physical(n_tau, valid[n_tau])
        if ok:
            return (n_tau, "OK")

    # All models failed physical validation — use simplest available
    simplest = min(valid)
    return (simplest, "all higher models unphysical")


# =====================================================================
#  Eddy-current per-run fit helper
# =====================================================================

@dataclass(frozen=True)
class EddyFitResult:
    """Result of single-exponential eddy-current fit on one run."""

    run_id: int
    I_nom: float
    branch: str
    B_inf: float          # asymptotic field [T]
    A: float              # eddy amplitude [T]
    tau: float            # time constant [turns]
    tau_err: float        # fit uncertainty on tau
    r2: float             # coefficient of determination
    n_turns: int          # turns used in fit
    n_outliers: int       # turns removed by MAD clip
    n_trimmed: int        # leading ramp turns trimmed (0 if no I_mean given)
    quality: str          # "GOOD", "WEAK_SIGNAL", "MARGINAL", "FIT_FAILED"
    reason: str           # human-readable explanation for quality


def fit_eddy_per_run(
    turns: np.ndarray,
    B1: np.ndarray,
    run_id: int,
    I_nom: float,
    branch: str,
    n_settled: int = 50,
    tau_bounds: tuple = (0.5, 500),
    mad_n_sigma: float = 5.0,
    I_mean: np.ndarray | None = None,
    I_ramp_threshold: float = 0.5,
) -> EddyFitResult:
    """Fit a single-exponential eddy model to one run's B1 data.

    Uses a two-pass approach: (1) fit on all data, (2) MAD on residuals
    to reject genuine spikes, (3) refit on cleaned data.  Classifies the
    result as ``GOOD`` (R² >= 0.9), ``WEAK_SIGNAL`` (|A| below noise
    floor), ``MARGINAL`` (everything else), or ``FIT_FAILED``.

    Parameters
    ----------
    turns : array
        Turn numbers (x-axis for the fit).
    B1 : array
        Dipole field values [T] for each turn.
    run_id : int
        Run identifier (for labelling).
    I_nom : float
        Nominal current [A].
    branch : str
        Hysteresis branch ("ascending" / "descending").
    n_settled : int
        Number of last turns used to estimate noise floor and B_inf.
    tau_bounds : tuple
        (lower, upper) bounds for tau in the fit.
    mad_n_sigma : float
        Number of MAD-scaled sigma for residual outlier rejection.
    I_mean : array or None
        Per-turn mean current [A].  If provided, leading turns where
        ``|I_mean - I_settled| > I_ramp_threshold`` are trimmed before
        fitting (removes ramp contamination from plateau detection).
    I_ramp_threshold : float
        Current deviation threshold [A] for ramp trimming (default 0.5).

    Returns
    -------
    EddyFitResult
    """
    from scipy.optimize import curve_fit as _curve_fit
    from scipy.stats import median_abs_deviation as _mad

    n_total = len(B1)

    # ── Trim leading ramp turns (if current data provided) ────────────
    n_trimmed = 0
    if I_mean is not None and len(I_mean) == len(B1):
        n_tail_I = min(n_settled, len(I_mean))
        I_settled = I_mean[-n_tail_I:].mean()
        ramp_flags = np.abs(I_mean - I_settled) > I_ramp_threshold
        # Only trim contiguous leading ramp turns
        first_ok = 0
        for i in range(len(ramp_flags)):
            if not ramp_flags[i]:
                first_ok = i
                break
        else:
            first_ok = len(ramp_flags)
        if first_ok > 0:
            turns = turns[first_ok:]
            B1 = B1[first_ok:]
            n_trimmed = first_ok

    if len(B1) < 10:
        return EddyFitResult(
            run_id=run_id, I_nom=I_nom, branch=branch,
            B_inf=np.nan, A=np.nan, tau=np.nan, tau_err=np.nan,
            r2=0.0, n_turns=n_total, n_outliers=0, n_trimmed=n_trimmed,
            quality="FIT_FAILED",
            reason=f"Only {len(B1)} turns after ramp trim",
        )

    # ── Trim leading "stuck" turns where field hasn't started moving ──
    # In bulk iron magnets the field can lag the current by many turns;
    # the first N turns may sit at the *previous* level's B1, violating
    # the exponential assumption.  Detect and trim them.
    n_tail_b = min(n_settled, len(B1))
    _nf = B1[-n_tail_b:].std()
    _transient = abs(B1[0] - B1[-n_tail_b:].mean())
    if _nf > 0 and _transient > 10 * _nf:
        stuck_thr = 5 * _nf
        n_stuck = 0
        for i in range(min(100, len(B1))):
            if abs(B1[i] - B1[0]) > stuck_thr:
                break
            n_stuck = i + 1
        if n_stuck > 5:
            turns = turns[n_stuck:]
            B1 = B1[n_stuck:]
            if I_mean is not None and len(I_mean) > n_trimmed + n_stuck:
                I_mean = I_mean[n_trimmed + n_stuck:]
            n_trimmed += n_stuck

    if len(B1) < 10:
        return EddyFitResult(
            run_id=run_id, I_nom=I_nom, branch=branch,
            B_inf=np.nan, A=np.nan, tau=np.nan, tau_err=np.nan,
            r2=0.0, n_turns=n_total, n_outliers=0, n_trimmed=n_trimmed,
            quality="FIT_FAILED",
            reason=f"Only {len(B1)} turns after stuck-field trim",
        )

    # ── Estimate B_inf and noise floor from settled region ─────────────
    n_tail = min(n_settled, len(B1))
    B_inf_est = B1[-n_tail:].mean()
    noise_floor = B1[-n_tail:].std()
    A_est = B1[0] - B_inf_est
    tau_est = 20.0

    # ── Pass 1: fit on all data ────────────────────────────────────────
    try:
        popt1, _ = _curve_fit(
            eddy_model, turns, B1,
            p0=[B_inf_est, A_est, tau_est],
            bounds=([-np.inf, -np.inf, tau_bounds[0]],
                    [np.inf, np.inf, tau_bounds[1]]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return EddyFitResult(
            run_id=run_id, I_nom=I_nom, branch=branch,
            B_inf=np.nan, A=np.nan, tau=np.nan, tau_err=np.nan,
            r2=0.0, n_turns=n_total, n_outliers=0, n_trimmed=n_trimmed,
            quality="FIT_FAILED",
            reason="curve_fit did not converge (pass 1)",
        )

    # ── MAD on residuals (removes spikes, not the transient) ──────────
    residuals = B1 - eddy_model(turns, *popt1)
    mad_val = _mad(residuals, scale="normal")
    if mad_val > 0:
        keep = np.abs(residuals) < mad_n_sigma * mad_val
    else:
        keep = np.ones(len(B1), dtype=bool)
    n_outliers = int(np.sum(~keep))
    turns_c = turns[keep]
    B1_c = B1[keep]

    if len(B1_c) < 10:
        return EddyFitResult(
            run_id=run_id, I_nom=I_nom, branch=branch,
            B_inf=np.nan, A=np.nan, tau=np.nan, tau_err=np.nan,
            r2=0.0, n_turns=n_total, n_outliers=n_outliers, n_trimmed=n_trimmed,
            quality="FIT_FAILED",
            reason=f"Only {len(B1_c)} turns after residual MAD clip",
        )

    # ── Pass 2: refit on cleaned data ─────────────────────────────────
    try:
        popt, pcov = _curve_fit(
            eddy_model, turns_c, B1_c,
            p0=popt1,
            bounds=([-np.inf, -np.inf, tau_bounds[0]],
                    [np.inf, np.inf, tau_bounds[1]]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        B_inf, A, tau = popt
        tau_err = perr[2]

        # R² on cleaned data
        B1_pred = eddy_model(turns_c, *popt)
        ss_res = np.sum((B1_c - B1_pred) ** 2)
        ss_tot = np.sum((B1_c - B1_c.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except (RuntimeError, ValueError):
        return EddyFitResult(
            run_id=run_id, I_nom=I_nom, branch=branch,
            B_inf=np.nan, A=np.nan, tau=np.nan, tau_err=np.nan,
            r2=0.0, n_turns=n_total, n_outliers=n_outliers, n_trimmed=n_trimmed,
            quality="FIT_FAILED",
            reason="curve_fit did not converge (pass 2)",
        )

    # ── Classify quality ───────────────────────────────────────────────
    if abs(A) < 3 * noise_floor:
        quality = "WEAK_SIGNAL"
        reason = (f"|A|={abs(A):.2e} T < 3*noise {noise_floor:.2e} T "
                  f"-- no detectable transient")
    elif r2 >= 0.9:
        quality = "GOOD"
        reason = f"R2={r2:.4f}"
    else:
        quality = "MARGINAL"
        reason = f"R2={r2:.4f}, |A|={abs(A):.2e} T"

    return EddyFitResult(
        run_id=run_id, I_nom=I_nom, branch=branch,
        B_inf=B_inf, A=A, tau=tau, tau_err=tau_err,
        r2=r2, n_turns=n_total, n_outliers=n_outliers, n_trimmed=n_trimmed,
        quality=quality, reason=reason,
    )


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
