"""Reusable utility functions for streaming rotating-coil analysis.

These functions extract common logic from the Jupyter analysis notebooks
so that notebook code stays concise and multiple notebooks can share the
same tested implementations.  They are designed for CERN accelerator-magnet
rotating-coil measurements across all machine complexes (LHC, SPS, PS, PSB,
transfer lines, test benches such as SM18).

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
"""

from __future__ import annotations

import numpy as np

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
