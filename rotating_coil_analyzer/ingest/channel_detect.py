"""Shared channel detection and validation for streaming and plateau readers.

This module centralises the heuristic-based column identification that was
previously duplicated in ``readers_streaming.py`` and ``readers_plateau.py``.

It also provides :class:`ColumnMapping` for explicit column assignment,
bypassing the heuristic when the operator knows the file layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnMapping:
    """Explicit column assignment override.

    Set individual fields to ``None`` (the default) to fall back to
    automatic detection for that column.  Set to an integer column index
    to force assignment.
    """

    time_col: Optional[int] = None
    flux_abs_col: Optional[int] = None
    flux_cmp_col: Optional[int] = None
    current_col: Optional[int] = None


# ---------------------------------------------------------------------------
# Robust range helper
# ---------------------------------------------------------------------------


def robust_range(x: np.ndarray) -> float:
    """Compute the robust dynamic range as *p99.5 - p0.5*.

    This is more resilient to outliers than ``max - min``.
    Returns ``nan`` for empty or all-NaN arrays.
    """
    if x.size == 0:
        return float("nan")
    try:
        return float(np.nanpercentile(x, 99.5) - np.nanpercentile(x, 0.5))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Flux channel detection
# ---------------------------------------------------------------------------


def detect_flux_channels(
    mat: np.ndarray,
    *,
    mapping: Optional[ColumnMapping] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, List[str]]:
    """Detect (or assign) absolute and compensated flux columns.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix, shape ``(n_samples, n_cols)`` with ``n_cols >= 3``.
    mapping : ColumnMapping, optional
        When ``flux_abs_col`` *and* ``flux_cmp_col`` are both set, those
        columns are used directly (no heuristic).

    Returns
    -------
    df_abs, df_cmp : np.ndarray
        1-D arrays of length ``n_samples``.
    abs_col, cmp_col : int
        Column indices that were selected.
    warnings : list of str
        Diagnostic messages (always populated, even on success).
    """
    warnings: List[str] = []

    # --- explicit override ---
    if (
        mapping is not None
        and mapping.flux_abs_col is not None
        and mapping.flux_cmp_col is not None
    ):
        abs_col = mapping.flux_abs_col
        cmp_col = mapping.flux_cmp_col
        df_abs = mat[:, abs_col].astype(np.float64, copy=False)
        df_cmp = mat[:, cmp_col].astype(np.float64, copy=False)
        warnings.append(
            f"explicit column mapping: df_abs=col{abs_col}, df_cmp=col{cmp_col}"
        )
        return df_abs, df_cmp, abs_col, cmp_col, warnings

    # --- auto-detect: larger robust range -> absolute channel ---
    c1 = mat[:, 1].astype(np.float64, copy=False)
    c2 = mat[:, 2].astype(np.float64, copy=False)
    r1 = robust_range(c1)
    r2 = robust_range(c2)

    if np.isfinite(r1) and np.isfinite(r2) and r2 > r1:
        warnings.append(
            "swapped flux columns: treated col2 as abs and col1 as cmp (by robust range)"
        )
        return c2, c1, 2, 1, warnings

    return c1, c2, 1, 2, warnings


# ---------------------------------------------------------------------------
# Current channel detection
# ---------------------------------------------------------------------------


def detect_current_channel(
    mat: np.ndarray,
    *,
    start_col: int = 3,
    mapping: Optional[ColumnMapping] = None,
    min_finite_frac: float = 0.9,
) -> Tuple[np.ndarray, int, List[str]]:
    """Detect (or assign) the main current column.

    Parameters
    ----------
    mat : np.ndarray
        Data matrix.
    start_col : int
        First column index to consider as a current candidate.
    mapping : ColumnMapping, optional
        When ``current_col`` is set, that column is used directly.
    min_finite_frac : float
        Minimum fraction of finite samples required for a candidate column.

    Returns
    -------
    I_main : np.ndarray
        1-D current array.
    col_idx : int
        Selected column index (or -1 if none found).
    warnings : list of str
    """
    warnings: List[str] = []
    n_rows, n_cols = mat.shape

    # --- explicit override ---
    if mapping is not None and mapping.current_col is not None:
        col_idx = mapping.current_col
        I_main = mat[:, col_idx].astype(np.float64, copy=False)
        warnings.append(f"explicit current column mapping: col{col_idx}")
        return I_main, col_idx, warnings

    # --- auto-detect: largest robust range among start_col.. ---
    if n_cols <= start_col:
        warnings.append("no current columns available (not enough columns)")
        return np.full(n_rows, np.nan, dtype=np.float64), -1, warnings

    ranges: List[Tuple[float, int]] = []
    for k in range(start_col, n_cols):
        c = mat[:, k].astype(np.float64, copy=False)
        n_finite = int(np.isfinite(c).sum())
        if n_finite < max(10, int(min_finite_frac * n_rows)):
            ranges.append((float("-inf"), k))
            continue
        ranges.append((robust_range(c), k))

    # Select by max range; tie-breaker: smallest column index
    best_range = max(r for r, _ in ranges) if ranges else float("-inf")
    best_ks = [k for r, k in ranges if np.isfinite(r) and abs(r - best_range) <= 0.0]
    best_k = min(best_ks) if best_ks else (ranges[0][1] if ranges else start_col)

    I_main = mat[:, best_k].astype(np.float64, copy=False)
    rng_txt = ", ".join(
        f"col{k}:{r:.6g}" for r, k in sorted(ranges, key=lambda ri: (-ri[0], ri[1]))
        if np.isfinite(r)
    )
    warnings.append(f"current candidate ranges (p99.5-p0.5): {rng_txt}")
    warnings.append(f"selected current col{best_k}")

    return I_main, best_k, warnings


# ---------------------------------------------------------------------------
# Post-detection validation
# ---------------------------------------------------------------------------


def validate_channel_assignment(
    df_abs: np.ndarray,
    df_cmp: np.ndarray,
    *,
    min_abs_range: float = 1e-15,
) -> List[str]:
    """Validate detected channel assignment and return warning strings.

    This function never raises -- it only produces diagnostic warnings.
    The caller decides whether to abort or continue.
    """
    warnings: List[str] = []

    r_abs = robust_range(df_abs)
    r_cmp = robust_range(df_cmp)

    if np.isfinite(r_abs) and np.isfinite(r_cmp):
        if r_cmp > r_abs:
            warnings.append(
                f"WARNING: cmp robust range ({r_cmp:.6g}) exceeds abs ({r_abs:.6g}). "
                "Channel assignment may be wrong. Consider explicit ColumnMapping."
            )
        if r_abs < min_abs_range:
            warnings.append(
                f"WARNING: abs channel robust range ({r_abs:.6g}) is extremely small. "
                "Signal may be absent or dominated by noise."
            )
    else:
        if not np.isfinite(r_abs):
            warnings.append("WARNING: abs channel robust range is non-finite.")
        if not np.isfinite(r_cmp):
            warnings.append("WARNING: cmp channel robust range is non-finite.")

    warnings.append(
        f"flux ranges (p99.5-p0.5): abs={r_abs:.6g}, cmp={r_cmp:.6g}"
    )

    return warnings
