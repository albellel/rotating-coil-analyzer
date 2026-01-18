from __future__ import annotations

"""Phase II preprocessing helpers.

This module implements *legacy-compatible* preprocessing steps that occur
*before* harmonic analysis (FFT).

Hard constraint (project-wide)
-----------------------------
No synthetic time is allowed. All computations that use time do so using the
measured acquisition time array as provided by the ingest layer. This module
never creates, repairs, interpolates, or extrapolates timestamps.

Implemented steps
-----------------
1) ``dit`` / ``di/dt`` correction (optional)

   The legacy analyzers apply a current-ramp correction on the incremental
   signal prior to integration/FFT. When the current is ramping and the mean
   current is sufficiently large, incremental samples are reweighted by:

     w_k = I_mean / I_k

   A turn is considered "on a ramp" if:
     - dI/dt > 0.1 A/s
     - mean(I) > 10 A

2) Drift correction and integration (optional)

   Two drift modes are supported:

   - "legacy" (C++):
       df0 = df - mean(df)
       flux = cumsum(df0)
       flux = flux - mean(flux)

     This matches the legacy C++ implementation exactly.

   - "weighted" (Bottura/Pentella):
       dt_k = t_k - t_{k-1}   (measured)
       df_k <- df_k - (sum(df)/sum(dt)) * dt_k
       flux = cumsum(df)

     This form is more appropriate when dt varies within a turn.

3) Provenance columns (recommended)

   For reproducibility, the GUI exports per-turn harmonic tables. When
   preprocessing options are enabled, it is beneficial to also store metadata
   per turn (e.g. whether di/dt correction was applied, the fitted dI/dt, drift
   mode, and drift offset estimates). These provenance columns do not change
   the physics; they record what was done.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


DriftMode = Literal["legacy", "weighted"]


@dataclass(frozen=True)
class DiDtResult:
    """Diagnostics for the ``dit`` / ``di/dt`` correction."""

    weights: np.ndarray  # (n_turns, Ns)
    applied: np.ndarray  # (n_turns,)
    I_mean_A: np.ndarray  # (n_turns,)
    dI_dt_A_per_s: np.ndarray  # (n_turns,)
    I_min_abs_A: np.ndarray  # (n_turns,)


@dataclass(frozen=True)
class DriftResult:
    """Diagnostics for drift correction."""

    mode: DriftMode
    applied: np.ndarray  # (n_turns,)
    total_time_s: Optional[np.ndarray] = None  # (n_turns,) for weighted mode
    offset_per_s: Optional[np.ndarray] = None  # (n_turns,) for weighted mode


def _validate_2d_same_shape(*arrays: np.ndarray) -> Tuple[int, int]:
    if not arrays:
        raise ValueError("No arrays provided")

    a0 = np.asarray(arrays[0])
    if a0.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a0.shape}")

    n_turns, Ns = a0.shape
    for i, a in enumerate(arrays[1:], start=1):
        ai = np.asarray(a)
        if ai.ndim != 2:
            raise ValueError(f"Array {i} is not 2D, got shape {ai.shape}")
        if ai.shape != (n_turns, Ns):
            raise ValueError(f"Array {i} shape mismatch: expected {(n_turns, Ns)}, got {ai.shape}")

    return n_turns, Ns


def estimate_linear_slope_per_turn(t_turns: np.ndarray, y_turns: np.ndarray) -> np.ndarray:
    """Least-squares slope per turn for y(t).

    Parameters
    ----------
    t_turns, y_turns:
        Arrays of shape (n_turns, Ns).

    Returns
    -------
    slope:
        Array of shape (n_turns,) with dy/dt slope from a least-squares fit.
        If the fit is not well-defined, slope is NaN.
    """

    _validate_2d_same_shape(t_turns, y_turns)

    t = np.asarray(t_turns, dtype=float)
    y = np.asarray(y_turns, dtype=float)

    t0 = np.mean(t, axis=1, keepdims=True)
    y0 = np.mean(y, axis=1, keepdims=True)

    dt = t - t0
    dy = y - y0

    den = np.sum(dt * dt, axis=1)
    num = np.sum(dt * dy, axis=1)

    slope = np.full(t.shape[0], np.nan, dtype=float)
    ok = den > 0.0
    slope[ok] = num[ok] / den[ok]
    return slope


def di_dt_weights(
    t_turns: np.ndarray,
    I_turns: np.ndarray,
    *,
    min_slope_A_per_s: float = 0.1,
    min_mean_I_A: float = 10.0,
    eps_I_A: float = 1e-12,
) -> DiDtResult:
    """Compute per-sample weights for the legacy ``dit`` / ``di/dt`` correction.

    A turn is corrected if:
      - dI/dt > min_slope_A_per_s
      - mean(I) > min_mean_I_A
      - all samples are finite and |I| > eps_I_A

    The weights are:
      w_k = I_mean / I_k

    Returns
    -------
    DiDtResult
        weights is (n_turns, Ns), applied is (n_turns,).
    """

    _validate_2d_same_shape(t_turns, I_turns)

    t = np.asarray(t_turns, dtype=float)
    I = np.asarray(I_turns, dtype=float)

    I_mean = np.mean(I, axis=1)
    slope = estimate_linear_slope_per_turn(t, I)
    I_min_abs = np.min(np.abs(I), axis=1)

    finite = np.all(np.isfinite(t), axis=1) & np.all(np.isfinite(I), axis=1)
    ok_I = I_min_abs > float(eps_I_A)
    on_ramp = (slope > float(min_slope_A_per_s)) & (I_mean > float(min_mean_I_A))

    applied = finite & ok_I & on_ramp

    weights = np.ones_like(I, dtype=float)
    if np.any(applied):
        weights[applied, :] = I_mean[applied, None] / I[applied, :]

    # Guard against NaN/inf weights.
    bad_w = ~np.all(np.isfinite(weights), axis=1)
    if np.any(bad_w):
        weights[bad_w, :] = 1.0
        applied = applied & (~bad_w)

    return DiDtResult(
        weights=weights,
        applied=applied,
        I_mean_A=I_mean,
        dI_dt_A_per_s=slope,
        I_min_abs_A=I_min_abs,
    )


def apply_di_dt_to_channels(
    df_abs_turns: np.ndarray,
    df_cmp_turns: np.ndarray,
    t_turns: np.ndarray,
    I_turns: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, DiDtResult]:
    """Apply ``di/dt`` weights to absolute and compensated incremental signals."""

    _validate_2d_same_shape(df_abs_turns, df_cmp_turns, t_turns, I_turns)

    res = di_dt_weights(t_turns=t_turns, I_turns=I_turns, **kwargs)
    df_abs_corr = np.asarray(df_abs_turns, dtype=float) * res.weights
    df_cmp_corr = np.asarray(df_cmp_turns, dtype=float) * res.weights

    return df_abs_corr, df_cmp_corr, res


def integrate_to_flux(
    df_turns: np.ndarray,
    *,
    drift: bool = False,
    drift_mode: DriftMode = "legacy",
    t_turns: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[DriftResult]]:
    """Integrate an incremental signal to flux, with optional drift correction.

    Parameters
    ----------
    df_turns:
        Incremental signal per turn, shape (n_turns, Ns).
    drift:
        Whether to apply drift correction.
    drift_mode:
        - "legacy": C++ expression (uniform dt)
        - "weighted": Bottura/Pentella dt-weighted correction using measured time
    t_turns:
        Required for drift_mode="weighted". Measured timestamps per turn.

    Returns
    -------
    flux_turns, diagnostics
        flux_turns has shape (n_turns, Ns). If drift is False, diagnostics is None.
    """

    _validate_2d_same_shape(df_turns)
    df = np.asarray(df_turns, dtype=float)

    if not drift:
        return np.cumsum(df, axis=1), None

    mode: DriftMode = drift_mode
    if mode not in ("legacy", "weighted"):
        raise ValueError(f"Unknown drift mode: {drift_mode!r}")

    if mode == "legacy":
        # Match the legacy C++ analyzer: cumsum(df-mean(df)) then recenter flux.
        df0 = df - np.mean(df, axis=1, keepdims=True)
        flux = np.cumsum(df0, axis=1)
        flux = flux - np.mean(flux, axis=1, keepdims=True)

        applied = np.all(np.isfinite(df), axis=1)
        return flux, DriftResult(mode="legacy", applied=applied)

    # weighted
    if t_turns is None:
        raise ValueError("t_turns is required for drift_mode='weighted'")

    _validate_2d_same_shape(df_turns, t_turns)
    t = np.asarray(t_turns, dtype=float)

    # dt_k = t_k - t_{k-1}; prepend makes dt[*,0] = 0.
    dt = np.diff(t, axis=1, prepend=t[:, :1])

    finite = np.all(np.isfinite(df), axis=1) & np.all(np.isfinite(dt), axis=1)
    total_time = np.sum(dt, axis=1)

    # Require positive total time; turns with non-positive total_time are not corrected.
    ok_time = np.isfinite(total_time) & (total_time > 0.0)
    applied = finite & ok_time

    df_corr = np.array(df, copy=True)
    offset = np.full(df.shape[0], np.nan, dtype=float)

    if np.any(applied):
        offset[applied] = np.sum(df[applied, :], axis=1) / total_time[applied]
        df_corr[applied, :] = df_corr[applied, :] - offset[applied, None] * dt[applied, :]

    flux = np.cumsum(df_corr, axis=1)

    return flux, DriftResult(mode="weighted", applied=applied, total_time_s=total_time, offset_per_s=offset)


def provenance_columns(
    n_turns: int,
    *,
    di_dt_enabled: bool,
    di_dt_res: Optional[DiDtResult],
    integrate_to_flux_enabled: bool,
    drift_enabled: bool,
    drift_mode: Optional[DriftMode],
    drift_abs: Optional[DriftResult],
    drift_cmp: Optional[DriftResult],
) -> Dict[str, Any]:
    """Build per-turn provenance columns for exported tables.

    This is a *traceability* helper: it does not change physics results.
    It records which preprocessing options were enabled, which turns were
    actually affected, and key per-turn diagnostics.

    Parameters
    ----------
    n_turns:
        Number of turns in the output tables (after trimming and turn dropping).
    di_dt_enabled:
        Whether the user enabled di/dt correction.
    di_dt_res:
        Diagnostics returned by :func:`di_dt_weights` / :func:`apply_di_dt_to_channels`.
        May be None (e.g. if no time array is available).
    integrate_to_flux_enabled:
        Whether integration to flux was enabled.
    drift_enabled:
        Whether drift correction was enabled (only meaningful when integration is enabled).
    drift_mode:
        Selected drift mode ("legacy" or "weighted") if enabled.
    drift_abs, drift_cmp:
        Diagnostics returned by :func:`integrate_to_flux` for absolute and compensated channels.

    Returns
    -------
    Dict[str, Any]
        Mapping suitable for building pandas DataFrames.

    Notes
    -----
    - Arrays are always length ``n_turns``.
    - String columns are stored as dtype=object arrays.
    - When a feature is disabled or diagnostics are not available, values are
      filled with sensible defaults (False or NaN).
    """

    n = int(n_turns)
    if n < 0:
        raise ValueError("n_turns must be non-negative")

    cols: Dict[str, Any] = {}

    # --- di/dt ---
    cols["preproc_di_dt_enabled"] = np.full(n, bool(di_dt_enabled), dtype=bool)

    if di_dt_res is not None and di_dt_res.applied.shape[0] == n:
        cols["preproc_di_dt_applied"] = np.asarray(di_dt_res.applied, dtype=bool)
        cols["preproc_dI_dt_A_per_s"] = np.asarray(di_dt_res.dI_dt_A_per_s, dtype=float)
        cols["preproc_di_dt_I_mean_A"] = np.asarray(di_dt_res.I_mean_A, dtype=float)
        cols["preproc_di_dt_I_min_abs_A"] = np.asarray(di_dt_res.I_min_abs_A, dtype=float)
    else:
        cols["preproc_di_dt_applied"] = np.zeros(n, dtype=bool)
        cols["preproc_dI_dt_A_per_s"] = np.full(n, np.nan, dtype=float)
        cols["preproc_di_dt_I_mean_A"] = np.full(n, np.nan, dtype=float)
        cols["preproc_di_dt_I_min_abs_A"] = np.full(n, np.nan, dtype=float)

    # --- integration / drift ---
    cols["preproc_integrate_to_flux"] = np.full(n, bool(integrate_to_flux_enabled), dtype=bool)
    cols["preproc_drift_enabled"] = np.full(n, bool(drift_enabled), dtype=bool)

    mode_str = ""
    if drift_enabled and drift_mode is not None:
        mode_str = str(drift_mode)
    cols["preproc_drift_mode"] = np.array([mode_str] * n, dtype=object)

    def _drift_block(prefix: str, diag: Optional[DriftResult]) -> Dict[str, Any]:
        block: Dict[str, Any] = {}
        if diag is not None and diag.applied.shape[0] == n:
            block[f"{prefix}preproc_drift_applied"] = np.asarray(diag.applied, dtype=bool)

            if diag.mode == "weighted":
                tt = diag.total_time_s if diag.total_time_s is not None else None
                off = diag.offset_per_s if diag.offset_per_s is not None else None
                block[f"{prefix}preproc_drift_total_time_s"] = (
                    np.asarray(tt, dtype=float) if tt is not None else np.full(n, np.nan, dtype=float)
                )
                block[f"{prefix}preproc_drift_offset_per_s"] = (
                    np.asarray(off, dtype=float) if off is not None else np.full(n, np.nan, dtype=float)
                )
            else:
                block[f"{prefix}preproc_drift_total_time_s"] = np.full(n, np.nan, dtype=float)
                block[f"{prefix}preproc_drift_offset_per_s"] = np.full(n, np.nan, dtype=float)
        else:
            block[f"{prefix}preproc_drift_applied"] = np.zeros(n, dtype=bool)
            block[f"{prefix}preproc_drift_total_time_s"] = np.full(n, np.nan, dtype=float)
            block[f"{prefix}preproc_drift_offset_per_s"] = np.full(n, np.nan, dtype=float)
        return block

    cols.update(_drift_block("absolute_", drift_abs))
    cols.update(_drift_block("compensated_", drift_cmp))

    return cols


def format_preproc_tag(
    *,
    di_dt_enabled: bool,
    integrate_to_flux_enabled: bool,
    drift_enabled: bool,
    drift_mode: Optional[str],
    include_dc: bool,
    main_order: Optional[int] = None,
) -> str:
    """Format a short, file-name-safe preprocessing tag.

    This tag is intended for export filenames so that files remain
    self-identifying when separated from logs.

    The tag intentionally encodes the \"main field order\" $m$ used for the
    phase reference (rotation). Even before $k_n$ is implemented, this prevents
    ambiguity when a dataset is analyzed with different $m$ values.

    Examples
    --------
    - di/dt on, main order m=2, integrate to flux, drift weighted, DC excluded:
      ``didt_on_m02_flux_dri_weighted_dc_off``
    - di/dt off, main order m=3, no integration (raw incremental signal), DC excluded:
      ``didt_off_m03_df_dc_off``
    """

    parts: list[str] = []
    parts.append("didt_on" if di_dt_enabled else "didt_off")

    if main_order is not None:
        try:
            m = int(main_order)
            if m > 0:
                parts.append(f"m{m:02d}")
        except Exception:
            # Ignore invalid main_order values.
            pass

    if integrate_to_flux_enabled:
        parts.append("flux")
        if drift_enabled:
            dm = (str(drift_mode).strip().lower() if drift_mode is not None else "")
            if dm not in ("legacy", "weighted"):
                dm = "unknown"
            parts.append(f"dri_{dm}")
        else:
            parts.append("dri_off")
    else:
        # Raw incremental signal path (no integration).
        parts.append("df")

    parts.append("dc_on" if include_dc else "dc_off")

    # Make file-name-safe.
    tag = "_".join(parts)
    tag = "".join(ch if (ch.isalnum() or ch in "_-." ) else "_" for ch in tag)
    return tag


def append_tag_to_path(path: str, tag: str) -> str:
    """Append a preprocessing tag to a filesystem path (before extension).

    If the base name already ends with the tag, the path is returned unchanged.
    """

    if not path:
        return path

    tag = str(tag).strip()
    if not tag:
        return path

    root, ext = os.path.splitext(path)
    if root.endswith("_" + tag) or root.endswith(tag):
        return path
    return f"{root}_{tag}{ext}"
