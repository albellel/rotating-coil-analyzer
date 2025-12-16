"""
Phase 1: validation utilities.

This module validates the standardized "raw measurement table" used by the analyzer.
It does not compute multipoles; it only verifies that the ingestion stage produced
clean, consistent arrays for later phases.

Canonical columns (required):
- time_s   : FDI time base [s] (typically from the ABS FDI time channel)
- dphi_abs : ABS FDI delta-flux samples [V·s] or equivalent integrator output
- dphi_cmp : CMP FDI delta-flux samples [V·s] or equivalent integrator output
- current_a: current signal already converted to amperes [A]

All additional columns are treated as "extras" and preserved.

Examples
--------
>>> import pandas as pd
>>> from rotating_coil_analyzer.scripts.validate_plateau import validate_core_table
>>> df = pd.DataFrame({"time_s":[0,1,2], "dphi_abs":[1,1,1], "dphi_cmp":[0.1,0.1,0.1], "current_a":[10,10,10]})
>>> rep = validate_core_table(df)
>>> rep.ok
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# Canonical required columns for Phase 1 and beyond.
CORE_COLS: Tuple[str, str, str, str] = ("time_s", "dphi_abs", "dphi_cmp", "current_a")


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of validating one table.

    Attributes
    ----------
    ok:
        True if no errors were found.
    errors:
        Fatal issues; caller should stop the pipeline.
    warnings:
        Non-fatal issues; caller may continue but should review.
    dropped_nonfinite_rows:
        If cleaning is enabled, how many rows were removed because of NaN/Inf in required columns.

    Examples
    --------
    >>> ValidationResult(ok=True, errors=[], warnings=[]).ok
    True
    """
    ok: bool
    errors: List[str]
    warnings: List[str]
    dropped_nonfinite_rows: int = 0

    def raise_if_errors(self) -> None:
        """
        Raise ValueError if errors exist.

        Examples
        --------
        >>> r = ValidationResult(ok=False, errors=["bad"], warnings=[])
        >>> try:
        ...     r.raise_if_errors()
        ... except ValueError:
        ...     pass
        """
        if self.errors:
            msg = "Validation failed:\n" + "\n".join(f"- {e}" for e in self.errors)
            raise ValueError(msg)


def default_column_names(ncols: int, extra_names: Sequence[str] | None = None) -> List[str]:
    """
    Build column names for a headerless numeric table.

    Parameters
    ----------
    ncols:
        Total number of columns detected.
    extra_names:
        Optional names for columns 5..n. If provided, length must be (ncols - 4).

    Returns
    -------
    list[str]
        Names using the canonical first 4 columns plus extras.

    Examples
    --------
    >>> default_column_names(4)
    ['time_s', 'dphi_abs', 'dphi_cmp', 'current_a']
    >>> default_column_names(6)
    ['time_s', 'dphi_abs', 'dphi_cmp', 'current_a', 'extra_1', 'extra_2']
    """
    if ncols < 4:
        raise ValueError(f"Expected >= 4 columns, got {ncols}.")

    names = list(CORE_COLS)
    n_extra = ncols - 4

    if n_extra == 0:
        return names

    if extra_names is None:
        names += [f"extra_{i+1}" for i in range(n_extra)]
        return names

    extra_names = list(extra_names)
    if len(extra_names) != n_extra:
        raise ValueError(f"extra_names length={len(extra_names)} but file has {n_extra} extra columns.")
    names += extra_names
    return names


def drop_nonfinite_rows(df: pd.DataFrame, cols: Iterable[str]) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where any of `cols` contains NaN/Inf.

    This is useful for some binary exports (or partially written files) that may contain
    padding rows. You indicated DAQ current is continuous (no NaNs), but FDI channels
    may still be affected by padding/invalid samples, so we clean based on required columns.

    Parameters
    ----------
    df:
        Input dataframe.
    cols:
        Columns that must be finite.

    Returns
    -------
    (clean_df, dropped_count)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> df = pd.DataFrame({"time_s":[0,1], "dphi_abs":[1,np.nan], "dphi_cmp":[0,0], "current_a":[10,10]})
    >>> clean, n = drop_nonfinite_rows(df, ["time_s","dphi_abs","dphi_cmp","current_a"])
    >>> (len(clean), n)
    (1, 1)
    """
    cols = list(cols)
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(df[c].to_numpy())
    dropped = int((~mask).sum())
    return df.loc[mask].reset_index(drop=True), dropped


def validate_time_monotonic(
    t: np.ndarray,
    *,
    allow_nonmonotonic: bool = False,
    max_backstep_s: float = 1e-3,
) -> tuple[list[str], list[str]]:
    """
    Validate that time is monotonic.

    Parameters
    ----------
    t:
        Time array.
    allow_nonmonotonic:
        If True, monotonicity violations become warnings; else errors.
    max_backstep_s:
        Tolerance for tiny negative dt (numeric noise).

    Returns
    -------
    (errors, warnings)

    Examples
    --------
    >>> import numpy as np
    >>> e,w = validate_time_monotonic(np.array([0.0, 1.0, 0.5]), allow_nonmonotonic=True)
    >>> len(e), len(w)
    (0, 1)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if t.size < 2:
        return errors, warnings

    dt = np.diff(t)
    back = np.where(dt < -max_backstep_s)[0]
    if back.size:
        msg = f"time_s is not monotonic (first backstep at rows {int(back[0])}->{int(back[0])+1})."
        if allow_nonmonotonic:
            warnings.append(msg)
        else:
            errors.append(msg)

    return errors, warnings


def detect_time_gaps(t: np.ndarray, *, gap_threshold_s: float = 2.0) -> list[int]:
    """
    Identify large time gaps. Later phases can use this to reset turn buffers.

    Parameters
    ----------
    t:
        Time array.
    gap_threshold_s:
        dt larger than this is considered a gap.

    Returns
    -------
    list[int]
        Indices of the first sample after each gap.

    Examples
    --------
    >>> import numpy as np
    >>> detect_time_gaps(np.array([0,1,2,10,11]), gap_threshold_s=3.0)
    [3]
    """
    if t.size < 2:
        return []
    gaps = np.where(np.diff(t) > gap_threshold_s)[0]
    return (gaps + 1).astype(int).tolist()


def validate_core_table(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str] = CORE_COLS,
    allow_time_nonmonotonic: bool = True,
    max_time_backstep_s: float = 1e-3,
) -> ValidationResult:
    """
    Validate a standardized raw measurement table.

    Parameters
    ----------
    df:
        DataFrame with canonical columns.
    required_cols:
        Required columns. Defaults to CORE_COLS.
    allow_time_nonmonotonic:
        If True, time issues are warnings; else errors.
    max_time_backstep_s:
        Allowed numeric tolerance for negative dt.

    Returns
    -------
    ValidationResult

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"time_s":[0,1], "dphi_abs":[1,1], "dphi_cmp":[0,0], "current_a":[10,10]})
    >>> validate_core_table(df).ok
    True
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}. Present={list(df.columns)}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # Verify numeric + finite for required columns.
    for c in required_cols:
        arr = df[c].to_numpy()
        if not np.issubdtype(arr.dtype, np.number):
            errors.append(f"Column '{c}' must be numeric; dtype={df[c].dtype}.")
            continue
        if not np.isfinite(arr).all():
            bad = np.where(~np.isfinite(arr))[0][:10].tolist()
            errors.append(f"Column '{c}' has non-finite values at rows {bad} (showing up to 10).")

    # Time monotonicity check (usually non-fatal at Phase 1).
    if not errors:
        t = df["time_s"].to_numpy()
        e2, w2 = validate_time_monotonic(
            t, allow_nonmonotonic=allow_time_nonmonotonic, max_backstep_s=max_time_backstep_s
        )
        errors.extend(e2)
        warnings.extend(w2)

    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)
