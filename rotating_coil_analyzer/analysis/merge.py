from __future__ import annotations

"""Abs/Cmp merge recommendation (diagnostic, user-approvable).

Your project requirement is:

*Never merge blindly.*

Accordingly, this module provides a *recommendation* engine that computes
diagnostics per harmonic order and returns a suggested per-order channel
selection. The caller (GUI / CLI) must present this to the user and only apply
the merge after explicit approval.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _mad(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Median absolute deviation (scaled to sigma for Gaussian, robust)."""
    x = np.asarray(x)
    med = np.nanmedian(x, axis=axis)
    dev = np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)
    return 1.4826 * dev


@dataclass(frozen=True)
class MergeDiagnostics:
    """Per-order diagnostics to support merge approval."""

    orders: np.ndarray  # (H,)
    noise_abs: np.ndarray  # (H,) robust sigma estimate
    noise_cmp: np.ndarray  # (H,) robust sigma estimate
    mismatch: np.ndarray  # (H,) median |abs-cmp|
    selected: np.ndarray  # (H,) int: 0 abs, 1 cmp
    flags: np.ndarray  # (H,) bitmask of diagnostic flags


# Flag bits
FLAG_MAIN_FORCED_ABS = 1 << 0
FLAG_BAD_CHANNEL = 1 << 1
FLAG_MISMATCH_LARGE = 1 << 2


def recommend_merge_choice(
    *,
    C_abs: np.ndarray,
    C_cmp: np.ndarray,
    magnet_order: int,
    orders: Optional[Sequence[int]] = None,
    prefer_cmp_if_better: float = 0.90,
    mismatch_tol_rel: float = 50.0,
) -> Tuple[np.ndarray, MergeDiagnostics]:
    """Recommend a per-order Abs/Cmp merge choice.

    Parameters
    ----------
    C_abs, C_cmp:
        Complex arrays of shape (n_turns, H). These should be *post-$k_n$* and
        typically *post-rotation*; whether you feed them through CEL/feeddown
        and/or normalization depends on what you want the merge to optimize.
        For "noise"-driven merging, using the *same stage you export* is
        usually the right choice.
    magnet_order:
        Main field order $m$. The recommendation **forces** order $m$ to Abs
        by default (project decision).
    orders:
        Optional harmonic order array of length H. If omitted, uses 1..H.
    prefer_cmp_if_better:
        Choose Cmp for a given order if its robust noise estimate is smaller
        than Abs by this multiplicative factor. Example: 0.90 means "Cmp must
        be at least 10% quieter".
    mismatch_tol_rel:
        If the Abs/Cmp median mismatch for an order exceeds
        ``mismatch_tol_rel * min(noise_abs, noise_cmp)``, the recommendation
        falls back to Abs and sets a mismatch flag.

    Returns
    -------
    choice, diagnostics
        choice is an int array of length H with 0 (abs) or 1 (cmp).
    """

    A = np.asarray(C_abs, dtype=complex)
    B = np.asarray(C_cmp, dtype=complex)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
        raise ValueError("C_abs and C_cmp must be the same 2D shape (n_turns, H)")

    n_turns, H = A.shape
    m = int(magnet_order)
    if not (1 <= m <= H):
        raise ValueError(f"magnet_order m must be in [1,{H}], got {magnet_order}")

    ords = np.arange(1, H + 1, dtype=int) if orders is None else np.asarray(list(orders), dtype=int)
    if ords.shape != (H,):
        raise ValueError(f"orders must have shape ({H},), got {ords.shape}")

    # Robust "noise" proxy: combine MAD of real and imag components.
    sig_abs = np.hypot(_mad(np.real(A), axis=0), _mad(np.imag(A), axis=0))
    sig_cmp = np.hypot(_mad(np.real(B), axis=0), _mad(np.imag(B), axis=0))

    mismatch = np.nanmedian(np.abs(A - B), axis=0)

    choice = np.zeros(H, dtype=int)
    flags = np.zeros(H, dtype=int)

    # Default: abs_main_cmp_others is a good legacy baseline.
    choice[:] = 1
    choice[m - 1] = 0
    flags[m - 1] |= FLAG_MAIN_FORCED_ABS

    # Decide each non-main order by comparing robust noise.
    for j in range(H):
        if j == (m - 1):
            continue

        a_ok = np.isfinite(sig_abs[j])
        b_ok = np.isfinite(sig_cmp[j])

        if not a_ok and not b_ok:
            choice[j] = 0
            flags[j] |= FLAG_BAD_CHANNEL
            continue
        if not b_ok:
            choice[j] = 0
            flags[j] |= FLAG_BAD_CHANNEL
            continue
        if not a_ok:
            choice[j] = 1
            flags[j] |= FLAG_BAD_CHANNEL
            continue

        # Prefer Cmp only if significantly quieter.
        if sig_cmp[j] < float(prefer_cmp_if_better) * sig_abs[j]:
            choice[j] = 1
        else:
            choice[j] = 0

        # If the two channels disagree far beyond what their noise suggests,
        # fall back to Abs and flag.
        denom = float(min(sig_abs[j], sig_cmp[j]))
        if denom > 0 and np.isfinite(denom):
            if mismatch[j] > float(mismatch_tol_rel) * denom:
                choice[j] = 0
                flags[j] |= FLAG_MISMATCH_LARGE

    diag = MergeDiagnostics(
        orders=ords,
        noise_abs=sig_abs,
        noise_cmp=sig_cmp,
        mismatch=mismatch,
        selected=choice,
        flags=flags,
    )

    return choice, diag
