"""FFT-based harmonic extraction for rotating-coil measurements.

Provides per-turn discrete Fourier transform with standard ``FFT/Ns``
normalisation and optional plateau-grouped statistical summaries.

Functions
---------
dft_per_turn
    Compute complex Fourier coefficients per turn on the implicit angular grid.
summarize_harmonics
    Mean and standard deviation of harmonics, optionally grouped by plateau id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class HarmonicsPerTurn:
    """Fourier coefficients per turn.

    Attributes
    ----------
    orders:
        Harmonic order vector ``[0,1,...,n_max]``.
    coeff:
        Complex Fourier coefficients of shape ``(n_turns, n_orders)``.
        Normalization: ``coeff = FFT(signal)/Ns``.
    """

    orders: np.ndarray
    coeff: np.ndarray


@dataclass(frozen=True)
class HarmonicsSummary:
    """Mean and standard deviation of harmonics.

    If ``by_plateau`` is False: arrays are ``(n_orders,)``.
    If ``by_plateau`` is True: arrays are ``(n_plateaus, n_orders)`` and the
    first axis corresponds to sorted unique plateau ids.
    """

    orders: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    by_plateau: bool
    plateau_ids: Optional[np.ndarray] = None


def dft_per_turn(signal_turns: np.ndarray, *, n_max: Optional[int] = None) -> HarmonicsPerTurn:
    r"""Compute discrete Fourier coefficients per turn.

    Parameters
    ----------
    signal_turns:
        Array of shape ``(n_turns, Ns)`` containing samples over angle within each turn.
        The function treats the second axis as the sample index within the turn.
    n_max:
        Maximum harmonic order to keep (inclusive). Default keeps all orders ``0..Ns-1``.

    Returns
    -------
    HarmonicsPerTurn
        Complex coefficients with normalization ``FFT/Ns``.

    Notes
    -----
    This computation does not depend on time. It is defined on the implicit angular grid
    \(\theta_k = 2\pi k/N_s\) associated with the sample index.
    """
    x = np.asarray(signal_turns)
    if x.ndim != 2:
        raise ValueError(f"signal_turns must be 2D (n_turns, Ns), got shape {x.shape}")

    n_turns, Ns = x.shape
    if Ns <= 0:
        raise ValueError("Ns must be > 0")

    fft = np.fft.fft(x, axis=1) / float(Ns)

    if n_max is None:
        n_max = Ns - 1
    n_max = int(n_max)
    if not (0 <= n_max <= Ns - 1):
        raise ValueError(f"n_max must be in [0, {Ns-1}], got {n_max}")

    orders = np.arange(n_max + 1, dtype=int)
    coeff = fft[:, : n_max + 1]

    return HarmonicsPerTurn(orders=orders, coeff=coeff)


def summarize_harmonics(
    harm: HarmonicsPerTurn,
    *,
    plateau_id_per_turn: Optional[np.ndarray] = None,
    include_dc: bool = True,
) -> Dict[str, HarmonicsSummary]:
    """Compute mean/std of harmonics, optionally grouped by plateau id.

    Parameters
    ----------
    harm:
        Per-turn coefficients.
    plateau_id_per_turn:
        Optional array of shape ``(n_turns,)``. If provided, this function returns both:
        - overall mean/std across all turns
        - per-plateau mean/std
    include_dc:
        Whether to include the DC (order 0) component in the summaries. Legacy metrology
        workflows typically treat order 0 as diagnostic only, and focus on orders >= 1.

    Returns
    -------
    dict
        Keys: ``"all"`` and optionally ``"by_plateau"``.
    """
    coeff = np.asarray(harm.coeff)
    if coeff.ndim != 2:
        raise ValueError("harm.coeff must be 2D")

    orders = np.asarray(harm.orders, dtype=int)
    if orders.ndim != 1 or orders.size != coeff.shape[1]:
        raise ValueError("harm.orders must be 1D and match harm.coeff second axis")

    if include_dc:
        sel = np.ones_like(orders, dtype=bool)
    else:
        sel = orders != 0

    out: Dict[str, HarmonicsSummary] = {}

    coeff_sel = coeff[:, sel]
    orders_sel = orders[sel]

    mean_all = np.nanmean(coeff_sel, axis=0)
    std_all = np.nanstd(coeff_sel, axis=0)
    out["all"] = HarmonicsSummary(
        orders=orders_sel,
        mean=mean_all,
        std=std_all,
        by_plateau=False,
        plateau_ids=None,
    )

    if plateau_id_per_turn is not None:
        pid = np.asarray(plateau_id_per_turn)
        if pid.ndim != 1 or pid.size != coeff.shape[0]:
            raise ValueError(
                f"plateau_id_per_turn must be 1D of length n_turns={coeff.shape[0]}, got shape {pid.shape}"
            )

        uniq = np.unique(pid[np.isfinite(pid)])
        uniq = np.sort(uniq)

        mean_p = np.full((len(uniq), coeff_sel.shape[1]), np.nan, dtype=complex)
        std_p = np.full((len(uniq), coeff_sel.shape[1]), np.nan, dtype=complex)

        for i, u in enumerate(uniq):
            mask = pid == u
            if not np.any(mask):
                continue
            mean_p[i, :] = np.nanmean(coeff_sel[mask, :], axis=0)
            std_p[i, :] = np.nanstd(coeff_sel[mask, :], axis=0)

        out["by_plateau"] = HarmonicsSummary(
            orders=orders_sel,
            mean=mean_p,
            std=std_p,
            by_plateau=True,
            plateau_ids=uniq,
        )

    return out
