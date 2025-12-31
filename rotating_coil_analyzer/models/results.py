from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TurnQC:
    """Quality-control flags at turn granularity.

    This structure is intentionally minimal. It is meant to be extended during Phase II.

    Attributes
    ----------
    ok:
        Boolean mask of shape ``(n_turns,)``.
    reason:
        Optional string array of shape ``(n_turns,)`` describing why a turn was excluded.
    """

    ok: np.ndarray
    reason: Optional[np.ndarray] = None


@dataclass(frozen=True)
class HarmonicsResult:
    """Container for harmonics computed from a single SegmentFrame.

    Attributes
    ----------
    run_id, segment:
        Identifiers.
    Ns:
        Samples per turn.
    orders:
        Harmonic order vector.
    coeff_per_turn:
        Complex coefficients per turn, shape ``(n_turns, n_orders)``.
    plateau_id:
        Optional plateau id per turn (MBA), shape ``(n_turns,)``.
    qc:
        Optional per-turn QC mask.
    """

    run_id: str
    segment: str
    Ns: int

    orders: np.ndarray
    coeff_per_turn: np.ndarray

    plateau_id: Optional[np.ndarray] = None
    qc: Optional[TurnQC] = None
