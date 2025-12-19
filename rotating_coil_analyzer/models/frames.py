from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class SegmentFrame:
    """
    In-memory representation of one segment file (one run_id + one aperture + one segment)
    after basic cleaning.

    Notes
    - 't' is always the time vector coming from the FDI (no synthetic time generation).
    - df columns are always float64.
    """
    source_path: Path
    run_id: str
    segment: str
    samples_per_turn: int
    n_turns: int
    df: pd.DataFrame
    warnings: Tuple[str, ...] = ()
    aperture_id: Optional[int] = None

    @property
    def n_samples(self) -> int:
        return int(len(self.df))
