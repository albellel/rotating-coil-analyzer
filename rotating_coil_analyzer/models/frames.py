from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class SegmentFrame:
    """
    Container for one segment's time-series as read from disk.

    IMPORTANT: We never "invent" a new time axis. If 't' exists, it comes from FDI
    (either embedded in corr_sigs or read from a separate raw_time file).
    """
    source_path: Path
    run_id: str
    segment: str
    samples_per_turn: int
    n_turns: int
    df: pd.DataFrame

    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def require_columns(self, *cols: str) -> None:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Present: {list(self.df.columns)}")

    @property
    def has_time(self) -> bool:
        return "t" in self.df.columns

    @property
    def n_rows(self) -> int:
        return int(len(self.df))
