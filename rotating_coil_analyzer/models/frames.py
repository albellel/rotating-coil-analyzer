from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SegmentFrame:
    source_path: Path
    run_id: str
    segment: str
    samples_per_turn: int
    n_turns: int
    df: pd.DataFrame  # columns: t, df_abs, df_cmp, I
    warnings: tuple[str, ...] = ()
