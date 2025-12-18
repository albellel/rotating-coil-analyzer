from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class SegmentFrame:
    source_path: Path
    segment: str
    run_id: str

    df: pd.DataFrame  # columns: t, df_abs, df_cmp, I
    extras: Optional[pd.DataFrame] = None

    samples_per_turn: Optional[int] = None
    n_turns: Optional[int] = None

    warnings: List[str] = field(default_factory=list)
