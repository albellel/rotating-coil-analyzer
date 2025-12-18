from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SegmentDef:
    name: str
    fdi_abs: int
    fdi_cmp: int
    length_m: Optional[float] = None


@dataclass(frozen=True)
class RunCatalog:
    run_id: str
    root_dir: Path
    parameters_path: Path

    samples_per_turn: int
    shaft_speed_rpm: float

    # Segment definitions coming from Parameters FDIs table
    segments: List[SegmentDef] = field(default_factory=list)

    # Actual segment files found on disk: segment_name -> file path
    corr_sigs_files: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementCatalog:
    root_dir: Path
    runs: Dict[str, RunCatalog] = field(default_factory=dict)
