from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SegmentSpec:
    aperture_id: int
    segment: str
    fdi_abs: int
    fdi_cmp: int
    length_m: Optional[float] = None


@dataclass(frozen=True)
class RunSpec:
    run_id: str


@dataclass(frozen=True)
class MeasurementCatalog:
    root_dir: Path
    parameters_path: Path
    samples_per_turn: int
    shaft_speed_rpm: float
    enabled_apertures: tuple[int, ...]
    segments: tuple[SegmentSpec, ...]
    runs: tuple[RunSpec, ...]
    # (run_id, segment) -> file path
    segment_files: dict[tuple[str, str], Path]
    warnings: tuple[str, ...] = ()
