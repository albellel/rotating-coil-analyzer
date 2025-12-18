from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SegmentSpec:
    """
    Segment definition from Parameters.txt (FDI mapping).

    segment_id: label used in filenames after 'Seg' (e.g. 'NCS', 'CS', '1', '2', ...)
    fdi_abs: index of the absolute FDI (0-based)
    fdi_cmp: index of the compensated FDI (0-based)
    length_m: optional segment length (if present in Parameters file), else None
    """
    segment_id: str
    fdi_abs: int
    fdi_cmp: int
    length_m: Optional[float] = None


@dataclass(frozen=True)
class MeasurementCatalog:
    """
    Catalog for one measurement folder (typically .../aperture1).

    segment_files maps (run_id, segment_id) -> Path to the corresponding *_corr_sigs_*.bin
    """
    root_dir: Path
    parameters_path: Path

    samples_per_turn: int
    shaft_speed_rpm: float

    enabled_apertures: List[int]
    segments: List[SegmentSpec]
    runs: List[str]

    segment_files: Dict[Tuple[str, str], Path]
    warnings: Tuple[str, ...] = ()

    def get_segment_file(self, run_id: str, segment_id: str) -> Path:
        key = (run_id, segment_id)
        if key not in self.segment_files:
            raise FileNotFoundError(f"No segment file for run='{run_id}' segment='{segment_id}'.")
        return self.segment_files[key]
