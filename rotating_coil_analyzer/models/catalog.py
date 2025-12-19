from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SegmentSpec:
    """
    Segment definition from Parameters.txt (FDI mapping), scoped by aperture.

    aperture_id: physical aperture number as used in filenames (e.g. 1, 2).
    segment_id: label used in filenames after 'Seg' (e.g. 'NCS', 'CS', '1', '2', ...)
    fdi_abs: index of the absolute FDI (0-based) from Parameters
    fdi_cmp: index of the compensated FDI (0-based) from Parameters
    length_m: optional segment length (if present in Parameters), else None
    """
    aperture_id: int
    segment_id: str
    fdi_abs: int
    fdi_cmp: int
    length_m: Optional[float] = None


@dataclass(frozen=True)
class MeasurementCatalog:
    """
    Phase-1 output: a filesystem-independent catalog of what exists in a measurement folder.

    Notes
    - segment_files is keyed by (run_id, aperture_id, segment_id) to avoid AP1/AP2 collisions.
    - When only one aperture is enabled, the GUI can treat aperture as an implementation detail,
      but the catalog keeps the physical aperture id because filenames do.
    """
    root_dir: Path
    parameters_path: Path
    parameters_root: Path

    samples_per_turn: int
    shaft_speed_rpm: float

    enabled_apertures: List[int]
    segments: List[SegmentSpec]
    runs: List[str]

    segment_files: Dict[Tuple[str, int, str], Path]
    warnings: Tuple[str, ...] = ()

    @property
    def logical_apertures(self) -> List[Optional[int]]:
        """
        If only one aperture is enabled, return [None] for a simpler UI identity.
        Otherwise return the physical aperture ids.
        """
        if len(self.enabled_apertures) <= 1:
            return [None]
        return list(self.enabled_apertures)

    def resolve_aperture(self, aperture_id: Optional[int]) -> int:
        """Map Optional aperture id (UI identity) to a physical aperture id."""
        if aperture_id is None:
            if not self.enabled_apertures:
                raise ValueError("No enabled apertures in catalog.")
            return int(self.enabled_apertures[0])
        return int(aperture_id)

    def segments_for_aperture(self, aperture_id: Optional[int]) -> List[SegmentSpec]:
        ap = self.resolve_aperture(aperture_id)
        return [s for s in self.segments if s.aperture_id == ap]

    def get_segment_file(self, run_id: str, aperture_id: Optional[int], segment_id: str) -> Path:
        ap = self.resolve_aperture(aperture_id)
        key = (run_id, ap, segment_id)
        if key not in self.segment_files:
            raise FileNotFoundError(f"No segment file for run='{run_id}' ap={ap} segment='{segment_id}'.")
        return self.segment_files[key]
