from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


Mode = Literal["abs", "cmp"]


@dataclass(frozen=True)
class SegmentSpec:
    segment: str
    fdi_abs: int
    fdi_cmp: int
    length_m: float
    aperture_id: Optional[int] = None  # None for single-aperture logical identity


@dataclass(frozen=True)
class ConnectionModel:
    comp_scheme_label: Optional[str] = None
    # Example: {(2,3): +1.0, (2,4): -1.0} meaning +2.3 -2.4
    fdi_connections: Dict[Tuple[int, int], float] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelSpec:
    segment: str
    mode: Mode
    fdi_id: int
    length_m: float
    aperture_id: Optional[int] = None
    aperture_token: Optional[int] = None  # preserve Ap_1 in filename even if aperture_id=None
    connection: ConnectionModel = field(default_factory=ConnectionModel)


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    magnet_name: Optional[str]
    date_yyyymmdd: Optional[str]
    time_hhmmss: Optional[str]
    cycle_name: Optional[str]
    aperture_token: Optional[int] = None


@dataclass
class MeasurementCatalog:
    root_dir: Path
    parameters_path: Path
    parameters: Dict[str, Any]

    samples_per_turn: int
    shaft_speed_rpm: Optional[float]

    enabled_apertures: List[int]
    segments: List[SegmentSpec]
    channels: List[ChannelSpec]
    runs: List[RunDescriptor]

    # (run_id, segment) -> file path
    segment_files: Dict[Tuple[str, str], Path] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)
