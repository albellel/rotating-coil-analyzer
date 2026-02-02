from __future__ import annotations

"""KnBundle and MergeResult: provenance-rich containers for kn calibration and merge workflow.

This module defines the data contracts between the Coil Calibration tab
and the Harmonic Merge tab.

Design goals
------------
- Full traceability: every exported result carries its provenance (source files,
  parameters, timestamps, compensation scheme, per-order merge choices).
- Immutable: all dataclasses are frozen to prevent accidental mutation.
- Serializable: metadata can be written to CSV headers or JSON sidecars.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from .kn_pipeline import SegmentKn


@dataclass(frozen=True)
class KnBundle:
    """Segment-level kn with full provenance metadata.

    This is the output of the Coil Calibration tab and the input
    to the Harmonic Merge tab.

    Attributes
    ----------
    kn : SegmentKn
        The actual calibration coefficients (kn_abs, kn_cmp, kn_ext).
    source_type : str
        Either "segment_txt" (loaded from file) or "head_csv" (computed from
        measurement-head geometry CSV).
    source_path : str
        Path to the source file (kn TXT or head CSV).
    timestamp : str
        ISO-8601 timestamp when this bundle was created.
    segment_id : str, optional
        Segment identifier (e.g., "Main", "A", "B") if known.
    aperture_id : int, optional
        Aperture number if applicable.
    head_abs_connection : str, optional
        For head_csv source: the absolute channel connection spec (e.g., "1.2").
    head_cmp_connection : str, optional
        For head_csv source: the compensated channel connection spec (e.g., "1.1-1.3").
    head_ext_connection : str, optional
        For head_csv source: the external channel connection spec (if any).
    head_warm_geometry : bool, optional
        For head_csv source: whether warm geometry scaling was used.
    head_n_multipoles : int, optional
        For head_csv source: number of multipoles computed.
    extra : dict, optional
        Additional metadata (user notes, etc.).
    """

    kn: SegmentKn
    source_type: str  # "segment_txt" | "head_csv"
    source_path: str
    timestamp: str

    # Context from Catalog tab
    segment_id: Optional[str] = None
    aperture_id: Optional[int] = None

    # Head-CSV specific provenance
    head_abs_connection: Optional[str] = None
    head_cmp_connection: Optional[str] = None
    head_ext_connection: Optional[str] = None
    head_warm_geometry: Optional[bool] = None
    head_n_multipoles: Optional[int] = None

    # Extensibility
    extra: Optional[Dict[str, Any]] = None

    @staticmethod
    def now_iso() -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export provenance as a flat dictionary (for CSV headers / JSON)."""
        d: Dict[str, Any] = {
            "kn_source_type": self.source_type,
            "kn_source_path": self.source_path,
            "kn_timestamp": self.timestamp,
            "kn_n_harmonics": len(self.kn.orders),
        }
        if self.segment_id is not None:
            d["kn_segment_id"] = self.segment_id
        if self.aperture_id is not None:
            d["kn_aperture_id"] = self.aperture_id
        if self.head_abs_connection is not None:
            d["kn_head_abs_connection"] = self.head_abs_connection
        if self.head_cmp_connection is not None:
            d["kn_head_cmp_connection"] = self.head_cmp_connection
        if self.head_ext_connection is not None:
            d["kn_head_ext_connection"] = self.head_ext_connection
        if self.head_warm_geometry is not None:
            d["kn_head_warm_geometry"] = self.head_warm_geometry
        if self.head_n_multipoles is not None:
            d["kn_head_n_multipoles"] = self.head_n_multipoles
        if self.extra:
            for k, v in self.extra.items():
                d[f"kn_extra_{k}"] = v
        return d


# Channel codes for per-order source map
CHANNEL_ABS = 0
CHANNEL_CMP = 1
CHANNEL_EXT = 2

CHANNEL_NAMES = {CHANNEL_ABS: "abs", CHANNEL_CMP: "cmp", CHANNEL_EXT: "ext"}


@dataclass(frozen=True)
class MergeResult:
    """Result of the harmonic merge workflow with full traceability.

    This is the output of the Harmonic Merge tab.

    Attributes
    ----------
    C_merged : np.ndarray
        Merged complex harmonics, shape (n_turns, H).
    orders : np.ndarray
        Harmonic orders [1, 2, ..., H], shape (H,).
    per_n_source_map : np.ndarray
        Per-order channel selection: 0=abs, 1=cmp, 2=ext. Shape (H,).
    compensation_scheme : str
        User-declared compensation scheme label (e.g., "A-C", "ABCD", "none").
        This is metadata for documentation; the actual compensation is already
        applied in the physical measurement.
    magnet_order : int
        Main field order m used for the merge.
    kn_provenance : KnBundle
        Full provenance of the kn calibration used upstream.
    merge_mode : str
        Merge mode that was applied (e.g., "abs_main_cmp_others", "custom").
    timestamp : str
        ISO-8601 timestamp when merge was applied.
    diagnostics : MergeDiagnostics, optional
        Noise/mismatch diagnostics if computed.
    C_abs : np.ndarray, optional
        Pre-merge absolute harmonics (for audit).
    C_cmp : np.ndarray, optional
        Pre-merge compensated harmonics (for audit).
    C_ext : np.ndarray, optional
        Pre-merge external harmonics if present.
    extra : dict, optional
        Additional metadata.
    """

    C_merged: np.ndarray
    orders: np.ndarray
    per_n_source_map: np.ndarray
    compensation_scheme: str
    magnet_order: int
    kn_provenance: KnBundle
    merge_mode: str
    timestamp: str

    # Optional audit data
    diagnostics: Optional[Any] = None  # MergeDiagnostics
    C_abs: Optional[np.ndarray] = None
    C_cmp: Optional[np.ndarray] = None
    C_ext: Optional[np.ndarray] = None

    extra: Optional[Dict[str, Any]] = None

    @property
    def n_turns(self) -> int:
        return int(self.C_merged.shape[0])

    @property
    def H(self) -> int:
        return int(self.orders.size)

    def source_map_as_names(self) -> list[str]:
        """Return per-order source as list of channel names."""
        return [CHANNEL_NAMES.get(int(c), f"unknown({c})") for c in self.per_n_source_map]

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export full provenance as flat dictionary."""
        d: Dict[str, Any] = {
            "merge_timestamp": self.timestamp,
            "merge_mode": self.merge_mode,
            "merge_compensation_scheme": self.compensation_scheme,
            "merge_magnet_order": self.magnet_order,
            "merge_n_turns": self.n_turns,
            "merge_n_harmonics": self.H,
        }
        # Include kn provenance
        d.update(self.kn_provenance.to_metadata_dict())
        # Per-order source map as comma-separated
        d["merge_per_n_source_map"] = ",".join(self.source_map_as_names())
        if self.extra:
            for k, v in self.extra.items():
                d[f"merge_extra_{k}"] = v
        return d

    def source_map_dataframe(self):
        """Return per-order source map as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            "n": self.orders,
            "source": self.source_map_as_names(),
            "source_code": self.per_n_source_map,
        })
