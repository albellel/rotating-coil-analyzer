"""Analysis profile -- bundles all pipeline-relevant configuration.

An AnalysisProfile groups every parameter that affects the analysis output
into one frozen dataclass.  It can be:

- Constructed from a MeasurementCatalog (auto-populate from Parameters.txt)
- Overridden field-by-field via ``dataclasses.replace()``
- Serialized to/from a dict for JSON provenance
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AnalysisProfile:
    """Frozen configuration for the full analysis pipeline.

    Required fields
    ---------------
    magnet_order : int
        Main field order m (1 = dipole, 2 = quadrupole, ...).
    r_ref_m : float
        Reference radius in metres.
    samples_per_turn : int
        Number of samples (encoder counts) per revolution.
    shaft_speed_rpm : float
        Rotation speed in RPM (absolute value).

    Optional fields (sensible defaults)
    ------------------------------------
    options : tuple of str
        Pipeline step tokens, subset of {"dit","dri","rot","cel","fed","nor"}.
    drift_mode : str
        "legacy" or "weighted".
    merge_mode : str
        Merge strategy for abs/cmp channels.
    legacy_rotate_excludes_last : bool
        If True, the rotation loop excludes the last harmonic (SM18 convention).
        If False, all harmonics are rotated (BTP8 / C++ convention).
    min_main_field_T : float
        Epsilon for safe division by the main field component.
    abs_calib : float
        Absolute calibration factor (legacy ``absCalib``).
    l_coil_m : float or None
        Coil length in metres (used for provenance/export, not pipeline math).
    skew_main : bool
        If True, use Im(main_field) for normalization instead of Re.
    """

    magnet_order: int
    r_ref_m: float
    samples_per_turn: int
    shaft_speed_rpm: float

    options: Tuple[str, ...] = ("dri", "rot")
    drift_mode: str = "legacy"
    merge_mode: str = "abs_upto_m_cmp_above"
    legacy_rotate_excludes_last: bool = True

    min_main_field_T: float = 1e-20
    abs_calib: float = 1.0
    l_coil_m: Optional[float] = None
    skew_main: bool = False
    max_zR: float = 1.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_catalog(
        cls,
        catalog: "MeasurementCatalog",
        **overrides: Any,
    ) -> AnalysisProfile:
        """Build a profile from a :class:`MeasurementCatalog` with optional overrides.

        Values populated from the catalog (ultimately from Parameters.txt):

        - ``magnet_order`` (falls back to 1 if catalog has ``None``)
        - ``samples_per_turn``
        - ``shaft_speed_rpm``

        Everything else uses class defaults unless overridden via keyword
        arguments.  Example::

            profile = AnalysisProfile.from_catalog(cat, r_ref_m=0.059,
                                                    options=("dri","rot","cel","fed"))
        """
        # Avoid circular import at module level
        from rotating_coil_analyzer.models.catalog import MeasurementCatalog  # noqa: F811

        base: Dict[str, Any] = dict(
            magnet_order=catalog.magnet_order or 1,
            samples_per_turn=catalog.samples_per_turn,
            shaft_speed_rpm=catalog.shaft_speed_rpm,
        )
        # r_ref_m has no catalog source -- require override or use a safe fallback
        if "r_ref_m" not in overrides:
            base["r_ref_m"] = 0.017  # 17 mm -- deliberately conservative
        base.update(overrides)
        # Ensure options is a tuple (callers may pass a list)
        if "options" in base and not isinstance(base["options"], tuple):
            base["options"] = tuple(base["options"])
        return cls(**base)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict (tuples become lists)."""
        d = asdict(self)
        d["options"] = list(d["options"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnalysisProfile:
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        d = dict(d)  # shallow copy
        if "options" in d and not isinstance(d["options"], tuple):
            d["options"] = tuple(d["options"])
        return cls(**d)
