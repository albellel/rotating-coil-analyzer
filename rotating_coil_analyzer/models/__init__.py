"""Models package - core data structures.

This package defines the core data models used throughout the analyzer:
- MeasurementCatalog: Discovered measurement folder structure
- SegmentFrame: Loaded segment data with metadata
- KnBundle: Calibration coefficients with provenance
- MergeResult: Merged harmonics with full traceability

Design principle:
- Models are immutable data containers (dataclasses with frozen=True where possible)
- All models carry provenance metadata for traceability
- No business logic in models - they are pure data holders
"""

from .profile import AnalysisProfile

__all__ = [
    "AnalysisProfile",
]
