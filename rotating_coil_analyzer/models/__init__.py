"""Models package - core data structures.

This package defines the core data models used throughout the analyzer:
- MeasurementCatalog (``models.catalog``): Discovered measurement folder structure
- SegmentFrame (``models.frames``): Loaded segment data with metadata
- AnalysisProfile (``models.profile``): Frozen pipeline configuration
- TurnQC / HarmonicsResult (``models.results``): Per-turn QC and harmonics containers

Note: the provenance-rich calibration/merge containers ``KnBundle`` and
``MergeResult`` live in :mod:`rotating_coil_analyzer.analysis.kn_bundle`
(not here), because they belong to the analysis/merge workflow.

Design principle:
- Models are immutable data containers (dataclasses with frozen=True where possible)
- All models carry provenance metadata for traceability
- No business logic in models - they are pure data holders
"""

from .profile import AnalysisProfile

__all__ = [
    "AnalysisProfile",
]
