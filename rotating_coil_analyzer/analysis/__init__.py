"""Analysis package for rotating coil data.

Design principle:
  - Ingest (Catalog) produces validated :class:`~rotating_coil_analyzer.models.frames.SegmentFrame` objects.
  - Analysis (Harmonics, Coil Calibration, Merge) consumes SegmentFrame and produces derived quantities.

Project-wide hard constraint:
  - No synthetic/modified time is allowed anywhere.

Accordingly, analysis functions are expressed on the *sample/turn index* and the
implicit angular grid per turn, not on any reconstructed time axis.

Legacy parity notes
-------------------
The legacy analyzers apply some optional preprocessing steps *before* FFT:

- ``dit`` / ``di/dt`` current-ramp correction (uses measured time + current)
- Drift correction (legacy uniform-Δt or Bottura/Pentella Δt-weighted)

These are implemented in :mod:`rotating_coil_analyzer.analysis.preprocess`.
"""

from .turns import TurnBlock, split_into_turns
from .fourier import dft_per_turn, summarize_harmonics
from .preprocess import (
    DriftMode,
    DriftResult,
    DiDtResult,
    estimate_linear_slope_per_turn,
    di_dt_weights,
    apply_di_dt_to_channels,
    integrate_to_flux,
    format_preproc_tag,
    append_tag_to_path,
    provenance_columns,
)

from .kn_pipeline import SegmentKn, LegacyKnPerTurn, load_segment_kn_txt, compute_legacy_kn_per_turn, merge_coefficients
from .merge import MergeDiagnostics, recommend_merge_choice
from .kn_head import HeadKnData, compute_head_kn_from_csv, compute_segment_kn_from_head, write_segment_kn_txt
from .kn_bundle import KnBundle, MergeResult, CHANNEL_ABS, CHANNEL_CMP, CHANNEL_EXT, CHANNEL_NAMES

__all__ = [
    "TurnBlock",
    "split_into_turns",
    "dft_per_turn",
    "summarize_harmonics",
    "DriftMode",
    "DriftResult",
    "DiDtResult",
    "estimate_linear_slope_per_turn",
    "di_dt_weights",
    "apply_di_dt_to_channels",
    "integrate_to_flux",
    "format_preproc_tag",
    "append_tag_to_path",
    "provenance_columns",

    # kn phase
    "SegmentKn",
    "LegacyKnPerTurn",
    "load_segment_kn_txt",
    "compute_legacy_kn_per_turn",
    "merge_coefficients",
    "MergeDiagnostics",
    "recommend_merge_choice",

    # kn head-csv computation
    "HeadKnData",
    "compute_head_kn_from_csv",
    "compute_segment_kn_from_head",
    "write_segment_kn_txt",

    # kn bundle and merge result (provenance-rich containers)
    "KnBundle",
    "MergeResult",
    "CHANNEL_ABS",
    "CHANNEL_CMP",
    "CHANNEL_EXT",
    "CHANNEL_NAMES",
]
