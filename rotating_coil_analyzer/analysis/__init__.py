"""Phase II analysis package.

Design principle:
  - Ingest (Phase I) produces validated :class:`~rotating_coil_analyzer.models.frames.SegmentFrame` objects.
  - Analysis (Phase II) consumes SegmentFrame and produces derived quantities.

Project-wide hard constraint:
  - No synthetic/modified time is allowed anywhere.

Accordingly, analysis functions are expressed on the *sample/turn index* and the
implicit angular grid per turn, not on any reconstructed time axis.
"""

from .turns import TurnBlock, split_into_turns
from .fourier import dft_per_turn, summarize_harmonics

__all__ = [
    "TurnBlock",
    "split_into_turns",
    "dft_per_turn",
    "summarize_harmonics",
]
