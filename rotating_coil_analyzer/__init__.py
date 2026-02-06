"""Rotating Coil Analyzer -- Python tooling for CERN accelerator-magnet rotating-coil measurements.

Designed for all CERN machine complexes (LHC, SPS, PS, PSB, transfer lines,
test benches such as SM18).

This package provides tools for:
- Ingesting rotating-coil acquisition files (streaming binary and plateau text formats)
- Splitting measurements into turns using strict time policies
- Computing per-turn Fourier harmonics
- Applying kn calibration coefficients
- Merging absolute/compensated harmonics with full provenance tracking
- Detecting current plateaus in streaming supercycle measurements
- Building per-turn results DataFrames for analysis and export

Key principles:
- No synthetic time: time must come from acquisition columns
- No interpolation: downsampling uses decimation only
- Full traceability: all operations are logged and auditable

Main subpackages:
- analysis: Harmonic computation, kn pipeline, merge logic, streaming utilities
- gui: Interactive ipywidgets GUI tabs
- ingest: File readers and measurement discovery
- models: Data models (SegmentFrame, MeasurementCatalog, KnBundle)
- validation: Golden reference validation utilities
"""

__all__ = []
