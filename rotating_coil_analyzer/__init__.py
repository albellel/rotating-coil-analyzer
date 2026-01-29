"""Rotating Coil Analyzer - Python tooling for rotating-coil magnetic measurements.

This package provides tools for:
- Ingesting rotating-coil acquisition files (streaming binary and plateau text formats)
- Splitting measurements into turns using strict time policies
- Computing per-turn Fourier harmonics
- Applying kn calibration coefficients
- Merging absolute/compensated harmonics with full provenance tracking

Key principles:
- No synthetic time: time must come from acquisition columns
- No interpolation: downsampling uses decimation only
- Full traceability: all operations are logged and auditable

Main subpackages:
- analysis: Harmonic computation, kn pipeline, merge logic
- gui: Interactive ipywidgets GUI tabs
- ingest: File readers and measurement discovery
- models: Data models (SegmentFrame, MeasurementCatalog, KnBundle)
- validation: Golden reference validation utilities
"""

__all__ = []
