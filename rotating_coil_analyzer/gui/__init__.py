"""GUI package - interactive ipywidgets interface.

This package provides the Jupyter notebook GUI with eight tabs:

0. Catalog: Browse and load measurement segments
1. Plateau Detection: Streaming supercycle plateau detection
2. Harmonics: Turn QC, FFT computation, amplitude plots
3. Coil Calibration: Load or compute kn coefficients
4. Harmonic Merge: Apply kn and merge Abs/Cmp channels (+ cel/fed diag, MAD, N_LAST)
5. Raw Signal Plots: Read-only exploration plots
6. Physics Plots: Hysteresis, TF, Ld, eddy-current settling
7. Comparison: Cross-measurement comparison

Entry point:
    from rotating_coil_analyzer.gui.app import build_gui
    gui = build_gui()

Design principles:
- Preview before apply: all destructive operations require confirmation
- No synthetic time: analysis uses turn/sample indexing
- Full provenance: all outputs carry metadata about their origin
"""
