# Rotating Coil Analyzer

Python tooling to ingest rotating-coil acquisition files (streaming binary and plateau text formats), split measurements into turns using a strict time policy, and compute per-turn Fourier harmonics with an interactive GUI.

This project is designed for **CERN accelerator-magnet** rotating-coil measurements across all machine complexes (LHC, SPS, PS, PSB, transfer lines, test benches such as SM18, ...), with a strong emphasis on **data integrity** and **traceability**.

---

## Key principles (non-negotiable)

### No synthetic time
The software **never creates synthetic time**.

- Time must come from acquisition columns in the input file(s).
- If time is missing or non-finite in a tail region, the affected samples/turns are **dropped**, not "fixed" by regenerating or aligning time.
- All trimming / dropping actions are **reported** to the user and require an explicit "preview -> apply" step in the GUI.

### No interpolation
Downsampling (when used for plotting) is **decimation only** (keep every Kth sample). No interpolation or resampling is performed.

---

## Current capabilities

### Supported input formats
- **Streaming binary** (`*.bin`): continuous acquisition data, "corr/generic" variants supported via `Parameters.txt` FDIs table mapping.
- **Plateau text** (`*_raw_measurement_data.txt`): DC plateau acquisition, multi-file plateau sequences concatenated in correct order, with plateau metadata propagated per turn.
- **Measurement-head CSV**: geometry files for computing kn calibration coefficients.

### Data model
- A discovered measurement folder is represented as a **MeasurementCatalog** (core API).
- A loaded segment is represented as a **SegmentFrame** (core API).
- Calibration coefficients are wrapped in **KnBundle** with full provenance.
- Merged harmonics are wrapped in **MergeResult** with full traceability.

---

## GUI overview

The GUI has five tabs:

### 1. Catalog
- Select a measurement folder
- Discover runs/segments via `Parameters.txt` and FDIs table mapping
- Load a selected segment and inspect diagnostics
- Preview waveforms

### 2. Harmonics
- Preview data-quality cuts (what will be trimmed/dropped)
- Apply cuts and compute FFT harmonics
- View amplitude vs. current plots
- View normal/skew vs. harmonic order per plateau

### 3. Coil Calibration
- Load kn from a TXT file, OR
- Compute kn from measurement-head geometry CSV
- Export computed kn to standard TXT format
- Outputs KnBundle with full provenance

### 4. Harmonic Merge
- Apply kn calibration to compute calibrated harmonics
- Select Abs/Cmp source per harmonic order
- Preset modes: "main from Abs, others from Cmp", etc.
- Post-merge normalisation: Tesla for n <= m, units for n > m (Bottura Section 3.7)
- Bottura 3.7 mixed-format CSV export via `mixed_format_table()`
- Record compensation scheme metadata
- Export with full traceability (kn provenance, per-n source map)

### 5. Plots
- Read-only exploration plots
- Plot any column vs. time
- Decimation-only downsampling
- Interactive zoom/pan via `%matplotlib widget` (ipympl backend)

---

## Installation

### Requirements
- Python 3.10+
- Common scientific stack: `numpy`, `pandas`, `matplotlib`, `ipywidgets`
- `ipympl` for interactive zoomable plots in Jupyter notebooks
- Jupyter environment (recommended for the GUI and analysis notebooks)

### Install (editable)
```bash
pip install -e .
```

This installs all dependencies including `ipympl` for interactive plots.

---

## Running the GUI

In a Jupyter notebook:

```python
%matplotlib widget  # Enable interactive zoomable plots

from rotating_coil_analyzer.gui.app import build_gui
gui = build_gui()
gui  # Display the GUI
```

---

## Notebooks

Example and analysis notebooks are in `rotating_coil_analyzer/notebooks/`:

### Tools
- `tools/analysis_gui.ipynb` -- Combined GUI (Catalog + Harmonics workflow)
- `tools/kn_from_mh_csv.ipynb` -- Compute kn calibration coefficients from measurement-head CSV

### SPS MBB dipole
- `SPS_MBB/2025-12-12_MBA/CS_harmonics.ipynb` -- CS segment harmonic analysis (Dec 2025 session)
- `SPS_MBB/2025-12-12_MBA/NCS_harmonics.ipynb` -- NCS segment harmonic analysis (Dec 2025 session)
- `SPS_MBB/2026-02-06_NCS_supercycle/NCS_harmonics.ipynb` -- NCS segment full analysis (Feb 2026 session)
- `SPS_MBB/2026-02-06_NCS_supercycle/NCS_supercycle.ipynb` -- Streaming supercycle analysis (Feb 2026)
- `SPS_MBB/2026-02-06_NCS_supercycle/b3_settling_200GeV.ipynb` -- Eddy-current b3 settling at MD1 plateau
- `SPS_MBB/comparisons/200GeV_vs_26GeV/comparison.ipynb` -- Cross-session comparison (200 GeV vs 26 GeV cycles)

### LEAR MC62 dipole
- `LEAR_MC62/00_system_check/` -- System check (10 turns)
- `LEAR_MC62/01_with_shims/` -- Staircase analysis + eddy-current settling (with shims, 1 Hz)
- `LEAR_MC62/02_without_shims/` -- Staircase analysis + eddy-current settling + FFMM parity (without shims, 1 Hz)
- `LEAR_MC62/03_2Hz_afternoon/` -- Streaming staircase analysis + eddy-current (2 Hz)
- `LEAR_MC62/04_2Hz_morning/` -- Reproducibility repeat (2 Hz, morning)
- `LEAR_MC62/comparisons/` -- Shims effect (01 vs 02) and reproducibility (03 vs 04)

### LIU BTP8 quadrupole
- `LIU_BTP8/2019-07-17/b3_sextupole.ipynb` -- b3 sextupole analysis
- `LIU_BTP8/2019-07-17/parity_validation.ipynb` -- Validation against legacy C++ results

### SM18 test bench
- `SM18/2024-12-04_parity/parity_validation.ipynb` -- Validation against legacy results

All notebooks use `%matplotlib widget` for interactive zoomable plots.

---

## Streaming Analysis Utilities

For **streaming (continuous) acquisition** measurements where the magnet current follows a machine supercycle, the package provides reusable utility functions in `rotating_coil_analyzer.analysis.utility_functions`:

| Function | Purpose |
|----------|---------|
| `compute_block_averaged_range` | Noise-robust within-turn current range (splits each turn into blocks, averages, takes max-min) |
| `detect_plateau_turns` | Three-rule plateau detection: (a) flat current, (b) starts on plateau, (c) ends on plateau |
| `classify_current` | Classify current value into a cycle-type label (injection, flat-top, ramp, ...). Default thresholds for SPS; fully customisable for other machines |
| `find_contiguous_groups` | Find contiguous runs of True in a boolean mask (e.g. injection plateau groups) |
| `process_kn_pipeline` | Full Kn pipeline in one call: dit -> drift -> FFT -> kn -> merge -> normalise |
| `build_harmonic_rows` | Convert pipeline results into a list of dicts, ready for `pd.DataFrame()` |
| `build_run_averages` | Per-run mean b3 with run ordering (for hysteresis / ramp analysis) |
| `diagnose_cel_fed` | Run pipeline with/without cel+fed, return diagnostic with SAFE/UNSAFE/MIXED recommendation |
| `ba_table_from_C` | Convert complex coefficients to legacy B/A DataFrame (all Tesla) |
| `mixed_format_table` | Bottura Section 3.7 mixed-format DataFrame (Tesla for n <= m, units for n > m) |

### Quick example

```python
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    process_kn_pipeline,
    build_harmonic_rows,
    build_run_averages,
    diagnose_cel_fed,
    ba_table_from_C,
    mixed_format_table,
)

# Block-averaged current range per turn (filters ADC noise)
I_range, I_blocks = compute_block_averaged_range(I_all, samples_per_turn=1024)

# Detect plateau turns (all three rules must pass)
info = detect_plateau_turns(I_blocks, I_mean, I_range, threshold=3.0)
plateau_mask = info["is_plateau"]

# Run the full Kn pipeline on plateau turns
result, C_merged, C_units, ok_main = process_kn_pipeline(
    flux_abs[plateau_mask], flux_cmp[plateau_mask],
    t[plateau_mask], I[plateau_mask],
    kn=kn, r_ref=0.02, magnet_order=1,
)

# Build a DataFrame
rows = build_harmonic_rows(result, C_merged, C_units, ok_main, magnet_order=1)
df = pd.DataFrame(rows)
```

### Custom current thresholds

The default thresholds in `classify_current` are tuned for SPS cycle structure. For other machines (PS, PSB, LHC, ...), pass a custom thresholds dictionary:

```python
# Example: PS Booster thresholds
psb_thresholds = {
    "zero": 10,
    "injection": 100,
    "flat-top": 500,
}
label = classify_current(I_value, thresholds=psb_thresholds)
```

---

## Running Tests

```bash
# Run all tests (121 tests)
python -m pytest rotating_coil_analyzer/tests/ -v

# Run specific test file
python -m pytest rotating_coil_analyzer/tests/test_kn_bundle.py -v

# Quick run
python -m pytest rotating_coil_analyzer/tests/ -x -q
```

---

## Kn File Format

The standard kn TXT format is whitespace-delimited columns:
```
Abs_Re  Abs_Im  Cmp_Re  Cmp_Im  [Ext_Re  Ext_Im]
```
- 4 columns: Absolute and Compensated channels
- 6 columns: Absolute, Compensated, and External channels

One row per harmonic order (n=1, 2, 3, ...).

---

## Compensation Scheme

**Important:** The compensation scheme (e.g., "A-C", "ABCD", "none") is **NOT inferable** from the measurement-head CSV file.

The MH CSV contains only coil geometry data (radius, angles, turns, magnetic surface, etc.), not wiring or connection metadata. The compensation scheme describes how coils are electrically connected to form the compensated channel.

**You must specify the compensation scheme explicitly** when:
- Computing kn from a measurement-head CSV (see `notebooks/tools/kn_from_mh_csv.ipynb`)
- Creating a KnBundle for the Harmonic Merge workflow

The scheme is stored in `KnBundle.extra["compensation_scheme"]` and propagated to all downstream exports.

Common compensation schemes:
- `"none"` or `"single"`: Single coil (no compensation)
- `"A-C"`: Two-coil difference (e.g., coil 1 minus coil 5)
- `"A-B-C-D"` or `"ABCD"`: Four-coil bucking (alternating sum/difference)
- `"custom"`: Non-standard wiring (document in notes)

---

## Project Structure

```
rotating_coil_analyzer/
├── analysis/               # Harmonic computation, kn pipeline, merge logic
│   ├── kn_pipeline.py      #   Core pipeline: dit -> drift -> FFT -> kn -> rot -> cel -> fed
│   ├── utility_functions.py #   Streaming analysis utilities (plateau detection, pipeline wrapper)
│   ├── preprocess.py        #   Drift correction, di/dt correction
│   ├── fourier.py           #   FFT-based harmonic extraction
│   ├── merge.py             #   Abs/Cmp channel merge recommendations
│   ├── kn_head.py           #   Kn computation from measurement-head CSV
│   └── kn_bundle.py         #   Provenance-rich kn container
├── gui/                    # ipywidgets GUI tabs
├── ingest/                 # File readers and measurement discovery
│   ├── readers_streaming.py #   Streaming binary reader
│   ├── readers_plateau.py   #   Plateau text reader
│   ├── channel_detect.py    #   Automatic flux/current channel detection
│   └── discovery.py         #   Measurement folder discovery
├── models/                 # Data models (SegmentFrame, MeasurementCatalog, AnalysisProfile)
├── notebooks/              # Jupyter analysis & example notebooks
├── tests/                  # Unit tests (121 tests)
└── validation/             # Golden reference validation (C++ parity)
```

---

## Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** -- Comprehensive reference: architecture, pipeline steps, models, GUI, function reference, cross-implementation comparison (Bottura, FFMM C++, Pentella).
- **[PARITY_REPORT.md](PARITY_REPORT.md)** -- Machine-precision parity validation against the legacy C++ analyzer (SM18, BTP8, MC62).
- **[Correction Options Reference](rotating_coil_analyzer/notebooks/correction_options_reference.md)** -- Option-by-option guide with cel/fed failure modes and benchmark comparison across all four implementations.

---

## Theory and References

The analysis algorithms follow the standard procedures described in:

- **Bottura, L.** -- *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets -- Part I: Harmonics* (included in `theory/` folder)

Key formulas implemented:
- FFT-based harmonic extraction: `f_n = 2 * FFT(flux) / N`
- Kn application: `C_n = f_n / conj(kn) * Rref^(n-1)`
- Phase rotation: `C_rotated = C * exp(-i * phi * k)` for all harmonics k=1..H (Bottura Eq. AIV.6)
- Center location (CEL) and feeddown corrections

The implementation achieves **machine-precision parity** (float64 rounding floor) with the legacy C++ analyzer on SM18 streaming (285,095 turns, B1 diff = 1.82e-12 T) and LIU BTP8 plateau (222 turns, B1/A1 at ~1e-18 T). See [PARITY_REPORT.md](PARITY_REPORT.md) for full details.
