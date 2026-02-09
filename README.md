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

### GUI & workflow notebooks
1. **02_analysis_gui.ipynb** -- Combined GUI (Catalog + Harmonics workflow)
2. **03_kn_from_mh_csv.ipynb** -- Compute kn calibration coefficients from measurement-head CSV

### Harmonic analysis notebooks
3. **b3_analysis_LIU_BTP8.ipynb** -- b3 sextupole analysis for a LIU BTP8 quadrupole
4. **b3_from_kn_20251212_171026_SPS_MBA_CS.ipynb** -- b3 from Kn for SPS MBA (CS segment)
5. **b3_from_kn_20251212_171026_SPS_MBA_NCS.ipynb** -- b3 from Kn for SPS MBA (NCS segment)
6. **analysis_20260206_142231_SPS_MBB_NCS.ipynb** -- Full harmonic analysis for SPS MBB dipole (NCS, single plateau)
7. **analysis_20260206_144537_SPS_MBB_NCS_supercycle.ipynb** -- Streaming supercycle analysis for SPS MBB dipole (NCS). Supercycle structure: LHC_pilot -> MD1 -> SFTPRO x20. Includes automatic plateau detection, hysteresis evolution tracking, within-plateau settling analysis (eddy currents vs current ramp). Key finding: MD1 is a true current plateau (drift < 0.1 A), SFTPRO is not (current ramps ~5 A).

### Eddy-current analysis notebooks
10. **eddy_current_b3_settling_200GeV.ipynb** -- Eddy-current b3 settling time at the MD1 injection plateau (exponential fit of sextupole decay after LHC excitation)

### Validation notebooks
8. **golden_standard_parity.ipynb** -- Validation against legacy C++ results (LIU BTP8 quadrupole)
9. **golden_standard_SM18_parity.ipynb** -- Validation against legacy results (SM18 test bench)

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

### Quick example

```python
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    process_kn_pipeline,
    build_harmonic_rows,
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
# Run all tests (96 tests)
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
- Computing kn from a measurement-head CSV (see `notebooks/03_kn_from_mh_csv.ipynb`)
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
├── tests/                  # Unit tests (96 tests)
└── validation/             # Golden reference validation (C++ parity)
```

---

## Documentation

Detailed documentation is in the `docs/` folder:

- **[Analysis Pipeline Reference](docs/analysis_pipeline.md)** -- Step-by-step pipeline documentation with formulas, code references, and cross-implementation comparison.
- **[Golden Standard Parity Report](docs/golden_standard_parity_report.md)** -- Methodology and results of validating against the legacy C++ analyzer.

---

## Theory and References

The analysis algorithms follow the standard procedures described in:

- **Bottura, L.** -- *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets -- Part I: Harmonics* (included in `theory/` folder)

Key formulas implemented:
- FFT-based harmonic extraction: `f_n = 2 * FFT(flux) / N`
- Kn application: `C_n = f_n / conj(kn) * Rref^(n-1)`
- Phase rotation: `C_rotated = C * exp(-i * phi * k)`
- Center location (CEL) and feeddown corrections

The implementation has been validated against the legacy C++ analyzer (ffmm/MatlabAnalyzerRotCoil.cpp) across multiple magnet types and CERN machine complexes.
