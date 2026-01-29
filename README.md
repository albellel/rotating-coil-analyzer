# Rotating Coil Analyzer

Python tooling to ingest rotating-coil acquisition files (streaming binary and plateau text formats), split measurements into turns using a strict time policy, and compute per-turn Fourier harmonics with an interactive GUI.

This project is designed for accelerator-magnet rotating-coil measurements, with a strong emphasis on **data integrity** and **traceability**.

---

## Key principles (non-negotiable)

### No synthetic time
The software **never creates synthetic time**.

- Time must come from acquisition columns in the input file(s).
- If time is missing or non-finite in a tail region, the affected samples/turns are **dropped**, not "fixed" by regenerating or aligning time.
- All trimming / dropping actions are **reported** to the user and require an explicit "preview → apply" step in the GUI.

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
- Interactive zoom/pan (when ipympl backend is available)

---

## Installation

### Requirements
- Python 3.10+ recommended
- Common scientific stack: `numpy`, `pandas`, `matplotlib`, `ipywidgets`
- Jupyter environment (recommended for the GUI notebooks)
- Optional: `ipympl` for interactive plots

### Install (editable)
```bash
pip install -e .
```

---

## Running the GUI

In a Jupyter notebook:

```python
%matplotlib widget  # Enable interactive plots (optional but recommended)

from rotating_coil_analyzer.gui.app import build_gui
gui = build_gui()
gui  # Display the GUI
```

---

## Notebooks

Example notebooks are in `rotating_coil_analyzer/notebooks/`:

1. **01_catalog_browser.ipynb** - Browse measurement catalogs
2. **02_phase2_gui.ipynb** - Harmonics computation workflow
3. **03_kn_from_mh_csv.ipynb** - Compute kn from measurement-head CSV

---

## Running Tests

```bash
# Run all tests
py -m pytest rotating_coil_analyzer/tests/ -v

# Run specific test file
py -m pytest rotating_coil_analyzer/tests/test_kn_bundle.py -v
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
├── analysis/          # Harmonic computation, kn pipeline, merge logic
├── gui/               # ipywidgets GUI tabs
├── ingest/            # File readers and discovery
├── models/            # Data models (SegmentFrame, MeasurementCatalog)
├── notebooks/         # Example Jupyter notebooks
├── tests/             # Unit tests
└── validation/        # Golden reference validation
```
