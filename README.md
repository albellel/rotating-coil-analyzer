# Rotating Coil Analyzer

Python tooling to ingest rotating-coil acquisition files (SM18 binary and MBA plateau text formats), split measurements into turns using a strict time policy, and compute per-turn Fourier harmonics with a simple two-tab GUI.

This project is designed for accelerator-magnet rotating-coil measurements, with a strong emphasis on **data integrity** and **traceability**.

---

## Key principles (non-negotiable)

### No synthetic time
The software **never creates synthetic time**.

- Time must come from acquisition columns in the input file(s).
- If time is missing or non-finite in a tail region, the affected samples/turns are **dropped**, not “fixed” by regenerating or aligning time.
- All trimming / dropping actions are **reported** to the user and require an explicit “preview → apply” step in the GUI.

---

## Current capabilities

### Supported input formats
- **SM18 binary** (`*.bin`): “corr/generic” variants supported via `Parameters.txt` FDIs table mapping.
- **MBA plateau text** (`*_raw_measurement_data.txt`): multi-file plateau sequences concatenated in correct order, with plateau metadata propagated per turn.

### Data model
- A discovered measurement folder is represented as a **MeasurementCatalog** (core API).
- A loaded segment is represented as a **SegmentFrame** (core API), holding:
  - the raw sample table (`SegmentFrame.df`)
  - metadata including `samples_per_turn`, `n_turns`, `aperture_id`, and (when available) `magnet_order`.

### Turn splitting and QC
- Turns are defined by `samples_per_turn` (no index pulse is assumed at present).
- Data-quality actions are *previewed* and then *applied*:
  - tail trimming to reach an integer number of turns
  - dropping turns with non-finite signals
  - enforcing time validity (optionally requiring strictly increasing time within each turn)

### Fourier harmonics (Phase II)
- Per-turn DFT up to a chosen maximum order.
- Optional **integrate differential signal → flux** (legacy convention).
- Optional **drift correction** (use only when needed).
- Phase reference uses the **main field order** (from `Parameters.txt` if available, otherwise user-selected).
- Normal/skew plots are shown per selected plateau (with plateau ID and mean current).

### Export
- Plot export: SVG or PDF.
- Table export: CSV.

---

## GUI overview

The GUI has two tabs:

### Phase I — Catalog
1. Select a measurement folder.
2. Discover runs/segments via `Parameters.txt` and FDIs table mapping.
3. Load a selected segment and inspect diagnostics.

If the measurement has only one aperture, the GUI shows **“single aperture measurement”** (not “none”).

### Phase II — Harmonics (FFT)
Workflow is explicit and safe:

1. **Preview data-quality cuts**
   - Shows what will be trimmed/dropped and why.
2. **Apply cuts and compute harmonics (FFT)**
   - Applies only what was previewed.
3. Plotting views:
   - **View 1:** Amplitude versus current for a selected harmonic and channel.
   - **View 2:** Normal/skew versus harmonic order for a selected plateau.

---

## Installation

### Requirements
- Python 3.10+ recommended
- Common scientific stack: `numpy`, `pandas`, `matplotlib`, `ipywidgets`
- Jupyter environment (recommended for the GUI notebooks)

### Install (editable)
```bash
pip install -e .
