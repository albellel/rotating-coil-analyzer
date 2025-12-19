# Rotating Coil Analyzer (Python) — Phase 1

This repository is an **offline Python analyzer** for rotating-coil magnetic measurements, inspired by CERN MMM / FFMM workflows.

Phase-1 scope is intentionally limited to:

- **Discovery / cataloging** of a measurement folder
- Parsing `Parameters.txt` (including `TABLE{...}` payloads with escaped `\t` / `\n`)
- **Locating segment files** (SM18 `corr_sigs` and `generic_corr_sigs`)
- Reading segment files (`.bin`, `.txt`, `.csv`) into a clean `pandas.DataFrame`
- Emitting **sanity checks** and convenient **debug plots** (GUI)

No FFT/multipoles/TurnQC/Kn calculations are part of Phase-1.

---

## Installation (editable)

From the repo root (same directory as `pyproject.toml`):

```bash
python -m pip install -U pip
python -m pip install -e .
```

Recommended for notebooks:

```python
%load_ext autoreload
%autoreload 2
```

---

## Expected file naming patterns (Phase-1)

Discovery supports both patterns:

- `<run_id>_corr_sigs_Ap_<ap>_Seg<seg>.bin`
- `<run_id>_generic_corr_sigs_Ap_<ap>_Seg<seg>.bin`

Also accepts `.txt` and `.csv` instead of `.bin`.

Where:
- `ap` is a **physical aperture id** (e.g., `1` or `2`)
- `seg` can be numeric (`3`) or string (`CS`, `NCS`, ...)

---

## Parameters.txt handling

Phase-1 searches for `Parameters.txt` in the selected folder **or up to 2 parent folders above**.

Required keys:
- `Parameters.Measurement.samples`
- `Parameters.Measurement.v`
- `Measurement.AP1.enabled`, `Measurement.AP2.enabled` (AP2 optional)
- `Measurement.AP1.FDIs` and `Measurement.AP2.FDIs` for enabled apertures

Strict policy: **no degraded mode**. If Parameters parsing fails, discovery fails loudly.

---

## Running the GUI (Jupyter / VS Code notebook)

In a notebook cell:

```python
from rotating_coil_analyzer.gui.app import build_catalog_gui
gui = build_catalog_gui()
gui
```

Notes:
- If you re-run the cell and see “duplicated GUI outputs”, use the notebook UI **Clear Outputs** for that cell.
- If you edit code but the kernel keeps using old imports, restart the kernel (or use autoreload).

---

## Sanity checks shown in the GUI

The reader emits `SegmentFrame.warnings`, including:
- selected binary format (dtype + number of columns)
- flux column assignment (abs/cmp)
- dt nominal check (from |v| and samples_per_turn)
- current candidate dynamic ranges
- duplicate current detection (`I1 == I2`, etc.)

---

## Tests (minimal ROI set)

Tests are in `rotating_coil_analyzer/tests` and use the standard library `unittest`.

Run from repo root:

```bash
python -m unittest discover -s rotating_coil_analyzer/tests -v
```

Covers:
- discovery filename regex: `corr_sigs` + `generic_corr_sigs`
- two-aperture collision prevention (segment_files key includes aperture)
- Parameters TABLE parsing with escaped sequences
- binary inference sanity (dt nominal check)
