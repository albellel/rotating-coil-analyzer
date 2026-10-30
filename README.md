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
- **Plateau text** (`*_raw_measurement_data.txt`): DC plateau acquisition, multi-file plateau sequences concatenated in correct order, with plateau metadata propagated per turn. Both the standard `_Run_<step>_I_<current>A_<seg>_` naming and the H/V steerer `_Run_<step>IH_<ih>IV_<iv>_<seg>_` naming (e.g. FFMM degauss) are recognised, and the parameters file may be named either `Parameters.txt` or `<magnet>_<timestamp>_Parameters.txt`.
- **Measurement-head CSV**: geometry files for computing kn calibration coefficients.

### Data model
- A discovered measurement folder is represented as a **MeasurementCatalog** (core API).
- A loaded segment is represented as a **SegmentFrame** (core API).
- Calibration coefficients are wrapped in **KnBundle** with full provenance.
- Merged harmonics are wrapped in **MergeResult** with full traceability.

---

## GUI overview

The GUI has eight tabs:

### 0. Catalog
- Select a measurement folder
- Discover runs/segments via `Parameters.txt` and FDIs table mapping
- Load a selected segment and inspect diagnostics
- Preview waveforms

### 1. Plateau Detection
- Detect current plateaus in streaming supercycle data
- Block-averaged range computation (filters ADC noise)
- Three-rule detection: flat current, starts on plateau, ends on plateau
- Visual overlay of detected plateaus on the current waveform

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

### 5. Raw Signal Plots
- Read-only time-series exploration
- Plot any column vs. time
- Decimation-only downsampling
- Interactive zoom/pan via `%matplotlib widget` (ipympl backend)

### 6. Physics Plots
- Hysteresis loops (B vs. I)
- Transfer function B1/I and differential inductance dB1/dI
- Eddy-current settling curves with exponential fit
- Per-run and per-plateau visualisations

### 7. Comparison
- Cross-measurement CSV comparison
- Load multiple exported CSV files and overlay harmonics
- Side-by-side normal/skew component plots

---

## Installation

Requirements: Python 3.10+, git. Everything else is installed by pip.

```bash
git clone https://gitlab.cern.ch/albellel/rotating-coil-analyzer.git
cd rotating-coil-analyzer
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"                 # numpy, pandas, scipy, matplotlib, ipywidgets, ipympl, pytest
python -m pytest                        # all tests must pass
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pip install -e ".[eddy]"` | `tools-for-data-analysis` (from gitlab.cern.ch, needs access to that repo) | eddy-current fits in the GUI *Physics Plots* tab |
| `pip install -e ".[pptx]"` | `python-pptx`, `Pillow` | `rotating_coil_analyzer.presentation` slide helpers |

Without `[eddy]` everything works except the eddy-fit button, which says so.

> **Windows:** if `git clone` fails with *Filename too long*, run
> `git config --global core.longpaths true` once, or clone into a short path
> such as `C:\code\`.

---

## Quick start (5 minutes)

**1. Run the bundled demo** (an FCC-ee quadrupole DC measurement shipped in the
repo, with FFMM's own results for comparison):

```bash
rca-analyze                     # or: python -m rotating_coil_analyzer.cli
```

It prints the settings it used (taken from the measurement's `Parameters.txt`:
m = 2, Rref = 10 mm, options), the mean $B_2$, centre and $b_n$, writes
per-turn and summary CSVs to `./rca_output/`, and the agreement with FFMM
($b_3$ to ~$10^{-13}$ units).

**2. Run it on your own measurement** — a measurement folder (with its
`Parameters.txt`) plus the segment kn file:

```bash
rca-analyze path/to/measurement_folder --kn path/to/Kn_values_Seg_Main.txt
rca-analyze path/to/folder --kn kn.txt --seg NCS --order 1 --rref 0.017 --options dri,rot
```

`--order`, `--rref`, `--skew/--normal` and `--options` override what
`Parameters.txt` says. If the main order or the reference radius cannot be
found and is not given, the run **stops with a message** instead of guessing.

**3. Interactive GUI** (Jupyter): see [Running the GUI](#running-the-gui).
**4. Scripted analysis** in your own notebook: see [GUIDE.md](GUIDE.md) §3.

### What your data must look like

The package reads the folder layout written by the CERN FFMM acquisition software:

- a parameters file `Parameters.txt` or `<magnet>_<timestamp>_Parameters.txt`
  in the folder or up to two levels above it, containing an FDIs table
  (`Measurement.AP1.FDIs` / `Parameters.MH.FDIs: TABLE{<segment>\t<abs FDI>\t<cmp FDI>\t...}`);
- data files, either
  - **streaming** `<run>_corr_sigs_Ap_<n>_Seg<id>.bin` (or `.txt/.csv`):
    columns `t, flux increment 1, flux increment 2, current(s)`, float64 or float32; or
  - **plateau** `<base>_Run_<step>_I_<current>A_<seg>_raw_measurement_data.txt`
    (one file per current step; also the H/V steerer form `..._Run_<step>IH_<ih>IV_<iv>_<seg>_...`);
- a **kn file**: whitespace-separated, one row per harmonic order n = 1..H,
  4 columns `Abs_Re Abs_Im Cmp_Re Cmp_Im` (or 6 with `Ext_Re Ext_Im`).
  It can also be computed from a measurement-head CSV
  (`notebooks/tools/kn_from_mh_csv.ipynb`).

The flux columns are **increments** from the digital integrator (FDI), not
voltage and not integrated flux. If your data come from another system, build
`(n_turns, Ns)` arrays yourself and call the pipeline directly (GUIDE.md §3.4).

### Settings the analysis needs — read from the file, never guessed

| Setting | Parameters.txt key | If missing |
|---|---|---|
| main order m | `magnetAnalyzer.magnetOrder`, else `magnetInfo.magnetType` (Dipole → 1, Quadrupole → 2, …) | must be given (`--order`, GUI field, `magnet_order=`) |
| reference radius | `magnetAnalyzer.refRadius` | must be given; values outside 1 mm–0.5 m are flagged |
| skew main | `magnetInfo.magnetNormSkew` | normal |
| options | `magnetAnalyzer.dit/dri/rot/cel/fed` | `dri,rot` (`nor` is always applied after the merge) |
| samples per turn | `Measurement.samples` | 512, with a WARNING |
| shaft speed | `Measurement.v`, else `Measurement.MotorAngularSpeed` | 60 rpm, with a WARNING (only used for the timing check) |

The pipeline also reports, without changing any number: `cel` requested but
not applicable, centre offset ≥ Rref (feed-down diverges), |B_m| > 20 T, main
angle at the ±π/2 wrap boundary (use `skew_main` or `encoder_offset_rad`), and
orders merged from a bucked compensated channel. In scripts these are Python
`UserWarning`s; in the GUI they go to the log.

### Large files (computational cost)

Measured on one 1.17 GB SM18 streaming file (57 022 turns × 512 samples,
15 orders):

| Stage | Time | Peak RAM above baseline |
|---|---|---|
| `StreamingReader.read` | 6.3 s | 4.2 GB (part is reclaimable page cache) |
| `process_kn_pipeline`, all turns, `dri,rot,cel,fed` | 2.2 s | 3.3 GB |

Time and memory scale linearly: an **11 GB** file takes **~1–1.5 min** of
computation but needs **~40 GB** to load plus **~30 GB** to analyse all turns
at once. On a smaller machine, analyse the turns in chunks (every turn is
processed independently):

```python
for s in range(0, tb.n_turns, 50_000):
    sl = slice(s, s + 50_000)
    res, Cm, Cu, ok = process_kn_pipeline(tb.df_abs[sl], tb.df_cmp[sl], tb.t[sl], tb.I[sl], kn, r_ref=R, magnet_order=m)
    ...  # append rows, then concatenate
```

---

## Running the GUI

In a Jupyter notebook:

```python
%matplotlib widget  # Enable interactive zoomable plots

from rotating_coil_analyzer.gui.app import build_gui
gui = build_gui()
gui  # Display the GUI
```

Typical session: **Catalog** → select the measurement folder and load a
segment (WARNINGs about missing settings appear in red) → **Coil
Calibration** → load the segment kn TXT → **Harmonic Merge** → check the main
order m and Rref that were pre-filled from `Parameters.txt` (they start at 0
and *Apply Kn* refuses if they are unknown) → *Apply Kn* → *Apply Merge* →
export. Diagnostics from the pipeline are written to the tab's log. If
`jupyter` is not installed yet: `pip install jupyterlab`.

---

## Notebooks

This is the **magnet-agnostic library repo**. The magnet-specific measurement
campaigns (SPS MBB, LEAR MC62, ...) and their analysis/R&D notebooks live in
their own repositories (`mbb-data-analysis`, `mc62-data-analysis`,
`mba-data-analysis`), which depend on this package. Only example/tool and
parity-validation notebooks remain here, under `rotating_coil_analyzer/notebooks/`:

### Tools (example / tutorial)
- `tools/analysis_gui.ipynb` -- Combined GUI (Catalog + Harmonics workflow)
- `tools/kn_from_mh_csv.ipynb` -- Compute kn calibration coefficients from measurement-head CSV

### Parity validation

These notebooks need the author's local `golden_standards/` datasets (large,
not distributed with the repo). From a fresh clone, FFMM parity is reproduced
by the bundled FCC-ee demo (`rca-analyze`) and its test
(`test_review_fixes_2026_10.py::test_fccee_end_to_end_matches_ffmm`).

- `LIU_BTP8/2019-07-17/b3_sextupole.ipynb` -- b3 sextupole analysis (quadrupole)
- `LIU_BTP8/2019-07-17/parity_validation.ipynb` -- Validation against legacy C++ results
- `SM18/2024-12-04_parity/parity_validation.ipynb` -- SM18 streaming parity against legacy results
- `Buckley_steerer/2026-06-23_degauss_parity/parity_validation.ipynb` -- Degauss staircase parity vs FFMM golden (dipole, machine precision) + degauss field decay
- `Buckley_steerer/2026-06-23_degauss_parity/gui_walkthrough.ipynb` -- Interactive GUI walkthrough on the degauss dataset (loaded natively, no renaming)

### Reference documents
- `notebooks/pipeline_reference.md` -- End-to-end pipeline explained: what each stage does, where it sits, and why (integrate-to-flux, dit, dri modes, Kn, rot, cel, fed, merge, nor)
- `notebooks/correction_options_reference.md` -- Option-by-option guide (cel/fed failure modes, benchmark across implementations)
- `notebooks/physics_reference.md` -- Physics background for the analysis quantities

All notebooks use `%matplotlib widget` for interactive zoomable plots. For a
hands-on walkthrough of the API, see [GUIDE.md](GUIDE.md).

---

## Streaming Analysis Utilities

For **streaming (continuous) acquisition** measurements where the magnet current follows a machine supercycle, the package provides reusable utility functions in `rotating_coil_analyzer.analysis.utility_functions`:

| Function | Purpose |
|----------|---------|
| `compute_block_averaged_range` | Noise-robust within-turn current range (splits each turn into blocks, averages, takes max-min) |
| `detect_plateau_turns` | Three-rule plateau detection: (a) flat current, (b) starts on plateau, (c) ends on plateau |
| `classify_current` | Classify current value into a cycle-type label (injection, flat-top, ramp, ...). Default thresholds for SPS; fully customisable for other machines |
| `find_contiguous_groups` | Find contiguous runs of True in a boolean mask (e.g. injection plateau groups) |
| `process_kn_pipeline` | Full Kn pipeline in one call: dit -> drift -> FFT -> kn -> merge -> normalise. Accepts `encoder_offset_rad` (constant angular offset applied before FFT) and `flip_signal_polarity` (negate flux before processing, replaces the old `FLIP_FIELD_SIGN` flag) |
| `build_harmonic_rows` | Convert pipeline results into a list of dicts, ready for `pd.DataFrame()` |
| `build_run_averages` | Per-run mean b3 with run ordering (for hysteresis / ramp analysis) |
| `diagnose_cel_fed` | Run pipeline with/without cel+fed, return diagnostic with SAFE/UNSAFE/MIXED recommendation |
| `diagnose_fdi_transitions` | Detect FDI stuck-channel issues at plateau boundaries |
| `ba_table_from_C` | Convert complex coefficients to legacy B/A DataFrame (all Tesla) |
| `mixed_format_table` | Bottura Section 3.7 mixed-format DataFrame (Tesla for n <= m, units for n > m) |

> **Eddy-current fitting** (`eddy_model`, `double_eddy_model`, `triple_eddy_model`,
> `validate_eddy_model_selection`, `EddyFitResult`, `fit_eddy_per_run`) and the
> continuous-time signal helpers now live in the companion package
> `tools_for_data_analysis` — import them from
> `tools_for_data_analysis.fitting.eddy`. They are not part of this package.

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

# Run the full Kn pipeline on plateau turns. The first two arguments are the
# flux INCREMENTS per turn (TurnBlock.df_abs / df_cmp), not integrated flux.
result, C_merged, C_units, ok_main = process_kn_pipeline(
    df_abs[plateau_mask], df_cmp[plateau_mask],
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
# Run all tests (pytest is in the [dev] extra; testpaths is set in pyproject.toml)
python -m pytest

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
├── gui/                    # ipywidgets GUI tabs (8 tabs)
│   ├── app.py               #   Tab assembly and build_gui() entry point
│   ├── harmonics.py         #   Tab 2: Turn QC, FFT, amplitude plots
│   ├── coil_calibration.py  #   Tab 3: Load / compute Kn
│   ├── harmonic_merge.py    #   Tab 4: Kn application, merge, CSV export
│   ├── plots.py             #   Tab 5: Raw signal time-series
│   ├── plateau_detection.py #   Tab 1: Streaming plateau detection
│   ├── physics_plots.py     #   Tab 6: Hysteresis, transfer fn, Ld, eddy settling
│   └── comparison.py        #   Tab 7: Cross-measurement CSV comparison
├── ingest/                 # File readers and measurement discovery
│   ├── readers_streaming.py #   Streaming binary reader
│   ├── readers_plateau.py   #   Plateau text reader
│   ├── channel_detect.py    #   Automatic flux/current channel detection
│   └── discovery.py         #   Measurement folder discovery
├── models/                 # Data models (SegmentFrame, MeasurementCatalog, AnalysisProfile)
├── presentation/           # PowerPoint report generation helpers
├── notebooks/              # Example/tool + parity-validation notebooks + reference docs
├── tests/                  # Unit + regression tests; tests/data holds the bundled FCC-ee demo
├── validation/             # Golden reference validation (C++ parity)
└── cli.py                  # `rca-analyze`: folder + kn file -> CSV (bundled demo without arguments)
docs/                       # Theory-to-code chapters + 11_bugfix_diary.md
scripts/
└── btp8_bruteforce_turns.py   # Brute-force turns-per-revolution search for BTP8
```

> Notebook generators and `nb_helpers` are no longer kept here. Programmatic
> notebook construction now uses `tools_for_data_analysis.helpers.notebook`, and the
> magnet campaign notebooks live in their own repos (`mbb-data-analysis`,
> `mc62-data-analysis`, `mba-data-analysis`).

---

## Documentation

- **[docs/](docs/README.md)** -- Theory-to-code documentation (consolidated 2026-08-20 from the retired LaTeX repo `RC-analyzer-documentation`): measurement principle, ingest and time policy, signal processing, FFT, kn calibration, corrections, merge/normalisation, validation, usage examples, and the [Bottura equation → function cross-reference with the documented deviations](docs/10_bottura_cross_reference.md). LaTeX-ready formulas.
- **[docs/12_implementations_and_their_errors.md](docs/12_implementations_and_their_errors.md)** -- How FFMM (native C++ and Matlab paths), the MMM-section Python analyzer and this package differ. The errors the earlier tools produce: last order not rotated, skew units ~1e16, ramp correction on rising ramps only, unguarded centring and normalisation, first-sample drift. What this package does about each.
- **[docs/11_bugfix_diary.md](docs/11_bugfix_diary.md)** -- Dated diary of every bug fix: symptom, cause, fix, whether earlier results are affected, verification.
- **[GUIDE.md](GUIDE.md)** -- Hands-on walkthrough of the GUI and of the programmatic streaming pipeline (the recipe the magnet repos use).
- **[DOCUMENTATION.md](DOCUMENTATION.md)** -- Architecture, data models, GUI, function reference, cross-implementation comparison (FFMM C++, MMM-section Python analyzer).
- **[PARITY_REPORT.md](PARITY_REPORT.md)** -- Machine-precision parity validation against the legacy C++ analyzer (SM18, BTP8, MC62).
- **[Correction Options Reference](rotating_coil_analyzer/notebooks/correction_options_reference.md)** -- Option-by-option guide with cel/fed failure modes and benchmark comparison across all four implementations.

---

## Theory and References

The analysis algorithms follow the standard procedures described in:

- **Bottura, L.** -- *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets -- Part I: Harmonics*, MTA-IN-97-007, CERN, 1997 (PDF in the author's literature repository `bibliography-review/magnetic_measurements/rotating_coil_harmonics/`; bib key `bottura1997`). The equation-by-equation transcription lives in the bibliography note `bibliography-review/magnetic_measurements/rotating_coil_harmonics/bottura1997_standard_analysis_field_quality_LHC_harmonics_notes.md`; the equation → function map with the documented deviations is [docs/10_bottura_cross_reference.md](docs/10_bottura_cross_reference.md).
- **Marusov, V.** (2013) -- *Measurement of a Time-Periodic Magnetic Field by Rotating Coil*, NIM-A 711, 121--123, doi:10.1016/j.nima.2013.01.042 (PDF in `bibliography-review/magnetic_measurements/rotating_coil_harmonics/`, bib key `marusov2013`). The plain-language guide `marusov_guide.md` moved with the LEAR MC62 campaign to the `mc62-data-analysis` repo.

Key formulas implemented:
- FFT-based harmonic extraction: `f_n = 2 * FFT(flux) / N`
- Kn application: `C_n = f_n / conj(kn) * Rref^(n-1)`
- Phase rotation: `C_rotated = C * exp(-i * phi * k)` for all harmonics k=1..H (Bottura Eq. AIV.6)
- Center location (CEL) and feeddown corrections
- Marusov temporal decomposition: `σ_{n,k} = (1/M) Σ_j C_n(j) exp(-i 2π k j / M)` — separates spatial and temporal content

The implementation achieves **machine-precision parity** (float64 rounding floor) with the legacy C++ analyzer on SM18 streaming (285,095 turns, B1 diff = 1.82e-12 T) and LIU BTP8 plateau (222 turns, B1/A1 at ~1e-18 T). See [PARITY_REPORT.md](PARITY_REPORT.md) for full details.

---

## License

MIT, see [LICENSE](LICENSE).
