# Rotating Coil Analyzer - Comprehensive Documentation

**Updated**: 2026-11-03 (checked against the code of that date; first written 2026-03-12)
**Version**: Current master branch

**Scope.** This file is the **architecture, data-model and function reference** of the
package. Where another document is the canonical home of a topic, this file only points to it:

| Topic | Canonical document |
|---|---|
| Install, GUI tabs, kn file format, CLI | [`README.md`](README.md) |
| Programmatic walkthrough | [`docs/GUIDE.md`](docs/GUIDE.md) |
| Theory to code (Bottura equations, the three deviations) | [`docs/`](docs/README.md), chapters 1-10 |
| Parity numbers against FFMM | [`docs/PARITY_REPORT.md`](docs/PARITY_REPORT.md) |
| FFMM / MMM analyzer compared stage by stage | [`docs/12_implementations_and_their_errors.md`](docs/12_implementations_and_their_errors.md) |
| Every fix, dated | [`docs/11_bugfix_diary.md`](docs/11_bugfix_diary.md) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Theoretical Foundation (Bottura)](#3-theoretical-foundation-bottura)
4. [Core Analysis Pipeline](#4-core-analysis-pipeline)
5. [Models & Data Structures](#5-models--data-structures)
6. [Ingest Module (File I/O)](#6-ingest-module-file-io)
7. [GUI Module](#7-gui-module)
8. [Validation & Testing](#8-validation--testing)
9. [Notebooks & Campaigns](#9-notebooks--campaigns)
10. [Parity with FFMM (C++ Golden Standard)](#10-parity-with-ffmm-c-golden-standard)
11. [Parity with the MMM Python analyzer](#11-parity-with-the-mmm-python-analyzer)
12. [Cross-Reference: Theory vs Implementation](#12-cross-reference-theory-vs-implementation)
13. [Known Limitations and Open Points](#13-known-limitations-and-open-points)
14. [Function Reference](#14-function-reference)

---

## 1. Executive Summary

The Rotating Coil Analyzer is a Python package for harmonic analysis of magnetic field measurements from rotating coil detectors at CERN. It transforms raw incremental flux signals into calibrated harmonic coefficients (Bn, An, bn, an) following the theoretical framework of Bottura's *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets* (MTA-IN-97-007).

### Key Design Principles

1. **No synthetic time**: All computations use measured acquisition time; never interpolated or generated
2. **Immutable data models**: All outputs use `@dataclass(frozen=True)` for safety and traceability
3. **Full provenance**: Source files, timestamps, options attached to every result
4. **FFMM parity**: Matches the official C++ reference implementation to **machine precision** (see [PARITY_REPORT.md](docs/PARITY_REPORT.md))
5. **Bottura-compliant**: Follows analytical theory without synthetic adjustments

### Package Structure

```
rotating_coil_analyzer/
  ├── analysis/           Core harmonic analysis pipeline
  │   ├── fourier.py         Per-turn DFT and harmonic summaries (dft_per_turn)
  │   ├── turns.py           Turn reshaping (1D -> 2D)
  │   ├── preprocess.py      Drift, di/dt corrections, preprocessing tags
  │   ├── kn_pipeline.py     Main Kn computation pipeline (HEART)
  │   ├── kn_head.py         Geometry-based Kn from measurement head CSV
  │   ├── merge.py           Abs/Cmp channel merge diagnostics (recommend_merge_choice)
  │   ├── kn_bundle.py       Provenance containers (KnBundle, MergeResult)
  │   └── utility_functions.py  High-level wrappers, plateau detection, diagnostics (eddy fitting lives in tools-for-data-analysis)
  ├── models/             Immutable data containers
  │   ├── catalog.py         Measurement catalog (filesystem discovery)
  │   ├── frames.py          In-memory segment data (SegmentFrame)
  │   ├── profile.py         Analysis configuration (AnalysisProfile)
  │   └── results.py         TurnQC, HarmonicsResult
  ├── ingest/             File discovery and readers
  │   ├── discovery.py       Measurement folder scanning
  │   ├── channel_detect.py  Heuristic channel identification
  │   ├── readers_streaming.py  Streaming (binary/text) reader
  │   └── readers_plateau.py    Plateau (DC) reader
  ├── gui/                8-panel Jupyter notebook GUI
  │   ├── app.py             Main application entry point (build_gui)
  │   ├── _dialogs.py        File/folder dialog helpers
  │   ├── harmonics.py       FFT & preprocessing panel
  │   ├── coil_calibration.py  Kn loading/computation panel
  │   ├── harmonic_merge.py    Channel merge & export panel
  │   ├── plots.py           Time-series exploration panel
  │   ├── plateau_detection.py Plateau detection panel (streaming supercycle)
  │   ├── physics_plots.py   Physics analysis panel (hysteresis, TF, Ld, eddy)
  │   ├── comparison.py      Measurement comparison panel
  │   └── log_view.py        HTML-based logging widget
  ├── validation/         Golden-standard validation tools
  │   └── golden_streaming.py  Streaming validation workflow (C++ parity)
  ├── cli.py              `rca-analyze`: folder + kn file -> CSV (bundled FCC-ee demo without arguments)
  └── tests/              unit + regression tests (run `python -m pytest`); tests/data = bundled demo data

notebooks/                Tool + parity-validation notebooks (repository top level, not package code)
docs/                     Theory-to-code chapters, guides, parity report and reference docs
golden_standards/         Local validation data (git-ignored, 8.7 GB)
```

---

## 2. Architecture Overview

### Data Flow

```
Raw Measurement Files
        │
        ▼
┌──────────────────────────────┐
│ INGEST: Discovery + Readers  │  Parameters.txt → MeasurementCatalog
│   StreamingReader (binary)   │  *.bin / *.txt → SegmentFrame
│   PlateauReader (DC text)    │  *_raw_measurement_data.txt → SegmentFrame
└──────────────────────────────┘
        │
        ▼  SegmentFrame (t, df_abs, df_cmp, I)
        │
┌──────────────────────────────┐
│ ANALYSIS: Kn Pipeline        │
│  1. dit (optional)           │  Reweight by I_mean/I_k
│  2. dri (optional)           │  Drift correction (legacy or weighted)
│  3. FFT                      │  2/Ns * FFT(flux)[1:H+1]
│  4. Kn calibration           │  C = (1/conj(Kn)) * Rref^k * F
│  5. rot (optional)           │  Rotate all harmonics by exp(-iφk)
│  6. cel (optional)           │  Center location from adjacent harmonics
│  7. fed (optional)           │  Feeddown via binomial expansion
│  8. nor (optional)           │  Normalize to 10^4 units
└──────────────────────────────┘
        │
        ▼  LegacyKnPerTurn (C_abs, C_cmp, phi, z, main_field)
        │
┌──────────────────────────────┐
│ POST-PROCESSING              │
│  merge_coefficients()        │  Select Abs/Cmp per harmonic order
│  safe_normalize_to_units()   │  Normalize to bn/an units
│  build_harmonic_rows()       │  Convert to DataFrame-ready dicts
│  plateau_summary()           │  Per-plateau statistics
│  (eddy fits: tools_for_data_analysis.fitting.eddy)
└──────────────────────────────┘
        │
        ▼  DataFrame with B_n(T), A_n(T), b_n(units), a_n(units)
```

### Pipeline Options

| Option | Step | Purpose | Mathematical Basis |
|--------|------|---------|-------------------|
| `dit` | Pre-FFT | Current-ramp correction | w_k = I_mean / I_k (quasi-static) |
| `dri` | Pre-FFT | Drift removal integration | flux = cumsum(df - mean(df)) - mean(cumsum(df)) |
| `rot` | Post-Kn | Main field alignment | C'_n = C_n * exp(-iφk), φ = arg(C_m)/m |
| `cel` | Post-rot | Center location | z_R = -C_{m-1} / ((m-1)*C_m) |
| `fed` | Post-cel | Feeddown correction | C'_n = Σ C(k,n) * z_R^(k-n) * C_k |
| `nor` | Post-fed | Normalization to units | c_n = 10^4 * C_n / B_m |

**Standard set**: `("dri", "rot", "cel", "fed")` — `nor` is handled post-merge by `safe_normalize_to_units()`

---

## 3. Theoretical Foundation (Bottura)

Reference: L. Bottura, *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets - Part I: Harmonics*, MTA-IN-97-007, CERN, 1997.

> **Canonical versions (2026-08-20):** the full equation transcription is the bibliography note
> `bibliography-review/magnetic_measurements/rotating_coil_harmonics/bottura1997_standard_analysis_field_quality_LHC_harmonics_notes.md`
> (its §11 maps equations to functions); the theory-to-code chapters are in [`docs/`](docs/README.md) and the
> equation → function table with the documented deviations is [`docs/10_bottura_cross_reference.md`](docs/10_bottura_cross_reference.md).
> The summary below is kept for orientation only.

### 3.1 Harmonic Field Expansion

**Eq. 1**: Complex field in the 2-D plane z = x + iy:
```
B(z) = By + iBx = Σ(n=1→∞) Cn * (z/Rref)^(n-1)
```

**Eq. 2-3**: Decomposition into normal and skew:
```
Cn = Bn + iAn           (Tesla, non-normalized)
cn = bn + ian           (units, normalized)
```

**Eq. 4**: Normalization for a magnet of main order m:
```
cn = (10^4 × Cn) / Bm = bn + ian
```

### 3.2 Reference Frame Transformations

**Eq. 5** (Translation/Feeddown):
```
Cn' = Σ(k=n→∞) [(k-1)! / ((n-1)!(k-n)!)] * Ck * (z/Rref)^(k-n)
```

**Eq. 6** (Rotation):
```
Cn' = Cn * e^(inθ)
```

### 3.3 Rotating Coil Measurement

**Eq. 12**: Flux through rotating coil:
```
Φ(θ) = L * Re[Σ(n=1) σn * Cn * e^(inθ)]
```

**Eq. 15**: Discrete Fourier Transform:
```
Ψn = Σ(k=1→N) Φk * e^(-2πi(n-1)(k-1)/N)
```

**Eq. AII.22**: Non-normalized harmonics from DFT:
```
Cn ≈ (Rref^(n-1) / Σn) * Ψ̄n
```

### 3.4 Drift Correction

**Eq. AII.12**: Voltage offset:
```
Voff = -(Σ ΔΦk) / t_{N+1}
```

**Eq. AII.14**: Corrected flux increments:
```
ΔΦk' = ΔΦk + Voff * Δtk
```

### 3.5 Main Field Reference Frame

**Eq. AIV.2-4**: Phase extraction and wrapping to [-π/2, π/2]:
```
θm = arg(Bm - iAm) / |Cm|
if θm > π/2:  θm -= π
if θm < -π/2: θm += π
```

**Eq. AIV.6**: Rotation to main field frame (ALL harmonics k=1..H):
```
Cn' = Cn * e^(inθm)
```

**Eq. AIV.8-9**: Normalization:
```
bn = (10^4 × Bn') / Bm'    (n = m+1 to Nm)
an = (10^4 × An') / Bm'
```

### 3.6 Center Location

**Eq. AIII.4**: For 2m-pole magnets (m > 1):
```
z = -Rref * C_{m-1} / ((m-1) * Cm)
```

**Eq. AIII.6**: Feeddown correction in centered frame:
```
Cn' = Σ(k=n→Nm) [(k-1)! / ((n-1)!(k-n)!)] * Ck * (z/Rref)^(k-n)
```

### 3.7 Dipole Center Location

Uses 20-pole cancellation (Eq. AIII.3): compensated harmonics n=10,11.
```
zR = -C_cmp[10] / (10 * C_cmp[11])
```
**Warning**: This uses high-order, weak signals — fragile at low current or poor SNR.

---

## 4. Core Analysis Pipeline

### 4.1 `compute_legacy_kn_per_turn()` — The Heart

**Location**: `analysis/kn_pipeline.py`

**Signature**:
```python
compute_legacy_kn_per_turn(
    *, df_abs_turns, df_cmp_turns, t_turns, I_turns,   # (n_turns, Ns) arrays
    kn, Rref_m, magnet_order,
    absCalib=1.0,
    options=("dri", "rot", "nor", "cel", "fed"),
    drift_mode="legacy",
    skew_main=False,
    eps_main=1e-20,
    legacy_rotate_excludes_last=False,
    dit_signed=False,
    max_zR=None,
    encoder_offset_rad=0.0,
    flip_signal_polarity=False,
) -> LegacyKnPerTurn
```

All arguments are keyword-only. `compute_from_profile(..., kn=, profile=)` is the same
pipeline driven by an `AnalysisProfile`. The function's own default `options` contains
`nor`; the standard set used by the notebooks and by `AnalysisProfile` leaves it out
(see section 2).

**Step-by-step pipeline** (per turn):

1. **dit** (if in options): Apply `di_dt_weights()` — reweight df by I_mean/I_k
   - Activation: |dI/dt| > 0.1 A/s AND |I_mean| > 10 A
   - `dit_signed=True`: FFMM C++ parity (ascending ramps only, positive current)
   - `dit_signed=False`: Both ramp directions (default)

2. **dri** (if in options): `integrate_to_flux(drift=True, drift_mode=...)`
   - Legacy mode (C++ exact): `flux = cumsum(df - mean(df)) - mean(cumsum(df))`
   - Weighted mode (Bottura AII.14): `df_k ← df_k - (Σdf/Σdt) * dt_k`

3. **FFT + scaling**:
   ```python
   f = (2.0 * FFT(flux)) / Ns   # legacy 2/N convention
   f = f[:, 1:H+1]              # drop DC, keep n=1..H
   ```

4. **Kn calibration**:
   ```python
   sens = (1 / conj(kn)) * Rref^[0..H-1]
   C = f * sens
   ```

5. **Snapshot**: Save C_abs_db, C_cmp_db (before rotation)

6. **rot** (if in options): Extract phase, rotate ALL harmonics k=1..H
   ```python
   φ = arg(C_abs[:, m-1])    # phase of main harmonic; skew_main: arg(-i * C_abs[:, m-1])
   # Wrap to [-π/2, π/2]  (turns within 0.05 rad of ±π/2 flagged in phi_near_wrap)
   φ_mech = φ / m
   C[:, k] *= exp(-i * φ_mech * (k+1))   for k=0..H-1
   ```
   - Default `legacy_rotate_excludes_last=False`: rotates ALL (Bottura AIV.6)
   - `legacy_rotate_excludes_last=True`: excludes the last harmonic H. This reproduces the
     loop `for (k = 1; k < nrHarmonics; ++k)` of the FFMM **native C++** analysis (production
     code, verified in source 2026-10-30, not only the SM18 export); use it for FFMM parity only
   - Two knobs act before this step: `flip_signal_polarity` negates all C right after kn (before
     the `C_abs_db` snapshot); `encoder_offset_rad` is a pre-rotation `exp(-i k offset)`.
     Both are described in `docs/06_corrections_rot_cel_fed.md`

7. **cel** (if in options):
   - Dipole (m=1): `zR = -C_cmp[:,9] / (10 * C_cmp[:,10])` (n=10,11 from compensated)
   - Quadrupole+ (m≥2): `zR = -C_abs[:,m-2] / ((m-1) * C_abs[:,m-1])` (robust)
   - `max_zR` clamp: if |zR| > max_zR, set zR=0 and flag in `zR_clamped`
   - `cel_applied` (per turn) says where it was actually applied; dipole with H < 11 → not applied, reported in `warnings`

8. **fed** (if in options): Binomial feeddown expansion
   ```python
   C'[n] = Σ(k=n→H-1) C(k,n) * zR^(k-n) * C[k]
   ```

9. **nor** (if in options): `C_units = C / main_field * 10000`

### 4.2 Post-Pipeline Functions

**`merge_coefficients()`**: Select Abs or Cmp per harmonic order
- Function default `abs_main_cmp_others`; `AnalysisProfile.merge_mode` default `abs_upto_m_cmp_above`
- Modes: `abs_all`, `cmp_all`, `abs_main_cmp_others`, `abs_upto_m_cmp_above`, `custom`
- Main harmonic always forced to Abs

**`safe_normalize_to_units()`**: Post-merge normalization
- `C_units = 10^4 * C / B_m`
- Returns `(C_units, ok_mask)` where ok = main field above threshold

**`recommend_merge_choice()`**: Diagnostic recommendation
- MAD-based noise estimation per channel per order
- Prefers Cmp if noise_cmp < 0.90 × noise_abs
- Flags mismatch > 50× noise
- Lives in `analysis/merge.py` (returns `MergeDiagnostics`)

### 4.3 Utility Functions

**Plateau Detection**:
- `compute_block_averaged_range()`: 10 blocks of ~100 samples, robust to ADC noise
- `detect_plateau_turns()`: Three rules — I_range < threshold, starts on plateau, ends on plateau
- `classify_current()`: Labels by cycle stage (zero/pre-ramp/injection/flat)

**Pipeline Wrapper**:
- `process_kn_pipeline()`: One-call wrapper: kn → merge → normalize
- `build_harmonic_rows()`: Converts to DataFrame-ready dicts with B_n/A_n/b_n/a_n columns

**Eddy-Current Fitting**: not in this package any more — `eddy_model`, `double_eddy_model`,
`triple_eddy_model`, `validate_eddy_model_selection`, `EddyFitResult`, `fit_eddy_per_run`
live in `tools_for_data_analysis.fitting.eddy` (no re-export here).

**Diagnostics**:
- `diagnose_cel_fed()`: Runs pipeline with/without cel+fed, recommends SAFE/UNSAFE/MIXED
- `diagnose_fdi_transitions()`: Detects stuck FDI channels at plateau boundaries

---

## 5. Models & Data Structures

All models use `@dataclass(frozen=True)` (immutable).

### 5.1 MeasurementCatalog

Filesystem-independent catalog of a measurement folder:
- `root_dir`, `parameters_path`, `samples_per_turn`, `shaft_speed_rpm`
- `enabled_apertures`, `segments: List[SegmentSpec]`, `runs: List[str]`
- `segment_files: Dict[(run_id, ap_id, seg_id) → Path]`

### 5.2 SegmentFrame

In-memory representation of one loaded segment:
- `source_path`, `run_id`, `segment`, `samples_per_turn`, `n_turns`
- `df: pd.DataFrame` with columns: `t`, `df_abs`, `df_cmp`, `I`, (optional: `plateau_id`, `plateau_step`, `plateau_I_hint`)
- **Invariant**: All float64, time from file (never synthetic), trimmed to integer turns

### 5.3 AnalysisProfile

Frozen pipeline configuration:
- Required: `magnet_order`, `r_ref_m`, `samples_per_turn`, `shaft_speed_rpm`
- Optional (default): `options` (`dri`, `rot`), `drift_mode` (`legacy`), `merge_mode` (`abs_upto_m_cmp_above`),
  `legacy_rotate_excludes_last` (False), `min_main_field_T` (1e-20), `abs_calib` (1.0), `l_coil_m` (None),
  `skew_main` (False), `dit_signed` (False), `max_zR` (None), `encoder_offset_rad` (0.0),
  `flip_signal_polarity` (False)
- `from_catalog(cat, **overrides)` takes `magnet_order` (magnetOrder, else magnetType),
  `r_ref_m` (refRadius) and `skew_main` (magnetNormSkew) from `Parameters.txt`;
  raises `ValueError` if `magnet_order` or `r_ref_m` is known neither from the file
  nor from the overrides (no silent m=1 / 17 mm since 2026-10-30).

### 5.4 LegacyKnPerTurn

Full pipeline result per turn:
- `orders`: harmonic orders 1..H
- `C_abs`, `C_cmp`: (n_turns, H) complex arrays (after all corrections)
- `C_abs_db`, `C_cmp_db`: Snapshot after kn, before rot/cel/fed/nor
- `phi_out_rad`, `phi_bad`, `zR`, `zR_clamped`, `z_m`, `x_m`, `y_m`: Rotation and center location
- `encoder_offset_rad`: the offset that was applied
- `main_field`, `main_field_db`: Main harmonic value
- `I_mean_A`, `dI_dt_A_per_s`, `duration_s`, `time_median_s`: Diagnostics
- `dit_applied`, `cel_applied`, `phi_near_wrap`: per-turn flags (None if the option was off)
- `warnings`: diagnostics (cel not applied, |zR| ≥ 1, |C_m| > 20 T, wrap boundary); `process_kn_pipeline` re-emits them as `UserWarning`

### 5.5 KnBundle / MergeResult

Full provenance containers:
- `KnBundle`: SegmentKn + source_type + source_path + timestamp + connection strings
- `MergeResult`: C_merged + per_n_source_map + kn_provenance + merge_mode

---

## 6. Ingest Module (File I/O)

### 6.1 Discovery

`MeasurementDiscovery.build_catalog()`:
1. Find Parameters.txt (up to 2 parent levels)
2. Parse samples_per_turn, shaft_speed_rpm, magnet_order, enabled_apertures
3. Parse FDIs tables (segment → FDI channel mapping)
4. Discover segment files (streaming: `*corr_sigs*.bin`; plateau: `*_raw_measurement_data.txt`)

### 6.2 Streaming Reader

`StreamingReader.read()`:
1. Auto-infer binary format (dtype × n_currents candidates)
2. Validate: strictly increasing time, dt matches nominal
3. Detect flux channels (larger robust_range = absolute)
4. Detect current channel (max robust_range among candidates)
5. Build SegmentFrame with raw time

### 6.3 Plateau Reader

`PlateauReader.read()`:
1. Find all plateau files matching representative pattern
2. Per-plateau: load, extract raw time, trim to integer turns
3. Concatenate, add plateau_id/step/I_hint metadata
4. Detect channels, build SegmentFrame

### 6.4 Channel Detection

`channel_detect.py`:
- `robust_range(x)`: p99.5 - p0.5 (robust to outliers)
- `detect_flux_channels()`: Larger robust_range → absolute
- `detect_current_channel()`: Max robust_range among candidates
- **Hard constraint**: No synthetic time in any reader

---

## 7. GUI Module

8-panel Jupyter notebook GUI with ipywidgets:

| Tab | Panel | Purpose |
|-----|-------|---------|
| 0 | Catalog | Load measurement folder, browse runs/segments |
| 1 | Plateau Detection | Detect current plateaus in streaming supercycle data using block-averaged current range with 3-rule detection |
| 2 | Harmonics | FFT computation, preprocessing options, amplitude/phase plots |
| 3 | Coil Calibration | Load/compute Kn from segment TXT or head CSV |
| 4 | Harmonic Merge | Apply Kn, select Abs/Cmp per order, normalize, export CSV |
| 5 | Raw Signal Plots | Interactive time-series exploration (no synthetic resampling) |
| 6 | Physics Plots | Hysteresis loops, transfer function (B/I), differential inductance (Ld = dB1/dI), eddy-current settling analysis |
| 7 | Comparison | Compare two exported CSV measurements side-by-side with statistical analysis |

**Plateau Detection** (`gui/plateau_detection.py`): Uses `compute_block_averaged_range`, `detect_plateau_turns`, `classify_current`, and `find_contiguous_groups` from `utility_functions.py`. Block-averaged range (10 blocks of ~100 samples) filters ADC noise. Three detection rules: (a) I_range < threshold, (b) start on plateau, (c) end on plateau.

**Physics Plots** (`gui/physics_plots.py`): Uses `plateau_summary`, `plot_hysteresis` and `build_run_averages` from `utility_functions.py`; the eddy fit (`fit_eddy_per_run`) comes from the optional companion package `tools-for-data-analysis` (extra `[eddy]`, imported lazily; the tab works without it, only the eddy fit is disabled). Provides hysteresis loop visualization (B vs I with up/down ramp separation), transfer function (B/I), differential inductance (Ld = dB1/dI from staircase plateau data), and eddy-current settling analysis with exponential fits.

**Comparison** (`gui/comparison.py`): Uses `diff_sigma` (and the level statistics built with `compute_level_stats`) from `utility_functions.py`. Loads two exported CSV files and computes per-operating-point differences with propagated uncertainties and sigma significance.

**Architecture**: Shared state dict passed via closures; callable getters for lazy evaluation. Debouncing (250ms) for VS Code stability. HTML-based logging (no ipywidgets.Output stacking).

---

## 8. Validation & Testing

### 8.1 Test Suite

- 170 tests in 23 files (2026-11-03), all passing: `python -m pytest` (`testpaths` in `pyproject.toml`)
- A fresh clone without `golden_standards/` and without the `[eddy]` extra: 166 pass, 4 are expected skips
- `.gitlab-ci.yml` runs the suite on the Acc-Py templates (no publish job until the CERN OSPO process is done)
- Fixes and the tests that pin them: `docs/11_bugfix_diary.md`

| Area | Test files |
|------|-----------|
| Merge, Kn and provenance | `test_harmonic_merge`, `test_kn_bundle`, `test_kn_loader_and_merge`, `test_mh_csv_compensation_scheme`, `test_kn_head_csv_vs_reference` |
| Pipeline correctness and guards | `test_safety_guards`, `test_kn_synthetic_pipeline`, `test_weighted_drift_known_answer`, `test_review_fixes_2026_10`, `test_turns_fourier` |
| Preprocessing and time policy | `test_preprocess`, `test_time_policy` |
| Ingest and channels | `test_channel_detect`, `test_reader`, `test_discovery`, `test_plateau`, `test_plateau_hv_format`, `test_streaming_utilities` |
| Profile and utilities | `test_analysis_profile`, `test_new_utility_functions` |
| GUI widgets and events | `test_gui_plot_widgets`, `test_plot_debounce`, `test_plot_clear_pattern`, `test_preview` |

### 8.2 Golden-Standard Validation

`validation/golden_streaming.py`:
- Discover golden streaming datasets and auto-match per-segment Kn files
- Run the full per-turn pipeline, compare per-harmonic against the C++ reference
  export (robust fixed-width parsing, time-aligned with positional fallback)
- Canonical output schema matching the legacy analyzer format

---

## 9. Notebooks & Campaigns

### 9.1 Inventory

This is the magnet-agnostic library repo. Only example/tool and parity-validation
notebooks live here; the magnet measurement campaigns moved to their own repos.

| Group | Notebooks | Type |
|-------|-----------|------|
| Tools | `tools/analysis_gui.ipynb`, `tools/kn_from_mh_csv.ipynb` | GUI demo, Kn-from-CSV utility |
| LIU BTP8 | `LIU_BTP8/2019-07-17/b3_sextupole.ipynb`, `parity_validation.ipynb` | Quadrupole (m=2), golden standard |
| SM18 | `SM18/2024-12-04_parity/parity_validation.ipynb` | Dipole (m=1), streaming parity |
| Buckley steerer | `Buckley_steerer/2026-06-23_degauss_parity/parity_validation.ipynb`, `gui_walkthrough.ipynb` | Degauss parity (FFMM), GUI walkthrough |

All paths are relative to the top-level `notebooks/` folder. Each campaign folder carries its own `report.md`.

Magnet campaign notebooks (SPS MBB, LEAR MC62, ...) live in the sibling repos
`mbb-data-analysis`, `mc62-data-analysis`, `mba-data-analysis`, which depend on
this package. Programmatic notebook construction in those repos uses
`tools_for_data_analysis.helpers.notebook` (mandatory — prevents encoding/f-string bugs).
For a hands-on API walkthrough see `docs/GUIDE.md`.

### 9.2 Magnet Parameters

Configurations used by the notebooks; the measured values and their provenance are owned by the
sibling repos (`results.md` there), not by this table.

| Magnet | R_ref [mm] | Order | Samples/Turn | Standard OPTIONS |
|--------|-----------|-------|--------------|-----------------|
| MC62 | 33 | 1 | 1024 (1 Hz) / 512 (2-4 Hz) | (dri, rot) — cel/fed UNSAFE |
| BTP8 | 59 | 2 | 512 | (dri, rot, cel, fed) — SAFE |
| SM18 | 50 | 1 | 512 | (dri, rot) + legacy_rotate_excludes_last=True |
| SPS MBB | 20 | 1 | 1024 | (dri, rot) or (dri, rot, cel, fed) |

### 9.3 Common Notebook Pattern

Every analysis notebook follows:
```
1. Configuration (paths, MAGNET_ORDER, R_REF, OPTIONS)
2. Load Kn calibration (load_segment_kn_txt)
3. Discover runs (discover_runs)
4. cel/fed safety diagnostic (diagnose_cel_fed)
5. Process pipeline (process_kn_pipeline or compute_legacy_kn_per_turn)
6. Merge & normalize (merge_coefficients + safe_normalize_to_units)
7. Plateau statistics (plateau_summary, build_run_averages)
8. Visualization (plot_hysteresis, harmonic spectra)
```

### 9.4 Key Findings from Notebooks

Orientation only; the numbers are owned by `docs/PARITY_REPORT.md` and by the `results.md` of the magnet repos.

- **Parity**: Python matches C++ reference at **machine precision** (see [PARITY_REPORT.md](docs/PARITY_REPORT.md))
- **Reproducibility**: Turn-to-turn CV < 0.5%
- **cel/fed fragility**: Dipole center location (n=10,11 compensated) is UNSAFE at low current/SNR
- **Quadrupole cel/fed**: Robust (uses C_{m-1}/C_m ratio, both strong signals)
- **Eddy settling**: 8-15 turns sufficient for exponential decay (SM18/BTP8), tau ~ 33-40 s for MC62 (iron relaxation)
- **4 Hz rotation** (MC62): 39% lower B1 noise, 49% lower b2 noise vs 1 Hz; rolling-std plateau detection required at fast rotation + slow ramp
- **Marusov 2D reconstruction**: Temporal DFT decomposition validated at machine epsilon (K=M identity). Per-turn averaging error is (T/τ)²/24 ≈ 2×10⁻⁶ (sub-ppm) — the 0.7% per-turn field change cancels by averaging symmetry (only curvature contributes). Phase coupling between full-stream and two-step validated: n=1 follows 3.6×k/M exactly (constant prefactor ≈ 2π/√3 at all k), impact on B1 ≈ 5 ppm. Temporal filtering (K~50) provides noise reduction. For MC62 4 Hz, Marusov confirms the per-turn pipeline is adequate rather than fixing a problem. Complete theory and results: `marusov_guide.md` in `mc62-data-analysis/notebooks/2026-03-04_4Hz_precycle/rotating_coil/`.
- **Multi-tau eddy fitting**: AICc selects 3-tau for B1/b2, 1-tau for b3 on MC62. Model validation residuals noise-limited (~3-4 × 10⁻⁴)

---

## 10. Parity with FFMM (C++ Golden Standard)

### 10.1 FFMM Pipeline Summary

The FFMM (Flexible Framework for Magnetic Measurements) C++ implementation:

```
1. DIT: if options contains "dit" && dI/dt > 0.1 && I_mean > 10:
      df *= I_mean / I_instantaneous
2. DRI: if options contains "dri":
      flux = cumsum(df - mean(df)) - mean(cumsum(df))
   else:
      flux = cumsum(df)
3. FFT: f = 2 * FFT(flux) / N; f = f[1:H+1]
4. KN:  C = (1/conj(kn)) * Rref^k * f
5. ROT: if "rot": C[k] *= exp(-i * φ * k) for k=1..H-1   (loop k < nrHarmonics)
6. CEL: if "cel": zR = -C[m-2]/((m-1)*C[m-1]) [or C_cmp[9]/(10*C_cmp[10]) for dipole]
7. FED: if "fed": C'[n] = Σ C(k,n) * zR^(k-n) * C[k]
8. NOR: if "nor": C /= B_main * 10000
```

### 10.2 Exact Parity Points

| Step | FFMM (C++) | Python Analyzer | Match? |
|------|-----------|-----------------|--------|
| dit activation | crr > 0.1 && cm > 10 | \|dI/dt\| > 0.1 && \|I_mean\| > 10 | **Yes** (dit_signed=True for FFMM parity) |
| dit weights | cm / c (element-wise) | I_mean / I_k | **Yes** |
| dri legacy | cumsum(df-mean(df)) - mean(cumsum(df)) | Same formula | **Yes** |
| FFT scaling | 2 * FFT / N | 2.0 * FFT / Ns | **Yes** |
| Kn application | 1/conj(kn) * Rref^ki | 1/conj(kn) * Rref^[0..H-1] | **Yes** |
| Phase wrapping | if > π/2: -= π; if < -π/2: += π | Same logic | **Yes** |
| Rotation range | k=1..nrHarmonics-1 (C++ loop, skips order H) | k=1..H (default, all); `legacy_rotate_excludes_last=True` reproduces the C++ loop | **Only with `legacy_rotate_excludes_last=True`** |
| CEL dipole | C_cmp(9) / (10*C_cmp(10)) | C_cmp[:,9] / (10*C_cmp[:,10]) | **Yes** |
| CEL quadrupole | C_abs(m-2) / ((m-1)*C_abs(m-1)) | C_abs[:,m-2] / ((m-1)*C_abs[:,m-1]) | **Yes** |
| Feeddown | Σ C(k,n) * zR^(k-n) * C[k] | Same binomial formula | **Yes** |
| Normalization | C / main_comp * 10000 | Same formula | **Yes** |

### 10.3 Known Differences

| Aspect | FFMM (C++) | Python Analyzer | Impact |
|--------|-----------|-----------------|--------|
| dit sign handling | Only activates when crr > 0 AND cm > 0 | Default: activates on both ramp directions; `dit_signed=True` for FFMM parity | Use `dit_signed=True` for exact parity |
| NaN protection in dit | Silent skip if c_dit has NaN/Inf | Same behavior (skip) | None |
| Factorial implementation | `int` (potential 32-bit overflow for n>12) | Python arbitrary precision | None practical (n≤15) |
| Channel merge | Separate output (no merge) | Configurable merge strategy | User chooses |
| Kn file format | `%Le` (long double) in C++ | Python float64 | Negligible precision difference |
| Impedance gain | Not explicitly handled | Not in pipeline (done upstream) | Consistent |

### 10.4 Validated Parity Results

Machine-precision parity has been achieved on all golden standard datasets. See [PARITY_REPORT.md](docs/PARITY_REPORT.md) for comprehensive results:

- **SM18**: B1 diff = 1.82e-12 T on 285,095 turns (streaming, all segments; `legacy_rotate_excludes_last=True`)
- **LIU BTP8**: B1/A1 at ~1e-18 T, B2 at ~5e-9 T (brute-force turn-selection recovery, 222 turns)
- **LEAR MC62**: Machine-precision B_main on streaming tests 03/04 (plateau-averaged comparison within µT)

---

## 11. Parity with the MMM Python analyzer

### 11.1 MMM Python analyzer: pipeline summary

An independent Python analyzer used in the MMM section (`rotcoil_lib.py`) implements:

```
1. di/dt correction (optional)
2. Drift correction
3. Integration: flux = cumsum(df)
4. Impedance gain: flux *= (Z_coil + Z_inst) / Z_inst
5. FFT: f_n = (2/N) * FFT(flux)[n]
6. Kn calibration: C_n = (rRef^(n-1) / conj(kn_n)) * f_n
7. Rotation: C *= exp(-i * φ_mech * n)
8. Center localization (optional)
9. Feeddown correction (optional)
10. Revolve_Z (optional, Cmp only)
11. Normalization: C_units = (C / B_main) * 1e4
```

### 11.2 Parity Points

| Feature | MMM Python analyzer | This package | Status |
|---------|----------|-----------------|--------|
| FFT formula | (2/N) * FFT(flux)[n] | (2/Ns) * FFT(flux)[1:H+1] | **Match** |
| Kn formula | rRef^(n-1) / conj(kn_n) * f_n | (1/conj(kn)) * Rref^k * f | **Match** |
| Rotation range | All harmonics k=1..H | All k=1..H (default) | **Match** |
| CEL formula | -rRef/(m-1) * C_{n-1}/C_n | Same | **Match** |
| Feeddown | Σ C(k,n) * (Dz/rRef)^(k-n) * C_k | Same binomial | **Match** |
| Normalization | (C / B_main) * 1e4 | 10^4 * C / B_m | **Match** |

### 11.3 MMM-analyzer features not in this package

| Feature | Description | Needed? |
|---------|-------------|---------|
| Impedance gain correction | flux *= (Z_coil + Z_inst) / Z_inst | **Not needed** — handled upstream by hardware/calibration |
| Revolve_Z | Flip signs of even b_n and odd a_n (Cmp only) | **Not needed** — convention-dependent, not physics |
| Encoder offset | Phi -= 2π/encStep post-rotation | **Not needed** — absorbed in Kn calibration |
| Ext channel | Third (external) coil channel | **Partially** — Python supports ext in KnBundle but not in main pipeline |
| Pulsed mode | Time-resolved harmonic analysis | **Not yet** — Python is DC/streaming only |
| Multi-segment | IntegralSegment, CentralSegment weighted merge | **Not yet** — currently single-segment only |

---

## 12. Cross-Reference: Theory vs Implementation

The equation-by-equation map from Bottura's procedure to the functions, with the three documented
deviations, is [`docs/10_bottura_cross_reference.md`](docs/10_bottura_cross_reference.md). The three
deviations are: (1) no forward/backward rotation averaging (AII.15-16); (2) dipole centring uses the
linearised 20-pole formula (AIII.3) instead of the 16-pole polynomial and cost function; (3)
`legacy_rotate_excludes_last=True` exists for FFMM parity only. The earlier "fully compliant" audit
table that stood here (2026-03-12) was removed because it contradicted these deviations.

What the package does not do, by design: no synthetic time, no interpolation or resampling of the
signals, no FFT windowing, no smoothing, no estimated calibration values.

---

## 13. Known Limitations and Open Points

Current as of 2026-11-03. The first four rows are the open software gaps; the rest are scope limits.

| Point | State | Where it is described |
|-------|-------|-----------------------|
| Dipole centring fragility | `cel` for m=1 uses the compensated n=10, 11 harmonics and is UNSAFE at low current or poor SNR. `diagnose_cel_fed()` classifies SAFE/UNSAFE/MIXED, `max_zR` clamps the offset (flag `zR_clamped`), `cel_applied` and `warnings` report what actually ran. There is **no automatic fallback** that disables cel/fed | `docs/06`, `docs/correction_options_reference.md` |
| Weighted drift (`drift_mode="weighted"`, Bottura AII.14) | Implemented and covered by a known-answer test; legacy stays the default. Not validated on streaming supercycle data (the legacy mode assumes uniform sampling) | `docs/03`, diary BF-2026-11-02-02 |
| External (third) coil channel | `KnBundle` carries it, the pipeline does not process it | `docs/12` |
| Forward/backward rotation averaging | Not implemented (deviation 1); each rotation is analysed independently | `docs/10` |
| Multi-segment merge | Not implemented (the MMM analyzer merges Integral and Central segments weighted by B*L); one segment per pipeline call | `docs/12` |
| Pulsed (time-resolved) mode | Not implemented; the package is DC and streaming only | `docs/12` |
| Impedance gain `(Z_coil + Z_inst)/Z_inst` | Not applied; assumed handled by the hardware or the calibration upstream | `docs/12` |
| `dit` on NaN/Inf | The correction is skipped silently, as in FFMM | `docs/03` |

---

## 14. Function Reference

### 14.1 Core Pipeline (`analysis/kn_pipeline.py`)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compute_legacy_kn_per_turn` | `(*, df_abs_turns, df_cmp_turns, t_turns, I_turns, kn, Rref_m, magnet_order, absCalib, options, drift_mode, skew_main, eps_main, legacy_rotate_excludes_last, dit_signed, max_zR, encoder_offset_rad, flip_signal_polarity)` → `LegacyKnPerTurn` | Full per-turn pipeline |
| `compute_from_profile` | `(*, df_abs_turns, df_cmp_turns, t_turns, I_turns, kn, profile)` → `LegacyKnPerTurn` | Profile-based wrapper |
| `merge_coefficients` | `(*, C_abs, C_cmp, magnet_order, mode, per_order_choice)` → `(C_merged, choice)` | Channel merge |
| `safe_normalize_to_units` | `(C, magnet_order, *, absCalib, skew_main, min_main_field)` → `(C_units, ok_mask)` | Post-merge normalization |
| `load_segment_kn_txt` | `(path)` → `SegmentKn` | Load Kn from text file |
| `bucked_cmp_orders` | `(kn, choice, *, rel_tol)` → `list[int]` | Orders taken from Cmp although the bucked Cmp channel cancels them (merge guard) |

### 14.2 Preprocessing (`analysis/preprocess.py`)

| Function | Purpose |
|----------|---------|
| `di_dt_weights(t_turns, I_turns, *, min_slope_A_per_s, min_mean_I_A, eps_I_A, signed)` | Compute dit correction weights (`DiDtResult`) |
| `apply_di_dt_to_channels(df_abs_turns, df_cmp_turns, t_turns, I_turns, **kwargs)` | Apply the dit weights to both channels (returns both arrays and the `DiDtResult`) |
| `integrate_to_flux(df_turns, *, drift, drift_mode, t_turns)` | Integrate df to flux with optional drift (`DriftResult`) |
| `estimate_linear_slope_per_turn(t, y)` | Least-squares slope per turn |
| `provenance_columns(n_turns, *, ...)` | Build provenance metadata |
| `format_preproc_tag(...)`, `append_tag_to_path(...)` | File-name tag describing the preprocessing options |

### 14.3 Utility Functions (`analysis/utility_functions.py`)

| Function | Purpose |
|----------|---------|
| `process_kn_pipeline(...)` | One-call wrapper: kn → merge → normalize |
| `build_harmonic_rows(...)` | Convert pipeline output to DataFrame rows |
| `discover_runs(run_dir, pcb_label)` | Parse run filenames |
| `plateau_summary(df, n_last, ...)` | Per-plateau statistics |
| `build_run_averages(df)` | Per-run b3 aggregation |
| `detect_plateau_turns(...)` | Three-rule plateau detection |
| `compute_block_averaged_range(I, Ns)` | Block-averaged current range |
| `classify_current(I, thresholds)` | Label current by cycle stage |
| `diagnose_cel_fed(...)` | CEL/FED safety diagnostic |
| `FdiTransitionCheck` | Diagnostic result dataclass for FDI stuck-channel detection |
| `diagnose_fdi_transitions(...)` | Detect FDI stuck-channel issues at plateau transitions |
| *(eddy models / fitting)* | Moved to `tools_for_data_analysis.fitting.eddy` (`eddy_model`, `double_eddy_model`, `triple_eddy_model`, `validate_eddy_model_selection`, `EddyFitResult`, `fit_eddy_per_run`) |
| `find_contiguous_groups(mask)` | Find contiguous True-groups in boolean mask |
| `ba_table_from_C(C, orders)` | Complex → B/A DataFrame |
| `mixed_format_table(C_merged, C_units, ...)` | Bottura 3.7 mixed format |
| `mad_sigma_clip(df, col, n_sigma)` | MAD-based outlier removal |
| `plot_hysteresis(ax, summ, ...)` | Hysteresis loop plot |
| `compute_level_stats(df, label)` | Mean/std of I, B1, b2, b3, TF per operating point |
| `diff_sigma(stats1, stats2, key)` | Difference with propagated error and sigma significance |
| `SPS_CURRENT_THRESHOLDS` | Default current thresholds dict for SPS cycle classification |

### 14.4 Ingest (`ingest/`)

| Function/Class | Purpose |
|----------------|---------|
| `MeasurementDiscovery.build_catalog(dir)` | Scan folder → MeasurementCatalog |
| `StreamingReader.read(path, ...)` | Binary/text → SegmentFrame |
| `PlateauReader.read(path, ...)` | DC text files → SegmentFrame |
| `robust_range(x)` | p99.5 - p0.5 robust range |
| `detect_flux_channels(mat, ...)` | Auto-detect abs/cmp columns |
| `detect_current_channel(mat, ...)` | Auto-detect current column |

### 14.5 Kn from Head Geometry (`analysis/kn_head.py`)

| Function | Purpose |
|----------|---------|
| `compute_head_kn_from_csv(csv_path, ...)` | Compute per-coil kn from head geometry |
| `compute_segment_kn_from_head(head, *, connections)` | Combine coils via connections |
| `parse_connection(conn)` | Parse "1.1-1.3+2*1.2" syntax |
| `write_segment_kn_txt(segment_kn, path)` | Write a segment kn TXT file |

### 14.6 Other Analysis Modules

| Function | Module | Purpose |
|----------|--------|---------|
| `split_into_turns(...)` → `TurnBlock` | `analysis/turns.py` | Reshape a 1-D stream into (n_turns, Ns) |
| `dft_per_turn(...)`, `summarize_harmonics(...)` | `analysis/fourier.py` | Per-turn DFT and harmonic summaries |
| `recommend_merge_choice(...)` → `MergeDiagnostics` | `analysis/merge.py` | Abs/Cmp recommendation per order |
| `run_golden_folder(...)`, `compare_units_table(...)` | `validation/golden_streaming.py` | Golden-standard comparison |

---

## Appendix A: Theory References

### Bottura (Primary Reference)
- **Bottura, L.** — *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets* (MTA-IN-97-007)
- Defines the complete harmonic analysis framework: FFT, Kn, rotation, CEL, feeddown, normalisation
- All pipeline steps follow this reference exactly

### Marusov (2013) — 2D Fourier Decomposition
- **Marusov, V.** — *Measurement of a Time-Periodic Magnetic Field by Rotating Coil*, NIM-A 711, pp. 121-123
- Extends the per-turn FFT analysis to a 2D decomposition in both angle (spatial, index n) and time (temporal, index k)
- The 2D coefficient σ_{n,k} captures the temporal evolution of each spatial harmonic
- Truncating to K temporal modes provides bandwidth-limited reconstruction
- **Per-turn averaging error**: B_avg - B(t_mid) = B''·T²/24 = A·(T/τ)²/24. The linear field change (first derivative) cancels by the symmetry of the averaging integral; only the curvature (second derivative) contributes. For MC62 4 Hz: (T/τ)²/24 ≈ 2×10⁻⁶ — sub-ppm.
- **Phase coupling validation**: Two-step ignores exp(-2πi·ks/(MNs)). For n=1, measured |Δσ|/|σ| = 3.6×k/M with constant prefactor (≈ 2π/√3) at all k — independently validates both implementations.
- Essential when τ/T < 10 (averaging error grows to > 4×10⁻⁵ of A); acts as noise filter when τ/T >> 10
- Our implementation applies temporal DFT to validated pipeline output C_n(j), guaranteeing K=M identity
- Complete theory, derivations and results: `marusov_guide.md` in `mc62-data-analysis/notebooks/2026-03-04_4Hz_precycle/rotating_coil/`

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Abs** | Absolute channel — single rotating coil (larger signal) |
| **Cmp** | Compensated channel — bucking/compensated coil pair (lower main harmonic) |
| **Ext** | External channel — third coil (optional) |
| **Bn** | Normal component of n-th harmonic (Tesla) |
| **An** | Skew component of n-th harmonic (Tesla) |
| **bn** | Normal component (units = 10^-4 of main field) |
| **an** | Skew component (units) |
| **Kn/Sn** | Complex sensitivity coefficient per harmonic (from coil geometry) |
| **Rref** | Reference radius (meters) |
| **CEL** | Center Location — compute coil center offset |
| **FED** | Feeddown correction — adjust for displaced center |
| **DRI** | Drift removal integration |
| **DIT** | dI/dt correction — current ramp reweighting |
| **ROT** | Rotation — align main field to real axis |
| **NOR** | Normalization — express in relative units |
| **zR** | Dimensionless center offset = z / Rref |
| **MAD** | Median Absolute Deviation (robust noise estimator) |
| **FDI** | Fast Digital Integrator (hardware) |
| **TF** | Transfer Function = B_main / I (T/kA) |

## Appendix C: File Format Reference

### Kn Segment TXT (4 or 6 columns)
```
Kn_abs_re  Kn_abs_im  Kn_cmp_re  Kn_cmp_im  [Kn_ext_re  Kn_ext_im]
```
One row per harmonic n=1..H.

### Raw Measurement Data (Plateau)
```
time  df_channel1  df_channel2  current1  [current2  ...]
```
Whitespace-separated, no header. One file per current level.

### Streaming Binary
Float32 or float64, little-endian. Columns: time, df_abs, df_cmp, [currents].

### Parameters.txt
Key-value format with TABLE{...} for structured data:
```
Parameters.Measurement.samples = 1024
Parameters.Measurement.v = -60.0
Measurement.AP1.FDIs = TABLE{NCS\t0\t2\t0.7\nCS\t1\t3\t0.7}
```

---

*End of documentation. Updated 2026-11-03; checked against the code (signatures, field lists, test layout, module list) and aligned with `docs/`.*
