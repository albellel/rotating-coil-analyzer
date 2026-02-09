# Analysis Pipeline Reference

## Overview

The rotating coil analysis pipeline transforms raw acquisition data (incremental flux samples, encoder counts, and current measurements) into calibrated magnetic field harmonics. It is designed for CERN accelerator magnets across all machine complexes (LHC, SPS, PS, PSB, transfer lines, test benches such as SM18).

The core implementation in `rotating_coil_analyzer/analysis/kn_pipeline.py` follows the legacy C++ analyzer semantics exactly (`ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp`).

**Inputs:**
- Incremental flux per sample, absolute and compensated channels (Wb)
- Encoder counts (for timing) or measured timestamps
- Current per sample (A)
- Kn calibration coefficients (complex, per harmonic order)
- Magnet parameters: order *m*, reference radius *R_ref*

**Outputs:**
- Calibrated complex harmonics C_n per turn, per channel (ABS/CMP)
- Derived quantities: rotation angle, centre location, main field
- Per-turn metadata: mean current, dI/dt, duration

## Pipeline Steps

The pipeline always executes in the following fixed order, regardless of how the option tokens are specified. Each step is controlled by a token in the `options` parameter.

### 1. di/dt Current-Ramp Correction (`dit`)

Reweights incremental flux samples to compensate for the changing current during a ramp. A turn is corrected when |dI/dt| > 0.1 A/s and |mean(I)| > 10 A.

**Formula:**

```
w_k = |I_mean| / I_k
df_corrected_k = df_k * w_k
```

The numerator uses |I_mean| so that the correction works for both positive and negative ramps. The denominator keeps the sign of I_k.

**Code:** `preprocess.py:139-207` (`di_dt_weights`), applied at `kn_pipeline.py:250-252`.

### 2. Drift Correction + Integration to Flux (`dri`)

Integrates the incremental signal to obtain flux, with optional drift removal. Two modes are supported.

#### Legacy mode (default)

Matches the C++ analyzer exactly. Reference: `MatlabAnalyzerRotCoil.cpp` line ~127.

```
flux = cumsum(df - mean(df)) - mean(cumsum(df))
```

Note: the subtracted term is `mean(cumsum(df))`, i.e. the mean of the *original* cumsum before drift correction. This is a subtle but critical detail.

**Code:** `preprocess.py:265-287`.

**Theory:** Bottura Eq. AII.14.

#### Weighted mode

Appropriate when dt varies within a turn (non-uniform encoder triggering).

```
offset = sum(df) / sum(dt)
df_corrected_k = df_k - offset * dt_k
flux = cumsum(df_corrected)
```

**Code:** `preprocess.py:289-315`.

**When to use each:** Legacy mode is exact for encoder-triggered acquisitions with uniform dt (e.g. BTP8). Weighted mode is preferable for systems with variable sample timing.

### 3. FFT

Computes Fourier coefficients from the integrated flux. Uses the standard 2/N normalisation.

```
f_n = 2 * FFT(flux)[n] / N_s     for n = 1, 2, ..., H
```

DC component (n=0) is dropped. The factor of 2 accounts for the two-sided spectrum.

**Code:** `kn_pipeline.py:265-270`.

**Theory:** Bottura Eq. 18.

### 4. Kn Calibration

Applies the complex coil-sensitivity coefficients and reference-radius scaling.

```
C_n = f_n / conj(kn_n) * R_ref^(n-1)
```

Where `kn_n` is the complex sensitivity of the coil for harmonic order *n*. The conjugate and power-of-R scaling follow the CERN convention.

**Code:** `kn_pipeline.py:272-285`.

**Theory:** Bottura Eq. 18.

A "DB snapshot" of C_abs and C_cmp is saved at this point (before rotation, feeddown, or normalisation).

### 5. Rotation (`rot`)

Aligns the harmonic phases to the main-field direction. The rotation reference is computed from the post-Kn calibrated main harmonic.

```
phi_m = angle(C_abs[m])
phi_wrapped = wrap_to_[-pi/2, +pi/2](phi_m)
phi = phi_wrapped / m

C_n = C_n * exp(-i * n * phi)     for n = 1, ..., H
```

Wrapping to [-pi/2, +pi/2] is done by adding or subtracting pi:

```python
if phi > pi/2:  phi -= pi
if phi < -pi/2: phi += pi
```

All harmonics n=1..H are rotated by default (`legacy_rotate_excludes_last=False`), matching Bottura Eq. AIV.6, the C++ analyzer, and the Pentella analyzer. Setting `legacy_rotate_excludes_last=True` excludes the last harmonic, which was an off-by-one in some legacy SM18 code.

**Code:** `kn_pipeline.py:291-308`.

**Theory:** Bottura Eq. AIV.5-6.

### 6. Centre Location (`cel`)

Computes the transverse position of the magnetic centre relative to the coil centre.

For m >= 2 (quadrupole and higher):

```
zR = -C_{m-1} / ((m-1) * C_m)          (dimensionless)
z = R_ref * zR                           (metres)
x = Re(z),  y = Im(z)
```

For dipoles (m=1), uses compensated harmonics n=10 and n=11 (legacy convention).

**Code:** `kn_pipeline.py:310-342`.

**Theory:** Bottura Eq. AIII.4.

### 7. Feeddown (`fed`)

Corrects harmonics for the off-centre measurement position using the binomial expansion.

```
C'_n = sum_{k=n}^{H-1} C(k,n) * zR^(k-n) * C_k
```

Where C(k,n) is the binomial coefficient "k choose n" and zR is the dimensionless centre from the CEL step.

**Code:** `kn_pipeline.py:343-362`.

**Theory:** Bottura Eq. AIII.6.

### 8. Normalisation (`nor`)

Scales all harmonics to units relative to the main field (1 unit = 10^-4 of the main field).

```
scale = 10000 / Re(C_m)
C_n = C_n * scale          for all n
```

If `skew_main=True`, Im(C_m) is used instead of Re(C_m).

**Important:** The in-pipeline `nor` option normalises **all** harmonics (including n <= m) in-place. This is used by SM18 and some validation workflows where the reference output is fully normalised. For the standard Bottura Section 3.7 record format (Tesla for n <= m, units for n > m), omit `nor` from the pipeline options and use `safe_normalize_to_units` post-merge instead — this is exactly what `process_kn_pipeline()` does by default.

**Code:** `kn_pipeline.py:364-374`.

**Theory:** Bottura Eq. AIV.8.

### 9. Channel Merge

Combines the absolute (ABS) and compensated (CMP) channels into a single harmonic set. This step is separate from the per-turn pipeline and uses `merge_coefficients()`.

The standard merge mode for parity validation is `abs_upto_m_cmp_above`:

- n <= m: use ABS (captures main field accurately)
- n > m: use CMP (lower noise through bucking)

Other modes: `abs_all`, `cmp_all`, `abs_main_cmp_others`, `custom`.

**Code:** `kn_pipeline.py:396-469`.

## Cross-Implementation Comparison

Five implementations have been compared step by step:

| Step | Python `kn_pipeline.py` | C++ `MatlabAnalyzerRotCoil.cpp` | MATLAB `RotatingCoilAnalysisTurn.m` | Pentella `rotcoil_lib.py` | Bottura theory |
|------|------------------------|---------------------------------|-------------------------------------|--------------------------|---------------|
| di/dt | `preprocess.py:139` | `computeHarmonics()` | `diDtCorrection()` | `correct_di_dt()` | Eq. AII.12 |
| Drift | `preprocess.py:265` | Line ~127 | `driftCorrection()` | `remove_drift()` | Eq. AII.14 |
| FFT | `kn_pipeline.py:265` | Line ~145 | `fft()` call | `np.fft.fft()` | Eq. 18 |
| Kn | `kn_pipeline.py:272` | Lines 149-156 | `applySensitivity()` | `apply_kn()` | Eq. 18 |
| Rotation | `kn_pipeline.py:299` | Lines 160-175 | `rotateHarmonics()` | `rotate()` | Eq. AIV.5-6 |
| CEL | `kn_pipeline.py:315` | Lines 180-195 | `centerLocation()` | `center_location()` | Eq. AIII.4 |
| Feeddown | `kn_pipeline.py:344` | Lines 200-220 | `feeddown()` | `feeddown()` | Eq. AIII.6 |
| Normalisation | `kn_pipeline.py:367` | Lines 225-235 | `normalize()` | `normalize()` | Eq. AIV.8 |
| Merge | `kn_pipeline.py:396` | Post-pipeline | Post-pipeline | Post-pipeline | N/A |

**All five implementations are mathematically identical for every pipeline step.** The only differences are language-specific (indexing conventions, array layout) and do not affect numerical results.

## Known Differences

### Rotation loop bounds (resolved)

The C++ code, Pentella analyzer, and Bottura Eq. AIV.6 all rotate ALL harmonics k=1..H. The Python default is now `legacy_rotate_excludes_last=False`, matching all three references. The `True` option is retained for legacy SM18 parity where the original C++ code had an off-by-one in the rotation loop.

### Normalisation source (resolved)

The standard Bottura Section 3.7 record format uses Tesla for n <= m and normalised units for n > m. The `process_kn_pipeline()` wrapper now handles this automatically:

1. Runs `compute_legacy_kn_per_turn` **without** `nor` — harmonics stay in Tesla
2. Merges channels in Tesla via `merge_coefficients`
3. Normalises post-merge via `safe_normalize_to_units` — produces units array
4. `build_harmonic_rows()` picks Tesla for n <= m and units for n > m

The in-pipeline `nor` option (which normalises ALL harmonics in-place) is still available for SM18 and validation workflows where the reference output is fully normalised.

### Pentella offset

The Pentella analyzer (`rotcoil_lib.py`) uses a different convention for the encoder offset (one sample shift). This does not affect parity when compared against the same raw data with the same offset convention.

### Turn count

Flux files typically contain more turns than the reference. The legacy software uses only the first `Parameters.Measurement.turns` turns per run (default 6 for BTP8). The remaining turns are warm-up or coast-down and are discarded.

## Code References

### Python (`kn_pipeline.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `load_segment_kn_txt` | 108 | Load Kn from text file |
| `compute_legacy_kn_per_turn` | 172 | Main pipeline entry point |
| `merge_coefficients` | 396 | ABS/CMP channel merge |
| `_wrap_arg_to_pm_pi_over_2` | 159 | Rotation angle wrapping |

### Python (`preprocess.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `di_dt_weights` | 139 | Compute di/dt correction weights |
| `apply_di_dt_to_channels` | 211 | Apply di/dt to both channels |
| `integrate_to_flux` | 229 | Drift correction + integration |
| `estimate_linear_slope_per_turn` | 105 | Least-squares dI/dt per turn |

### C++ (`MatlabAnalyzerRotCoil.cpp`)

| Section | Line (approx.) | Purpose |
|---------|----------------|---------|
| Drift correction | ~127 | `cumsum(df - mean(df)) - mean(cumsum(df))` |
| FFT | ~145 | `2*fft/N` |
| Kn application | 149-156 | `1/conj(kn) * R^(n-1)` |
| Rotation | 160-175 | Phase alignment loop |
| CEL | 180-195 | Centre location |
| Feeddown | 200-220 | Binomial expansion |
| Normalisation | 225-235 | `10000/Re(C_m)` |

## Streaming Analysis Utilities

For **streaming (continuous) acquisition** measurements -- where the magnet current follows a machine supercycle rather than holding at individual DC plateaus -- additional utilities are provided in `rotating_coil_analyzer/analysis/utility_functions.py`.

These utilities address the key challenge: during streaming, only turns acquired on **flat current plateaus** produce reliable harmonics. Turns during ramps or transitions must be excluded.

### Plateau Detection

The plateau detection pipeline uses a **block-averaged** approach to filter out sample-level ADC noise:

#### Step 1: Block-averaged current range (`compute_block_averaged_range`)

Each turn (typically 1024 samples) is split into *N* blocks (default 10). Each block is averaged to a single value, then the range (max - min) of the block means is computed:

```
I_blocks[turn, k] = mean(I[turn, k*block_sz : (k+1)*block_sz])
I_range[turn] = max(I_blocks[turn, :]) - min(I_blocks[turn, :])
```

This measures real current drift or ramp contamination while filtering out sample-level noise spikes. Raw `max(I) - min(I)` over 1024 samples is dominated by ADC noise and would reject even perfectly flat plateaus.

**Code:** `utility_functions.py:51-87` (`compute_block_averaged_range`)

#### Step 2: Three-rule plateau detection (`detect_plateau_turns`)

A turn is accepted as "on a plateau" only if **all three** rules pass:

| Rule | Condition | Purpose |
|------|-----------|---------|
| **(a)** | `I_range < threshold` | Current must be flat throughout the turn |
| **(b)** | `\|I_blocks[0] - I_mean\| < threshold` | Turn must **start** on the plateau |
| **(c)** | `\|I_blocks[-1] - I_mean\| < threshold` | Turn must **end** on the plateau |

Rules (b) and (c) reject turns that straddle a ramp-to-plateau or plateau-to-ramp boundary. A turn that passes rule (a) but fails (b) or (c) is flagged as "boundary-rejected".

**Code:** `utility_functions.py:90-136` (`detect_plateau_turns`)

#### Step 3: Current-level classification (`classify_current`)

Each plateau turn is classified by its mean current into a machine cycle-type label. The default thresholds are tuned for SPS:

| Label | Current range (A) |
|-------|------------------|
| `zero` | < 50 |
| `pre-ramp` | 50 -- 200 |
| `injection` | 200 -- 500 |
| `flat-low` | 500 -- 2000 |
| `flat-mid` | 2000 -- 4000 |
| `flat-high` | > 4000 |

Custom thresholds can be provided for other machines (PS, PSB, LHC, ...) via the `thresholds` parameter.

**Code:** `utility_functions.py:155-189` (`classify_current`)

### Pipeline Wrapper

#### `process_kn_pipeline`

Wraps the three core pipeline functions into a single call:

```
compute_legacy_kn_per_turn  -->  merge_coefficients  -->  safe_normalize_to_units
```

Returns the full `LegacyKnPerTurn` result, merged complex coefficients, normalised units, and the `ok_main` boolean mask.

**Code:** `utility_functions.py:235-315` (`process_kn_pipeline`)

#### `build_harmonic_rows`

Converts pipeline results into a list of dicts (one per turn) suitable for `pd.DataFrame()`. Each row contains:

- Per-turn scalars: `time_s`, `I_mean_A`, `ok_main`, `phi_rad`, `x_mm`, `y_mm`
- Bn/An (T) for orders <= magnet_order
- bn/an (units) for higher orders
- Optional extra columns (e.g. `global_turn`, `label`, `I_range_A`)

**Code:** `utility_functions.py:318-375` (`build_harmonic_rows`)

### General-Purpose Utilities

#### `find_contiguous_groups`

Finds contiguous runs of `True` values in a boolean array. Used to identify injection plateaus, flat-top groups, and other contiguous turn sequences in the supercycle structure.

Returns a list of `(start, end)` tuples (inclusive indices), optionally filtered by minimum group length.

**Code:** `utility_functions.py:196-228` (`find_contiguous_groups`)

### Run-Level Aggregation and Export

#### `build_run_averages`

Computes per-run mean b3 with run ordering, suitable for hysteresis and ramp analysis. Requires columns `run`, `I_mean_A`, `I_nom_A`, `b3_units`, and `turn_in_run` in the input DataFrame.

**Code:** `utility_functions.py:382-404` (`build_run_averages`)

#### `ba_table_from_C`

Converts complex harmonic coefficients to a legacy B/A DataFrame (all values in Tesla). Convention: B_n = Re(C_n), A_n = Im(C_n). Shared by the GUI Harmonic Merge tab and analysis notebooks.

**Code:** `utility_functions.py:411-440` (`ba_table_from_C`)

#### `mixed_format_table`

Builds a Bottura Section 3.7 mixed-format DataFrame: Tesla for n <= m (main and lower orders), normalised units for n > m (higher orders). When the `nor` option was active in the pipeline, all harmonics are exported as units. Shared by the GUI Harmonic Merge tab and analysis notebooks.

**Code:** `utility_functions.py:443-493` (`mixed_format_table`)

### Code References

| Function | Line | Purpose |
|----------|------|---------|
| `compute_block_averaged_range` | 51 | Noise-robust current range per turn |
| `detect_plateau_turns` | 90 | Three-rule plateau detection |
| `classify_current` | 155 | Current-level classification |
| `find_contiguous_groups` | 196 | Contiguous True-run finder |
| `process_kn_pipeline` | 235 | Full pipeline wrapper |
| `build_harmonic_rows` | 318 | DataFrame row builder |
| `build_run_averages` | 382 | Per-run mean b3 with run ordering |
| `ba_table_from_C` | 411 | Complex coefficients to legacy B/A DataFrame |
| `mixed_format_table` | 443 | Bottura 3.7 mixed-format DataFrame |

## References

- L. Bottura, "Rotating Coil Measurements", CERN Accelerator School (CAS) proceedings
- ffmm C++ framework: `ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp`
- MATLAB Coder path: `RotatingCoilAnalysisTurn.m`
