# Analysis Pipeline Reference

## Overview

The rotating coil analysis pipeline transforms raw acquisition data (incremental flux samples, encoder counts, and current measurements) into calibrated magnetic field harmonics. The implementation in `rotating_coil_analyzer/analysis/kn_pipeline.py` follows the legacy C++ analyzer semantics exactly (`ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp`).

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

**Code:** `preprocess.py:139-207` (`di_dt_weights`), applied at `kn_pipeline.py:243-244`.

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

**Code:** `kn_pipeline.py:258-262`.

**Theory:** Bottura Eq. 18.

### 4. Kn Calibration

Applies the complex coil-sensitivity coefficients and reference-radius scaling.

```
C_n = f_n / conj(kn_n) * R_ref^(n-1)
```

Where `kn_n` is the complex sensitivity of the coil for harmonic order *n*. The conjugate and power-of-R scaling follow the CERN convention.

**Code:** `kn_pipeline.py:264-277`.

**Theory:** Bottura Eq. 18.

A "DB snapshot" of C_abs and C_cmp is saved at this point (before rotation, feeddown, or normalisation).

### 5. Rotation (`rot`)

Aligns the harmonic phases to the main-field direction. The rotation reference is computed from the post-Kn calibrated main harmonic.

```
phi_m = angle(C_abs[m])
phi_wrapped = wrap_to_[-pi/2, +pi/2](phi_m)
phi = phi_wrapped / m

C_n = C_n * exp(-i * n * phi)     for n = 1, ..., H-1
```

Wrapping to [-pi/2, +pi/2] is done by adding or subtracting pi:

```python
if phi > pi/2:  phi -= pi
if phi < -pi/2: phi += pi
```

The last harmonic (n=H) is excluded from rotation by default (`legacy_rotate_excludes_last=True`), matching the C++ loop bounds.

**Code:** `kn_pipeline.py:283-303`.

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

**Code:** `kn_pipeline.py:306-329`.

**Theory:** Bottura Eq. AIII.4.

### 7. Feeddown (`fed`)

Corrects harmonics for the off-centre measurement position using the binomial expansion.

```
C'_n = sum_{k=n}^{H-1} C(k,n) * zR^(k-n) * C_k
```

Where C(k,n) is the binomial coefficient "k choose n" and zR is the dimensionless centre from the CEL step.

**Code:** `kn_pipeline.py:332-350`.

**Theory:** Bottura Eq. AIII.6.

### 8. Normalisation (`nor`)

Scales all harmonics to units relative to the main field (1 unit = 10^-4 of the main field).

```
scale = 10000 / Re(C_m)
C_n = C_n * scale          for all n
```

If `skew_main=True`, Im(C_m) is used instead of Re(C_m).

**Code:** `kn_pipeline.py:352-361`.

**Theory:** Bottura Eq. AIV.8.

### 9. Channel Merge

Combines the absolute (ABS) and compensated (CMP) channels into a single harmonic set. This step is separate from the per-turn pipeline and uses `merge_coefficients()`.

The standard merge mode for parity validation is `abs_upto_m_cmp_above`:

- n <= m: use ABS (captures main field accurately)
- n > m: use CMP (lower noise through bucking)

Other modes: `abs_all`, `cmp_all`, `abs_main_cmp_others`, `custom`.

**Code:** `kn_pipeline.py:384-458`.

## Cross-Implementation Comparison

Five implementations have been compared step by step:

| Step | Python `kn_pipeline.py` | C++ `MatlabAnalyzerRotCoil.cpp` | MATLAB `RotatingCoilAnalysisTurn.m` | Pentella `rotcoil_lib.py` | Bottura theory |
|------|------------------------|---------------------------------|-------------------------------------|--------------------------|---------------|
| di/dt | `preprocess.py:139` | `computeHarmonics()` | `diDtCorrection()` | `correct_di_dt()` | Eq. AII.12 |
| Drift | `preprocess.py:265` | Line ~127 | `driftCorrection()` | `remove_drift()` | Eq. AII.14 |
| FFT | `kn_pipeline.py:258` | Line ~145 | `fft()` call | `np.fft.fft()` | Eq. 18 |
| Kn | `kn_pipeline.py:269` | Lines 149-156 | `applySensitivity()` | `apply_kn()` | Eq. 18 |
| Rotation | `kn_pipeline.py:295` | Lines 160-175 | `rotateHarmonics()` | `rotate()` | Eq. AIV.5-6 |
| CEL | `kn_pipeline.py:310` | Lines 180-195 | `centerLocation()` | `center_location()` | Eq. AIII.4 |
| Feeddown | `kn_pipeline.py:336` | Lines 200-220 | `feeddown()` | `feeddown()` | Eq. AIII.6 |
| Normalisation | `kn_pipeline.py:355` | Lines 225-235 | `normalize()` | `normalize()` | Eq. AIV.8 |
| Merge | `kn_pipeline.py:384` | Post-pipeline | Post-pipeline | Post-pipeline | N/A |

**All five implementations are mathematically identical for every pipeline step.** The only differences are language-specific (indexing conventions, array layout) and do not affect numerical results.

## Known Differences

### Rotation loop bounds

The C++ code rotates harmonics k=1..H-1, leaving the last harmonic unrotated. The Python implementation replicates this via `legacy_rotate_excludes_last=True` (default). Setting it to `False` would rotate all H harmonics.

### Normalisation source

The reference output uses a mixed format: Tesla for n <= m, normalised units for n > m. To reproduce this, run the pipeline *without* the `nor` option, then normalise n > m manually after merging:

```python
b_n = C_merged[n].real / C_merged[m].real * 10000
```

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
| `merge_coefficients` | 384 | ABS/CMP channel merge |
| `_wrap_arg_to_pm_pi_over_2` | 159 | Rotation angle wrapping |

### Python (`preprocess.py`)

| Function | Line | Purpose |
|----------|------|---------|
| `di_dt_weights` | 139 | Compute di/dt correction weights |
| `apply_di_dt_to_channels` | 210 | Apply di/dt to both channels |
| `integrate_to_flux` | 228 | Drift correction + integration |
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

## References

- L. Bottura, "Rotating Coil Measurements", CERN Accelerator School (CAS) proceedings
- ffmm C++ framework: `ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp`
- MATLAB Coder path: `RotatingCoilAnalysisTurn.m`
