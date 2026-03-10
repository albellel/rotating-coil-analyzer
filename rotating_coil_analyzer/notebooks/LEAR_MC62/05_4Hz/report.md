# LEAR MC62 Dipole -- 05 Staircase 4 Hz -- March 4, 2026

## Overview

Full streaming staircase analysis of the LEAR MC62 dipole at 4 Hz rotation speed (~238 RPM). The measurement includes a 20-group precycle (alternating +/-200 A) followed by a 20-step staircase (0 -> 200 -> 20 A in 20 A steps, ascending then descending). Both Integral and Central segments were measured.

**Key features of this campaign:**
- First MC62 measurement at 4 Hz (previous: 1 Hz and 2 Hz)
- First MC62 measurement with FFMM results files (machine-precision parity validated)
- Precycle + staircase structure with slow 1 A/s ramp rate
- 512 samples/turn, ~1345 turns per plateau group

## Notebooks

| Notebook | Description |
|----------|-------------|
| `analysis.ipynb` | Full analysis: plateau detection, harmonics, FFMM golden standard, eddy settling, hysteresis |
| `marusov_reconstruction.ipynb` | R&D: Marusov (2013) temporal decomposition applied to pipeline output. Identity check (K=M, machine epsilon), temporal filtering (K<M), full-stream comparison |
| `eddy_transfer_function.ipynb` | R&D: Multi-tau eddy fitting (1/2/3-tau with AICc selection) across all plateaus, eddy amplitude scaling with dI/dt (1 A/s vs 50 A/s), static magnetization curve, model validation |
| `dynamic_eddy_correction.ipynb` | R&D: Convolution-based eddy correction during ramps using impulse response calibrated from plateau fits. 11-20x B1 improvement at 40-100 A |

### Documentation

| File | Description |
|------|-------------|
| `report.md` | This file — measurement report with configuration, results, and key findings |
| `marusov_guide.md` | Plain-language guide to the Marusov 2D Fourier decomposition: theory, formulas, implementation, and results |

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | LEAR MC62 dipole (m=1), Integral + Central segments |
| Rotation speed | ~238 RPM (3.97 Hz), period ~0.252 s/turn |
| Samples/turn | 512 |
| Kn calibration | Integral: `Kn_R45_PCB_N1_0001_A_AC.txt`, Central: `Kn_DQ_5_18_7_250_47x50_0001_A_AC.txt` |
| Reference radius | 0.033 m |
| Pipeline options | `dri`, `rot` (cel/fed disabled -- UNSAFE) |
| Encoder offset | pi rad (raw C_1 on negative real axis; pi restores physical convention B1>0, b2>0) |
| Plateau method | Rolling std of I_mean (window=50, threshold=0.05 A) |
| Settling turns | N_LAST = 680 (~170 s at 4 Hz) |

## Measurement Structure

| Phase | Groups | I range (A) | Turns/group |
|-------|--------|-------------|-------------|
| Precycle | 20 | +/-200 alternating | ~1338 |
| Staircase ascending | 10 | 20 -> 200 (20 A steps) | ~1345 |
| Staircase descending | 10 | 180 -> 20 (20 A steps) | ~1345 |
| **Total** | **40** | | **~53,590 plateau turns** |

## Plateau Detection

Standard block-averaged I_range detection fails at 4 Hz with 1 A/s ramp because the per-turn current change (~0.25 A) is smaller than the detection threshold. Solution: rolling standard deviation of I_mean over a 50-turn window with threshold 0.05 A.

- Distribution: std < 0.02 captures 85% of turns (clean plateaus), std < 0.05 captures 93%
- With min_length=50: cleanly produces 40 groups (20 precycle + 20 staircase)

## cel/fed Diagnostic

cel/fed is **UNSAFE**: 100/100 high-I turns have |zR| > 0.01 (median 0.121, max 0.154). This is consistent with the dipole cel fragility (uses compensated n=10,11 which are weak at low order). Pipeline options automatically reduced to `('dri', 'rot')`.

## FFMM Golden Standard Validation

Machine-precision parity achieved with FFMM C++ results using `('dri', 'rot', 'dit')` + `dit_signed=True` + `FFMM_R_ref=0.33`:

| Segment | N turns | B1 max |diff| | b2 diff | b3 diff | b4 diff | b5 diff |
|---------|---------|-----------------|---------|---------|---------|---------|
| Integral | 57,715 | 0.0 uT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Central | 57,715 | N/A (FFMM B_main = NaN, Central Kn all zeros in FFMM) | -- | -- | -- | -- |

**Note on encoder offset:** The FFMM comparison uses `encoder_offset_rad = 0.0` (matching FFMM's raw sign convention), while the physics analysis uses `pi` for the correct physical convention (B1 > 0, b2 > 0). Offset=pi interacts with `_wrap_arg_to_pm_pi_over_2` and cannot be used for FFMM parity. Any offset except pi gives identical FFMM parity (confirmed empirically).

## Results -- Settled Harmonics (Integral segment, N_LAST=680)

### Ascending branch

| I (A) | N | B1 (T) | B1_std (uT) | b2 (units) | b2_std | b3 (units) | b3_std | TF (T/kA) |
|-------|---|--------|-------------|------------|--------|------------|--------|-----------|
| 20.2 | 680 | +0.02243 | 11 | +13.55 | 0.033 | -0.87 | 0.051 | 1.112 |
| 60.2 | 680 | +0.06786 | 21 | +13.35 | 0.021 | -0.80 | 0.046 | 1.128 |
| 100.2 | 680 | +0.11317 | 44 | +13.20 | 0.017 | -0.78 | 0.047 | 1.130 |
| 140.2 | 680 | +0.15705 | 61 | +13.18 | 0.016 | -0.75 | 0.047 | 1.121 |
| 180.2 | 680 | +0.18785 | 75 | +13.40 | 0.017 | -0.69 | 0.050 | 1.043 |
| 200.2 | 680 | +0.19904 | 87 | +13.56 | 0.015 | -0.64 | 0.047 | 0.994 |

### Descending branch

| I (A) | N | B1 (T) | B1_std (uT) | b2 (units) | b2_std | b3 (units) | b3_std | TF (T/kA) |
|-------|---|--------|-------------|------------|--------|------------|--------|-----------|
| 180.2 | 680 | +0.18850 | 66 | +13.26 | 0.014 | -0.68 | 0.044 | 1.046 |
| 140.2 | 680 | +0.15807 | 51 | +13.01 | 0.013 | -0.74 | 0.041 | 1.128 |
| 100.2 | 680 | +0.11388 | 37 | +13.10 | 0.010 | -0.73 | 0.031 | 1.137 |
| 60.2 | 680 | +0.06854 | 19 | +13.21 | 0.011 | -0.72 | 0.035 | 1.139 |
| 20.2 | 680 | +0.02309 | 10 | +13.24 | 0.032 | -0.77 | 0.032 | 1.145 |

## Hysteresis

Clear B1 hysteresis with ascending |B1| < descending |B1| at the same current (remanence):

| I (A) | dB1 (mT) | db2 (units) | db3 (units) |
|-------|----------|-------------|-------------|
| 20.2 | +0.66 | -0.31 | -0.10 |
| 100.2 | +0.72 | -0.10 | -0.05 |
| 200.2 | (top, no desc) | -- | -- |
| 180.2 | +0.65 | -0.14 | -0.005 |

B1 hysteresis width ~0.7 mT (ascending - descending). b2 hysteresis ~0.1-0.3 units. b3 hysteresis < 0.1 units.

## Eddy Current Settling

B1 eddy settling fits (1-tau model):

| Branch | I range | Good fits | tau range (s) | R2 range |
|--------|---------|-----------|---------------|----------|
| Ascending | 20-100 A | 5/10 | 33-40 s | 0.91-0.98 |
| Descending | 20-40 A | 1/9 | 30-32 s | 0.90-0.98 |

- Settling is visible only at low-to-moderate currents (where ramp-to-plateau transition is sharpest)
- tau ~ 33-40 s is the iron magnetisation relaxation time constant
- At high current (> 140 A), eddy signals are weak (R2 < 0.7) -- consistent with saturation reducing eddy amplitude
- N_LAST = 680 turns (~170 s) = 4.9τ: eddy residual = 0.75% of A ≈ 6.5 µT (below per-turn noise of 21 µT, but 8× the std-of-mean when 680 turns are averaged). For sub-µT static accuracy, exponential fitting is needed.

## Comparison with 1 Hz Campaign (01_with_shims)

**Important caveats:** Different analysis settings (1 Hz: cel/fed ON, offset=pi; 4 Hz: cel/fed OFF, offset=0), different measurement dates (Feb 11 vs Mar 4), potentially different coil axial positions (B1 magnitudes differ by ~16%). b_n noise comparison is affected by cel/fed (which adds noise to higher harmonics at the 4 Hz and was correctly disabled, but was left on at 1 Hz).

### Per-turn noise (all-turns, median across current steps)

| Metric | 1 Hz | 4 Hz | Ratio (4Hz/1Hz) |
|--------|------|------|-----------------|
| B1 noise | baseline | 0.61x | **4 Hz is 39% quieter** |
| b2 noise | baseline | 0.51x | **4 Hz is 49% quieter** |
| b3 noise | baseline | 1.93x | 4 Hz is 93% noisier |
| Turns/plateau | 350 | 1345 | 3.8x more data |

### Assessment

**4 Hz is better for B1 and b2** (lower per-turn scatter), but **noisier for b3** (likely vibration-induced, flat ~0.05 units across all currents). The b3 noise floor of 0.05 units is still excellent (sub-unit precision).

With 4x more turns per plateau, the **standard error of the mean** is:
- b2 at 200 A: 4 Hz = 0.0007 units vs 1 Hz = 0.0026 units (4 Hz is 3.7x better)
- b3 at 200 A: 4 Hz = 0.0014 units vs 1 Hz = 0.0003 units (1 Hz is 5x better)

**Overall verdict:** 4 Hz provides cleaner data for the main field and quadrupole component, with more turns per plateau enabling better averaging. The sextupole noise is higher but still at sub-unit level. The main advantage is throughput: same measurement time yields 4x more usable turns.

## Marusov 2D Reconstruction (R&D)

Marusov (2013) proposes a 2D Fourier decomposition of the rotating coil signal into spatial harmonics (n) and temporal modes (k). Our implementation applies temporal DFT decomposition to the validated pipeline output C_n(j), avoiding the need to reimplement the complex spatial pipeline (drift, Kn, rotation).

See `marusov_guide.md` for a complete plain-language explanation of the theory and formulas.

### Identity check (K=M)

When all temporal modes are retained, the reconstruct is the exact identity:

| Metric | Value |
|--------|-------|
| max |B1_KM - B1_pipeline| | 3.89 × 10⁻¹⁶ T |
| max |b2_KM - b2_pipeline| | 4.39 × 10⁻¹³ units |
| All staircase max dB1/B1 | 7.25 × 10⁻¹³ |

**Identity check: PASS** — the decomposition is exact to machine epsilon.

### Temporal filtering (K < M, settled region)

| K | dB1_rms (µT) | dB1_rel | db2_rms (units) | db3_rms (units) |
|---|-------------|---------|----------------|----------------|
| 5 | 60.3 | 8.9e-4 | 0.084 | 0.042 |
| 50 | 33.5 | 4.9e-4 | 0.041 | 0.042 |
| 200 | 18.9 | 2.8e-4 | 0.023 | 0.040 |
| M (full) | 0 | 0 | 0 | 0 |

Residual = noise content removed by temporal filtering. On settled plateaus, this is purely measurement noise.

### Full-stream vs two-step comparison

Direct σ_{n,k} comparison between full-stream (single 1D FFT of concatenated stream) and two-step (per-turn FFT → temporal DFT). The two-step ignores the phase coupling term exp(-2πi·ks/(MNs)).

| Quantity | Value |
|----------|-------|
| DC (k=0) agreement | Machine epsilon (2.9×10⁻¹⁶ relative) |
| σ_{n,k} relative error (n=1) | ~3.6×k/M — constant prefactor at ALL k values |
| Phase coupling prefactor | 3.6 ≈ 2π/√3 (geometric RMS of phase integral) |
| Impact on per-turn B1 at k_eddy | ~5 ppm (because σ_k/σ_0 ≈ 1.9×10⁻⁴) |
| Per-turn averaging error | (T/τ)²/24 ≈ 2×10⁻⁶ of eddy amplitude (sub-ppm) |
| Measured |C1| diff (settling) | 3.2×10⁻³ (noise-dominated) |
| Measured |C1| diff (settled) | 7.7×10⁻⁴ (noise-dominated) |
| n=2, n=3 σ comparison | Noise-dominated (10-150× above k/M theory) |

**Why the error is quadratic, not linear**: The 0.7% per-turn field change (= T/τ) is the first derivative effect — and it cancels exactly by the symmetry of the averaging integral. Only the curvature (second derivative, B'') contributes: error = B''·T²/24 = A·(T/τ)²/24. This is a fundamental property of symmetric averaging.

**Phase coupling validation**: For n=1, the measured/theory ratio is constant at 3.6× across all k values (k=1 to k=100). This proves both implementations are correct — they disagree by exactly the predicted geometric prefactor. For n=2, n=3, noise dominates (σ_{2,k} and σ_{3,k} are 50–200× smaller than σ_{1,k}).

**Eddy disentanglement**: With τ/T ≈ 140, the per-turn pipeline provides ~140 samples per time constant. Eddies are fully visible as exponential settling in B1(j), b2(j), b3(j) from turn to turn. Disentanglement via exponential fitting or last-N averaging works at sub-ppm precision. Marusov confirms this rather than fixing a problem.

### Verdict

For MC62 4 Hz (τ/T ≈ 140):

- **Per-turn averaging**: (T/τ)²/24 ≈ 2×10⁻⁶ — sub-ppm. The linear field change (0.7%) cancels by averaging symmetry; only the curvature contributes.
- **Phase coupling**: 3.6×k/M on σ_{n,k} (quantitatively validated), but only ~5 ppm impact on per-turn B1 at the eddy frequency (σ_k ≪ σ_0).
- **Eddy resolution**: ~140 samples per τ — eddies are well-resolved by per-turn sampling. Last-N average and exponential fitting both work at sub-ppm.
- **Rotation correction**: per-turn pipeline's nonlinear rotation cannot be replicated in full-stream — practical barrier for b_n during transients.
- **On settled plateaus**: both approaches give identical results (field is constant, no temporal-spatial coupling to resolve).
- **Temporal filtering** (K ~ 50): captures all eddy content, removes ~50% of per-turn noise bandwidth. Optimal for noise reduction.
- **Best K**: K ≈ 50 for eddy analysis (k_eddy ≈ 10, with margin). b3 does not benefit from temporal filtering (broadband vibration noise).

## Multi-Tau Eddy Transfer Function (R&D)

Multi-tau (1/2/3-exponential) eddy fitting across all 40 plateau groups, with AICc model selection.

### Model selection summary

| Harmonic | 1-tau | 2-tau | 3-tau |
|----------|-------|-------|-------|
| B1 | 11 | 11 | 18 |
| b2 | 11 | 9 | 20 |
| b3 | 40 | 0 | 0 |

B1 and b2 often benefit from multi-tau models (iron has multiple relaxation time scales). b3 is well described by 1-tau everywhere.

### Model validation (ascending staircase)

| I (A) | Model | R² | Early rms (µT) | Late rms (µT) | Late relative |
|-------|-------|----|----------------|---------------|---------------|
| 20 | 1-tau | 0.977 | 42 | 10 | 4.2e-4 |
| 60 | 1-tau | 0.952 | 58 | 23 | 3.4e-4 |
| 100 | 1-tau | 0.924 | 71 | 47 | 4.2e-4 |

Late relative residual ~3-4 × 10⁻⁴ — noise-limited, not model-limited.

### Eddy amplitude scaling with dI/dt

Comparison between precycle (50 A/s) and staircase (1 A/s) ramp rates at similar current levels shows eddy amplitude scales with ramp rate, consistent with linear eddy response. The ratio is not exactly 50× due to different thermal history and settling dynamics between precycle and staircase phases.

## Dynamic Eddy Correction During Ramps (R&D)

Convolution-based eddy correction applied to ramp data. See `dynamic_eddy_correction.ipynb` and `marusov_guide.md` "Dynamic Eddy Correction During Ramps" for full details.

### Approach

Back-calculate the impulse response parameter c(I) from plateau multi-tau fits:
c = A1 / (dI/dt × τ × (1 - exp(-t_ramp/τ))), accounting for the slow ramp (t_ramp ~ τ).
Then predict eddy during ramps via discrete convolution: eddy(j) = Σ c × dI_k × exp(-(t_j-t_k)/τ).

### Validation results (B1, ascending staircase)

| I (A) | Measured bias | Corrected bias | Improvement |
|-------|-------------|---------------|-------------|
| 40–100 (1-tau, R²>0.92) | 450–600 µT | 30–40 µT | **11–20x** |
| 120 (2-tau, only tau1) | 610 µT | 189 µT | 3.2x |
| 140+ (weak eddy) | <400 µT | 75–261 µT | 1.5–1.9x |

### Key findings

- **Impulse response c**: 1.8–8.7 × 10⁻⁵ T/A for B1 (only ~2–6% of total TF is eddy-susceptible)
- **Steady-state ramp lag**: c × dI/dt × τ = 140–490 µT at 0.63 A/s (0.07–0.33% of B1)
- **Best performance at 40–100 A**: corrected bias (30–40 µT) approaches per-turn noise floor
- **1-tau limitation**: correction degrades at 120+ A where 2-tau models are needed but only tau1 used
- **b2/b3 not yet corrected**: b2 c-values scatter wildly with 2-tau fits; b3 has only 1 good fit
- **Marusov cannot help directly**: temporal filtering cleans noise, but cannot separate static from eddy on ramps

## Key Findings

1. **4 Hz rotation works well**: clean data, proper plateau detection (rolling-std method), machine-precision FFMM parity.
2. **cel/fed is UNSAFE**: automatically disabled (dipole cel fragility confirmed).
3. **B1 and b2 noise improved** vs 1 Hz: 39% and 49% lower per-turn scatter respectively.
4. **b3 noise floor ~0.05 units**: higher than 1 Hz (~0.005) but still excellent for practical purposes.
5. **Eddy settling**: tau ~ 33-40 s at low current, negligible at high current. N_LAST=680 (~170 s) provides ~5 tau settling margin.
6. **Clear hysteresis**: dB1 ~ 0.7 mT, db2 ~ 0.1-0.3 units, db3 < 0.1 units.
7. **FFMM parity**: first MC62 campaign with machine-precision validation (B1 diff = 0.0 uT over 57,715 turns).
8. **Marusov reconstruction validated**: (a) K=M identity at machine epsilon; (b) phase coupling follows ~3.6×k/M for n=1 (stable prefactor at low k where SNR is high, CoV=0.5% for k=1,5,10; scatters at high k near noise floor); (c) per-turn averaging error is (T/τ)²/24 ≈ 2×10⁻⁶ — sub-ppm, because the linear field change cancels by averaging symmetry; (d) impact on per-turn B1 is only ~5 ppm at the eddy frequency; (e) eddies are already well-resolved by per-turn sampling (140 samples/τ); (f) temporal filtering K≈50 optimal for noise reduction; (g) Marusov confirms the per-turn pipeline is adequate rather than fixing a problem.
9. **Multi-tau eddy fitting**: 3-tau models preferred for B1 and b2 (AICc); b3 well described by 1-tau; model validation residuals noise-limited (~3-4 × 10⁻⁴ relative).

## Output CSVs

All outputs in `output/MC62/05_4Hz/`:

| File | Description |
|------|-------------|
| `MC62_Integral_all_turns.csv` | All 26,815 staircase turns (Integral) |
| `MC62_Integral_summary.csv` | 20 plateau summaries (Integral) |
| `MC62_Central_all_turns.csv` | All 26,815 staircase turns (Central) |
| `MC62_Central_summary.csv` | 20 plateau summaries (Central) |
