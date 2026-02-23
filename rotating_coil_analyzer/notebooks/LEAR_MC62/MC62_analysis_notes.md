# MC62 — LEAR C-shaped Dipole — Analysis Notes

## Magnet Description

- **Name**: MC62
- **Type**: Red bulk C-shaped dipole
- **Machine**: LEAR antiproton decelerator, CERN
- **Gap**: 100 mm (half-gap = 50 mm)
- **Excitation range**: 0 to +/-200 A (warm magnet, no superconducting coils)

## Rotating Coil Setup

- **Coil diameter**: >90 mm (fits the 100 mm gap)
- **Two PCBs measured simultaneously**:
  - **Integral (R45)**: long PCB, 30 harmonics, k1_abs = 1.4475,
    compensated suppression ~12900x.  Measures the integrated field
    over the full magnet length (including fringe fields).
  - **Central (DQ)**: small PCB, 15 harmonics, k1_abs = 0.0320,
    compensated suppression very high.  Measures the local field in
    the magnet centre only.
- **Rotation**: -60 rpm, 1024 samples/turn, 10 turns per plateau
- **Kn calibration**: external files used (`Kn values/` folder); the
  in-test Kn files (`Kn_values_Seg_*.txt`) are all zeros and must
  **not** be used.
- **Raw data format**: 4 columns per file — `[time_s, flux_ch1, flux_ch2, I_DCCT_A]`.
  Column 3 is the measured DCCT current in Amperes.

## A-C Compensation Verification

The rotating coil PCBs use an A-C (absolute-compensated) winding design:
the **A coil** (absolute) picks up the full field, while the **C coil**
(compensated) is wound to cancel the main harmonic (n=1 for a dipole),
leaving only the higher-order field errors.

### Kn Calibration Check

| PCB | k1_abs | k1_cmp | Suppression (k1_abs/k1_cmp) | k2_cmp / k2_abs |
|-----|--------|--------|----------------------------:|:---------------:|
| Integral (R45) | 1.4475 | 1.12e-4 | ~12,900x | 1.000 |
| Central (DQ) | 3.196e-2 | 1.39e-17 | ~2.3e15x | 1.000 |

**Both PCBs are correctly compensated:**

- **k1_cmp is strongly suppressed**: the compensated winding effectively
  cancels the dipole fundamental.  The R45 achieves a practical
  suppression of ~13000x (good, typical for a PCB coil).  The DQ
  achieves essentially perfect suppression — k1_cmp ≈ 0 (numerical
  noise in the calibration file).
- **Higher harmonics pass through**: k2_cmp ≈ k2_abs for both PCBs
  (ratio = 1.000), confirming that the compensation only targets the
  main harmonic and does not attenuate the field-error signal.

### Why Central Compensated Still Fails (SNR = 3)

The compensation is physically correct — the problem is **signal level**:

1. The DQ PCB is ~45x smaller than R45 (k1_abs = 0.032 vs 1.448),
   so all flux signals are proportionally weaker.
2. After the essentially perfect suppression, the compensated residual
   (the higher-harmonic content) is astronomically small.
3. At FDI gain = 100, this residual is below the ADC noise floor.

This is not a compensation defect — it is an instrumentation gain
problem.  With FDI gain increased to ~10000 (see gain recommendations
below), the compensated channel should become usable.

---

## Test `00_test` — First RC Measurement (2026-02-11)

### Current Cycle

    0 -> +100 -> +200 -> +100 -> 0 -> -100 -> -200 -> -100 -> 0  [A]

9 runs (Run_00 to Run_08), each at a fixed current plateau with 10
turns.  Total measurement time ~3446 s (~57 min).  Ramp rate 30 A/s
with ~430 s gaps between runs (ramp + settling).

Pre-cycle: 5 repetitions up to +/-200 A before the measurement runs.

### Pipeline Configuration

- Options: `("dri", "rot", "cel", "fed")` — drift, rotation, centre
  location, feed-down corrections all enabled.
- Drift mode: `"legacy"` (C++ compatible).
- Merge strategy:
  - **Integral (R45)**: `abs_upto_m_cmp_above` — B1 from absolute
    channel, higher harmonics (n>1) from compensated.
  - **Central (DQ)**: `abs_all` — all harmonics from absolute channel
    only.  The compensated channel has SNR ~3 (unusable); see the
    *FDI Gain & Signal Analysis* section below.
- Channel detection: `robust_range` on the 200 A run determines
  which flux column is absolute vs compensated.  No swap needed for
  either PCB (column 1 = absolute, column 2 = compensated).

### Averaging

Last 8 out of 10 turns per plateau are averaged (first 2 skipped for
eddy-current settling).  Configurable via `N_LAST_TURNS`.

---

## Reference Radius

### What it is

R_ref is the radius at which harmonic coefficients are **evaluated**
(normalised).  It is a user choice related to the magnet aperture and
beam envelope — **not** a coil property.  The coil winding geometry
is already encoded in the Kn calibration values and handled internally
by the pipeline.

### How it affects the numbers

- **B1 (main field in Tesla)**: completely unaffected by R_ref.
- **Transfer function B1/I**: completely unaffected by R_ref.
- **Relative harmonics** bn, an (units of 1e-4 relative to B1):
  scale as `(R_new / R_old)^(n-1)`.  Changing R_ref from 10 mm to
  33 mm multiplies b2 by 3.3x, b3 by 10.9x, b4 by 35.9x, etc.

### Choice for MC62

    R_ref = 33 mm  =  2/3 of 50 mm half-gap

This follows the standard CERN convention of 2/3 of the aperture
radius, representing the region where field quality matters for the
beam.

---

## Large b2 (Quadrupole Component) in C-shaped Dipoles

### Physics

A C-shaped dipole has inherent **left-right asymmetry**: the iron yoke
is open on one side (the gap opening) and closed on the other.  This
breaks the mid-plane mirror symmetry that an H-shaped (window-frame)
dipole would have.

The quadrupole component b2 measures exactly this: a **linear field
gradient** across the horizontal aperture.  The field is systematically
stronger on the closed yoke side and weaker on the open side.

In a symmetric H-dipole, b2 cancels by geometry.  In a C-dipole it
does not — b2 is an "allowed" harmonic of the C geometry.

### Measured values (R_ref = 33 mm, after `abs_all` merge for Central)

| PCB | b2 at 100 A | b2 at 200 A |
|-----|-------------|-------------|
| Integral (R45) | -149.6 units | -148.2 units |
| Central (DQ) | 0.4 units | 3.0 units |

The integral coil sees the full asymmetry integrated over the magnet
length (~-148 units, very large).  The central coil, sampling only the
magnet centre where the C-yoke is most symmetric, sees much less b2
(~3 units).  This difference is physically expected and consistent.

The b2 is remarkably stable across the current range (148-150 units),
indicating it is dominated by geometry rather than saturation.

---

## Hysteresis

### Observation

At the same nominal current, the field differs between ascending and
descending branches due to iron magnetisation history:

| I [A] | B1 ascending [T] | B1 descending [T] | Delta [mT] |
|------:|------------------:|-------------------:|-----------:|
| +100 | -0.1316 | -0.1367 | 5.0 |
| -100 | +0.1316 | +0.1367 | 5.0 |

The ~5 mT hysteresis at 100 A (~3.8% of B1) is typical for a warm
iron-dominated dipole.

### Remanent field at 0 A

The residual field at zero current depends on the magnetisation history:

| Run | Branch | B1 (integral) [mT] | B1 (central) [mT] |
|-----|--------|--------------------:|-------------------:|
| 00 | initial | +1.12 | +1.81 |
| 04 | after +200 A | -1.82 | -3.03 |
| 08 | after -200 A | +1.75 | +2.90 |

The sign of the remanent field reflects the last excitation direction.
The central coil sees a larger remanent field (~3 mT) than the integral
(~1.8 mT) because the central region retains magnetisation more
strongly than the fringe regions.

---

## Transfer Function

### Definition

    TF = B1 / I   [T/kA]

### Saturation

| |I| [A] | TF integral [T/kA] | TF central [T/kA] |
|---------:|-------------------:|-------------------:|
| 100 | -1.316 | -2.160 |
| 200 | -1.147 | -1.883 |

The transfer function drops from 1.32 to 1.15 T/kA (integral) between
100 A and 200 A — a **13% reduction**, showing clear iron saturation
at 200 A.  The central coil shows a similar ~13% drop.

The central TF is larger than integral because B1_central > B1_integral:
the local field in the magnet centre is stronger than the length-averaged
integrated field (which includes end regions where the field drops off).

### Symmetry

The transfer function is symmetric between positive and negative
excitation (as expected for a dipole with symmetric iron):

    TF(+100 A) = -1.316,  TF(-100 A) = -1.316  (identical)
    TF(+200 A) = -1.147,  TF(-200 A) = -1.148  (identical within noise)

---

## Sextupole (b3)

| PCB | b3 at 100 A | b3 at 200 A |
|-----|-------------|-------------|
| Integral (R45) | -2.4 units | -2.7 units |
| Central (DQ) | 2.3 units | 1.7 units |

For the integral coil, b3 is small and negative (~-2.7 units at
R_ref = 33 mm) with no significant current dependence or hysteresis.
For the central coil (using `abs_all` merge), b3 is ~1.7 units at
200 A — small but now reliably measured from the absolute channel.
The previous values near zero were artifacts of the noisy compensated
channel (SNR=3).

In a dipole, the sextupole is an "allowed" harmonic (for a C-shape,
both even and odd harmonics are allowed due to the broken symmetry)
but its amplitude is naturally much smaller than b2.

---

## Key Differences: Integral vs Central PCB

| Quantity | Integral (R45) | Central (DQ) | Explanation |
|----------|---------------|--------------|-------------|
| B1 at 200 A | 0.229 T | 0.377 T | Integral averages over full length incl. fringe |
| b2 at 200 A | -148 units | 3 units | Central samples symmetric midplane region |
| TF at 100 A | 1.32 T/kA | 2.16 T/kA | Same physics as B1 difference |
| Remanent B1 | ~1.8 mT | ~3.0 mT | Central iron retains more magnetisation |
| Harmonics | 30 | 15 | Larger coil resolves more harmonics |

---

## Consistency with Bottura Standard Analysis (MTA-IN-97-007)

The analysis pipeline follows the procedure described in L. Bottura,
*"Standard Analysis Procedures for Field Quality Measurement of the
LHC Magnets — Part I: Harmonics"*, MTA-IN-97-007, CERN (1997, rev. 2000).

### Pipeline Steps (Bottura reference)

| Step | Bottura ref. | Description |
|------|-------------|-------------|
| 1. Drift correction | Eq. AII.12–14 | Remove integrator DC offset (linear drift in flux) |
| 2. DFT + spectrum folding | Eq. AII.19–22 | Fourier decomposition of flux per turn |
| 3. Harmonic extraction via Kn | Eq. AII.22 | Convert flux spectra to field harmonics using coil sensitivities |
| 4. Centre localization (CEL) | Eq. AIII.1 (dipole) | Find magnetic centre by zeroing non-allowed high-order harmonics |
| 5. Feed-down correction (FED) | Eq. AIII.6 | Translate harmonics to the magnetic centre frame |
| 6. Rotation correction (ROT) | Eq. AIV.2–6 | Rotate into main-field reference frame (A_m = 0) |
| 7. Normalization | Eq. AIV.8–9 | Convert to units (1e-4 relative to B_m) |
| 8. Merge abs/cmp channels | Section 3.7 | B1 from absolute, higher harmonics from compensated |

All steps are consistent with the Bottura standard.  The merge step
(step 8) is modified for the Central PCB (`abs_all`) due to the
unusable compensated channel — see the FDI section below.

### Dipole-Specific Notes

- **Allowed harmonics** for a standard symmetric dipole: n = 1, 3, 5,
  7, 9, ... (odd).  For the MC62 C-shape, the broken left-right symmetry
  makes **all harmonics allowed** (both even and odd), which is why
  b2 is large.
- **Centre localization** for a dipole (m=1): Bottura prescribes a
  7th-degree polynomial (Eq. AIII.1) zeroing the 16-pole (n=8) or
  20-pole (n=10).  Our implementation uses a first-order linear
  approximation: `zR = -C_cmp[10] / (10 * C_cmp[11])`, which is the
  same first-order feed-down inversion but applied between compensated
  harmonics n=10 and n=11 (legacy C++ choice).  This is valid when the
  shaft offset is small relative to R_ref.  For quadrupoles and above,
  the standard linear formula (Bottura Eq. AIII.4) is used.

---

## Correction Impact Analysis (Run 02, 200 A)

### Why each correction exists

| Correction | Physics | Bottura ref. |
|------------|---------|-------------|
| **Drift** (`dri`) | Integrator electronics have a DC voltage offset that accumulates linearly in the flux signal over each turn | Eq. AII.12–14 |
| **Rotation** (`rot`) | Coil encoder index pulse is not aligned with the magnet field direction; harmonics must be rotated into the main-field frame | Eq. AIV.2–6 |
| **Centre location** (`cel`) | Coil shaft axis does not coincide with the magnetic centre; offset creates spurious feed-down between harmonics | Eq. AIII.1–5 |
| **Feed-down** (`fed`) | Translate all harmonics from the measured (off-centre) frame to the magnetic centre, removing spurious contributions from higher-order terms feeding into lower-order ones | Eq. AIII.6 |

### Measured impact at 200 A (last 8 turns averaged)

**Integral PCB** (merge: `abs_upto_m_cmp_above`):

| Options | B1 [T] | b2 [units] | b3 [units] |
|---------|-------:|-----------:|-----------:|
| (none) | -0.223130 | -153.67 | -3.95 |
| + drift | -0.223131 | -153.67 | -3.95 |
| + drift + rotation | -0.229322 | -148.24 | -2.75 |
| + drift + rot + centre | -0.229322 | -148.24 | -2.75 |
| + drift + rot + cel + fed (full) | -0.229322 | -148.24 | -2.75 |

**Central PCB** (merge: `abs_all`):

| Options | B1 [T] | b2 [units] | b3 [units] |
|---------|-------:|-----------:|-----------:|
| (none) | -0.374698 | 3.29 | 1.20 |
| + drift | -0.374698 | 3.26 | 1.14 |
| + drift + rotation | -0.376684 | 2.99 | 1.72 |
| + drift + rot + centre | -0.376684 | 2.99 | 1.72 |
| + drift + rot + cel + fed (full) | -0.376684 | 2.99 | 1.72 |

### Incremental correction contributions

| Correction | Integral dB1 | Integral db2 | Integral db3 | Central dB1 | Central db2 | Central db3 |
|------------|:------------:|:------------:|:------------:|:-----------:|:-----------:|:-----------:|
| + drift | -0.2 uT | +0.00 | -0.00 | +0.6 uT | -0.03 | -0.05 |
| + rotation | **-6192 uT** | **+5.43** | **+1.20** | **-1986 uT** | **-0.27** | **+0.58** |
| + centre loc | 0 | 0 | 0 | 0 | 0 | 0 |
| + feed-down | 0 | 0 | 0 | 0 | 0 | 0 |

### Interpretation

1. **Rotation is the dominant correction** — it shifts B1 by ~6.2 mT
   (2.8% of B1) on the Integral coil and ~2.0 mT (0.5%) on Central.
   It also changes b2 by +5.4 units and b3 by +1.2 units on the
   Integral coil.  This is physically expected: the coil encoder
   reference is not aligned with the dipole field direction, so
   rotation into the main-field frame (Bottura Eq. AIV.6) redistributes
   the signal between normal and skew components.  **This correction is
   essential and must always be enabled.**

2. **Drift correction is negligible** (<1 uT on B1, <0.05 units on
   harmonics).  This indicates clean integrator electronics with
   minimal DC offset during the 1 s turn period.  The correction is
   computationally cheap and should be kept enabled as a safeguard.

3. **Centre location has zero impact**.  For a dipole (m=1), the CEL
   algorithm uses compensated harmonics n=10 and n=11 to estimate the
   shaft offset via a first-order linear formula:
   `zR = -C_cmp[10] / (10 * C_cmp[11])`.
   In this measurement, the computed `|zR|` exceeds the `max_zR = 0.01`
   safety threshold and is clamped to zero.  This happens because the
   high-order compensated harmonics are dominated by noise rather than
   real magnetic signal — especially for the Central PCB (SNR=3 on
   compensated), but even for the Integral PCB where n=10 and n=11 are
   at the sensitivity limit (k10 ≈ 4.6e-13, k11 ≈ 1.9e-14).
   **The max_zR guard correctly prevents noise amplification through
   the feed-down Taylor expansion.**

4. **Feed-down has zero impact** because it depends on CEL: with zero
   shaft offset, the Taylor expansion (Bottura Eq. AIII.6) reduces to
   the identity transform — `C'_n = C_n` for all n.

5. **Practical significance**: even if the shaft offset were 1% of
   R_ref (= 0.33 mm), the dominant feed-down from b3 to b2 would be
   only ~0.03 units — invisible against the -148 units geometric b2.
   The magnetic centre is not operationally critical for a standalone
   warm dipole test.

### Decision Rationale

All four corrections `("dri", "rot", "cel", "fed")` are kept enabled
in the pipeline configuration, following the Bottura standard procedure:

- **rot** is essential and has large impact.
- **dri** is negligible here but is standard practice and costs nothing.
- **cel + fed** have zero impact in this measurement due to the max_zR
  safety clamp, but they are kept enabled so that the pipeline is ready
  for future measurements where a better compensated channel (with
  higher FDI gain) may yield a usable centre localisation.  The max_zR
  guard ensures they cannot introduce noise artifacts.

### The `max_zR` Safety Clamp

The `max_zR = 0.01` parameter is **not prescribed by Bottura** — it is
a numerical safety guard in our implementation.  Bottura's Eq. AIII.1
uses a cost function (Eq. AIII.2) to select the physical root from the
polynomial, which serves an analogous purpose of rejecting unphysical
solutions.

**What it does**: if the computed dimensionless centre offset `|zR|`
(= |delta_z / R_ref|) exceeds `max_zR`, it is set to zero for that turn.
This disables both CEL and feed-down for that turn.

**Why it is necessary**: the feed-down Taylor expansion computes
`C'_n = sum_{k>=n} comb(k,n) * zR^{k-n} * C_k` (Bottura Eq. AIII.6).
When `|zR|` is large due to noise, the binomial amplification factor
`comb(k,n) * |zR|^{k-n}` becomes enormous — e.g. for H=15,
`comb(14,7) * |0.5|^7 ≈ 27x` — causing catastrophic noise injection
from high harmonics into all lower harmonics.

**Does it invalidate the measurement?**  No — it **protects** it:

- In this measurement, CEL cannot determine the magnetic centre because
  the compensated harmonics n=10, n=11 are pure noise (insufficient coil
  sensitivity at these orders).
- Without the clamp, a random noise-driven `zR` would corrupt every
  harmonic via the Taylor expansion.
- With the clamp, `zR = 0` → feed-down is the identity → harmonics are
  preserved exactly as measured.
- The clamp is conservative: `max_zR = 0.01` corresponds to a maximum
  credible offset of 0.33 mm (1% of R_ref = 33 mm).  Shaft offsets
  beyond this for a well-centred rotating coil in a 100 mm gap are
  unphysical.
- If future measurements with improved FDI gains produce clean n=10, n=11
  signals, the CEL will yield `|zR| < 0.01` naturally (a real 0.3 mm
  offset) and the feed-down correction will activate automatically.

---

## File Locations

    measurements/2026_02_11_MC62/
      00_test/                          <- this test
        20260211_104646_test_REDMAGNET/
          20260211_105610_REDMAGNET/    <- 9 run files per PCB
          REDMAGNET_*_Parameters.txt    <- hardware config
      Kn values/
        Kn_R45_PCB_N1_0001_A_AC.txt    <- integral PCB calibration
        Kn_DQ_5_18_7_250_47x50_0001_A_AC.txt  <- central PCB calibration

    rotating_coil_analyzer/notebooks/
      analysis_20260211_104646_LEAR_MC62_00_test.ipynb  <- analysis notebook

    output/
      MC62_00_test_analysis_notes.md                   <- this file

---

## FDI Gain & Signal Analysis

### FDI Setup

Four FDI (Flux Digitizer Interface) channels are used, two per PCB:

| Channel | PCB | Type | FDI Gain |
|---------|-----|------|----------|
| ch1 (col 1) | Integral (R45) | Absolute | 1 |
| ch2 (col 2) | Integral (R45) | Compensated | 100 |
| ch1 (col 1) | Central (DQ) | Absolute | 1 |
| ch2 (col 2) | Central (DQ) | Compensated | 100 |

### Signal Level Analysis (200 A Plateau)

The signal quality was assessed by comparing each channel's `robust_range`
(block-averaged peak-to-peak flux variation per turn) to the noise floor
estimated from the 0 A run.

| PCB | Channel | robust_range @200 A | Noise @0 A | SNR |
|-----|---------|--------------------:|-----------:|----:|
| Integral (R45) | Absolute | 4.097e-3 | 2.069e-5 | 198 |
| Integral (R45) | Compensated | 1.604e-4 | 4.403e-7 | 365 |
| Central (DQ) | Absolute | 1.483e-4 | 7.409e-7 | 200 |
| Central (DQ) | Compensated | 9.227e-8 | 2.874e-8 | **3** |

### Problem: Central Compensated Channel (SNR = 3)

The Central PCB compensated channel has an SNR of only ~3, making it
effectively unusable for harmonic analysis.  This is caused by:

1. **Small coil**: the DQ PCB is much smaller than R45, producing
   ~27x less absolute flux (k1_abs = 0.032 vs 1.448).
2. **Very high suppression**: the DQ compensated winding has a
   suppression ratio of ~2.3e15x (vs ~12900x for R45), so the
   compensated signal at the output is astronomically small.
3. **Low gain**: the FDI gain of 100 is insufficient to amplify
   this tiny residual signal above the ADC noise floor.

### Decision: Use `abs_all` Merge for Central

Because the Central compensated channel carries no usable information,
the analysis pipeline uses `merge_mode="abs_all"` for the Central PCB:
**all harmonics** (including n>1) are derived from the absolute channel
only.  This means the Central harmonics do not benefit from compensated
bucket suppression, but the absolute signal has adequate SNR (~200) and
provides reliable results.

The Integral PCB retains the standard `merge_mode="abs_upto_m_cmp_above"`
(B1 from absolute, higher harmonics from compensated), since both its
channels have excellent SNR (>190).

### Gain Recommendations for Next Measurements

The `00_test` measurement was taken with **abs gain = 1, cmp gain = 100**.
The recommendations below are for **future measurements** based on the
signal analysis of this test.

**Hardware constraint**: the FDI software allows only one **global**
gain for all absolute channels and one global gain for all compensated
channels.  Available gains: 0.1, 0.2, 0.4, 0.5, 1, 2, 4, 5, 10, 20,
40, 50, 100.  Maximum gain = 100.

The global gain must be chosen so that the **largest signal** (always
the Integral PCB, which has ~28x more absolute flux and ~1740x more
compensated flux than the Central PCB) does not saturate the ADC.

**Physical signal levels at 200 A (gain=1 equivalent):**

| Channel | Integral (R45) | Central (DQ) | Ratio |
|---------|---------------:|-------------:|------:|
| Absolute | 4.097e-3 | 1.483e-4 | 28x |
| Compensated | 1.604e-6 | 9.23e-10 | 1740x |

**Empirical clipping tests (Integral absolute at 200 A):**

| Abs gain | Result |
|:--------:|--------|
| 10 | **Clips hard** — square-wave saturation at ~0.001 Wb |
| 5 | **Clips slightly** — only the tips cut at ~0.002 Wb |
| 4 | **OK** — no clipping |

**Compensated overflow tests:**

The "o" (overflow) on the FDI display was observed during current
**ramps** at cmp gain=100 AND at cmp gain=50.  This confirms the
overflow is caused by the large dB/dt during ramps, not by the plateau
signal.  The plateau data at gain=100 was verified clean (all 92,160
samples checked — zero stuck values, zero consecutive identical samples
at min/max, stable per-turn range).  Since ramp data never enters the
measurement files, the overflow indicator is cosmetic and does not
affect the analysis.

**Recommended FDI settings:**

| Setting | `00_test` (current) | Next measurements | Effect on Integral | Effect on Central |
|---------|:-------------------:|:-----------------:|:------------------:|:-----------------:|
| Abs gain | 1 | **4** | SNR 198 → ~800 | SNR 200 → ~800 |
| Cmp gain | 100 | **100** (keep) | SNR 365 (unchanged) | SNR 3 (unusable, unchanged) |

**Rationale:**

- **Abs = 4**: provides 20% headroom below the clipping point observed
  at gain=5, while improving SNR by 4x on both PCBs.
- **Cmp = 100** (keep): the overflow is ramp-only and harmless.
  Keeping gain=100 preserves the full compensated SNR=365 for the
  Integral PCB, giving the best possible harmonic precision.
  Reducing to 50 would halve the SNR for no analytical benefit.

**Merge mode validation (abs_upto_m_cmp_above vs abs_all, Integral PCB):**

A direct comparison on the `00_test` data (cmp gain=100, SNR=365)
confirms that the compensated channel delivers dramatically better
harmonic precision:

| Harmonic | Scatter with cmp (avg std) | Scatter with abs_all (avg std) | Cmp advantage |
|----------|:--------------------------:|:------------------------------:|:-------------:|
| b2 | 0.13 units | 0.71 units | 5.5x better |
| b3 | 0.06 units | 0.73 units | 12x better |

The compensated winding suppresses the dominant B1 signal, giving the
ADC much better dynamic range for the small field errors.  The `abs_all`
mode also shows a systematic ~1.2 unit shift in mean values, likely from
residual coil geometry errors that the compensated design cancels.

**Conclusion**: `abs_upto_m_cmp_above` remains the correct merge mode
for the Integral PCB at cmp gain=100.  The compensated channel is 5-12x
more precise than the absolute channel for higher harmonics.

**Central compensated is a fundamental limitation:**

The Central (DQ) compensated channel has SNR ≈ 3 at gain = 100 (the
maximum available).  This cannot be improved with current FDI hardware.
The root cause is the combination of a small coil (45x less flux than
R45), essentially perfect compensation (suppression ≈ 2.3e15x), and
maximum gain capped at 100.  The `abs_all` merge mode is therefore
**mandatory** for the DQ PCB and will remain so unless the FDI hardware
is upgraded to support higher gains or an external pre-amplifier is
added.
