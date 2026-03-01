# SPS MBB Dipole — 200 GeV vs 26 GeV Comparison

## Overview

Compares harmonic results between two measurement sessions of the same
SPS MBB dipole (NCS segment) at 200 GeV and 26 GeV beam energies.  Both
sessions have identical supercycle structure (**LHC_pilot -> MD1 -> SFTPRO
x 20**) but were measured ~30 min apart with different magnetisation
histories.

The notebook addresses three physics questions:

1. **Eddy-current settling** — how fast does the field transient decay
   after the ramp to injection, and how many exponential components are
   needed?
2. **Magnetic hysteresis** — does the injection field depend on the
   preceding cycle (200 GeV vs 26 GeV)?
3. **Eddy / hysteresis separation** — how do we prove the observed offset
   is static hysteresis and not residual eddy currents?

---

## Configuration

| Parameter | Value |
|-----------|-------|
| 200 GeV session | `2026_02_06/01_200_extended` |
| 26 GeV session | `2026_02_06/03_26_extended` |
| Segment | NCS (non-connection side) |
| Operating points | MD1 injection (~301 A) and SFTPRO flat-top (~4,815 A) |
| Settling correction | Last 18 of ~23 injection turns per supercycle |
| End-SC trim | 1 turn per SC (ramp contamination) |
| Min SC turns | 15 (skip partial fragments) |
| Outlier removal | 5-sigma MAD clipping on B1 |
| cel/fed | Disabled (UNSAFE diagnostic: median |zR| = 0.051) |
| Pipeline options | `("dri", "rot")` |

## Data Summary

| Property | 200 GeV | 26 GeV |
|----------|---------|--------|
| Total turns | 1,061 | 1,064 |
| Injection turns (settled) | 360 | 360 |
| Flat-high turns (SFTPRO) | 72 | 69 |
| Supercycles matched | 20 | 20 |

---

## Notebook Structure

| Section | Cell ID(s) | Content |
|---------|-----------|---------|
| 1. Configuration | `config` | Paths, pipeline options, thresholds |
| 2. Imports | `imports` | Libraries, Kn loading |
| 3. cel/fed diagnostic | `cel-fed` | Safety check → UNSAFE, cel/fed disabled |
| 4. Helper | `helper` | `load_and_process()` — SC grouping, settling, sigma clip |
| 5. Load data | `load-both` | Both datasets processed |
| 6. Current profiles | `current-profiles` | I(t) overview, 1x2 |
| 7. Per-SC injection | `per-sc-inj` | B1, b2, b3 per SC (error bars) |
| **7b. Eddy settling** | `sec7b-hdr`, `eddy-settling-B1` | **2x2: traces, overlay, 2-tau fit** |
| 8. SFTPRO flat-top | `sftpro-per-sc` | Per-SC at high current |
| 9. Summary stats | `summary-stats` | Mean/std table |
| 10. Comparison plots | `comparison-plots` | B1, b2, b3 time series |
| 11. Box plots | `boxplots` | Distribution comparison |
| 12. Significance | `diff-table` | Sigma test: B1 (22σ), b3 (4.8σ) |
| **12b. Hysteresis** | `9edeyrijch`, `77uo8su1kuv` | **Per-SC delta B1, delta b3 (bar charts)** |
| **12c. Eddy/hyst separation** | `zvg4i48taxk`, `bhk6s38ufav` | **delta(t) flatness, decomposition** |
| 13. Export | `export` | CSV output |

---

## 1. Eddy-Current Settling Theory

### Origin

When the magnet current changes rapidly — here from the LHC pilot flat-top
(~5785 A) down to the injection plateau (~301 A) — **eddy currents** are
induced in every conductive loop that threads the changing flux: yoke
laminations, beam screen, coil wedges, end plates, etc.  Each such loop *i*
has its own *L/R* time constant

    tau_i = L_i / R_i

and generates a transient field perturbation that decays exponentially.

### Multi-exponential model

The total eddy-current contribution to the main dipole field is

    dB1(t) = Sum_i  A_i * exp(-t / tau_i)

| Symbol | Meaning |
|--------|---------|
| t | time since the start of the injection plateau (s) |
| A_i | amplitude of the i-th eddy-current component (µT) — the field it contributes at t = 0 |
| tau_i | decay time constant (s) of the i-th component |

**No offset term** appears because dB1 is defined as the deviation from
the settled (long-time) mean, so dB1 → 0 by construction.

### Models tested

| Model | Expression | Free parameters |
|-------|------------|-----------------|
| 1-tau | A1 * exp(-t/tau1) | 2 |
| 2-tau | A1 * exp(-t/tau1) + A2 * exp(-t/tau2) | 4 |
| 3-tau | A1 * exp(-t/tau1) + A2 * exp(-t/tau2) + A3 * exp(-t/tau3) | 6 |

Selection by **R²** (coefficient of determination).  Overfitting is flagged
when two tau values are within 20% of each other with opposite-sign
amplitudes (near-degenerate pair).

### Physical interpretation

The initial field perturbation at t = 0 is

    dB1(0) = Sum_i  A_i

Each A_i quantifies how much field the i-th eddy-current path "injects"
into the aperture:

- **Fast component** (tau < 0.5 s): thin structures (beam screen, copper wedges)
- **Slow component** (tau ~ 2–5 s): bulk yoke laminations

### Worked example — 200 GeV, 2-tau fit

The best fit gives:

    dB1(t) = -5.1 µT * exp(-t / 0.035 s)  +  (-17.2 µT) * exp(-t / 2.48 s)
              ~~~~~~ fast ~~~~~~~~~~~~~~      ~~~~~~~~ slow ~~~~~~~~~~~~~~~

At the moment the current settles on the injection plateau, eddy currents
add dB1(0) = -22.3 µT (≈ -193 ppm of B1 = 115.6 mT) to the dipole
field.  The two components then decay at very different rates:

    t (s)    Fast (tau=0.035 s)   Slow (tau=2.48 s)    Total        % of dB1(0)
    -----    ------------------   -----------------    ----------   -----------
    0.0      -5.1 µT              -17.2 µT             -22.3 µT     100 %
    0.035    -1.9 µT              -17.0 µT             -18.8 µT      84 %
    0.1      -0.3 µT              -16.5 µT             -16.8 µT      75 %
    1.0       ~0                  -11.5 µT             -11.5 µT      52 %
    2.0       ~0                   -7.7 µT              -7.7 µT      35 %
    5.0       ~0                   -2.3 µT              -2.3 µT      10 %
    7.1       ~0                   -1.0 µT              -1.0 µT       4.5 %
    10        ~0                   -0.3 µT              -0.3 µT       1.4 %

**Key observations:**

- **Fast component** (tau = 35 ms, A = -5.1 µT): decays within a fraction
  of one turn — unresolvable at 1 s sampling, so its amplitude is
  effectively an extrapolation.  Likely originates from thin conductive
  structures (beam screen, wedge copper).

- **Slow component** (tau = 2.48 s, A = -17.2 µT): dominates the visible
  settling.  Carries 77% of the initial perturbation and takes ~3*tau =
  7.4 s to drop below 5% of dB1(0).  Consistent with eddy currents in the
  bulk yoke laminations.

- **Settling cut-off**: skipping the first ~5 turns (N_LAST_TURNS_INJ = 18
  out of ~23) leaves a residual ~2.3 µT — well within the turn-to-turn
  measurement noise (sigma ~ 5 µT).

- **26 GeV comparison**: similar time constants (tau_fast = 37 ms,
  tau_slow = 3.02 s), slightly smaller initial perturbation
  (dB1(0) = -19.6 µT) — consistent with the same magnet under a different
  excitation history.

### Fit results

Both datasets select **2-tau** as the best model (3-tau flagged as overfit
for 200 GeV):

| Dataset | Best model | tau_fast (s) | A_fast (µT) | tau_slow (s) | A_slow (µT) | dB1(0) (µT) | R² |
|---------|-----------|-------------|-------------|-------------|-------------|-------------|-----|
| 200 GeV | 2-tau | 0.035 | -5.1 (23%) | 2.48 | -17.2 (77%) | -22.3 | 0.966 |
| 26 GeV | 2-tau | 0.037 | -4.1 (21%) | 3.02 | -15.5 (79%) | -19.5 | 0.962 |

Turn period: T_turn = 1.000 s for both datasets.

---

## 2. Hysteresis Comparison Theory

### B–H hysteresis in accelerator magnets

In a ferromagnetic yoke the relationship between field B and excitation
current I is not single-valued: it depends on the **magnetic history**
(the B–H hysteresis loop).  At any given operating current I_0, the field
B(I_0) differs depending on whether the magnet was previously excited to
a higher or lower maximum current I_max.

### Cycle-dependent remanent magnetisation

The two measurement campaigns share the same injection current
(I_inj ≈ 301 A) but differ in the preceding cycle's peak excitation:

| Campaign | Preceding peak | Expected remanent state |
|----------|---------------|------------------------|
| 200 GeV | ~5785 A (LHC pilot) | Iron driven further into saturation → larger remanent M_r → return branch sits **lower** on the B–H plane |
| 26 GeV | ~1700 A (26 GeV cycle) | Lower saturation → smaller remanent M_r → return branch sits **higher** |

### Hysteresis width definition

The hysteresis width at injection is defined as

    delta_B1_hyst = <B1>_200GeV  -  <B1>_26GeV

and similarly for the relative sextupole

    delta_b3_hyst = <b3>_200GeV  -  <b3>_26GeV

A negative delta_B1 means the 200 GeV cycle produces a **lower** field at
injection — consistent with the deeper-saturation return branch.

### Why higher harmonics are affected

The iron yoke's permeability mu_r is spatially non-uniform (pole tips vs
mid-plane vs return yoke).  When mu_r changes with the remanent state, the
harmonic content of the field also shifts.  The sextupole (b3) is
particularly sensitive because the n = 3 harmonic originates from the
boundary between the pole tip and the yoke gap, where saturation effects
are strongest.

### Statistical significance

Only quantities exceeding 3 sigma are considered real:

    sigma = |delta| / sqrt(std1²/N1 + std2²/N2)

where std1, std2 are the turn-to-turn standard deviations and N1, N2 are
the number of settled turns.  High sigma means reliably detected, not
necessarily large.

---

## 3. Separating Eddy Currents from Hysteresis

### The ambiguity

Both eddy currents and hysteresis produce a field perturbation at injection.
A naive comparison of the two datasets' mean B1 cannot, by itself, tell
which effect dominates — unless we exploit their different **time
signatures**.

### Distinguishing properties

| Property        | Eddy currents                       | Hysteresis                        |
|-----------------|-------------------------------------|-----------------------------------|
| Time dependence | Decays exponentially within each SC | Constant offset between datasets  |
| Origin          | Faraday induction (dI/dt in conductive loops) | Remanent magnetisation (B–H loop in iron yoke) |
| Scope           | Within-dataset transient: dB1(t) → 0 | Between-dataset static offset     |
| Current dependence | Proportional to dI/dt before plateau | Depends on I_max of preceding cycle |

### Decomposition

The per-turn offset between datasets decomposes as:

    delta(t) = delta_B1_hyst  +  [eddy_200(t) - eddy_26(t)]
               ~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               constant         "differential eddy" (decays)

If the eddy-current parameters (A, tau) were identical for both datasets,
the differential eddy would be zero and delta(t) = constant = pure
hysteresis at all times.

In practice, the two datasets have slightly different eddy parameters
(different dB1(0) and tau) because they have different excitation histories
before the injection plateau.

### Predicted differential eddy contamination

Using the 2-tau fit parameters:

    t (s)   eddy_200    eddy_26    diff       note
    -----   --------    -------    ------     ----
      0     -22.3 µT    -19.5 µT   -2.8 µT   200 GeV has larger dB1(0)
      1     -11.5 µT    -11.1 µT   -0.4 µT   converging
      2      -7.7 µT     -8.0 µT   +0.3 µT   sign flip (26 GeV has slower tau)
      5      -2.3 µT     -3.0 µT   +0.7 µT   settled region starts here
     10      -0.3 µT     -0.6 µT   +0.3 µT
     18      -0.0 µT     -0.0 µT   +0.0 µT   both negligible

The sign flip at t ≈ 1.5 s occurs because the 26 GeV eddy has a longer
tau_slow (3.02 s vs 2.48 s).  At early times the 200 GeV eddy is larger
(bigger amplitude); at later times the 26 GeV eddy persists longer (slower
decay).

### Quantitative conclusion

Over the settled window (t >= 5 s):

    Mean differential eddy:     +0.2 µT
    Measured hysteresis offset: -7.4 µT
    Eddy contamination:          2.7% of the hysteresis (opposite sign)

The differential eddy is **positive** while the hysteresis is **negative**,
so they partially cancel — the **true hysteresis is slightly larger** than
the measured offset (approximately -7.6 µT corrected vs -7.4 µT raw in the
settled region).

### Verification: per-turn delta(t) plot

The notebook (section 12c) plots the actual per-turn offset
delta(t) = <B1>_200(t) - <B1>_26(t) across the full injection plateau
(~23 turns, averaged over 20 SCs).

**Result**: delta(t) is **flat** across the plateau within error bars.
The data points scatter around -7.4 µT from turn 5 to turn 22 with no
visible exponential trend.  Only the first 1-2 turns show a slightly more
negative offset (~-10 µT), consistent with the predicted -2.8 µT
differential eddy at t = 0 added to the -7.4 µT hysteresis.

The green prediction curve (hysteresis + differential eddy from fits)
matches the data well — confirming the decomposition is quantitatively
consistent.

**Bottom line**: the -7.6 µT offset is genuine hysteresis from the iron
yoke's remanent magnetisation, not residual eddy currents.

---

## Results

### Statistical Significance (Section 12)

#### At Injection (~301 A)

| Quantity | 200 GeV | 26 GeV | Delta | Significance | Verdict |
|----------|---------|--------|-------|-------------|---------|
| B1 (T) | 0.115633 | 0.115641 | -7.6 µT | 22.2 sigma | **REAL** |
| b2 (units) | -1.125 | -1.132 | +0.007 | 0.5 sigma | no evidence |
| b3 (units) | +0.177 | +0.198 | -0.021 | 4.8 sigma | **REAL** |
| TF (T/kA) | 0.3843 | 0.3843 | ~0 | 1.7 sigma | no evidence |

#### At SFTPRO Flat-Top (~4,815 A)

| Quantity | 200 GeV | 26 GeV | Delta | Significance | Verdict |
|----------|---------|--------|-------|-------------|---------|
| B1 (T) | 1.793681 | 1.793817 | -136 µT | 1.7 sigma | no evidence |
| b2 (units) | -0.945 | -0.964 | +0.020 | 1.9 sigma | no evidence |
| b3 (units) | +0.345 | +0.355 | -0.011 | 1.5 sigma | no evidence |
| TF (T/kA) | 0.3726 | 0.3726 | ~0 | 2.8 sigma | suggestive |

### Hysteresis Analysis (Section 12b)

Only B1 and b3 at injection exceed 3 sigma.  Per-supercycle matched
analysis (20 SC pairs):

| Quantity | delta (hyst) | Std across SCs | Stability (std/|mean|) | Relative |
|----------|-------------|---------------|----------------------|----------|
| B1 | -7.6 ± 2.0 µT | 2.0 µT | 26% | -66 ppm |
| b3 | -0.021 ± 0.044 units | 0.044 units | 209% | — |

**Interpretation:**

- **B1**: The hysteresis offset is small (-7.6 µT ≈ 66 ppm) but highly
  reproducible across 20 supercycles (26% relative spread).  The 200 GeV
  cycle produces a lower field at injection, consistent with the
  deeper-saturation return branch of the B–H loop.

- **b3**: The mean offset (-0.021 units) is statistically significant when
  averaged over 360 turns, but the per-supercycle variability (std = 0.044)
  is larger than the mean — the offset is real on average but noisy
  cycle-to-cycle.

- **All other quantities** (b2, TF, everything at SFTPRO) are not
  statistically distinguishable between the two sessions.

### Eddy / Hysteresis Separation (Section 12c)

| Test | Result |
|------|--------|
| Per-turn delta(t) shape | **Flat** across plateau (no exponential trend) |
| Differential eddy in settled window | +0.2 µT (2.7% of hysteresis, opposite sign) |
| Predicted delta(t) vs data | Good agreement (green curve matches red dots) |
| Conclusion | Offset is **genuine hysteresis**, not eddy contamination |

---

## Key Findings

1. **Eddy-current settling** is well described by a **2-tau model**:
   - Fast component (tau ~ 35 ms, ~20% of dB1): sub-turn decay, likely
     beam screen or copper wedges.  Unresolvable at 1 s sampling.
   - Slow component (tau ~ 2.5–3.0 s, ~80% of dB1): visible settling over
     ~7 turns, consistent with bulk yoke laminations.
   - Initial perturbation: -22 µT (200 GeV), -20 µT (26 GeV) ≈ 170–190 ppm.
   - Settling cut-off (skip first 5 turns) leaves ~2.3 µT residual, within
     the 5 µT noise floor.

2. **Injection B1 hysteresis**: -7.6 µT (22.2 sigma, 66 ppm).  The 200 GeV
   cycle produces a lower field, consistent with the deeper-saturation
   return branch.  Reproducible across 20 SCs (26% relative spread).

3. **Injection b3 hysteresis**: -0.021 units (4.8 sigma).  Real on average
   but noisy cycle-to-cycle (std = 0.044 > mean = 0.021).

4. **SFTPRO** quantities are not significantly different — at high current
   the iron is re-saturated and the magnetic history is largely erased.

5. **Eddy / hysteresis separation** confirmed: the per-turn offset delta(t)
   is flat across the plateau.  Differential eddy contamination is only
   2.7% of the hysteresis and of opposite sign.  The offset is genuine
   iron hysteresis.

## Observations

1. The two datasets were measured ~30 min apart with different
   magnetisation histories, so differences reflect cumulative history
   effects (hysteresis), not instrumental drift.

2. The high B1 significance (22.2 sigma) is driven by very low
   turn-to-turn scatter (5 µT std) combined with 360 turns per dataset.
   The hysteresis itself is tiny (66 ppm) but the measurement precision
   is sufficient to detect it clearly.

3. The b3 hysteresis offset, while real in the global mean, has large
   per-SC variance — a longer measurement (more supercycles) would
   improve confidence in the per-SC stability.

4. The 26 GeV eddy decays slightly slower than the 200 GeV eddy
   (tau_slow = 3.02 s vs 2.48 s).  This may reflect the different
   iron permeability state (less saturated yoke → higher mu_r → longer
   L/R time constant), but the difference is within the fit uncertainty.

### cel/fed Safety Diagnostic

This notebook includes a `diagnose_cel_fed()` check that verifies the
centre-location and feeddown corrections (cel/fed) are reliable.  The
diagnostic compares pipeline results with and without cel/fed, flags turns
with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation.
For this dataset, cel/fed was found **UNSAFE** (median |zR| = 0.051,
100% of turns exceed threshold) and was disabled.  See
`correction_options_reference.md` for background.

## Output Files

- `output/2026_02_06/compare_200_vs_26/plateau_harmonics_200GeV.csv` (562 rows)
- `output/2026_02_06/compare_200_vs_26/plateau_harmonics_26GeV.csv` (592 rows)
- `output/2026_02_06/compare_200_vs_26/plateau_harmonics_200GeV_settled.csv` (432 rows)
- `output/2026_02_06/compare_200_vs_26/plateau_harmonics_26GeV_settled.csv` (429 rows)
- `output/2026_02_06/compare_200_vs_26/per_supercycle_injection_200GeV.csv` (20 rows)
- `output/2026_02_06/compare_200_vs_26/per_supercycle_injection_26GeV.csv` (20 rows)
- `output/2026_02_06/compare_200_vs_26/summary_comparison_settled.csv` (4 rows)
