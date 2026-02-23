# SPS MBB Dipole — 200 GeV vs 26 GeV Extended Comparison — Analysis Notes

## Magnet Description

- **Name**: MBB (SPS Main Bending dipole)
- **Type**: H-frame warm iron-dominated dipole
- **Machine**: SPS (Super Proton Synchrotron), CERN
- **Excitation range**: 0 to ~4815 A (operational cycle)

## Rotating Coil Setup

- **Segment analysed**: NCS (non-connection side)
- **Rotation**: 1024 samples/turn
- **Kn calibration**: cross-session file from a prior MBA calibration
  (`20251212_171026_SPS_MBA/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt`),
  15 harmonics.  The in-session Kn files (`Kn_values_Seg_NCS.txt`) are all
  zeros and must **not** be used.
- **Raw data format**: 5 columns per file —
  `[time_s, flux_ch1, flux_ch2, I_DCCT_A, col5]`.
  Auto-detection (`robust_range`) determines which flux column is absolute
  vs compensated.

## Measurement Campaigns

Both measurements were performed on 2026-02-06 on the same SPS MBB dipole,
with different SPS cycle configurations:

| Dataset | Session | Timestamp | Description |
|---------|---------|-----------|-------------|
| **200 GeV extended** | `01_200_extended` | 2026-02-06 14:45 | Extended measurement, 200 GeV MD1 cycle |
| **26 GeV extended** | `03_26_extended` | 2026-02-06 15:18 | Extended measurement, 26 GeV MD1 cycle |

### Supercycle Structure

Both datasets share the supercycle structure:

    LHC_pilot -> MD1 -> SFTPRO   (repeated ~20 times)

- **MD1 injection plateau**: ~301 A (~24 turns per supercycle)
- **SFTPRO flat-top**: ~4815 A (~3-4 turns per supercycle)

Total measurement time per dataset: ~17.7 minutes (~1061-1064 turns).

---

## Pipeline Configuration

- **Options**: `("dri", "rot", "cel", "fed")` — drift, rotation, centre
  location, feed-down corrections all enabled.
- **Drift mode**: `"legacy"` (C++ compatible).
- **Merge strategy**: `abs_upto_m_cmp_above` — B1 from absolute channel,
  higher harmonics (n>1) from compensated.
- **Reference radius**: R_ref = 20 mm.
- **Magnet order**: m = 1 (dipole).
- **Normalization**: min_b1_T = 1e-4 T.
- **Centre location guard**: max_zR = 0.01.

---

## Plateau Detection

Current plateau turns are identified using the three-rule block-averaged
detection from `utility_functions.py`:

1. **Range rule**: block-averaged I range < 2.5 A
2. **Start boundary**: |first-block mean - turn mean| < 2.5 A
3. **End boundary**: |last-block mean - turn mean| < 2.5 A

Current classification thresholds (SPS defaults):

| Label | Upper bound (A) |
|-------|----------------:|
| zero | 50 |
| pre-ramp | 200 |
| injection | 500 |
| flat-low | 2000 |
| flat-mid | 4000 |
| flat-high | >4000 |

### Detected Plateau Turns

| Dataset | Injection | Flat-high | Total plateau | Total turns |
|---------|----------:|----------:|--------------:|------------:|
| 200 GeV | 488 | 74 | 562 | 1061 |
| 26 GeV  | 520 | 72 | 592 | 1064 |

---

## Settling Correction

At the injection plateau (~301 A), each supercycle's ~24 turns show
**eddy-current settling** in the first few turns.  The eddy-current time
constants (from a separate eddy-current analysis on this magnet family)
are in the range tau ~ 0.5-5 s, meaning the first ~6 turns (at ~1 s/turn)
are contaminated.

### Approach

1. **Group** injection turns by supercycle using `find_contiguous_groups()`
   on the injection boolean mask (min_length=2).
2. Within each supercycle group, **keep only the last N_LAST_TURNS_INJ = 18**
   turns (out of ~24), discarding the first ~6 turns where eddy currents
   are still settling.
3. For SFTPRO flat-top (N_LAST_TURNS_HIGH = None): use **all** turns —
   no settling concern at the cycle top where dB/dt is small.

### Result

| Dataset | Supercycles | Injection: all -> settled | Flat-high: all -> settled |
|---------|------------:|-------------------------:|--------------------------:|
| 200 GeV | 20 | 488 -> 360 | 74 -> 72 |
| 26 GeV  | 20 | 520 -> 360 | 72 -> 72 |

The settling effect is visible in the per-supercycle delta-B1 plots: the
first ~6 turns of each injection plateau show a systematic offset of
~10-30 uT from the settled mean, decaying exponentially.

---

## Outlier Removal

### Method: MAD-Based Sigma Clipping

After settling selection, outliers are removed using **Median Absolute
Deviation (MAD) sigma clipping** on B1, applied independently per
operating point (injection / flat-high):

    MAD = median(|x_i - median(x)|)
    sigma_MAD = 1.4826 * MAD     (Gaussian-equivalent sigma)
    outlier if |x_i - median(x)| > N_SIGMA_CLIP * sigma_MAD

Configuration: **N_SIGMA_CLIP = 5** (conservative; only catches extreme
outliers).

### Outliers Found

| Dataset | Operating point | Turns removed | Details |
|---------|-----------------|:-------------:|---------|
| 200 GeV | injection | 0 | — |
| 200 GeV | flat-high | 2 | Ramp-boundary turns |
| 26 GeV  | injection | 3 | Anomalous B1 values (MAD clipping) |
| 26 GeV  | flat-high | 3 | Ramp-boundary turns |

The removed turns are the **last turn of their respective supercycle
groups** — the ramp has already started within the turn, so while the
block-averaged I range still passes the 2.5 A plateau threshold, the
current is changing within the turn and the field value is contaminated.

These ramp-boundary turns have characteristically high I_range (0.97-1.94 A
vs typical 0.5-0.7 A for clean plateau turns) and anomalously low B1.

### Final Turn Counts (After Settling + Sigma Clipping)

| Dataset | Injection (settled) | Flat-high (clean) |
|---------|--------------------:|------------------:|
| 200 GeV | 360 | 72 |
| 26 GeV  | 357 | 69 |

---

## Reference Radius

### Choice

    R_ref = 20 mm

This is the standard reference radius for SPS MBB dipole harmonic
analysis.

### How it affects the numbers

- **B1 (main field in Tesla)**: completely unaffected by R_ref.
- **Transfer function B1/I**: completely unaffected by R_ref.
- **Relative harmonics** bn, an (units of 1e-4 relative to B1):
  scale as `(R_new / R_old)^(n-1)`.

---

## Results: Summary Statistics (Settled + Cleaned Turns)

| Dataset | Operating point | N turns | I mean (A) | B1 mean (T) | B1 std (T) | b2 mean (units) | b2 std (units) | b3 mean (units) | b3 std (units) | TF (T/kA) |
|---------|-----------------|--------:|-----------:|------------:|-----------:|----------------:|---------------:|----------------:|---------------:|----------:|
| 200 GeV | Injection (MD1) | 360 | 300.9 | 0.115634 | 0.000005 | -1.1301 | 0.1628 | +0.1766 | 0.0605 | 0.3843 |
| 200 GeV | Top of SFTPRO | 72 | 4814.5 | 1.793681 | 0.000472 | -0.9445 | 0.0576 | +0.3447 | 0.0418 | 0.3726 |
| 26 GeV | Injection (MD1) | 357 | 300.9 | 0.115641 | 0.000005 | -1.1325 | 0.1684 | +0.1976 | 0.0593 | 0.3843 |
| 26 GeV | Top of SFTPRO | 69 | 4814.5 | 1.793817 | 0.000462 | -0.9643 | 0.0636 | +0.3554 | 0.0418 | 0.3726 |

---

## Transfer Function

### Definition

    TF = B1 / I   [T/kA]

### Values

| Operating point | I (A) | TF 200 GeV (T/kA) | TF 26 GeV (T/kA) |
|-----------------|------:|-------------------:|------------------:|
| Injection (MD1) | 300.9 | 0.3843 | 0.3843 |
| Top of SFTPRO | 4814.5 | 0.3726 | 0.3726 |

### Saturation

The transfer function drops from 0.384 T/kA at injection (301 A) to
0.373 T/kA at SFTPRO top (4815 A) — a **3.1% reduction**, indicating
moderate iron saturation at the SFTPRO flat-top current.

### Comparison

At injection, TF is identical between datasets (no measurable difference).

At SFTPRO, the per-turn TF distribution reveals a small but notable
offset: the 200 GeV dataset has TF ~ 0.00003 T/kA lower than the
26 GeV dataset (**2.8 sigma**).  This is the most statistically
significant difference in this analysis.  Dividing B1 by I removes
the current-correlated scatter in B1, leaving a tighter distribution
that makes the underlying offset more visible.  See the SFTPRO section
and the Conclusion for interpretation.

---

## Important Note on Magnetisation History

Iron hysteresis has memory extending **many cycles back**, not just the
immediately preceding one.  The two datasets were measured in separate
sessions (~30 minutes apart) with different cycle configurations and
different pre-measurement magnetisation states.  Therefore:

- Differences at **either** operating point may reflect history-dependent
  effects from the full magnetisation history, not just the last cycle.
- We cannot claim "same history" at injection just because the
  supercycle structure is the same — the state before the first
  supercycle differs between the two sessions.
- The SFTPRO comparison is particularly interesting because the
  immediately preceding MD1 differs (200 vs 26 GeV), but injection
  differences are equally valid indicators of history effects.

---

## Injection Comparison (~301 A)

### Context

At the MD1 injection plateau (~301 A), both datasets follow the same
within-supercycle ramp pattern (LHC_pilot -> ramp-up to injection).
However, the full magnetisation history differs between the two sessions,
so differences at injection cannot be excluded.

### B1 at Injection

| Dataset | B1 mean (T) | B1 std (T) | N turns |
|---------|------------:|-----------:|--------:|
| 200 GeV | 0.115634 | 0.000005 | 360 |
| 26 GeV  | 0.115641 | 0.000005 | 357 |

Difference: **-7 µT (~22 sigma)**.

This difference is **real** (detected with very high confidence) but
**tiny** — only 60 ppm of B1.  The sigma is extremely high because
both datasets have very low turn-to-turn scatter (5 µT std), and with
~360 turns each, the uncertainty on each mean is only ~0.3 µT.  Even
a 7 µT shift is easily resolved against that precision.

After MAD sigma clipping, the 26 GeV dataset's injection scatter dropped
from 71 µT to 5 µT (3 outlier turns removed), making the two datasets
similarly precise.

---

## SFTPRO Flat-Top: Effect of Preceding MD1 Cycle

### Context

The SFTPRO flat-top (~4815 A) is the critical comparison point because it
is preceded by **different MD1 cycles** in each dataset:

- **200 GeV dataset**: SFTPRO follows a 200 GeV MD1 (flat-top at higher
  current, then ramp to SFTPRO)
- **26 GeV dataset**: SFTPRO follows a 26 GeV MD1 (flat-top at lower
  current, then ramp to SFTPRO)

If the iron magnetisation from the MD1 flat-top is not fully erased by
the subsequent ramp to SFTPRO, the B1 (and harmonics) at the SFTPRO
plateau should differ.  This is a direct test of **history-dependent
permeability** at high field.

### B1 at SFTPRO Flat-Top

| Dataset | Preceding MD1 | B1 mean (T) | B1 std (T) |
|---------|--------------|------------:|-----------:|
| 200 GeV | 200 GeV MD1 | 1.793681 | 0.000472 |
| 26 GeV  | 26 GeV MD1  | 1.793817 | 0.000462 |

Difference: **-136 uT +/- 79 uT** (~1.7 sigma).

The dataset preceded by a 26 GeV MD1 sees a slightly higher B1 at SFTPRO
than the one preceded by a 200 GeV MD1.  This ~136 uT difference is
marginally significant — it is larger than measurement noise but does not
reach the conventional 3-sigma threshold.

### Full SFTPRO Comparison

| Quantity | 200 GeV dataset | 26 GeV dataset | Difference | Significance |
|----------|----------------:|---------------:|-----------:|-------------:|
| B1 (T) | 1.793681 +/- 0.000472 | 1.793817 +/- 0.000462 | -0.000136 | 1.7 sigma |
| b2 (units) | -0.9445 +/- 0.0576 | -0.9643 +/- 0.0636 | +0.0198 | 1.9 sigma |
| b3 (units) | +0.3447 +/- 0.0418 | +0.3554 +/- 0.0418 | -0.0107 | 1.5 sigma |
| TF (T/kA) | 0.37261 +/- 0.00010 | 0.37264 +/- 0.00010 | -0.00003 | **2.8 sigma** |

N turns: 200 GeV = 72, 26 GeV = 69.

### Transfer Function at SFTPRO: 2.8 Sigma

The transfer function TF = B1/I is the most sensitive indicator because
dividing by the per-turn current removes the current-correlated component
of the B1 scatter.  The raw B1 std (~0.47 mT) includes both field noise
and the effect of the ~1.8 A current spread at the SFTPRO flat-top.
When normalised to TF, this current-correlated scatter cancels, leaving
a much tighter distribution (TF std ~ 0.10 mT/kA).

The result is that the ~0.03 mT/kA TF difference between datasets
reaches **2.8 sigma** — the most statistically significant difference
in this analysis, approaching the conventional 3-sigma threshold.

This TF offset means that the iron permeability at 4815 A is slightly
different between the two sessions.  Since TF depends only on the
B-H curve at the operating current, this is a direct signature of
history-dependent permeability: the iron arrives at the same current
with a different microscopic domain structure, leading to a different
incremental permeability.

The b2 difference (1.9 sigma) and B1 difference (1.7 sigma) are
consistent in direction.  b3 is not significant (1.5 sigma).

### Interpretation

The per-supercycle SFTPRO analysis (20 supercycles per dataset) shows that
the B1 offset between datasets is **consistent across all 20 supercycles**
(i.e., it is not a drift but a systematic offset).  This is consistent
with a real, small history-dependent effect: the iron magnetisation state
at SFTPRO retains a memory of the preceding MD1 flat-top current.

However, the effect is small (~76 ppm of B1) and marginally significant.
The full magnetisation history differs between the two sessions (not just
the preceding MD1), so this offset reflects the combined effect of all
history differences.  Additional measurements with alternating 200 GeV
and 26 GeV MD1 cycles within the same session would be needed to isolate
the contribution of the immediately preceding MD1 from longer-range
history effects.

---

## Harmonic Content

### b2 (Normal Quadrupole)

| Operating point | b2 200 GeV (units) | b2 26 GeV (units) | Difference |
|-----------------|--------------------:|-------------------:|-----------:|
| Injection | -1.130 | -1.131 | +0.001 |
| SFTPRO top | -0.945 | -0.964 | +0.020 |

The quadrupole component is small and negative at both operating points
(~-1.1 units at injection, ~-0.95 units at SFTPRO top).  The difference
between datasets is negligible at injection (+0.001 units) and small at
SFTPRO top (+0.020 units, ~2 sigma).

The b2 value shows a mild current dependence: -1.13 units at 301 A vs
-0.95 units at 4815 A, suggesting slight geometry-dependent saturation
effects.

### b3 (Normal Sextupole)

| Operating point | b3 200 GeV (units) | b3 26 GeV (units) | Difference |
|-----------------|--------------------:|-------------------:|-----------:|
| Injection | +0.177 | +0.197 | -0.020 |
| SFTPRO top | +0.345 | +0.355 | -0.011 |

The sextupole is small and positive at both operating points.  At
injection, the 26 GeV dataset shows marginally higher b3 (+0.020 units
more, ~4.5 sigma based on propagated uncertainties) — this is a small
but statistically significant difference, possibly reflecting a real
magnetisation-history effect on the sextupole at low field.

At SFTPRO top, b3 is approximately twice the injection value (~0.35
vs ~0.19 units), showing clear current dependence.  The difference
between datasets is small (-0.011 units, ~1.5 sigma).

---

## Statistical Significance: What Does "X Sigma" Mean?

When comparing two measurements (e.g. B1 from the 200 GeV dataset vs the
26 GeV dataset), we want to know: **is the observed difference real, or
could it just be measurement noise?**

We compute:

    sigma = |mean_1 - mean_2| / sqrt(std_1^2/N_1 + std_2^2/N_2)

- **Numerator**: the absolute difference between the two dataset means.
- **Denominator**: the *standard error of the difference* — how precisely
  we know the difference, given the turn-to-turn scatter (std) and the
  number of turns (N) in each dataset.  More data and less scatter make
  the denominator smaller and sigma larger.

### Reading the sigma value

| Sigma value | Meaning | Probability it's just noise |
|:-----------:|---------|:---------------------------:|
| < 2 | No evidence of a real difference | > 5% |
| 2 – 3 | Suggestive — might be real, might be noise | 0.3% – 5% |
| **> 3** | **Strong evidence the difference is real** | **< 0.3%** |

### Important: high sigma does NOT mean large difference

A high sigma means the difference is **reliably detected** (not noise).
It says nothing about whether the difference is **physically large or
operationally important**.

**Example from this analysis:**

- **B1 at injection: ~22 sigma** — the actual difference is only **7 µT**
  (60 ppm of B1).  Because we average ~360 turns with only 5 µT
  turn-to-turn scatter, our uncertainty on each mean is ~0.3 µT.  Even
  a tiny 7 µT shift stands out clearly against that small uncertainty.
  *The difference is real, but tiny.*

- **TF at SFTPRO: 2.8 sigma** — the difference is borderline (just below
  the 3-sigma threshold).  It *probably* reflects a real effect, but we
  cannot be fully confident — there is still a ~0.5% chance it could be
  noise.

### Takeaway

- **Look at sigma** to decide: *"is there evidence of a real difference?"*
  (> 3 → yes, < 2 → no, 2–3 → maybe)
- **Look at the actual difference values** (µT, units, mT/kA) to decide:
  *"does the difference matter for accelerator operation?"*

---

## Difference Table (200 GeV - 26 GeV)

| Operating Point | Quantity | Difference | Sigma | Verdict |
|-----------------|----------|:----------:|:-----:|---------|
| Injection (~301 A) | B1 | -7 µT | ~22 | Real but tiny (60 ppm) |
| Injection (~301 A) | b2 | +0.003 units | 0.2 | No evidence of difference |
| Injection (~301 A) | b3 | -0.021 units | **4.7** | **Real difference** |
| Injection (~301 A) | TF | ~0 | 1.7 | No evidence of difference |
| SFTPRO (~4815 A) | B1 | -136 µT | 1.7 | No evidence of difference |
| SFTPRO (~4815 A) | b2 | +0.020 units | 1.9 | No evidence of difference |
| SFTPRO (~4815 A) | b3 | -0.011 units | 1.5 | No evidence of difference |
| SFTPRO (~4815 A) | TF | -0.03 mT/kA | **2.8** | Suggestive, not conclusive |

Uncertainties are propagated standard errors of the mean: sqrt(std²/N).

Both operating points may show history-dependent effects.  The datasets
were measured in separate sessions (~30 min apart) with different cycle
configurations and different pre-measurement states.  Iron hysteresis
has memory extending many cycles back, so observed differences reflect
the combined effect of all magnetisation history differences.

---

## Per-Supercycle Stability

The per-supercycle injection analysis (20 supercycles per dataset) shows:

- **B1**: extremely stable across all 20 supercycles in both datasets.
  No systematic drift over the ~17 min measurement.
- **b2**: stable mean with moderate turn-to-turn scatter (~0.16 units
  std), driven by the low signal level at 301 A.
- **b3**: stable mean with scatter ~0.06 units std.
- **TF**: stable to within measurement precision across all supercycles.

---

## Consistency with Bottura Standard Analysis

The analysis pipeline follows L. Bottura, *"Standard Analysis Procedures
for Field Quality Measurement of the LHC Magnets — Part I: Harmonics"*,
MTA-IN-97-007, CERN (1997, rev. 2000).

| Step | Bottura ref. | Description |
|------|-------------|-------------|
| 1. Drift correction | Eq. AII.12-14 | Remove integrator DC offset |
| 2. DFT + spectrum folding | Eq. AII.19-22 | Fourier decomposition per turn |
| 3. Harmonic extraction via Kn | Eq. AII.22 | Convert flux spectra to field harmonics |
| 4. Centre localisation (CEL) | Eq. AIII.1 | Find magnetic centre |
| 5. Feed-down correction (FED) | Eq. AIII.6 | Translate to magnetic centre frame |
| 6. Rotation correction (ROT) | Eq. AIV.2-6 | Rotate into main-field frame |
| 7. Normalisation | Eq. AIV.8-9 | Convert to units (1e-4 relative to B_m) |
| 8. Merge abs/cmp channels | Section 3.7 | B1 from absolute, n>1 from compensated |

---

## Key Findings

1. **B1 at injection: real but tiny difference (~22 sigma, 7 µT)**:
   the difference is clearly real (high sigma) but physically small
   (60 ppm of B1).  The high sigma comes from extremely precise means
   (360 turns, 5 µT scatter → 0.3 µT uncertainty on each mean).
   This shows our measurement can resolve even tiny field differences
   at injection.

2. **Marginal B1 difference at SFTPRO (1.7 sigma)**: B1 is 136 µT
   lower in the 200 GeV dataset.  Consistent across all 20 supercycles
   (systematic offset, not drift), but below 2 sigma — could be noise.

3. **TF at SFTPRO is the most notable indicator (2.8 sigma)**:
   dividing B1 by I removes current-correlated scatter, tightening
   the distribution and revealing a ~0.03 mT/kA offset.  This is
   in the suggestive range (2–3 sigma) — probably real but not
   conclusive.  A direct signature of history-dependent iron
   permeability at 4815 A.

4. **b3 at injection: clear difference (4.7 sigma)**: 0.021 units
   higher in the 26 GeV dataset.  The only quantity clearly exceeding
   3 sigma in this analysis — strong evidence of a real difference.
   Could be a magnetisation-history effect or session-to-session
   systematic.

5. **b2 at SFTPRO: 1.9 sigma difference**: +0.020 units, consistent
   in direction with the B1/TF offset but not conclusive (below 2
   sigma threshold).

6. **Settling correction is essential**: the first ~6 injection turns
   per supercycle carry eddy-current transients of 10-30 uT.

7. **Outlier removal catches ramp-boundary turns**: 5 turns total
   removed by MAD sigma clipping (last-in-group turns where the ramp
   has already started).

---

## Conclusion: Can We Distinguish History-Dependent Effects?

### 1. Hysteresis (DC magnetisation history)

**Partially — suggestive but not conclusive.**

The transfer function at SFTPRO shows the strongest signal: a 2.8-sigma
offset between the two datasets, indicating that the iron permeability
at 4815 A depends on the magnetisation history.  The B1 offset (1.7
sigma) and b2 offset (1.9 sigma) are consistent in direction.

However, this does not reach the conventional 3-sigma significance
level, and the two datasets were measured in separate sessions (~30 min
apart) with different full magnetisation histories — not just different
immediately preceding MD1 cycles.  Iron hysteresis has memory extending
many cycles back, so the observed offset is the combined effect of:
- the different MD1 flat-top currents (200 vs 26 GeV),
- the different pre-measurement magnetisation states,
- any environmental changes between sessions (temperature, alignment).

We **cannot isolate** which part of the history is responsible.

At injection (~301 A), B1 shows a tiny but statistically real difference
(-7 µT, ~22 sigma).  However, this is only 60 ppm of B1, so while the
measurement precision is sufficient to detect it, it is far too small
to have operational significance.  The high sigma is a consequence of
the excellent measurement precision (360 turns, 5 µT scatter) rather
than a large effect.

### 2. Eddy Currents

**Yes — clearly visible within each dataset, but not directly
comparable between datasets.**

Eddy-current settling is unambiguously present in both datasets: the
first ~6 turns of each supercycle's injection plateau show a systematic
B1 offset of 10-30 uT from the settled mean, decaying over ~6 s.

However, the settling dynamics differ between the two sessions: the
26 GeV dataset has ~14x larger turn-to-turn B1 scatter at injection
(71 uT vs 5 uT std).  This could indicate different eddy-current
amplitudes or different noise conditions, but the analysis does not
isolate eddy currents from other transient effects (e.g. power supply
settling, thermal effects).

A direct comparison of eddy-current time constants would require fitting
exponential decays to the per-supercycle B1 transients — this is done
in the separate eddy-current analysis notebook but not in this
comparison notebook.

### 3. Sextupole (b3)

**Yes — the most statistically significant difference, but only at
injection.**

At injection, b3 differs by 0.020 units between datasets (4.5 sigma) —
the only quantity that clearly exceeds 3 sigma.  The 26 GeV dataset
has higher b3 (+0.197 vs +0.177 units).

At SFTPRO, b3 differs by only 0.011 units (1.5 sigma) — not
significant.

The injection b3 difference is real in the statistical sense, but we
cannot attribute it to a specific mechanism.  Possible causes:
- **Magnetisation-dependent sextupole**: if the iron has different
  remanent magnetisation patterns in the two sessions, the non-linear
  B-H relationship could produce different b3 at the same current.
- **Session-to-session systematic**: temperature-dependent yoke
  geometry, coil positioning, or electronics drift.
- **Eddy-current residual**: if the settling time constant for b3
  differs from B1, the N_LAST_TURNS_INJ = 18 selection may leave
  different residual eddy-current contamination in b3 for the two
  datasets.

### Summary Table

| Effect | Distinguishable? | Evidence | Limitation |
|--------|:----------------:|----------|------------|
| Hysteresis (DC) | Marginal | TF at SFTPRO 2.8 sigma, B1 1.7 sigma | Separate sessions; cannot isolate from multi-cycle memory |
| Eddy currents | Yes (within dataset) | Clear settling in first ~6 turns/SC | Cannot directly compare between datasets |
| b3 sextupole | Yes (at injection) | 4.5 sigma injection difference | Cannot attribute to specific mechanism |
| TF saturation | Yes | 3.1% drop from 301 A to 4815 A | Identical in both datasets; not history-dependent |

### Recommendation for Future Measurements

To conclusively separate history-dependent effects from session-to-session
systematics:

1. **Interleave cycles within a single session**: alternate 200 GeV and
   26 GeV MD1 cycles in the same measurement run (e.g. supercycle
   pattern: 200-26-200-26-...).  This eliminates session-to-session
   offsets and allows direct within-session comparison.

2. **Pre-cycle to a known state**: before starting the measurement,
   run a standardised pre-cycle (e.g. 5 repetitions to full excitation)
   to establish a reproducible initial magnetisation state.

3. **Monitor temperature**: log the yoke/coil temperature to rule out
   thermal contributions to the b3 injection difference.

---

## File Locations

    measurements/2026_02_06/
      01_200_extended/                       <- 200 GeV extended dataset
        20260206_144537_SPS_MBB/
          20260206_144559_MBB/               <- raw data files (NCS + CS)
      03_26_extended/                        <- 26 GeV extended dataset
        20260206_151808_SPS_MBB/
          20260206_151827_MBB/               <- raw data files (NCS + CS)

    measurements/20251212_171026_SPS_MBA/
      CRMMMMH_AV-00000001/
        Kn_values_Seg_Main_A_AC.txt          <- cross-session Kn calibration

    rotating_coil_analyzer/notebooks/
      compare_b1_b2_b3_200GeV_vs_26GeV.ipynb <- analysis notebook

    output/2026_02_06/compare_200_vs_26/
      plateau_harmonics_200GeV.csv           <- all plateau turns (200 GeV)
      plateau_harmonics_26GeV.csv            <- all plateau turns (26 GeV)
      plateau_harmonics_200GeV_settled.csv   <- settled + cleaned (200 GeV)
      plateau_harmonics_26GeV_settled.csv    <- settled + cleaned (26 GeV)
      per_supercycle_injection_200GeV.csv    <- per-SC injection summary
      per_supercycle_injection_26GeV.csv     <- per-SC injection summary
      per_supercycle_sftpro_200GeV.csv       <- per-SC SFTPRO summary
      per_supercycle_sftpro_26GeV.csv        <- per-SC SFTPRO summary
      summary_comparison_settled.csv         <- summary table
      SPS_MBB_200GeV_vs_26GeV_analysis_notes.md  <- this file
