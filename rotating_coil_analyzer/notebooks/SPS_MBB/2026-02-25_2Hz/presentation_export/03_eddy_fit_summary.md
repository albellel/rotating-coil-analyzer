# Eddy-Current Fit Summary

## Outlier removal matters

The first few turns after the ramp can contain **ramp artifacts** (turns where
the DAQ trigger overlaps the tail of the current ramp). These outliers inflate
SS_res and distort both R2 and tau. A two-pass 3.5x MAD residual clip removes
them reliably (typically 5-6 of ~50 turns).

| Example | Before cleaning | After cleaning |
|---------|----------------|---------------|
| Body B1, 200 GeV | R2 = 0.60, tau = 8.1 s | R2 = 0.99, tau = 3.8 s |
| Body B1, 26 GeV | R2 = 0.66, tau = 7.0 s | R2 = 1.00, tau1 = 2.1 s, tau2 = 5.1 s |

The "before" R2 is misleadingly low: the fit already captures the trend, but
a handful of outlier points dominate SS_res. The "after" tau values are
physically consistent across energies and with the fringe results.

**All results below are after outlier removal.**

## Best-fit time constants (cleaned)

### 200 GeV MD1

| Segment | Quantity | Model | R2 | B_inf | A1 | tau1 (s) | A2 | tau2 (s) |
|---|---|---|---|---|---|---|---|---|
| Body | B1 (T) | 1-tau | 0.9924 | -0.116256 | 0.000178 | 3.817 | | |
| Body | b2 (units) | 1-tau | 0.1155 | 0.017 | 0.044 | 2.551 | | |
| Body | b3 (units) | 1-tau | 0.4344 | -0.226 | 0.080 | 2.607 | | |
| Fringe | B1 (T) | 2-tau | 0.9983 | -0.019621 | -0.000206 | 1.307 | -0.000072 | 4.799 |
| Fringe | b2 (units) | 1-tau | 0.8317 | 1.617 | -0.092 | 12.129 | | |
| Fringe | b3 (units) | 2-tau | 0.9989 | 5.173 | -0.932 | 0.948 | -0.513 | 3.443 |

### 26 GeV MD1

| Segment | Quantity | Model | R2 | B_inf | A1 | tau1 (s) | A2 | tau2 (s) |
|---|---|---|---|---|---|---|---|---|
| Body | B1 (T) | 2-tau | 0.9964 | -0.116257 | 0.000123 | 2.129 | 0.000097 | 5.103 |
| Body | b2 (units) | 1-tau | 0.0229 | -0.904 | 0.937 | 1000.0 | | |
| Body | b3 (units) | 1-tau | 0.5236 | -0.229 | 0.084 | 3.472 | | |
| Fringe | B1 (T) | 2-tau | 0.9996 | -0.019625 | -0.000114 | 1.555 | -0.000086 | 4.248 |
| Fringe | b2 (units) | 1-tau | 0.8419 | 1.047 | -0.096 | 2.350 | | |
| Fringe | b3 (units) | 2-tau | 0.9967 | 5.054 | -0.557 | 0.619 | -0.922 | 2.240 |

## Physical interpretation

| Component | Material | Expected tau | Observed |
|-----------|----------|-------------|----------|
| Beam pipe | Stainless steel | ~0.5-1 s | tau1 = 0.6-1.6 s (all B1 and b3 fits) |
| Yoke laminations | Si-steel | ~2-5 s | tau2 = 2.2-5.1 s (all B1 and b3 fits) |

Both segments see the same two eddy sources. The 2-tau model is selected by
AICc wherever the signal-to-noise ratio permits:

- **B1**: 2-tau in the fringe (both energies) and in the body at 26 GeV.
  At 200 GeV (fewer turns), the body 2-tau fit collapses to an absurd
  tau2 = 510 s, so AICc correctly picks 1-tau (tau = 3.8 s, a weighted
  average of the two true time constants).
- **b3**: 2-tau in the fringe (both energies). Body b3 is too noisy
  (R2 ~ 0.4-0.5) -- the eddy amplitude ~0.08 units is barely above the
  turn-to-turn scatter.
- **b2**: 1-tau everywhere. Body b2 fits are garbage (R2 < 0.12) -- no
  detectable eddy signal. Fringe b2 at 200 GeV has a suspicious
  tau = 12.1 s (vs 2.4 s at 26 GeV for the same iron); likely a 1-tau
  fit artifact on weak/noisy data.

## Why body R2 can be low despite a "good-looking" fit

R2 = 1 - SS_res / SS_tot measures the fraction of data variance explained by
the model. When the eddy amplitude is small relative to the turn-to-turn noise
(body b3: 0.08 units amplitude vs ~0.1 units scatter), R2 is low even if the
model correctly captures the trend. The eye is a natural low-pass filter and
tracks the smooth curve through scatter; R2 does not.

After outlier removal, body B1 R2 jumps to 0.99+ because the eddy amplitude
(~180 uT) becomes large relative to the remaining noise. Body b3 stays at
R2 ~ 0.5 because the signal-to-noise ratio in units is intrinsically low
(same eddy in Tesla, but divided by the large body B1).

## Settling requirement

**The fringe b3 is the critical quantity** for SPS operations: it settles with
tau1 = 0.6-0.95 s and tau2 = 2.2-3.4 s. At N_last = 18 turns (= 9 s at 2 Hz),
the eddy residual is exp(-9/3.4) = 7% of initial amplitude, or ~0.07 units --
well below the SPS tolerance of ~1 unit.

## Model selection: AICc

For weak signals (body b2, b3), AICc correctly picks 1-tau (or rejects fitting
entirely). For strong signals (fringe B1, b3; body B1), it selects 2-tau when
the data can resolve both time constants. 3-tau is never justified -- it never
improves AICc over 2-tau, consistent with only two main conducting structures.
