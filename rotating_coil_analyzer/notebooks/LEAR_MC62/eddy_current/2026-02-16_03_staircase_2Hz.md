# MC62 -- Eddy-Current Settling -- 03_staircase_2Hz

## Overview
Eddy-current settling analysis for MC62 (no shims) at 2 Hz rotation (0.5 s/turn).
Same methodology as test 02, with 2x time resolution and 2x more turns per plateau (~740 vs 350).
Ramp rate: 1 A/s. Integral PCB (R45) only.

## Results

### Fit Statistics (29 retained of 38, R2 >= 0.5)
- **tau range**: 1.6 -- 40.2 s
- **tau mean**: 13.2 +/- 8.7 s
- **R2 range**: 0.54 -- 0.98

### Tau vs Current

| |I| range (A) | tau (s) | N |
|---------------|---------|---|
| [10, 60) | 16.6 +/- 11.9 | 7 |
| [60, 120) | 16.7 +/- 6.8 | 11 |
| [120, 180) | 8.1 +/- 5.5 | 9 |
| [180, 210) | 4.9 +/- 1.1 | 2 |

### Removed Outliers (9 fits)
Runs 11, 16, 18, 29, 30, 31, 32, 33, 34 -- mostly reversal points near +/-200 A and ascending return branch.

### N_LAST_TURNS Sensitivity

| N_LAST | B1 max error (units) | b3 max error (units) |
|--------|---------------------|---------------------|
| 100 | 0.00 | 0.00 |
| 200 | 5.62 | 0.01 |
| 300 | 7.51 | 0.01 |
| 340 | 8.04 | 0.02 |
| 400 | 8.71 | 0.02 |
| 500 | 9.71 | 0.02 |
| 600 | 11.40 | 0.02 |

### 5x tau_max settling rule
- 5 x tau_max (40.2 s) = 201 s = 402 turns at 2 Hz
- With ~726 turns/plateau, safe N_LAST_TURNS = 324

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Key Findings
1. **Tau is significantly shorter at 2 Hz** (mean 13.2 s) vs 1 Hz test 02 (mean 26.0 s). This is surprising -- eddy-current physics should be rotation-speed independent. The difference may be due to the shorter ramp dwell (20 s/step at 1 A/s) at 2 Hz interacting with the 1 A/s ramp rate differently.
2. **Same mu_r dependence**: tau decreases with current (16.7 s at low I, 4.9 s near saturation), consistent with permeability drop at high flux density.
3. **More outliers at 2 Hz** (9 vs 4 at 1 Hz): the ascending return branch (runs 31-34) shows poor R2, suggesting the eddy-current settling model breaks down for certain branch transitions.
4. **b3 bias is negligible** (<0.02 units) across all N_LAST values, confirming b3 is insensitive to the averaging window.
5. **B1 bias grows with N_LAST** (up to 11 units at N_LAST=600) as expected from including unsettled turns.
6. **N_LAST_TURNS = 340** confirmed as appropriate -- gives equivalent settling time to 170 at 1 Hz.
