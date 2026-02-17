# MC62 -- Eddy-Current Settling -- 02_staircase_without_shims

## Overview
Eddy-current settling analysis for MC62 without shims. Same methodology as test 01.

## Results

### Fit Statistics (34 retained of 38, R2 >= 0.5)
- **tau range**: 12.5 -- 35.5 s
- **tau mean**: 26.0 +/- 8.1 s
- **R2 range**: 0.78 -- 0.98

### Tau vs Current

| |I| range (A) | tau (s) | N |
|---------------|---------|---|
| [10, 60) | 30.9 +/- 2.8 | 8 |
| [60, 120) | 32.1 +/- 2.6 | 12 |
| [120, 180) | 19.8 +/- 6.0 | 10 |
| [180, 210) | 12.9 +/- 0.4 | 4 |

### Removed Outliers (4 fits)
Runs 11, 12, 31, 32 -- reversal points near +/-200 A.

### N_LAST_TURNS Sensitivity
| N_LAST | B1 max error (units) | b3 max error (units) |
|--------|---------------------|---------------------|
| 100 | 0.88 | 0.02 |
| 170 | 2.97 | 0.03 |
| 250 | 8.27 | 0.03 |

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Key Findings
1. **Tau values match test 01** closely (26.0 vs 26.4 s mean), confirming shims do not affect eddy-current dynamics.
2. **Same mu_r dependence**: 33 s at low I, 13 s near saturation.
3. **N_LAST_TURNS = 170** confirmed as appropriate (b3 bias < 0.03 units).
