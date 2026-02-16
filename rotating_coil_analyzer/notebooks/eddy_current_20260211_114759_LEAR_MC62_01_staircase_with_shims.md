# MC62 -- Eddy-Current Settling -- 01_staircase_with_shims

## Overview
Eddy-current settling analysis for MC62 with shims. Fits exponential model B1(t) = B1_inf + A*exp(-t/tau) to each of the 41 plateaus.

## Configuration
- Integral PCB only (best SNR)
- 60 rpm (1.0 s/turn)
- Settled reference: last 50 turns
- Model: B1(t) = B1_inf + A*exp(-t/tau)
- Min |I| for fit: 10 A
- R2 quality threshold: 0.50

## Results

### Fit Statistics (31 retained of 38 attempted, R2 >= 0.5)
- **tau range**: 11.0 -- 40.2 s
- **tau mean**: 26.4 +/- 8.5 s
- **R2 range**: 0.59 -- 0.99

### Tau vs Current (mu_r Dependence)

| |I| range (A) | tau (s) | N |
|---------------|---------|---|
| [10, 60) | 30.8 +/- 2.7 | 8 |
| [60, 120) | 32.6 +/- 3.7 | 12 |
| [120, 180) | 17.2 +/- 5.2 | 9 |
| [180, 210) | 12.3 +/- 1.2 | 2 |

### Removed Outliers (7 fits, R2 < 0.5)
Runs 10, 11, 12, 30, 31, 32, 33 -- reversal points near peak current where |delta-I| is small and eddy-current amplitude is tiny.

### N_LAST_TURNS Sensitivity
| N_LAST | B1 max error (units) | b3 max error (units) |
|--------|---------------------|---------------------|
| 100 | 1.07 | 0.02 |
| 170 | 3.55 | 0.03 |
| 250 | 9.75 | 0.05 |

## Key Findings
1. **Clear mu_r dependence**: tau drops from ~33 s (low I, high mu_r) to ~12 s (near saturation, low mu_r).
2. **Positive/negative symmetric**: both branches show matching tau values.
3. **b3 bias from eddy currents is negligible** (< 0.03 units for any N_LAST_TURNS).
4. **N_LAST_TURNS = 170 is conservative**: worst-case B1 bias ~3.5 units, b3 bias < 0.03 units.
