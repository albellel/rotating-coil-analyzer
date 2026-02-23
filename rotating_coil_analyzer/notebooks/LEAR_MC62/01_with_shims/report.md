# Test 01 -- MC62 With Shims -- Feb 11, 2026

## Overview
Full hysteresis staircase measurement of MC62 with shimming plates installed. Current cycle: 0 -> +200 -> 0 -> -200 -> 0 A in 20 A steps (41 plateaus, 350 turns each at -60 rpm).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, with shims) |
| Rotation | 1 Hz (-60 rpm), 1024 samples/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 170 |
| Turns/plateau | 350 |
| Pipeline | `dri`, `rot` (cel/fed auto-disabled -- UNSAFE) |
| Integral merge | `abs_upto_m_cmp_above` |
| Central merge | `abs_all` (compensated SNR too low) |

## Data Summary
- 41 integral + 41 central runs
- 14,350 total rows per PCB (350 turns x 41 runs, run 23 has 351)
- Timeline: 0--26,124 s
- All 41 runs succeeded (Integral PCB)

## Key Results (Integral PCB)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +100 (asc) | 0.129 | -14.44 | -17.44 | 1.286 |
| +200 (asc) | 0.229 | -14.29 | -17.34 | 1.144 |
| -200 (desc) | -0.229 | -14.24 | -17.32 | 1.140 |

- **b2**: ~-151 units (large, C-shape asymmetry)
- **b3**: ~-17 units (stable across staircase)
- **TF**: peaks 1.38 T/kA at 20 A, drops to 1.14 at 200 A (saturation)
- **Hysteresis**: ~0.7 mT at 100 A between ascending/descending

### Inductance Analysis
- **L_app = B1/I** (transfer function, secant of B-H)
- **L_diff = dB1/dI** (differential inductance, local slope of B-H)
- L_diff drops faster than L_app at saturation; ratio L_diff/L_app < 1 quantifies saturation degree

### Central PCB Data Quality
Several Central runs show corrupt B1 values (e.g., +34 T, +3,065 T). Due to extremely low compensated-signal SNR. Central PCB results only trusted at high |I| (>= 100 A).

## Eddy-Current Settling

Exponential model: B1(t) = B1_inf + A*exp(-t/tau), Integral PCB only.

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
Runs 10, 11, 12, 30, 31, 32, 33 -- reversal points near peak current where |delta-I| is small.

### N_LAST_TURNS Sensitivity

| N_LAST | B1 max error (units) | b3 max error (units) |
|--------|---------------------|---------------------|
| 100 | 1.07 | 0.02 |
| 170 | 3.55 | 0.03 |
| 250 | 9.75 | 0.05 |

## Key Findings

1. **b2 ~ -151 units**: large systematic quadrupole from C-shape asymmetry. Stable across current.
2. **Clear mu_r dependence on tau**: drops from ~33 s (low I, high mu_r) to ~12 s (near saturation).
3. **Positive/negative symmetric**: both branches show matching tau values.
4. **b3 bias from eddy currents is negligible** (< 0.03 units for any N_LAST_TURNS).
5. **N_LAST_TURNS = 170 is conservative**: worst-case B1 bias ~3.5 units, b3 bias < 0.03 units.
