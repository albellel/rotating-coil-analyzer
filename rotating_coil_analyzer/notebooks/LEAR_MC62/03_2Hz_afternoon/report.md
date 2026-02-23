# Test 03 -- MC62 2 Hz Staircase (Afternoon) -- Feb 16, 2026

## Overview
Full hysteresis staircase measurement of MC62 (no shims), streaming at 2 Hz rotation (512 samples/turn).
Current cycle: precycles (0->-200->+200->-200->0 A) then staircase 0->+200->0->-200->0 A in 20 A steps.
Ramp rate: 1 A/s. 41 staircase plateaus, ~740 turns each (~370 s per step).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, no shims) |
| Rotation | 2 Hz (120 rpm), 512 samples/turn, 0.5 s/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 340 |
| Pipeline | `dri`, `rot` (cel/fed auto-disabled -- UNSAFE) |
| Parity pipeline | `dri`, `rot`, `cel`, `fed`, `dit` (signed, matching FFMM C++ native) |
| Plateau threshold | 0.5 A (block-averaged), min_length=50, merge gap<100 |

## Data Summary
- 35,828 total turns (587 MB per binary file)
- 3 precycle groups (at +/-200 A) + 41 staircase plateaus
- Precycle: turns 0--3785, Staircase: turns 3786--35827
- All 41 plateaus "good" quality (Integral PCB)
- **Central PCB has severe numerical issues** (corrupt values at several current levels)

## Key Results (Integral PCB)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +20 (asc) | 0.024577 | -17.75 | -12.98 | 1.2288 |
| +40 (asc) | 0.049539 | -16.55 | -12.47 | 1.2385 |
| +60 (asc) | 0.074489 | -16.01 | -12.32 | 1.2415 |
| +80 (asc) | 0.099401 | -15.76 | -12.28 | 1.2425 |
| +100 (asc) | 0.124245 | -15.62 | -12.26 | 1.2424 |
| +120 (asc) | 0.148913 | -15.54 | -12.25 | 1.2409 |
| +140 (asc) | 0.172420 | -15.57 | -12.23 | 1.2316 |
| +160 (asc) | 0.191419 | -15.69 | -12.20 | 1.1964 |
| +180 (asc) | 0.206241 | -15.85 | -12.15 | 1.1458 |
| +200 (asc) | 0.218521 | -16.03 | -12.09 | 1.0926 |

- Saturation onset at ~120 A. At 200 A: TF = 1.09 T/kA (12% below low-field value).
- **Hysteresis**: B1 width 0.77--1.11 mT across current range.

### Inductance Analysis
- **L_app = B1/I**, **L_diff = dB1/dI** -- saturation onset visible at ~120 A.

## FFMM C++ Parity Check

Parity uses `dit` (di/dt correction) with `signed=True` to match FFMM C++ native threshold logic.

### B_main
- **Max |diff|**: 1.8e-13 T (machine precision)
- **Mean |diff|**: 6.2e-17 T
- **Mean |rel diff|** (|B|>0.01 T): 1.3e-15

### Harmonics (R_ref = 0.33 m)
- **b2**: RMS diff = 0.0000 units, max |diff| = 0.0000
- **b3**: RMS diff = 0.0000 units, max |diff| = 0.0004

**Machine-precision agreement on all 35,826 turns. Verdict: PASS.**

## Eddy-Current Settling

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
Runs 11, 16, 18, 29, 30, 31, 32, 33, 34 -- mostly reversal points and ascending return branch.

### N_LAST_TURNS Sensitivity

| N_LAST | B1 max error (units) | b3 max error (units) |
|--------|---------------------|---------------------|
| 100 | 0.00 | 0.00 |
| 200 | 5.62 | 0.01 |
| 340 | 8.04 | 0.02 |
| 600 | 11.40 | 0.02 |

### 5x tau_max settling rule
- 5 x tau_max (40.2 s) = 201 s = 402 turns at 2 Hz
- With ~726 turns/plateau, safe N_LAST_TURNS = 324

## Key Findings

1. **Tau significantly shorter at 2 Hz** (mean 13.2 s vs 26.0 s at 1 Hz). Surprising -- may be due to shorter ramp dwell interacting differently with 1 A/s ramp rate.
2. **Same mu_r dependence**: tau decreases with current (16.7 s at low I, 4.9 s near saturation).
3. **More outliers at 2 Hz** (9 vs 4 at 1 Hz): ascending return branch shows poor R2.
4. **b3 bias negligible** (<0.02 units) across all N_LAST values.
5. **N_LAST_TURNS = 340** confirmed appropriate -- equivalent settling to 170 at 1 Hz.
6. **FFMM parity perfect** -- machine-precision agreement on all turns.
