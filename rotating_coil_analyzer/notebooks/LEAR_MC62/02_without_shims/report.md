# Test 02 -- MC62 Without Shims -- Feb 12, 2026

## Overview
Full hysteresis staircase measurement of MC62 without shimming plates. Current cycle: 0 -> +200 -> 0 -> -200 -> 0 A in 20 A steps (41 plateaus, 350 turns each at -60 rpm).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, without shims) |
| Rotation | 1 Hz (-60 rpm), 1024 samples/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 170 |
| Pipeline | `dri`, `rot` (cel/fed auto-disabled -- UNSAFE) |

## Data Summary
- 41 integral + 41 central runs, 14,351 turns per PCB
- All runs succeeded, all plateaus "good" quality

## Key Results (Integral PCB)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +100 (asc) | 0.129 | -15.20 | -17.53 | 1.286 |
| +200 (asc) | 0.226 | -15.31 | -17.43 | 1.132 |
| +100 (desc) | 0.129 | -14.89 | -17.33 | 1.294 |
| -200 (desc) | -0.226 | -15.34 | -17.42 | 1.132 |

- **b2 ~ -15 units** (without shims) vs **-151 units** (with shims in test 01)
- **b3 ~ -17 units**: consistent between shim/no-shim tests
- **TF** and **hysteresis** patterns match test 01 closely

### Inductance Analysis
- **L_app = B1/I**, **L_diff = dB1/dI** -- same saturation signature as test 01.

### Central PCB
Similar corrupt-value issues at low current as test 01.

## Eddy-Current Settling

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

## FFMM C++ Validation

Both pipelines process the same raw run-based data (41 plateaus x 350 turns).

- FLIP_FIELD_SIGN = False (raw sign for comparison)
- FFMM options: `dri rot nor cel fed dit`
- Our OPTIONS: `dri rot cel fed` (`dit` N/A on plateaus)

### Best Averaging Window
FFMM averages all 350 turns per plateau. At N_LAST = 350:

| Metric | Value |
|--------|-------|
| B_main RMS (Integral, |I|>=10 A) | 0.6 uT |
| b3 RMS (Integral) | 0.000 units |
| Our default (N_LAST=170) RMS | 72.4 uT (eddy-current settling exclusion) |

### Per-Harmonic RMS Residuals (N_LAST=350, |I|>=10 A)

| Order | RMS bn | RMS an |
|-------|--------|--------|
| n=2 | 0.003 | 0.001 |
| n=3 | 0.000 | 0.000 |
| n=4--10 | 0.000 | 0.000 |
| n=11--15 | 0.001--0.003 | 0.001--0.002 |

**All harmonics < 0.003 units -- machine precision parity. Verdict: PASS.**

## Key Findings

1. **Shims removal reduced |b2| from ~151 to ~15 units** -- shims were worsening field quality.
2. **Tau values match test 01** closely (26.0 vs 26.4 s), confirming shims do not affect eddy-current dynamics.
3. **N_LAST_TURNS = 170** confirmed appropriate (b3 bias < 0.03 units).
4. **Sub-microtesla B_main parity** with FFMM C++ when matching the same averaging window (350 turns).
5. The 72 uT difference at N_LAST=170 is intentional (we exclude settling turns; FFMM does not).
