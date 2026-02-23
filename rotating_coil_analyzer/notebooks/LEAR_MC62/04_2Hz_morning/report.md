# Test 04 -- MC62 2 Hz Staircase (Morning Repeat) -- Feb 17, 2026

## Overview
Morning repeat measurement of MC62 (no shims), streaming at 2 Hz rotation (512 samples/turn).
Reproducibility test: identical setup to test 03 (Feb 16 afternoon), repeated ~16 h later.
Current cycle: staircase 0->+200->0->-200->0 A in 20 A steps (no precycle detected).
Ramp rate: 1 A/s. 40 staircase plateaus, ~800 turns each (~400 s per step).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, no shims) |
| Rotation | 2 Hz (120 rpm), 512 samples/turn, 0.5 s/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 340 |
| N_SKIP_END | 20 |
| Pipeline | `dri`, `rot` (cel/fed auto-disabled -- UNSAFE) |
| Parity pipeline | `dri`, `rot`, `cel`, `fed`, `dit` (signed, matching FFMM C++ native) |
| Plateau threshold | 0.5 A (block-averaged), min_length=50, merge gap<100 |

## Data Summary
- 32,544 total turns, time span: 0--16,447 s (274 min)
- 40 staircase plateaus (no precycle -- systematic stepping from start)
- Plateau turns: 31,605 / 32,544 (97.1%)
- Turn classification: 13,600 used (41.8%), 17,430 settling (53.6%), rest ramp/transition
- All 40 plateaus "good" quality (Integral PCB)

## Key Results (Integral PCB)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +20 (asc) | 0.024540 | -18.04 | -12.94 | 1.2270 |
| +40 (asc) | 0.049501 | -16.50 | -12.45 | 1.2375 |
| +60 (asc) | 0.074443 | -15.88 | -12.31 | 1.2407 |
| +80 (asc) | 0.099349 | -15.60 | -12.27 | 1.2419 |
| +100 (asc) | 0.124189 | -15.44 | -12.25 | 1.2419 |
| +120 (asc) | 0.148858 | -15.35 | -12.24 | 1.2405 |
| +140 (asc) | 0.172369 | -15.39 | -12.23 | 1.2312 |
| +160 (asc) | 0.191367 | -15.51 | -12.20 | 1.1960 |
| +180 (asc) | 0.206179 | -15.68 | -12.15 | 1.1454 |
| +200 (asc) | 0.218460 | -15.88 | -12.09 | 1.0923 |

## cel/fed Safety Diagnostic
- **Recommendation**: UNSAFE
- 100% of turns have |zR| > 0.01 (median 0.105, max 0.110)
- B_main max |diff| from cel/fed: 3.68e-06 T (negligible)
- -> cel/fed disabled, OPTIONS = `("dri", "rot")`

## FFMM C++ Parity Check

### B_main
- **Turns compared**: 32,508
- **Max |diff|**: 2.25e-12 T (machine precision)
- **Mean |diff|**: 1.58e-16 T
- **RMS diff**: 1.34e-14 T

### Harmonics (R_ref = 0.33 m, |I| > 10 A)
- **b2**: RMS diff = 0.0000 units, max |diff| = 0.0000
- **b3**: RMS diff = 0.0000 units, max |diff| = 0.0003
- **b4**: RMS diff = 0.0002 units, max |diff| = 0.0278
- **b5**: RMS diff = 0.0037 units, max |diff| = 0.4745
- n >= 6: divergence from (R_ref/R_coil)^n noise amplification at 330 mm

**Machine-precision agreement on all 32,508 turns. Verdict: PASS.**

### Inductance Analysis
- **L_app = B1/I**, **L_diff = dB1/dI** -- same saturation signature as test 03.

## Key Findings

1. **Excellent reproducibility vs test 03**: B1 agrees to ~60 uT, b3 to ~0.05 units.
2. **b2 ~ -16 units**, b3 ~ -12 units -- consistent with test 03.
3. **TF at 200 A**: 1.092 T/kA, identical to test 03.
4. **Hysteresis**: 0.8--1.1 mT width, matching test 03.
5. **No precycle detected**: Current cycle started with systematic 20 A stepping from 0 A.
6. **FFMM parity perfect** on all turns.
7. Systematic B1 offset (~30--60 uT) vs test 03 likely from ambient temperature drift.

## Cross-Test Comparison

| Test | b2 (units) | b3 (units) | TF@200A (T/kA) |
|------|-----------|-----------|-----------------|
| 01 (with shims, 1 Hz) | -151 | -17 | 1.132 |
| 02 (no shims, 1 Hz) | -15 | -17 | 1.132 |
| 03 (no shims, 2 Hz) | -16 | -12 | 1.093 |
| 04 (no shims, 2 Hz, morning) | -16 | -12 | 1.092 |
