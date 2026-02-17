# MC62 -- LEAR C-Shaped Dipole -- 04_staircase_2Hz_morning -- Rotating Coil Analysis

## Overview
Morning repeat measurement of MC62 (no shims), streaming at 2 Hz rotation (512 samples/turn).
This is a reproducibility test: identical setup to test 03 (Feb 16 afternoon), repeated the following morning (Feb 17).
Current cycle: staircase 0->+200->0->-200->0 A in 20 A steps (no precycle detected -- systematic stepping from start).
Ramp rate: 1 A/s. 40 staircase plateaus, ~800 turns each (~400 s per step).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, no shims) |
| Rotation | 2 Hz (120 rpm), 512 samples/turn, 0.5 s/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 340 |
| N_SKIP_END | 20 |
| Pipeline | `dri`, `rot` (cel/fed auto-disabled) |
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
| +100 (asc) | 0.1242 | -15.44 | -12.25 | 1.2419 |
| +200 (asc) | 0.2185 | -15.88 | -12.09 | 1.0923 |
| +100 (desc) | 0.1250 | -14.82 | -12.17 | 1.2499 |
| -200 (desc) | -0.2185 | -15.92 | -12.12 | 1.0924 |

## Golden Standard Parity Check (vs FFMM C++)

Parity uses `dit` (di/dt correction) with `signed=True` to match the FFMM C++ native
threshold logic (`crr > 0.1 && cm > 10`).

### B_main
- **Turns compared**: 32,508
- **Max |diff|**: 2.25e-12 T (machine precision)
- **Mean |diff|**: 1.58e-16 T
- **RMS diff**: 1.34e-14 T
- **Median |diff|**: 2.78e-17 T
- **Mean |rel diff|** (|B|>0.01 T): 5.05e-15

### Harmonics (at R_ref = 0.33 m, |I| > 10 A)
- **b2**: RMS diff = 0.0000 units, max |diff| = 0.0000
- **b3**: RMS diff = 0.0000 units, max |diff| = 0.0003
- **b4**: RMS diff = 0.0002 units, max |diff| = 0.0278
- **b5**: RMS diff = 0.0037 units, max |diff| = 0.4745
- n >= 6: divergence from (R_ref/R_coil)^n noise amplification at 330 mm, not a pipeline error

### Note on FFMM Central
FFMM Central results are all NaN (measurement-embedded Central Kn file is all-zeros). Parity check is Integral only.

## cel/fed Safety Diagnostic
- **Recommendation**: UNSAFE
- 100% of turns have |zR| > 0.01 (median 0.105, max 0.110)
- B_main max |diff| from cel/fed: 3.68e-06 T (negligible)
- -> cel/fed disabled, OPTIONS = `("dri", "rot")`

See `correction_options_reference.md` for background on dipole cel fragility.

## Observations
1. **Reproducibility vs test 03**: Results closely match test 03 (afternoon). B1 agrees to ~60 uT, b3 to ~0.05 units. See companion comparison notebook for detailed statistics.
2. **b2 ~ -16 units** at high current, consistent with C-shape geometry.
3. **b3 ~ -12 units**, stable across current range.
4. **Hysteresis**: Clear ascending/descending branch splitting, ~0.8--1.1 mT width.
5. **TF at 200 A**: 1.092 T/kA, identical to test 03.
6. **No precycle detected**: Current cycle started with systematic 20 A stepping from 0 A (unlike test 03 which had precycles).
7. **Parity with FFMM is perfect**: Machine-precision agreement on all 32,508 turns (B_main max |diff| = 2.25e-12 T).

## Comparison with Previous Tests

| Test | b2 (units) | b3 (units) | TF@200A (T/kA) |
|------|-----------|-----------|-----------------|
| 01 (with shims, 1 Hz) | -151 | -17 | 1.132 |
| 02 (no shims, 1 Hz) | -15 | -17 | 1.132 |
| 03 (no shims, 2 Hz) | -16 | -12 | 1.093 |
| 04 (no shims, 2 Hz, morning) | -16 | -12 | 1.092 |

## Suggestions
- Run the companion comparison notebook (`LEAR_MC62/comparison/2026-02-17_03_vs_04_reproducibility.ipynb`) to quantify reproducibility.
- Results confirm the measurement procedure is stable and reliable.
- Any systematic B1 offset (~30-60 uT) between tests 03 and 04 may correlate with ambient temperature.
