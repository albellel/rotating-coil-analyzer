# MC62 -- LEAR C-Shaped Dipole -- 04_staircase_2Hz_morning -- Rotating Coil Analysis

## Overview
Morning repeat measurement of MC62 (no shims), streaming at 2 Hz rotation (512 samples/turn).
This is a reproducibility test: identical setup to test 03 (Feb 16 afternoon), repeated the following morning (Feb 17).
Current cycle: precycles (0->-200->+200->-200->0 A) then staircase 0->+200->0->-200->0 A in 20 A steps.
Ramp rate: 1 A/s. Expected ~41 staircase plateaus, ~740 turns each (~370 s per step).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, no shims) |
| Rotation | 2 Hz (120 rpm), 512 samples/turn, 0.5 s/turn |
| R_ref | 0.033 m |
| N_LAST_TURNS | 340 |
| Pipeline | `dri`, `rot`, `cel`, `fed` (legacy drift) |
| Parity pipeline | `dri`, `rot`, `cel`, `fed`, `dit` (signed, matching FFMM C++ native) |
| Plateau threshold | 0.5 A (block-averaged), min_length=50, merge gap<100 |

## Data Summary
- *(to be filled after execution)*
- Total turns: TBD
- Precycle groups + staircase plateaus: TBD
- Plateau quality: TBD

## Key Results (Integral PCB)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +100 (asc) | TBD | TBD | TBD | TBD |
| +200 (asc) | TBD | TBD | TBD | TBD |
| +100 (desc) | TBD | TBD | TBD | TBD |
| -200 (desc) | TBD | TBD | TBD | TBD |

## Golden Standard Parity Check (vs FFMM C++)

Parity uses `dit` (di/dt correction) with `signed=True` to match the FFMM C++ native
threshold logic (`crr > 0.1 && cm > 10`).

### B_main
- *(to be filled after execution)*

### Harmonics
- *(to be filled after execution)*

### Note on FFMM Central
FFMM Central results are expected to be all NaN (measurement-embedded Central Kn is all-zeros). Parity check is Integral only.

## cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check. See `correction_options_reference.md` for background.
- Diagnostic result: *(to be filled after execution)*

## Observations
1. **Reproducibility**: Compare results with test 03 (Feb 16) to assess day-to-day measurement stability.
2. **b2**: Expected ~-13 units at high current (consistent with C-shaped dipole asymmetry).
3. **b3**: Expected ~-12 units.
4. **Hysteresis**: Expected similar ascending/descending branch structure.
5. **Central PCB**: May still show numerical issues (same PCB calibration).

## Comparison with Previous Tests

| Test | b2 (units) | b3 (units) | TF@200A (T/kA) |
|------|-----------|-----------|-----------------|
| 01 (with shims, 1 Hz) | -151 | -17 | 1.132 |
| 02 (no shims, 1 Hz) | -15 | -17 | 1.132 |
| 03 (no shims, 2 Hz) | -13 | -12 | 1.093 |
| 04 (no shims, 2 Hz, morning) | TBD | TBD | TBD |

## Suggestions
- Run the companion comparison notebook (`LEAR_MC62/comparison/2026-02-17_03_vs_04_reproducibility.ipynb`) to quantify reproducibility.
- If results match test 03, this confirms the measurement procedure is stable and reliable.
- Any systematic differences may point to thermal drift (magnet temperature difference morning vs afternoon).
