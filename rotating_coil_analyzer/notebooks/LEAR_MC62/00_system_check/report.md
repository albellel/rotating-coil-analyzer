# MC62 -- LEAR C-Shaped Dipole -- 00_test -- Rotating Coil Analysis

## Overview
Initial test measurement of the MC62 LEAR C-shaped bulk dipole. Short staircase: 0 -> +100 -> +200 -> +100 -> 0 -> -100 -> -200 -> -100 -> 0 A (9 plateaus, 10 turns each).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole) |
| Magnet order | 1 (dipole) |
| R_ref | 0.033 m |
| Samples/turn | 1024 |
| N_LAST_TURNS | 8 (of 10 total) |
| Sign flip | Yes |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |

## Data Summary
- 9 integral runs, 9 central runs
- 10 turns per run, 90 total rows per PCB
- Timeline: 0--3,446 s
- All 9 runs succeeded, all marked "good" quality

## Key Results (Integral PCB, last 8 turns averaged)

| I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|-------|--------|-----------|-----------|-----------|
| +100 (asc) | 0.131 | -153 | -1.1 | 1.306 |
| +200 (asc) | 0.228 | -152 | -1.6 | 1.138 |
| +100 (desc) | 0.136 | -152 | -1.3 | 1.356 |
| -200 (desc) | -0.228 | -152 | -1.6 | 1.140 |

## Observations
1. **Large systematic b2 ~ -152 units**: inherent C-shaped geometry asymmetry.
2. **Small b3 ~ -1 to -2 units** at excitation for both PCBs.
3. **Hysteresis** visible: 5 mT gap at 100 A between ascending/descending branches.
4. **Transfer function**: saturation onset visible (TF drops from 1.31 to 1.14 T/kA between 100 and 200 A).
5. **Central PCB**: B1 systematically ~1.65x higher than integral (local field vs field integral/length).
6. Only 10 turns per plateau limits eddy-current settling -- most turns are still settling.

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Suggestions
1. This is a **quick system check**, not a production measurement. Use the 350-turn staircase tests (01, 02) for quantitative analysis.
2. The N_LAST_TURNS=8 (of 10) likely includes settling transient.
