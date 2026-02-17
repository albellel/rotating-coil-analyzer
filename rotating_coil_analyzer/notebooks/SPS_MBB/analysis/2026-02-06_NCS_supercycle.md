# SPS MBB Dipole -- NCS Streaming Supercycle Analysis

## Overview
Full streaming supercycle analysis of the SPS MBB dipole (NCS segment). The measurement captures 20 repetitions of the supercycle structure: LHC_pilot -> MD1 -> SFTPRO. Automatic plateau detection identifies flat-current turns for harmonic analysis.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1) |
| Segment | NCS |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (15 harmonics) |
| Reference radius | 0.02 m |
| Plateau threshold | I range < 2.5 A |
| Supercycle structure | LHC_pilot -> MD1 -> SFTPRO x20 |

## Data Summary

| Item | Value |
|------|-------|
| Raw data shape | 1,086,464 x 5 |
| Total turns | 1,061 |
| Time span | 1,061 s (17.7 min) |
| Current range | -0.2 to 5,750.4 A |
| Plateau turns | 599 / 1,061 |
| Injection turns (MD1, 301 A) | 488 |
| Flat-high turns (SFTPRO, 4,815 A) | 74 |
| Ramp turns | 462 |
| Supercycles detected | 20 |

## Results

### Per Plateau Type

| Plateau | N | I mean (A) | B1 (T) | b2 (units) | b3 (units) | TF (mT/A) |
|---------|---|-----------|--------|-----------|-----------|-----------|
| MD1 injection | 488 | 300.9 | +0.1157 | -1.11 | +0.22 | 0.384 |
| SFTPRO flat-top | 74 | 4,814.5 | +1.7938 | -0.91 | +0.38 | 0.373 |

### Current Flatness
- **MD1 injection: TRUE PLATEAU** -- I = 300.91 +/- 0.090 A (p-p = 0.56 A), drift = +0.023 A over 23 turns
- **SFTPRO flat-top: NOT A TRUE PLATEAU** -- current ramps ~5 A over 3-4 turns

### Hysteresis Evolution (20 supercycles)
- MD1 injection b3 peak-to-peak across SCs: 0.161 units, std: 0.039 units
- SFTPRO b3 peak-to-peak: 0.132 units, std: 0.038 units
- B1 drift at MD1 (last5 - first5): -2.4 uT (negligible)

### Drift Correction
- Legacy vs weighted b3 RMS diff: 2.54e-04 units (negligible)

## Key Findings

1. **MD1 injection is the only reliable plateau** for eddy-current and hysteresis studies.
2. **SFTPRO is NOT a true flat-top** at coil-turn resolution: the current increases by ~5 A across the 3-4 detected turns.
3. **LHC pilot peak (~5,593 A) has no stable flat-top** -- ramp-only, no plateau turns detected.
4. **Eddy-current settling at MD1 is negligible**: B1 shift from turn 0 to turn 2 is only +13.6 uT; b3 bias < 0.003 units.
5. All 562 turns (488 injection + 74 flat-high) passed ok_main at 100%.

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Suggestions

1. **Use MD1 injection for systematic measurements**: stable current, negligible eddy currents, 24 turns/SC.
2. **Treat SFTPRO as a snapshot, not a plateau**: any "settling" analysis at SFTPRO reflects current ramp, not eddy currents.
3. **Longer plateaus needed**: LHC pilot peak has no flat-top at all. If high-field harmonics are needed, request a dedicated flat-top from the machine coordinator.

## Output Files
- `output/.../MBB_NCS_streaming_plateau_legacy.csv` (562 rows)
- `output/.../MBB_NCS_streaming_plateau_weighted.csv` (562 rows)
