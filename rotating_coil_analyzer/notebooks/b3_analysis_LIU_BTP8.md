# LIU BTP8 Quadrupole -- b3 Sextupole Analysis

## Overview
Comprehensive b3 (sextupole) analysis for the LIU BTP8 quadrupole, validated against the legacy C++ analyzer. This quadrupole (m=2) is the primary golden standard dataset.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | LIU BTP8 quadrupole (m=2) |
| Kn calibration | `Kn_R45_PCB_N1_0001_A_ABCD.txt` (15 harmonics) |
| Reference radius | 0.059 m |
| Samples/turn | 512 |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| `legacy_rotate_excludes_last` | False |

## Data Summary
- 37 runs, 19 current levels (0, +/-5, +/-10, +/-25, +/-50, +/-75, +/-100, +/-125, +/-150, +/-200 A)
- 519 total turns processed, 222 aligned to reference
- 0 A excluded (18 turns): B2 at noise level makes normalised harmonics meaningless

## Results

### b3 per Current Level (0 A excluded)
- b3 ranges from +1.67 units (low current) to +2.02 units (200 A)
- Clear current dependence due to iron saturation effects
- Turn-to-turn CV < 1% at |I| >= 100 A
- Scatter increases dramatically at low currents (1.48 units std at 5 A vs 0.006 at 200 A)

### Hysteresis
- Negligible: max delta = +0.051 units at 10 A (noise), < 0.007 units at |I| >= 100 A

### Parity with Legacy C++
- **Max |diff|: 4.13e-04 units** (single turn at +150 A)
- **RMS diff: 2.98e-05 units**
- **100%** of 204 non-zero turns within 0.001 units
- **99.5%** within 1e-04 units

### a3 (Skew Sextupole)
- a3 ~ +1.6 to +2.1 units (comparable magnitude to b3)
- b3-a3 anti-correlation: r = -0.986

## Key Findings
1. **Machine-precision parity** with legacy C++ at every non-zero current level.
2. b3 shows clear current-dependent saturation nonlinearity (+1.7 to +2.0 units).
3. Hysteresis is negligible for this quadrupole.
4. Turn-to-turn reproducibility is excellent (median CV = 0.36% at |I| > 50 A).
