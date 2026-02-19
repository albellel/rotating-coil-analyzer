# MC62 -- LEAR C-Shaped Dipole -- 01_staircase_with_shims -- Rotating Coil Analysis

## Overview
Full hysteresis staircase measurement of MC62 with shimming plates installed. Current cycle: 0 -> +200 -> 0 -> -200 -> 0 A in 20 A steps (41 plateaus, 350 turns each at -60 rpm).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, with shims) |
| R_ref | 0.033 m |
| N_LAST_TURNS | 170 |
| Turns/plateau | 350 |
| Pipeline | `dri`, `rot`, `cel`, `fed` (legacy drift) |
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

## Central PCB Data Quality Issues
Several Central runs show corrupt B1 values (e.g., +34 T, +3,065 T, -5.5e6 T). This is due to extremely low compensated-signal SNR. Central PCB results should only be trusted at high |I| (>= 100 A).

## Observations
1. **b2 ~ -151 units**: large systematic quadrupole from C-shape asymmetry. Stable across all current levels.
2. **b3 ~ -17 units**: stable sextupole, with slight current dependence.
3. **Hysteresis**: ~0.7 mT at 100 A between ascending/descending.
4. **With shims** vs without: compare with notebook 02 to evaluate shimming effectiveness.

### Inductance Analysis (Section 10b)
Compares apparent and differential inductance from the B-H curve:
- **L_app = B1/I** (Transfer Function, proportional to apparent inductance = secant of B-H)
- **L_diff = dB1/dI** (proportional to differential inductance = local slope of B-H)

In a linear (unsaturated) magnet, L_app = L_diff = const. As iron saturates, L_diff drops faster than L_app because it tracks the local slope while L_app tracks the secant. The ratio L_diff/L_app < 1 quantifies the degree of saturation. Computed via `np.gradient(B1, I)` per branch (ascending/descending separately to preserve hysteresis information).

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Suggestions
1. **Central PCB needs outlier filtering** before use at low currents.
2. **Compare with 02_without_shims** to quantify shimming effect on b2.
