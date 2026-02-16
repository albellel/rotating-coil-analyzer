# MC62 -- LEAR C-Shaped Dipole -- 02_staircase_without_shims -- Rotating Coil Analysis

## Overview
Full hysteresis staircase measurement of MC62 without shimming plates. Current cycle: 0 -> +200 -> 0 -> -200 -> 0 A in 20 A steps (41 plateaus, 350 turns each at -60 rpm).

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | MC62 (red bulk C-shaped dipole, without shims) |
| R_ref | 0.033 m |
| N_LAST_TURNS | 170 |
| Pipeline | `dri`, `rot`, `cel`, `fed` (legacy drift) |

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

## Observations
1. **b2 ~ -15 units** (without shims) vs **-151 units** (with shims in test 01). The shimming INCREASED the quadrupole dramatically -- the shims may need repositioning.
2. **b3 ~ -17 units**: consistent between shim/no-shim tests.
3. **TF** and **hysteresis** patterns match test 01 closely (as expected -- shims affect field quality, not main field).
4. **Central PCB** has similar corrupt-value issues at low current as in test 01.

## Suggestions
1. **Compare b2 systematically** between 01 (with shims) and 02 (without shims) to evaluate shimming effectiveness.
2. **The b2 ~ -152 with shims vs -15 without** is counterintuitive -- verify shim orientation and position.
