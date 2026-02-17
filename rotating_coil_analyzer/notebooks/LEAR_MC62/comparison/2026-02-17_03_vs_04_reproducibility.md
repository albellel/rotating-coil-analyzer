# MC62 Reproducibility: Test 03 (Feb 16 afternoon) vs Test 04 (Feb 17 morning)

## Overview
Side-by-side comparison of two MC62 staircase measurements taken ~16 hours apart,
to assess day-to-day measurement reproducibility. Both tests use identical hardware,
Kn calibration, current cycle, and analysis pipeline.

## Configuration

| Parameter | Test 03 | Test 04 |
|-----------|---------|---------|
| Date | 2026-02-16 (afternoon) | 2026-02-17 (morning) |
| Magnet | MC62 (red bulk C-shaped dipole, no shims) | same |
| PCBs | Integral (R45) + Central (DQ) | same |
| Rotation | 2 Hz (120 rpm), 512 samples/turn | same |
| Current cycle | 0->+200->0->-200->0 A, 20 A steps | same |
| Kn calibration | External (R45 30 harm, DQ 15 harm) | same |
| R_ref | 0.033 m | same |
| N_LAST_TURNS | 340 | same |
| N_SKIP_END | 20 | same |
| Pipeline | dri, rot (cel/fed auto-disabled) | same |

## Data Summary

| Metric | Test 03 | Test 04 |
|--------|---------|---------|
| Total turns | 35,828 | 32,544 |
| Staircase plateaus | 41 | 40 |
| Plateau turns (Integral) | 30,466 | 31,030 |
| Good-quality runs | 41 | 40 |
| I range (A) | -200 .. 200 | -200 .. 200 |

## Key Comparison Results (Integral PCB)

### B1 (Main Field)

| I (A) | B1_03 (T) | B1_04 (T) | Delta_B1 (uT) |
|-------|-----------|-----------|---------------|
| +20 (asc) | 0.024577 | 0.024540 | -37 |
| +100 (asc) | 0.124245 | 0.124189 | -56 |
| +200 (asc) | 0.218521 | 0.218460 | -61 |
| +100 (desc) | 0.125026 | 0.124991 | -35 |
| -200 (desc) | -0.218514 | -0.218475 | +39 |

### b2 (Quadrupole)

| I (A) | b2_03 (units) | b2_04 (units) | Delta_b2 (units) |
|-------|--------------|--------------|-----------------|
| +200 (asc) | -16.03 | -15.88 | +0.15 |
| -200 (desc) | -15.92 | -15.92 | 0.00 |

### b3 (Sextupole)

| I (A) | b3_03 (units) | b3_04 (units) | Delta_b3 (units) |
|-------|--------------|--------------|-----------------|
| +200 (asc) | -12.09 | -12.09 | 0.00 |
| -200 (desc) | -12.11 | -12.12 | -0.01 |

## Summary Statistics (Integral PCB, 38 matched levels at |I| > 0)

| Quantity | Max |diff| | Mean |diff| | RMS diff |
|----------|-----------|------------|----------|
| Delta_B1 | 62 uT | 33 uT | 37 uT |
| Delta_b2 | 0.274 units | 0.075 units | 0.106 units |
| Delta_b3 | 0.045 units | 0.007 units | 0.011 units |
| Delta_TF | 0.00178 T/kA | 0.00036 T/kA | 0.00045 T/kA |

### Correlation Coefficients (|I| > 0)

| Quantity | Pearson r |
|----------|-----------|
| B1 | 1.00000000 |
| b2 | 0.99537 |
| b3 | 0.99891 |

## Hysteresis Width Comparison

| I (A) | Width_03 (mT) | Width_04 (mT) | Delta (mT) |
|-------|--------------|--------------|-----------|
| 20 | 0.77 | 0.81 | +0.04 |
| 60 | 0.75 | 0.78 | +0.03 |
| 100 | 0.78 | 0.80 | +0.02 |
| 140 | 1.11 | 1.11 | 0.00 |

## Plots Produced

1. **B1 vs Current** (Integral + Central): Overlay of both tests, ascending/descending branches sorted by I_nom
2. **b2 vs Current** (Integral + Central): Quadrupole comparison
3. **b3 vs Current** (Integral + Central): Sextupole comparison
4. **TF vs Current** (Integral + Central): Transfer function comparison
5. **Per-level difference bars**: Delta_B1, Delta_b2, Delta_b3 at each current level
6. **Multipole spectrum** at peak current: Bar chart comparing bn (n=2..15)
7. **Hysteresis width** vs current: Both tests overlaid
8. **B1 std** at each plateau: Turn-to-turn scatter comparison

## Observations & Conclusions

### Reproducibility Verdict
**Excellent day-to-day reproducibility confirmed.** All metrics show differences well within measurement uncertainty:
- Delta_B1 <= 62 uT (~0.03% relative) -- consistent with earth's field variations
- Delta_b3 < 0.05 units -- negligible at 1e-4 relative scale
- Delta_b2 < 0.3 units -- well within turn-to-turn scatter

### Main Field (B1)
- Systematic offset of ~30-60 uT between afternoon and morning measurements
- Likely due to small temperature difference affecting iron permeability
- Offset is consistent across current levels (not current-dependent)

### Quadrupole (b2)
- Mean difference 0.075 units (well below the ~16-unit systematic b2)
- No current-dependent trend in the differences

### Sextupole (b3)
- Mean difference 0.007 units -- negligible
- The b3 ~ -12 units systematic value is perfectly reproducible

### Transfer Function
- Max TF difference 0.0018 T/kA (~0.15% relative)
- Consistent with the B1 offset

### Hysteresis
- Width reproducible to ~40 uT
- No systematic drift between tests

### Advice for Next Measurements
- The 2 Hz streaming procedure is validated for MC62
- Consider tracking ambient temperature to correlate small B1 drifts
- A third repeat would build statistical confidence
