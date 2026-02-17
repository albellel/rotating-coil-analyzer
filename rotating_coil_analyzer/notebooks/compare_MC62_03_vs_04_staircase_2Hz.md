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
| Pipeline | dri, rot, cel, fed (legacy drift) | same |

## Data Summary
*(To be filled after execution.)*

| Metric | Test 03 | Test 04 |
|--------|---------|---------|
| Total turns | TBD | TBD |
| Staircase plateaus | TBD | TBD |
| Good-quality runs | TBD | TBD |

## Key Comparison Results (Integral PCB)

### B1 (Main Field)
| I (A) | B1_03 (T) | B1_04 (T) | dB1 (T) |
|-------|-----------|-----------|---------|
| +100 (asc) | TBD | TBD | TBD |
| +200 (asc) | TBD | TBD | TBD |
| -200 (desc) | TBD | TBD | TBD |

### b2 (Quadrupole)
| I (A) | b2_03 (units) | b2_04 (units) | db2 (units) |
|-------|--------------|--------------|-------------|
| +200 (asc) | TBD | TBD | TBD |
| -200 (desc) | TBD | TBD | TBD |

### b3 (Sextupole)
| I (A) | b3_03 (units) | b3_04 (units) | db3 (units) |
|-------|--------------|--------------|-------------|
| +200 (asc) | TBD | TBD | TBD |
| -200 (desc) | TBD | TBD | TBD |

## Per-Harmonic Differences at Peak Current
*(Bar chart of bn (n=2..15) side-by-side at |I|=200 A.)*

## Hysteresis Width Comparison
*(Plot of B1 ascending - descending vs |I| for both tests.)*

## Turn-to-Turn Scatter
*(B1_std comparison at each plateau level.)*

## Summary Statistics
*(To be filled after execution.)*

| Metric | Value |
|--------|-------|
| Max |dB1| | TBD T |
| Max |db2| | TBD units |
| Max |db3| | TBD units |
| RMS dB1 | TBD T |
| B1 correlation | TBD |

## Observations & Advice

### Reproducibility Verdict
*(Overall assessment after execution.)*

### Main Field (B1)
- Expected: differences < 10 uT (earth field level) indicate excellent reproducibility.

### Quadrupole (b2)
- Expected: sub-unit differences for a stable magnet.

### Sextupole (b3)
- Expected: sub-unit differences.

### Transfer Function
- Any systematic TF shift could indicate temperature-dependent permeability.

### Hysteresis
- Width should be nearly identical if the magnet's domain structure is well-established.

### Advice for Next Measurements
- If reproducibility is confirmed, this validates the 2 Hz streaming procedure for MC62.
- Consider running a third repeat (e.g., next morning) to build statistics.
- Track ambient temperature to correlate any small drifts.
