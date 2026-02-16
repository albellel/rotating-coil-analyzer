# MC62 -- Validation: Python Analyzer vs FFMM C++ Pipeline

## Overview
Validates our Python rotating-coil analyzer against the reference FFMM C++ pipeline for the MC62 02_staircase_without_shims measurement. Both pipelines process the same raw data.

## Configuration
- FLIP_FIELD_SIGN = False (raw sign for comparison)
- FFMM reference: `MC62_Integral_Average_results.txt`, `MC62_Central_Average_results.txt`

## Results

### Best Averaging Window
FFMM averages all 350 turns per plateau (confirmed by sweep). At N_LAST = 350:

| Metric | Value |
|--------|-------|
| B_main RMS (Integral, |I|>=10 A) | 0.6 uT |
| b3 RMS (Integral) | 0.000 units |
| Our default (N_LAST=170) RMS | 72.4 uT (explained by eddy-current settling exclusion) |

### Per-Harmonic RMS Residuals (N_LAST=350, |I|>=10 A)

| Order | RMS bn | RMS an |
|-------|--------|--------|
| n=2 | 0.003 | 0.001 |
| n=3 | 0.000 | 0.000 |
| n=4--10 | 0.000 | 0.000 |
| n=11--15 | 0.001--0.003 | 0.001--0.002 |

**All harmonics < 0.003 units -- machine precision parity.**

### FFMM Central PCB Issues
5 of 41 FFMM Central rows have corrupt B_main (|B_main| > 1 T). Both our pipeline and FFMM produce corrupt values for the Central PCB at low currents.

## Verdict
- **Normal harmonics (bn): PASS** -- all < 1 unit
- **Skew harmonics (an): PASS** -- all < 1 unit
- **Overall: PASS**

## Observations
1. **Sub-microtesla B_main agreement** when matching the same averaging window (350 turns).
2. The 72 uT difference at N_LAST=170 is intentional (we exclude settling turns; FFMM does not).
3. **Central PCB validation unreliable** -- both pipelines have corrupt values at low current.

## Suggestions
1. The `max_zR` clamping that was removed was the correct decision -- without it, all harmonics match to machine precision (0.003 units worst case).
2. Consider adding a Central PCB SNR check to flag unreliable plateaus automatically.
