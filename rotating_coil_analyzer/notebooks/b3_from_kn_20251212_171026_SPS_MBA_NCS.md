# SPS MBA Dipole -- NCS Segment Harmonic Analysis

## Overview
Analysis of the SPS MBA dipole NCS (non-compensated side) segment from session `20251212_171026_SPS_MBA`. Processes 125 plateau files with both legacy and weighted drift correction modes.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBA dipole (m=1) |
| Segment | NCS |
| Session | `20251212_171026_SPS_MBA` |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (AC compensation) |
| Reference radius | 0.02 m |
| Coil length | 0.47 m |
| Samples/turn | 1024 |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Merge mode | `abs_upto_m_cmp_above` |

## Data Summary

| Item | Value |
|------|-------|
| NCS files processed | 125 |
| Total turns | 3,502 from 125 runs |
| Current range | 0--3100 A (63 levels, 50 A steps) |
| Non-zero turns | 3,444 |
| Failed files | 0 |
| ok_main pass rate | 100% (3,502/3,502) |

## Results

### Main Field (B1)
- B1 range (I != 0): [0.0197, 1.1875] T
- B1 at max current (~3100 A): 1.187 T
- Legacy vs weighted B1 RMS difference: 1.82e-07 T (negligible)

### b3 (Sextupole)
- b3 range (I != 0): [-2.56, +0.35] units
- b3 mean: +0.002 units
- b3 std: 0.338 units
- Low-current bias: b3 ~ -2.1 at 50 A, crosses zero around 550 A
- High-current plateau: b3 ~ +0.08 to +0.18 units (stable) for I > 750 A

### Drift Correction Comparison
- b3 RMS difference (legacy vs weighted): 0.000094 units
- b3 max |difference|: 0.002 units
- Drift modes agree to sub-milli-unit precision across all harmonics

### Hysteresis
- At low current (< 300 A): ramp-up gives higher b3 than ramp-down (delta up to +0.29 units)
- This reverses at some high-current levels (delta down to -0.10 units at 1850 A)
- Hysteresis magnitude decreases with |I|

### Reproducibility
- Turn-to-turn b3 CV at |I| > 50 A: < 1%
- Worst scatter at 50 A (CV ~ 13%) as expected from lower SNR

## Observations

1. **Clean data**: Unlike the CS segment, the NCS segment has no extreme outliers. All 3,502 turns produce physically reasonable B1 values.
2. **Excellent drift parity**: Legacy and weighted drift modes agree to < 0.002 units for b3, confirming both algorithms are equivalent for this dataset.
3. **Current-dependent b3**: The b3 varies from -2.1 units at 50 A to +0.18 units at ~1650 A, showing clear saturation-driven nonlinearity.
4. **Small hysteresis**: b3 hysteresis width is < 0.3 units, primarily visible at low currents.

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Suggestions

1. **Benchmark dataset**: The NCS segment could serve as a clean reference dataset for pipeline validation (no outliers, 100% ok_main, excellent drift parity).
2. **Report medians alongside means**: For robustness against potential future outliers.

## Output Files
- `output/20251212_171026_SPS_MBA/MBA_NCS_computed_legacy_drift.csv` (3,502 rows)
- `output/20251212_171026_SPS_MBA/MBA_NCS_computed_weighted_drift.csv` (3,502 rows)
