# SPS MBB Dipole -- Dec 12, 2025

## Overview
Analysis of the SPS MBB dipole from measurement session `20251212_171026_SPS_MBA`. Both CS (connection side) and NCS (non-connection side) segments are processed through the full Kn pipeline.

**Note:** DAQ segment labels were swapped in all campaigns up to 2026-02-25 (what the DAQ called "CS" was physically the NCS side, and vice versa). For this pre-2Hz campaign, both segments were inside the magnet body so the swap had no practical impact on results.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1) |
| Session | `20251212_171026_SPS_MBA` |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (AC compensation) |
| Reference radius | 0.02 m |
| Coil length | 0.47 m |
| Samples/turn | 1024 |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Merge mode | `abs_upto_m_cmp_above` |
| Min B1 threshold | 1e-4 T |

## Data Summary

| Item | CS | NCS |
|------|----|----|
| Files processed | 125 | 125 |
| Total turns | 3,502 | 3,502 |
| Current range | 0--3100 A (63 levels, 50 A steps) | same |
| Non-zero turns | 3,444 | 3,444 |
| Failed files | 0 | 0 |
| ok_main pass rate | -- | 100% |

## CS Segment

### Outlier Rejection
CS contains sporadic turns with extreme unphysical B1 values (up to ~7.4e+11 T) from ADC glitches. MAD-based sigma clipping (`mad_sigma_clip()`) on B1, 5-sigma threshold, applied per operating point.

### Results
- B1 at max current (~3100 A): ~1.19 T
- b3 range: approximately -0.5 to +0.1 units
- Clean levels (500--700 A): b3 ~ -0.3 units, std ~ 0.2 units
- Legacy vs weighted drift modes agree to sub-milli-unit precision after outlier removal

## NCS Segment

### Results
- B1 range (I != 0): [0.0197, 1.1875] T
- **Clean data**: no extreme outliers (unlike CS)
- Legacy vs weighted B1 RMS diff: 1.82e-07 T (negligible)
- b3 mean: +0.002 units, std: 0.338 units
- Low-current bias: b3 ~ -2.1 at 50 A, crosses zero around 550 A
- High-current plateau: b3 ~ +0.08 to +0.18 units (stable) for I > 750 A
- b3 RMS diff (legacy vs weighted): 0.000094 units

### Hysteresis
- At low current (< 300 A): ramp-up gives higher b3 than ramp-down (delta up to +0.29 units)
- Reverses at some high-current levels (delta down to -0.10 units at 1850 A)
- Hysteresis magnitude decreases with |I|

## Key Findings

1. **AC compensation** suppresses dipole sensitivity by ~20,000x in the compensated channel.
2. **CS requires MAD outlier rejection** -- ADC glitches produce extreme B1 values.
3. **NCS is clean** and could serve as a benchmark reference dataset (100% ok_main, excellent drift parity).
4. **Excellent drift-mode parity**: legacy and weighted agree to < 0.002 units for all harmonics.
5. **Current-dependent b3**: varies from -2.1 units at 50 A to +0.18 units at ~1650 A (saturation-driven nonlinearity).

## Output Files
- `output/20251212_171026_SPS_MBA/MBB_{CS,NCS}_computed_{legacy,weighted}_drift.csv`
