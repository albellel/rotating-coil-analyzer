# SPS MBB Dipole -- NCS Single-Plateau Analysis

## Overview
Analysis of a single NCS streaming file from the SPS MBB dipole measurement session `20260206_142231`. This file contains a streaming supercycle acquisition at a nominal current of 100 A.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1) |
| Segment | NCS |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (cross-session from MBA, same coil) |
| Reference radius | 0.02 m |
| Samples/turn | 1024 |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Min B1 threshold | 1e-4 T |

## Data Summary

| Item | Value |
|------|-------|
| Raw data shape | 118,784 x 5 |
| Total turns | 116 |
| Nominal current | 100 A (single level) |
| Actual current range | -0.3 to 301.1 A (streaming supercycle) |
| ok_main pass rate | 100% (116/116) |

## Results

Because this is a streaming acquisition (not a DC plateau), the 116 turns span a full current ramp (0 to ~300 A and back). Statistics computed across all turns are dominated by ramp turns where current varies widely.

### Per-Turn Statistics (all 116 turns at nominal 100 A)
- B1 mean: +4.79 T, std: 51.2 T (extreme scatter from ramp turns)
- b3 mean: +618 units, std: 6,604 units
- b3 median: -31.2 units (more representative of plateau behavior)

### Drift Correction
- Legacy vs weighted B1 RMS diff: ~499 T (diverges on ramp turns)

## Observations

1. **Streaming supercycle data**: This notebook processes a raw streaming file without plateau detection. All 116 turns are treated equally, including ramp turns where the current varies by hundreds of amperes. This inflates all statistics.
2. **Median vs mean divergence**: The median b3 (-31.2 units) is far more representative of plateau behavior than the mean (+618 units), confirming that outlier ramp turns dominate the mean.
3. **Not suitable for quantitative harmonic analysis without plateau detection**: See the companion supercycle notebook for proper plateau-aware analysis.

## Suggestions

1. **Use this notebook as a data quality preview only**, not for harmonic analysis. The supercycle notebook (`analysis_20260206_144537_SPS_MBB_NCS_supercycle.ipynb`) applies proper plateau detection and is the correct tool for quantitative analysis.
2. **Add per-current-bin statistics** if this notebook is to be used for streaming data exploration.

## Output Files
- `output/.../MBB_NCS_computed_legacy_drift.csv` (116 rows)
- `output/.../MBB_NCS_computed_weighted_drift.csv` (116 rows)
