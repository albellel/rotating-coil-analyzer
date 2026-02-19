# SPS MBB Dipole -- CS Segment Harmonic Analysis

## Overview
Analysis of the SPS MBB dipole CS (compensated side) segment from measurement session `20251212_171026_SPS_MBA`. Processes 125 plateau files through the full Kn pipeline with both legacy and weighted drift correction modes.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1) |
| Segment | CS |
| Session | `20251212_171026_SPS_MBA` |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (AC compensation) |
| Reference radius | 0.02 m |
| Coil length | 0.47 m |
| Samples/turn | 1024 |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Merge mode | `abs_upto_m_cmp_above` |
| Min B1 threshold | 1e-4 T |

## Kn Calibration
- 15 harmonics (n=1..15)
- AC compensation suppression ratio (n=1): 20,268x
- |Kn_abs(n=1)| = 1.154e-01, |Kn_cmp(n=1)| = 5.696e-06

## Data Summary

| Item | Value |
|------|-------|
| Total raw files | 250 (both CS+NCS) |
| CS files processed | 125 |
| Total turns | 3,502 from 125 runs |
| Current range | 0--3100 A (63 levels, 50 A steps) |
| 0 A turns excluded | 58 |
| Non-zero turns | 3,444 |
| Failed files | 0 |

## Outlier Rejection

The CS segment contains sporadic turns with extreme unphysical B1 values (up to ~7.4e+11 T) from ADC glitches or digitizer overflow. These corrupt all derived harmonics and make the inductance L_diff plot unusable (gradient spikes).

**Method**: MAD-based sigma clipping (`mad_sigma_clip()`) on `B1_T`, applied per operating point (`I_nom_A`) with a 5-sigma threshold. The same keep-mask is applied to both legacy and weighted DataFrames since they represent identical physical turns. All downstream analysis (B1 overview, drift comparison, b3, hysteresis, inductance, export) automatically uses cleaned data.

## Results

### Main Field (B1)
- B1 at max current (~3100 A): ~1.19 T
- After outlier rejection, all remaining turns have physically reasonable B1 values

### b3 (Sextupole)
- b3 range across non-zero levels: approximately -0.5 to +0.1 units
- Clean levels (e.g., 500-700 A): b3 ~ -0.3 units with std ~ 0.2 units

### Drift Correction Comparison
Legacy and weighted drift modes agree to sub-milli-unit precision for b3 across all current levels after outlier removal.

## Observations

1. **AC compensation** works as expected, suppressing dipole sensitivity by ~20,000x in the compensated channel.
2. **MAD outlier rejection** removes corrupt turns with extreme B1 values, cleaning up all downstream statistics and inductance plots.
3. **Well-behaved levels** (500-700 A) show excellent drift-mode agreement and low turn-to-turn scatter (std ~ 0.2 units).
4. All 125 files processed without pipeline errors.

### Inductance Analysis (Section 8b)
Compares apparent and differential inductance from the ramp B-H curve:
- **L_app = B1/I** (Transfer Function, proportional to apparent inductance = secant of B-H)
- **L_diff = dB1/dI** (proportional to differential inductance = local slope of B-H)

Per-run B1 averages are split into ascending/descending branches at the peak current. The ascending branch corresponds to the initial ramp-up (virgin curve), the descending branch to the ramp-down. L_diff is computed via `np.gradient(B1, I)` per branch. Saturation is visible as L_diff dropping below L_app at high current, and the ratio L_diff/L_app quantifies the saturation level.

### cel/fed Safety Diagnostic
This notebook includes a `diagnose_cel_fed()` check that verifies the centre-location and feeddown corrections (cel/fed) are reliable. The diagnostic compares pipeline results with and without cel/fed, flags turns with |zR| > 1% of R_ref, and provides a SAFE/MIXED/UNSAFE recommendation. See `correction_options_reference.md` for background.

## Suggestions

1. **Investigate raw data**: The extreme outliers at specific turns likely indicate ADC glitches, digitizer overflow, or corrupt data blocks. Trace back to the raw flux waveforms to identify the root cause.
2. **Flag affected levels**: Mark levels with high std/median ratio as potentially contaminated in the summary table.

## Output Files
- `output/20251212_171026_SPS_MBA/MBB_CS_computed_legacy_drift.csv` (3,502 rows)
- `output/20251212_171026_SPS_MBA/MBB_CS_computed_weighted_drift.csv` (3,502 rows)
