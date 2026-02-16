# SPS MBB Dipole -- 200 GeV vs 26 GeV Comparison

## Overview
Compares harmonic results between two measurement sessions of the SPS MBB dipole (NCS segment) at 200 GeV and 26 GeV beam energies. Both sessions have identical supercycle structure (LHC_pilot -> MD1 -> SFTPRO x20) but were measured ~30 min apart.

## Configuration

| Parameter | Value |
|-----------|-------|
| 200 GeV session | `2026_02_06/01_200_extended` |
| 26 GeV session | `2026_02_06/03_26_extended` |
| Operating points | MD1 injection (~301 A) and SFTPRO flat-top (~4,815 A) |
| Settling correction | Last 18 of ~24 injection turns per supercycle |
| Outlier removal | 5 MAD-sigma clipping on B1 |

## Data Summary

| Property | 200 GeV | 26 GeV |
|----------|---------|--------|
| Total turns | 1,061 | 1,064 |
| Injection turns (settled) | 360 | 357 |
| Flat-high turns | 72 | 69 |
| Sigma-clip removals | 0 | 3 |

## Results

### At Injection (~301 A)

| Quantity | 200 GeV | 26 GeV | Delta | Significance |
|----------|---------|--------|-------|-------------|
| B1 (T) | 0.115644 | 0.115649 | -5.6 uT | 14.5 sigma |
| b2 (units) | -1.107 | -1.112 | +0.006 | 0.5 sigma |
| b3 (units) | +0.222 | +0.230 | -0.008 | 1.7 sigma |

### At SFTPRO Flat-Top (~4,815 A)

| Quantity | 200 GeV | 26 GeV | Delta | Significance |
|----------|---------|--------|-------|-------------|
| B1 (T) | 1.793825 | 1.793925 | -100 uT | 1.3 sigma |
| b2 (units) | -0.905 | -0.933 | +0.029 | 2.7 sigma |
| b3 (units) | +0.385 | +0.378 | +0.007 | 0.9 sigma |

## Key Findings

1. **Injection B1** shows the only statistically significant difference: -5.6 uT (14.5 sigma). This is real but extremely small (~50 ppm of 0.1156 T).
2. **SFTPRO b2** shows a suggestive difference (+0.029 units, 2.7 sigma) but is not conclusive.
3. **All other quantities** are not statistically distinguishable between the two sessions.
4. The high B1 sigma is driven by very low turn-to-turn scatter (5 uT std) combined with ~360 turns per dataset.

## Observations

1. The two datasets were measured ~30 min apart with different magnetisation histories, so differences reflect cumulative history effects.
2. b3 and b2 differences are well within measurement uncertainty.
3. The tiny B1 difference at injection may reflect residual magnetisation or temperature drift.

## Output Files
- `output/.../compare_200_vs_26/summary_comparison_settled.csv` (4 rows)
- Per-supercycle and per-turn CSV files for both sessions
