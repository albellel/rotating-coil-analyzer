# SPS MBB Dipole -- Eddy-Current b3 Settling at 200 GeV

## Overview
Investigates b3 (sextupole) eddy-current settling at the MD1 injection plateau (~301 A) of the SPS MBB dipole. Fits exponential decay model B1(t) = B1_inf + A * exp(-t/tau) to each supercycle's injection plateau.

## Configuration

| Parameter | Value |
|-----------|-------|
| Extended session | `2026_02_06/01_200_extended` |
| Original session | `2026_02_06/02_200_original` |
| Plateau threshold | I range < 3.0 A |
| Min injection turns for fit | 5 |
| Kn harmonics | 15 |

## Data Summary

| Dataset | Raw turns | Plateau turns | Injection groups | Injection turns fitted |
|---------|-----------|--------------|------------------|----------------------|
| Extended | 1,061 | 604 | 20 supercycles | 480 (24/SC) |
| Original | 90 | 40 | 0 | 0 (no injection detected) |

## Results

### Per-Supercycle Fits (Extended)
- 20/20 supercycles fitted
- tau mean: 302.4 s, std: 456.7 s, **median: 5.0 s**
- R-squared range: 0.002 to 0.327
- Several supercycles hit the upper tau bound (1,000 s)

### Global Fit (All 480 Points)
- tau = 0.41 +/- 0.56 s
- R-squared: **0.024** (no explanatory power)

## Key Findings

1. **The eddy-current b3 settling is at or below the noise floor** for this magnet at the MD1 injection level (~301 A).
2. The exponential model explains < 2.5% of the b3 variance (R2 = 0.024).
3. The b3 turn-to-turn scatter within each supercycle (0.03-0.07 units std) is comparable to the total amplitude range, swamping any exponential signal.
4. The **Original dataset** yielded zero injection groups (plateau turns were at a different current level).

## Observations

1. **Null result**: The eddy-current settling signal in b3 is too small to measure with this magnet/excitation level/rotation speed.
2. The per-supercycle fits are wildly inconsistent (tau ranges from 0.1 s to 1,000 s), confirming the exponential model is not appropriate for this data.
3. This contrasts with the MC62 dipole where tau = 12-36 s and R2 > 0.98 -- the SPS MBB is a laminated warm magnet with much faster eddy-current decay at 301 A.

## Suggestions

1. **Higher excitation levels may show measurable settling**: At 4,815 A (SFTPRO), mu_r is lower and the eddy-current amplitude relative to noise may be different, but the short SFTPRO flat-top (3-4 turns) prevents fitting.
2. **Dedicated measurements with longer plateaus** at higher current could improve the SNR for eddy-current characterisation.
3. Consider reporting this as a null result: b3 settling time < 1 turn (< 1 s) at 301 A.

## Output Files
- `output/.../eddy_current_b3_settling/b3_injection_extended.csv` (480 rows)
- `output/.../eddy_current_b3_settling/b3_fits_per_supercycle_extended.csv` (20 rows)
- `output/.../eddy_current_b3_settling/b3_global_fit_summary.csv` (1 row)
