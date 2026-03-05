# SPS MBB Dipole -- CS Supercycle -- Feb 6, 2026

## Overview
Full streaming supercycle analysis of the SPS MBB dipole (CS segment). The measurement captures 20 repetitions of the supercycle structure: LHC_pilot -> MD1 -> SFTPRO. Automatic plateau detection identifies flat-current turns for harmonic analysis. Extended sessions at both 200 GeV and 26 GeV beam energies.

**Note:** The segment was originally labelled "NCS" in the DAQ but is physically the
connection side (CS). Labels corrected 2026-03-05.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `CS_harmonics.ipynb` | Single-file preview (data quality check, not for quantitative analysis) |
| `NCS_supercycle.ipynb` | Full supercycle analysis with plateau detection (archived, uses old label) |
| `200_extended_NCS.ipynb` | Extended 200 GeV session analysis (archived, uses old label) |
| `26_extended_NCS.ipynb` | Extended 26 GeV session analysis (archived, uses old label) |
| `b3_settling.ipynb` | Eddy-current b3 settling analysis |
| `b3_settling_200GeV.ipynb` | Eddy-current b3 settling at 200 GeV (detailed) |

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1), CS segment |
| Kn calibration | `Kn_values_Seg_Main_A_AC.txt` (15 harmonics, cross-session from Dec 2025) |
| Reference radius | 0.02 m |
| Plateau threshold | I range < 2.5 A |
| Supercycle | LHC_pilot -> MD1 -> SFTPRO x20 |

## Data Summary (Supercycle)

| Item | Value |
|------|-------|
| Total turns | 1,061 |
| Current range | -0.2 to 5,750.4 A |
| Plateau turns | 599 / 1,061 |
| MD1 injection turns (301 A) | 488 |
| SFTPRO flat-top turns (4,815 A) | 74 |
| Supercycles detected | 20 |

## Supercycle Results

| Plateau | N | I mean (A) | B1 (T) | b2 (units) | b3 (units) | TF (mT/A) |
|---------|---|-----------|--------|-----------|-----------|-----------|
| MD1 injection | 488 | 300.9 | +0.1157 | -1.11 | +0.22 | 0.384 |
| SFTPRO flat-top | 74 | 4,814.5 | +1.7938 | -0.91 | +0.38 | 0.373 |

### Current Flatness
- **MD1 injection: TRUE PLATEAU** -- I = 300.91 +/- 0.090 A
- **SFTPRO flat-top: NOT A TRUE PLATEAU** -- current ramps ~5 A over 3-4 turns

### Supercycle Stability
- MD1 injection b3 peak-to-peak across SCs: 0.161 units, std: 0.039 units
- B1 drift at MD1 (last5 - first5): -2.4 uT (negligible)

## Eddy-Current b3 Settling

### At MD1 Injection (~301 A)
- **Null result**: eddy-current b3 settling is at or below the noise floor.
- Exponential model explains < 2.5% of variance (R2 = 0.024).
- Turn-to-turn scatter (0.03--0.07 units std) swamps any exponential signal.
- Contrasts with MC62 (bulk iron) where tau = 12--36 s and R2 > 0.98. The SPS MBB is laminated -- much faster eddy-current decay.
- **Conclusion**: b3 settling time < 1 turn (< 1 s) at 301 A.

## Key Findings

1. **MD1 injection is the only reliable plateau** for eddy-current and hysteresis studies.
2. **SFTPRO is NOT a true flat-top** at coil-turn resolution.
3. **LHC pilot peak (~5,593 A) has no stable flat-top** -- ramp-only.
4. **Eddy-current settling at MD1 is negligible**: B1 shift < 14 uT; b3 bias < 0.003 units.
5. **Laminated yoke**: eddy currents undetectable in b3, consistent with fast L/R decay in thin laminations.
