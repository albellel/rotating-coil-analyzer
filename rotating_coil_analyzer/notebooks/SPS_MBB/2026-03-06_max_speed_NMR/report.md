# SPS MBB Dipole -- Max Speed + NMR -- Mar 6, 2026

## Overview

Full streaming supercycle analysis of the SPS MBB dipole at maximum rotation speed (~176 RPM / 2.93 Hz), with simultaneous NMR/Hall probe acquisition. Two MD1 supercycles were recorded: 200 GeV (full ramp) and 26 GeV (flat top only). Both body and fringe segments were measured with dedicated Kn calibration files per segment.

**Key feature of this campaign:** first measurement with corrected DAQ segment labels (body/fringe match physical segments) and per-segment Kn files from the session itself (not cross-session).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `200GeV_analysis.ipynb` | Full analysis: plateau detection, harmonics, eddy settling, FFMM golden standard (200 GeV MD1) |
| `26GeV_analysis.ipynb` | Same analysis for 26 GeV MD1 |
| `comparison.ipynb` | B1, b2, b3 comparison between 200 GeV and 26 GeV, multipole spectrum, statistical significance |
| `NMR_data.ipynb` | NMR (Caylar) and Hall probe visualization from H5 files; locked/unlocked display, correlation |

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1), body + fringe segments |
| Rotation speed | ~176 RPM (2.93 Hz), period ~0.341 s/turn |
| Kn calibration | Per-segment: `Kn_values_Seg_body.txt`, `Kn_values_Seg_fringe.txt` (from session) |
| Reference radius | 0.02 m |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Plateau threshold | I range < 2.5 A (block-averaged, 10 blocks) |
| Settling turns (injection) | N_LAST = 18 (~6.1 s at 176 RPM) |

## Measurement Sessions

| Session | Supercycle | H5 file (NMR) |
|---------|------------|---------------|
| `20260306_152236_SPS_MBB` | 200 GeV (MD1 full ramp) | `20260306_152447_...md1full_.h5` |
| `20260306_153553_SPS_MBB` | 26 GeV (MD1 flat) | `20260306_153650_...md1flat_.h5` |

## Segment Labelling

From the 4 Hz campaign onwards (2026-03-05), DAQ segment labels are **correct**:
- **body** = physical body segment (inside magnet yoke), `is_fringe=False`
- **fringe** = physical fringe segment (partially outside yoke), `is_fringe=True`

Raw flux peak-to-peak ratio confirms the assignment: body/fringe ~ 6x at flat-top.

## Kn File Assignment

The two Kn files (`Kn_values_Seg_body.txt` and `Kn_values_Seg_fringe.txt`) are calibrations for two coil positions (Z = -280 mm and Z = +280 mm). They differ by only ~0.03% at n=1 (manufacturing tolerance). A Kn swap is **undetectable from rotating coil data alone** because the two coils are physically near-identical.

The golden standard comparison (below) confirms that FFMM uses the same Kn assignment, so **the current mapping is consistent with FFMM**.

For independent verification, the body B1 at LHC top (~2.009 T) matches the NMR locked value (~2.028 T) to ~1%, confirming the body coil sits inside the magnet gap.

## Golden Standard (FFMM Parity)

Both notebooks achieve **machine-precision parity** with the FFMM C++ results using `('dri', 'rot')` + `legacy_rotate_excludes_last=True`:

| Segment | N turns | B1 diff | b2 diff | b3 diff | b4 diff | b5 diff |
|---------|---------|---------|---------|---------|---------|---------|
| body (200 GeV) | 1907 | 0.0 uT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fringe (200 GeV) | 1907 | 0.0 uT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| body (26 GeV) | 1980 | 0.0 uT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fringe (26 GeV) | 1980 | 0.0 uT | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Data Summary

### 200 GeV MD1

| Item | Value |
|------|-------|
| Total turns (body) | 1964 |
| Plateau turns (body) | 1473 |
| Settled turns (body) | 358 (injection: 339, flat-high: 19) |
| Total turns (fringe) | 1910 |

### 26 GeV MD1

| Item | Value |
|------|-------|
| Total turns (body) | 2034 |
| Plateau turns (body) | 1588 |
| Settled turns (body) | 375 (injection: 356, flat-high: 19) |
| Total turns (fringe) | 1980 |

## Results -- Settled Harmonics

### Body (main field)

| Energy | Plateau | N | I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|--------|---------|---|-------|--------|------------|------------|-----------|
| 200 GeV | Injection | 339 | 300.9 | -0.1159 | -0.13 | -0.21 | 0.385 |
| 200 GeV | Flat-top | 19 | 4815.8 | -1.7811 | +0.09 | +0.02 | 0.370 |
| 26 GeV | Injection | 356 | 300.9 | -0.1156 | +0.09 | -0.01 | 0.384 |
| 26 GeV | Flat-top | 19 | 4815.9 | -1.7813 | +0.09 | +0.02 | 0.370 |

### Fringe

| Energy | Plateau | N | I (A) | B1 (T) | b2 (units) | b3 (units) | TF (T/kA) |
|--------|---------|---|-------|--------|------------|------------|-----------|
| 200 GeV | Injection | 336 | 300.9 | -0.0197 | -1.21 | +4.42 | 0.065 |
| 200 GeV | Flat-top | 19 | 4815.8 | -0.2909 | +0.39 | +7.18 | 0.060 |
| 26 GeV | Injection | 360 | 300.9 | -0.0196 | -1.49 | +4.89 | 0.065 |
| 26 GeV | Flat-top | 19 | 4815.9 | -0.2909 | +0.75 | +7.12 | 0.060 |

## NMR / Hall Probe

The Caylar NMR teslameter (with integrated Hall probe) acquired simultaneously during both supercycles. The NMR locks only at high field (> ~1.6 T, i.e. SFTPRO flat-top and LHC top).

| Supercycle | Total samples | Locked samples | Hall range (T) | |NMR| range locked (T) |
|------------|---------------|----------------|----------------|------------------------|
| 200 GeV | 86,091 | 1,138 (1.3%) | 0.80 -- 2.17 | 1.63 -- 2.03 |
| 26 GeV | 92,604 | 1,105 (1.2%) | 1.66 -- 2.17 | 1.79 -- 2.03 |

### NMR vs Body B1 at LHC Top

| Measurement | Value |
|-------------|-------|
| Body B1 max (rotating coil) | 2.009 T |
| NMR max (locked) | 2.028 T |
| Hall probe max | 2.172 T |
| NMR / Body B1 ratio | 0.99 |

The NMR tracks the body B1 to ~1%. The difference is expected: B1 is integrated over the 0.47 m coil length while NMR is a point measurement. The Hall probe reads ~8% higher, likely due to calibration or positioning differences.

## Eddy Current Settling

### Body segment

Eddy settling at injection is weak (laminated yoke):
- B1: 1-tau fit, R2 ~ 0.6 (200 GeV) / 0.67 (26 GeV) -- marginal
- b2: R2 < 0.01 -- no detectable eddy
- b3: R2 ~ 0.06 (200 GeV) / 0.008 (26 GeV) -- no detectable eddy

### Fringe segment

Eddy settling is stronger in the fringe (amplified by small B1):
- B1: 2-tau fit, R2 ~ 0.98 (200 GeV) / 0.95 (26 GeV), tau1 ~ 1.1 s
- b3: 1-tau fit, R2 ~ 0.95 (200 GeV) / 2-tau R2 ~ 0.78 (26 GeV), tau ~ 1.4 s

## Key Findings

1. **Max speed works**: 176 RPM (~2.93 Hz) produces clean data with machine-precision FFMM parity on all harmonics.
2. **Body B1 matches expectations**: 116 mT (injection), 1.78 T (SFTPRO), 2.01 T (LHC top).
3. **NMR confirms body assignment**: NMR locked value (~2.03 T) matches body B1 max (~2.01 T) to 1%.
4. **DAQ labels now correct**: body and fringe labels match physical segments (verified by raw flux ratio ~6x).
5. **Kn assignment consistent with FFMM**: golden standard validates the Kn-to-segment mapping. However, a Kn swap is inherently undetectable from rotating coil data alone (0.03% difference at n=1).
6. **Fringe eddy currents dominate**: settling is measurable in the fringe (tau ~ 1 s) but negligible in the laminated body.
7. **200 GeV vs 26 GeV**: body harmonics are consistent across energies; injection b2 shows a small difference (-0.13 vs +0.09 units) suggesting hysteresis sensitivity.

## Output CSVs

All outputs in `output/MBB/2026-03-06_max_speed_NMR/`:

| File | Description |
|------|-------------|
| `200GeV/MBB_{body,fringe}_streaming_plateau.csv` | All plateau turns |
| `200GeV/MBB_{body,fringe}_streaming_settled.csv` | Settled turns only |
| `200GeV/eddy_fits_mean.csv` | Multi-tau eddy fit parameters |
| `26GeV/...` | Same structure for 26 GeV |
| `compare_200_vs_26/summary_comparison_settled.csv` | Cross-energy comparison |
