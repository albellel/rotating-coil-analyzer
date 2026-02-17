# MC62 Shims Effect: Test 01 (with shims) vs Test 02 (without shims)

## Overview
Side-by-side comparison of MC62 measurements with and without iron shimming plates,
to quantify the effect of shims on field quality. Both tests use identical hardware,
Kn calibration, current cycle, and analysis pipeline.

## Configuration

| Parameter | Test 01 | Test 02 |
|-----------|---------|---------|
| Date | 2026-02-11 | 2026-02-12 |
| Magnet | MC62 (red bulk C-shaped dipole, with shims) | MC62 (no shims) |
| PCBs | Integral (R45) + Central (DQ) | same |
| Rotation | 1 Hz (-60 rpm), 1024 samples/turn | same |
| Current cycle | 0->+200->0->-200->0 A, 20 A steps | same |
| Kn calibration | External (R45 30 harm, DQ 15 harm) | same |
| R_ref | 0.033 m | same |
| N_LAST_TURNS | 170 | same |
| Pipeline | dri, rot (cel/fed auto-disabled) | same |

## Data Summary

| Metric | Test 01 (with shims) | Test 02 (without shims) |
|--------|---------------------|------------------------|
| Staircase plateaus | 41 | 41 |
| Total turns (Integral) | 14,350 | 14,351 |
| Good-quality runs | 41 | 41 |
| I range (A) | -200 .. 200 | -200 .. 200 |

## Shims Effect Summary (Integral PCB, |I| > 0)

Differences computed as Delta = Test 02 (no shims) - Test 01 (with shims).

| Quantity | Max |diff| | Mean |diff| | RMS diff |
|----------|-----------|------------|----------|
| Delta_B1 | 0.008334 T | 0.005320 T | 0.005724 T |
| Delta_b2 | 136.4 units | 132.8 units | 132.8 units |
| Delta_b3 | 14.9 units | 14.7 units | 14.7 units |
| Delta_TF | 0.074 T/kA | 0.059 T/kA | 0.061 T/kA |

### Correlation Coefficients (|I| > 0)

| Quantity | Pearson r |
|----------|-----------|
| B1 | 0.99991 |
| b2 | -0.515 |
| b3 | 0.920 |

## Key Findings

1. **Dominant b2 shift (~133 units)**: Removing shims reduced |b2| from ~151 units to ~15 units. The shims were dramatically increasing the quadrupole component rather than correcting it -- suggesting incorrect shim orientation or position.

2. **b3 shift (~15 units)**: Removing shims also changed b3 from ~-17 to ~-12 units. This is a secondary effect through iron saturation redistribution.

3. **B1 shift (~5 mT)**: Small change in main field magnitude, as expected -- shims affect field quality (homogeneity) more than total flux.

4. **TF shift (~0.06 T/kA)**: Transfer function changed by ~5%, consistent with the iron redistribution affecting the magnetic circuit reluctance.

5. **Negative b2 correlation (r = -0.515)**: The current-dependent b2 pattern is inverted between shim and no-shim configurations, confirming the shims fundamentally alter the field symmetry rather than just shifting it.

## Plots Produced

1. **B1 vs Current** (Integral + Central): Overlay of both tests, ascending/descending branches sorted by I_nom
2. **b2 vs Current** (Integral + Central): Shows the dramatic b2 shift from shim removal
3. **b3 vs Current** (Integral + Central): Secondary harmonic shift
4. **TF vs Current** (Integral + Central): Transfer function comparison
5. **Per-level difference bars** (2x2): Delta_B1, Delta_b2, Delta_b3, Delta_TF at each current level
6. **Multipole spectrum** at peak current: Bar chart comparing bn (n=2..15)

## Observations

1. The b2 ~ -151 units WITH shims vs -15 units WITHOUT is counterintuitive. For a C-shaped dipole, b2 arises from broken mid-plane symmetry. The shims appear to have worsened rather than corrected this asymmetry.

2. Both configurations show stable b2 and b3 across the current range (no strong current dependence), indicating the harmonics are dominated by geometry rather than saturation effects.

3. Central PCB shows the same qualitative effects but with more noise at low currents due to poor compensated-signal SNR.

## Recommendations

1. **Verify shim placement**: The large b2 increase with shims suggests they may be oriented incorrectly or positioned asymmetrically.
2. **The no-shims configuration (Test 02) has better field quality** based on lower |b2|.
3. **Further shimming studies** should target b2 reduction from the -15 unit baseline.
