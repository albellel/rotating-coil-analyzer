# Harmonic Spectra

## Layout

| Column | Shows |
|--------|-------|
| **Body + Fringe + Integrated** | Signed b_n for all three segments (linear). Shows why integrated b₃ can be large: fringe b₃ ≈ 5 units dominates despite short length. |
| **Body (log)** | |b_n| on log scale. Downward slope = real multipoles; horizontal plateau = noise floor. |
| **Fringe (log)** | Same for fringe. Higher noise floor in units (weaker B₁), but dominant harmonics stand out more. |
| **Integrated** | What the beam sees. Body dominates by 12.5× in length and 6× in field. |

## Reading the log plot

Real multipoles decay as |b_n| ∝ (R_ref/R)^n — a straight line on log scale.
Noise adds a constant floor. The transition from slope to plateau tells you
up to which order the measurement is trustworthy.

## Noise floor and harmonic detection

The **red dashed line** is the estimated measurement noise floor, computed from
the **supercycle-to-supercycle scatter** of each b_n:

    sigma_n = std_SC(mean(b_n)),    noise floor line = median_n(sigma_n)

The line is a visual reference for the typical noise level. But each harmonic
is coloured or greyed out based on its own **per-harmonic SNR**:

    SNR_n = |mean(b_n)| / sigma_n

A harmonic is coloured (detected) if SNR_n > 2, i.e. its mean value is more
than twice its own supercycle scatter.

This avoids the pitfall of a single global threshold: a harmonic sitting on the
smooth decay curve (e.g. b_5 in the body) may have a small absolute value but
is highly **repeatable** across supercycles — low sigma_n → high SNR →
correctly identified as real signal. Conversely, a harmonic with a larger
absolute value but huge scatter is correctly flagged as unreliable.

## 200 GeV MD1

![Harmonics 200 GeV MD1](harmonics_200GeV_MD1.png)

## 26 GeV MD1

![Harmonics 26 GeV MD1](harmonics_26GeV_MD1.png)

## Body vs Fringe: b₂

![b2 body vs fringe](b2_body_vs_fringe.png)

## Body vs Fringe: b₃

![b3 body vs fringe](b3_body_vs_fringe.png)

## Key observations

- **First column (linear)**: fringe b₃ ≈ 5 units dwarfs body b₃ ≈ −0.2 units.
  The integrated b₃ is small because the body dominates in length and field.

- **Body log**: the body has excellent field quality — harmonics are small,
  reaching noise quickly. Only b₃ (and maybe b₅) are clearly above noise.
  This is **expected for a good dipole**: clean iron geometry with minimal
  higher-order content.

- **Fringe log**: more harmonics above noise because B₁ is 6× weaker,
  amplifying b_n. These describe the **fringe field shape** (end effects),
  not the beam-relevant integrated field.

- At **flat-top** (high B₁), noise floor drops and more harmonics are reliable.

- At **injection** (low B₁), noise rises. Body harmonics near noise does
  **not** mean bad field quality — it means quality exceeds measurement
  sensitivity at this current.

- **Noise floor line**: the red dashed line is the median of sigma_n — a
  visual reference. The colour/grey classification uses each harmonic's own
  sigma_n (per-harmonic SNR > 2), so a small but repeatable harmonic on the
  decay curve is correctly detected even if it sits near the line.
