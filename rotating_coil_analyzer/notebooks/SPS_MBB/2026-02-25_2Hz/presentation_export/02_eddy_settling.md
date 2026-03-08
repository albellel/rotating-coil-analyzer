# Eddy-Current Settling

## Physics

When the current ramp ends and the injection plateau begins, the changing flux
has induced **eddy currents** in the beam pipe and yoke laminations. These decay
exponentially:

B(t) = B_∞ + Σ A_i · exp(-t/τ_i)

The time constants τ_i depend on geometry and resistivity of the conducting
structures (beam pipe ~1 s, yoke laminations ~2–4 s).

**Why it matters:** at injection (26 GeV MD1) the beam circulates for several
seconds. If b₃ is still drifting, the chromaticity changes and can cause beam
losses.

**Body vs fringe:** both see similar eddy amplitudes *in Tesla*, but the fringe
B₁ is 6× weaker, so the normalised b_n = C_n/B₁ × 10⁴ is amplified
~6× in the fringe. The fringe drives the settling requirement.

## Outlier removal

The first few turns can contain **ramp artifacts** (DAQ trigger overlapping
the current ramp tail). These are removed with a two-pass 3.5x MAD residual
clip (typically 5-6 of ~50 turns). This is critical: without it, body B1 R2
drops from 0.99 to 0.60 and tau is distorted (8 s instead of 3.8 s).

## Uncertainty band

We average N ~ 35 identical supercycles. The shaded band is
+/- 1 SEM = +/- sigma/sqrt(N): the uncertainty on the **mean** settling curve,
not the raw scatter.

## 200 GeV MD1

![Eddy settling 200 GeV MD1](eddy_settling_200GeV_MD1.png)

## 26 GeV MD1

![Eddy settling 26 GeV MD1](eddy_settling_26GeV_MD1.png)
