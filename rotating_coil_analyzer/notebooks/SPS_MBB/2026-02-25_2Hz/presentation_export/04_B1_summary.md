# Dipole Field B₁ Summary

Before looking at normalised harmonics, it is important to know the
**absolute dipole field** B₁ in each segment. Since b_n = C_n / B₁ × 10⁴,
a small B₁ (fringe) amplifies both signal and noise in units.

## Units

- **Body and Fringe**: B₁ in Tesla (T) -- the field averaged over the
  coil's 470 mm sensitive length.
- **Integrated**: the true field integral ∫B₁ dl in Tesla-metres (Tm),
  i.e. B_body × L_body + 2 × B_fringe × L_fringe. This is what the beam
  sees (total bending strength).

## Summary table

| Energy | Plateau | B1 Body (T) | B1 Fringe (T) | B1 Integrated (Tm) | Ratio Body/Fringe |
|---|---|---|---|---|---|
| 200 GeV MD1 | Injection (300 A) | -0.116255 | -0.019622 | 0.702797 | 5.9 |
| 200 GeV MD1 | Flat-top (4800 A) | -1.781715 | -0.291179 | 10.762071 | 6.1 |
| 26 GeV MD1 | Injection (300 A) | -0.116255 | -0.019625 | 0.702804 | 5.9 |
| 26 GeV MD1 | Flat-top (4800 A) | -1.781749 | -0.291240 | 10.762328 | 6.1 |

## All segments combined

![B1 summary](B1_summary.png)

## Body only

![B1 body](B1_body.png)

## Fringe only

![B1 fringe](B1_fringe.png)

## Integrated field (Tm)

![B1 integrated](B1_integrated.png)

## Body vs Fringe comparison

![B1 body vs fringe](B1_body_vs_fringe.png)

## Key observations

- The body/fringe ratio is ~6:1 at both energies and both plateaus,
  confirming the coil straddles the magnet end (one segment inside the yoke,
  one outside).
- The integrated field at flat-top is ~10.76 Tm, corresponding to an
  effective magnetic length L_mag = B_int / B_body ~ 6.04 m (vs yoke
  length 6.26 m -- the 0.22 m difference is the fringe fall-off region).
- B₁ values are consistent across the two energy cycles (same magnet,
  same current plateaus), as expected.
