# 1. Measurement principle

Theory source: L. Bottura, *Standard Analysis Procedures for Field Quality
Measurement of the LHC Magnets — Part I: Harmonics*, MTA-IN-97-007 (rev. 2000),
bib key `bottura1997`. Equation numbers below follow the ⭐⭐⭐ note
`bibliography-review/magnetic_measurements/bottura1997_standard_analysis_field_quality_LHC_harmonics_notes.md`
(§2 for Eqs. 1–22, §4 for Appendix I). Nothing is re-derived here.

## 1.1 Complex multipole expansion (Eqs. 1–4)

The 2-D transverse field is written as a power series in $z = x + iy$ at the
reference radius $R_{ref}$:

$$
B(z) = B_y + i\,B_x = \sum_{n=1}^{\infty} C_n \left(\frac{z}{R_{ref}}\right)^{n-1},
\qquad C_n = B_n + i\,A_n \ [\mathrm{T}].
$$

$B_n$ is the **normal** and $A_n$ the **skew** component of order $n$
($n=1$ dipole, $n=2$ quadrupole, …). For a normal magnet of main order $m$
the relative coefficients are

$$
c_n = 10^4\,\frac{C_n}{B_m} = b_n + i\,a_n \quad [\text{units}],
$$

where $B_m$ is taken in the frame in which the main skew component vanishes
(Appendix IV). One unit $= 10^{-4}$ of the main field.

> **Convention caveat (checked against the code).** The old LaTeX wrote
> $B_n \propto \mathrm{Re}(C_n)$, $A_n \propto -\mathrm{Im}(C_n)$. The code does
> **not** negate the imaginary part: after calibration `C_abs`/`C_cmp` are
> complex arrays with $B_n = \mathrm{Re}(C_n)$ and $A_n = \mathrm{Im}(C_n)$
> (`utility_functions.ba_table_from_C`, `build_harmonic_rows`). The sign
> conventions live entirely in the complex sensitivity $k_n$ and in the
> rotation step, exactly as in Bottura's worked example (note §8 d).

## 1.2 Frame transformations (Eqs. 5–6)

Translation by $\Delta z$ (feed-down law, used by the `fed` step):

$$
C'_n = \sum_{k=n}^{\infty} \frac{(k-1)!}{(n-1)!\,(k-n)!}\; C_k
\left(\frac{\Delta z}{R_{ref}}\right)^{k-n}.
$$

Rotation by $\theta$ (used by the `rot` step):

$$
C'_n = C_n\, e^{i n \theta}.
$$

## 1.3 Flux and voltage of a rotating coil (Eqs. 7–14)

Modelling the coil as two filaments of length $L$ rotating rigidly, the linked
flux at angle $\theta$ is (Eq. 12)

$$
\psi(\theta) = L\,\mathrm{Re}\!\left[\sum_{n=1}^{\infty}
\frac{\chi_n}{n R_{ref}^{\,n-1}}\; C_n\, e^{i n \theta}\right],
$$

with the complex geometric factors $\chi_n$ (Eq. 13) fixed by the filament
radii, opening angle and initial phase. The pickup voltage (Eq. 14),

$$
V = -\frac{\partial \psi}{\partial t}
= -L\,\mathrm{Re}\!\left[\sum_n \frac{\chi_n e^{in\theta}}{n R_{ref}^{\,n-1}}
\left(\frac{\partial C_n}{\partial t} + i\,n\,C_n\,\frac{\partial\theta}{\partial t}\right)\right],
$$

contains a **field-change term** and a **coil-rotation term**. The DC (plateau)
procedure assumes the first is zero over a turn; on ramps it is not, which is
why drift correction is a per-turn operation that must not be confused with
the physical flux change (see chapter 3).

## 1.4 What the hardware delivers

The acquisition chain at CERN (FDI / VME integrators) does **not** deliver
$V(t)$: it integrates the coil voltage between consecutive encoder triggers
and delivers **flux increments** $\Delta\psi_k$ per angular step (Bottura
§3.2, AII.8–11). In this package those increments are the `df_abs` / `df_cmp`
columns of a `SegmentFrame`. Consequently:

- the flux is reconstructed by a **running sum** (AII.17–18), `np.cumsum`,
  not by trapezoidal integration — this is the documented
  "already-integrated discrete increments" exception to the global
  integration rule (see chapter 3);
- the measured time stamps `t` are used only for diagnostics, the `dit`
  correction and the weighted drift mode; the angular grid is **implicit in
  the sample index** (chapter 2).

## 1.5 From DFT to harmonics (Eqs. 15–19, AII.20–22)

Sampling $\psi$ at $N$ equally spaced angles and taking the DFT $\Psi_n$
(Eq. 15), Bottura shows (Appendix I, Eq. 17–18) that for an ideal coil of
$N_t$ turns

$$
C_n \approx \frac{2}{N}\,\frac{R_{ref}^{\,n-1}}{\kappa_n}\,\Psi_{n+1},
\qquad \kappa_n = \frac{N_t L \chi_n}{n},
$$

$\kappa_n$ being the **complex coil sensitivity** (real for a perfect radial
coil, imaginary for a perfect tangential one; a gain-weighted sum
$\sum_s g_s \kappa_n^s$ for a bucked set, Eq. 22). Equivalently, with the
folded spectrum $\Xi_n$ (AII.20–21), $C_n \approx R_{ref}^{\,n-1}\,\Xi_n/\kappa_n$
(AII.22). Chapter 4 shows that the code's `2*FFT/Ns` is identical to the fold
for a real signal; chapter 5 covers the sensitivity $k_n$ the code uses.

## 1.6 Goal of the analysis

Given per-turn increments $\Delta\psi_k$ for an absolute and a compensated
channel, calibrated $k_n$, $R_{ref}$ and the main order $m$, produce

- $C_n = B_n + iA_n$ in tesla at $R_{ref}$ in the **main-field frame**
  (rotated so that $A_m = 0$, optionally centred on the magnetic axis),
- $c_n = b_n + i a_n$ in units for $n > m$,
- per turn, with no averaging across turns inside the pipeline (aggregation is
  an explicit later step: `plateau_summary`, `summarize_harmonics`).
