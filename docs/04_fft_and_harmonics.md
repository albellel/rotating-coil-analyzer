# 4. Fourier analysis and harmonic extraction

Code: `analysis/kn_pipeline.compute_legacy_kn_per_turn` (FFT + scaling step),
`analysis/fourier.dft_per_turn` (uncalibrated per-turn coefficients used by
the GUI Harmonics tab and `summarize_harmonics`). Bottura counterpart:
Eq. 15–18, Appendix I, AII.19–22.

## 4.1 Angular grid

Each turn provides $\psi_k$, $k = 0,\ldots,N_s-1$, on the implicit uniform
grid $\theta_k = 2\pi k/N_s$ (chapter 2). No re-parameterisation by time or
by encoder angle happens: the old LaTeX's
"$\theta(t) = \theta_0 + \omega t$" is **not** used anywhere. Uniform rotation
within a turn is an assumption of the acquisition (encoder-triggered
sampling), not something the code reconstructs.

## 4.2 DFT and scaling

`numpy.fft.fft` along the sample axis, no window, no zero padding:

$$
\Psi_{n+1} = \sum_{k=0}^{N_s-1}\psi_k\,e^{-2\pi i\,n k/N_s}
\qquad (\text{Bottura Eq. 15 / AII.19, 1-based }\Psi),
$$

$$
f_n = \frac{2}{N_s}\,\Psi_{n+1}, \qquad n = 1,\ldots,H,
$$

i.e. `f = 2*fft(flux, axis=1)/Ns`, then `f[:, 1:H+1]` — the DC bin is
dropped and orders $1\ldots H$ are kept, $H$ being the number of rows of the
$k_n$ file (typically 15, Bottura's $N_m$). The pipeline requires
$N_s \ge H + 1$; in practice $N_s = 512$ or $1024 \gg 2H$, so the Nyquist bin
never enters.

**Equivalence with Bottura's fold (AII.20–21).** For a real $\psi$,
$\Psi_{N-n+1} = \Psi^*_{n+1}$, hence

$$
\Xi_n = \frac{\Psi_{n+1} + \Psi^*_{N-n+1}}{N} = \frac{2\,\Psi_{n+1}}{N} = f_n .
$$

The `2/N_s` scaling **is** the folded, normalised spectrum; $f_n$ is the
physical flux amplitude of order $n$ in Vs (note §8 d).

`fourier.dft_per_turn` (GUI "uncalibrated" view) uses the plain `FFT/Ns`
normalisation and keeps order 0 — a factor 2 smaller than $f_n$ and
diagnostic only; the calibrated pipeline always uses $2/N_s$.

## 4.3 From $f_n$ to field harmonics

With the complex sensitivity $k_n$ (chapter 5), Bottura AII.22
$C_n = R_{ref}^{\,n-1}\,\Xi_n/\kappa_n$ becomes in the code

$$
C_n = \frac{R_{ref}^{\,n-1}}{\overline{k_n}}\; f_n \qquad [\mathrm{T}],
$$

`C = f * (1/conj(kn)) * Rref**(n-1)` (`sens_abs`, `sens_cmp`). The complex
**conjugate** follows the FFMM / Pentella convention for how the segment
$k_n$ files store the sensitivity phase; for a purely radial coil $k_n$ is
real and the two agree trivially. This convention is parity-proven to machine
precision (`PARITY_REPORT.md` §5.3).

$R_{ref}$ enters here and **only** here in the pipeline (the centre-location
step works with the dimensionless $z_R = \Delta z/R_{ref}$, chapter 6), so a
single reference radius is guaranteed throughout.

## 4.4 Normal and skew components

$$
B_n = \mathrm{Re}(C_n), \qquad A_n = \mathrm{Im}(C_n).
$$

No sign flip is applied to the imaginary part (see chapter 1.1 for the
correction of the old LaTeX). The phase $\varphi_n = \arg C_n$ is preserved
(the GUI plots amplitude and phase per order); only the main-order phase is
used downstream, for the rotation step.

## 4.5 Main order and units

The user supplies the main order $m$ (`magnet_order`: 1 dipole, 2 quadrupole,
3 sextupole, …; `AnalysisProfile` reads the default from `Parameters.txt`).
Normalised coefficients (Bottura Eq. 4, AIV.8–9),

$$
b_n = 10^4\,\frac{B_n}{B_m}, \qquad a_n = 10^4\,\frac{A_n}{B_m},
$$

are produced **after** rotation and merge by `safe_normalize_to_units`
(chapter 7), or in-pipeline by the `nor` option. By construction, after
rotation $b_m = 10^4$ and $a_m = 0$. `skew_main=True` normalises by $A_m$
instead (legacy `skw` option for skew magnets).

## 4.6 Per-turn results, aggregation later

`compute_legacy_kn_per_turn` returns `LegacyKnPerTurn` with `(n_turns, H)`
complex arrays `C_abs`, `C_cmp` (after all enabled corrections) and the
snapshots `C_abs_db`, `C_cmp_db` taken right after calibration (before
rotation/cel/fed/nor) — the "DB" record of the legacy analyzer. Averaging
(per plateau, per run, last-$N$ turns) is an explicit downstream step:
`plateau_summary`, `summarize_harmonics`, `build_run_averages`.

## 4.7 What is deliberately absent

No windowing, smoothing, resampling, interpolation, zero padding or
time-domain filtering (Bottura AII "6.8 Filtering" is left undefined in the
note too). Bottura's AC time-periodic extension (Marusov 2013,
`marusov2013`) is not part of the per-turn pipeline; it was studied as an R&D
post-processing of the per-turn $C_n(j)$ in `mc62-data-analysis`.
