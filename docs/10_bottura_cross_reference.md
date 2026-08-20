# 10. Bottura MTA-IN-97-007 → code: cross-reference and documented deviations

Canonical equation → function map for this repository (supersedes
`DOCUMENTATION.md` §3, §12–13, which now point here). Equation numbers follow
the ⭐⭐⭐ note
`bibliography-review/magnetic_measurements/bottura1997_standard_analysis_field_quality_LHC_harmonics_notes.md`
(bib key `bottura1997`); its §11 carries the same map from the theory side and
its §12 the flags about the source text. Verdicts and parity numbers for other
repos: hub `bibliography-review/coordinator/hubs/rotating-coil-analyzer.md` §2–3.

## 10.1 Equation → implementation

| Bottura | Content | Code | Status |
|---|---|---|---|
| Eq. 1–3 | $B(z) = \sum C_n (z/R_{ref})^{n-1}$, $C_n = B_n + iA_n$ | conceptual; `ba_table_from_C`, `build_harmonic_rows` ($B_n = \mathrm{Re}$, $A_n = \mathrm{Im}$) | as written |
| Eq. 4, AIV.8–9 | $c_n = 10^4 C_n/B_m$ | `safe_normalize_to_units`; in-pipeline `nor` | exact |
| Eq. 5, AIII.6 | translation / feed-down | `kn_pipeline` `fed` step (`math.comb`) | exact, truncated at $H$ |
| Eq. 6, AIV.6 | rotation $C_n e^{in\theta}$, **all orders** | `kn_pipeline` `rot` step | exact by default; see deviation 3 |
| Eq. 12–13, 19–21 | $\chi_n$, $\kappa_n$ radial/tangential | `kn_head.py` (`_csi_n` finite-winding form, orientation factor) | as written + Deniau finite-winding correction |
| Eq. 22 | bucked set $\kappa_n = \sum g_s\kappa_n^s$ | `kn_head.parse_connection`, `compute_segment_kn_from_head` | exact |
| Eq. 15, AII.19 | DFT of $\psi_k$ | `np.fft.fft` in `compute_legacy_kn_per_turn`; `fourier.dft_per_turn` | exact, no window |
| Eq. 17–18, AI.20, AII.20–22 | $C_n = \tfrac{2}{N}R_{ref}^{n-1}\Psi_{n+1}/\kappa_n$ (fold) | `f = 2*FFT/Ns`, `C = f*Rref**(n-1)/conj(kn)` | exact (fold ≡ $2/N$ for real $\psi$); conjugate = legacy file convention |
| AII.1–2 | uniform encoder angles | implicit sample-index grid, `turns.split_into_turns` | as written |
| AII.3–7 | pulse time, mid-time, speed | measured `t` kept; `duration_s`, `dI_dt_A_per_s` diagnostics | time used for diagnostics only |
| AII.8–11 | counts → Vs, voltage | done upstream (FDI); readers take increments as is | not in package |
| AII.12–14 | offset / drift, $\Delta t$-weighted | `preprocess.integrate_to_flux`: `weighted` = AII.12–14 verbatim; `legacy` = uniform-$\Delta t$ C++ form | exact (two modes) |
| AII.15–16 | forward/backward average, $\varepsilon_k$ | — | **not implemented** (deviation 1) |
| AII.17–18 | running-sum flux | `np.cumsum` | exact |
| AII.23 | bucking ratio $\beta_n$ | `merge.recommend_merge_choice` MAD-noise comparison | functional stand-in, ratio not computed |
| AIII.1–2 | dipole centring, 16-pole polynomial + cost | — | not implemented (deviation 2) |
| AIII.3 | dipole centring, 20-pole | `cel`, $m=1$: $z_R = -C^{cmp}_{10}/(10\,C^{cmp}_{11})$ | **linearised** (deviation 2) |
| AIII.4–5 | $2m$-pole centring | `cel`, $m \ge 2$: $z_R = -C^{abs}_{m-1}/((m-1)C^{abs}_m)$ | exact |
| AIV.1–5 | $|C_m|$, $\varphi_m$, wrap to $\pm\pi/2$, $\alpha_m = \varphi_m/m$ | `_wrap_arg_to_pm_pi_over_2`, `phi_out = wrap(arg C_m)/m` | exact |
| AIV.7 | gradient $g_m = B_m/R_{ref}^{m-1}$ | not computed (trivial from output) | — |
| §3.3 order | dit? → drift → integrate → DFT → fold → $\kappa$ | same order; `dit` is an FFMM addition, not in Bottura | as written + `dit` |
| §3.4 | AC procedure (no drift, fwd/bwd average) | — | not implemented; analyst decides on `dri` for ramps |
| §3.7 | abs up to $m$, cmp above; store centre + angle | `merge_coefficients("abs_upto_m_cmp_above")`, `mixed_format_table`, `x_mm/y_mm/phi_rad` | exact |
| §3.8 | quality checks (speed ripple, offset, fwd/bwd error, bucking) | `dt` nominal check in `StreamingReader`; `DriftResult.offset_per_s`; merge diagnostics | partial |

## 10.2 The three documented deviations

### Deviation 1 — no forward/backward rotation averaging (AII.15–16)

Bottura pairs a forward and a backward rotation and averages
$\Delta\psi_k = (\Delta\psi^+_k - \Delta\psi^-_k)/2$ to cancel linear
systematic errors to first order — and makes this the *only* drift handling
in the AC (ramp) procedure. The package processes every rotation
independently; all drift rejection rests on the per-turn `dri` step
(AII.12–14), which is strictly valid only at constant field. Consequences:
on ramps the analyst must judge whether `dri` is acceptable; the
forward/backward error indicator $\varepsilon_k$ is not available. Open flag
in the hub §4: bidirectional DC bench campaigns (`mbb-data-analysis`,
`mc62-data-analysis`) could quantify what the averaging would buy.

### Deviation 2 — dipole centring uses the linearised 20-pole (AIII.3)

The main text prescribes the 16-pole (AIII.1) solved as a 7th-degree
polynomial with the root chosen by the cost function AIII.2 (20-pole AIII.3
given as the alternative). The code uses the 20-pole, **linearised** to
first order, from the compensated channel:
$z_R = -C^{cmp}_{10}/(10\,C^{cmp}_{11})$ — the legacy C++/FFMM choice,
parity-proven. It is Bottura-compliant in the sense the note allows (either
non-allowed order, compensated harmonics preferred) but it is fragile: at
low field or poor SNR the two weak high-order signals give unphysical
$|z_R|$ and `fed` then corrupts all lower orders. Mitigations: `max_zR`
clamp, `diagnose_cel_fed()` (SAFE/UNSAFE/MIXED), and Bottura's own rule
to centre only at moderate-to-high field in steady state. An automatic
fallback (disable `cel`/`fed` when UNSAFE) is proposed, not implemented.

### Deviation 3 — `legacy_rotate_excludes_last` is SM18-parity only

`legacy_rotate_excludes_last=True` stops the rotation loop at $n = H-1$,
reproducing an off-by-one in the historical SM18 C++ export. AIV.6 rotates
all orders; so do the current FFMM C++ path and Pentella. The flag has no
theoretical basis and must be left at its default `False` for physics; it is
set to `True` only in the SM18 golden-standard comparison
(`PARITY_REPORT.md` §1.2; observable as an $\sim 10^{-7}$ difference on
order 15 in the Buckley degauss parity).

## 10.3 Smaller differences and gaps (not deviations from Bottura)

| Item | Note |
|---|---|
| `dit` current-ramp reweighting | FFMM legacy step, absent from Bottura; `dit_signed` selects C++ parity thresholds |
| Conjugate in $1/\overline{k_n}$ | phase convention of the legacy $k_n$ files; identical for real (radial) $k_n$; parity-proven |
| Legacy `dri` subtracts `mean(cumsum(df))` | affects only the discarded DC bin; kept for bit-for-bit C++ parity |
| Weighted `dri` mode | implemented and unit-tested, not yet validated end-to-end on speed-ripple data |
| External (ext) channel | read into `SegmentKn`/`KnBundle`, not processed by the pipeline |
| Multi-segment $B\cdot L$-weighted merge, pulsed/time-resolved mode, impedance-gain correction | Pentella/FFMM features not in this package (`DOCUMENTATION.md` §13.2) |
| Noise → harmonic precision (`bottura1997b`) | not used as a formal uncertainty model; per-turn scatter (MAD) is the empirical stand-in |

## 10.4 What is verified

Every equation in §10.1 marked "exact" is covered by the golden-standard
comparisons in `PARITY_REPORT.md` (SM18 streaming to $\sim10^{-12}$ T, LIU
BTP8 and Buckley degauss at the float64 floor) and by the synthetic tests
(chapter 8). No windowing, smoothing, resampling, interpolation, synthetic
time or synthetic current exists anywhere in the pipeline.
