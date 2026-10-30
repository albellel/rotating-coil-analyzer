# 12. The analysis implementations used so far, how they differ, and what this package fixes

Rotating-coil data at CERN have been analysed with several implementations of
the same theory. This chapter says what each one does, where they differ, and
which **errors** they produce, and how this package handles each error. Every
statement about another implementation was read from its source code (version
noted) or measured on a golden dataset. The theory itself is not re-derived;
see chapter 10 and the `bottura1997` note.

## 12.1 The implementations

| Name used here | What it is | Version examined | Role |
|---|---|---|---|
| **Bottura** | L. Bottura, *Standard Analysis Procedures for Field Quality Measurement of the LHC Magnets — Part I: Harmonics*, MTA-IN-97-007 (1997) | equation numbering of the `bottura1997` note | the theoretical reference |
| **FFMM native** | CERN FFMM acquisition software, C++ function `RotatingCoilAnalyzerFunc` in `src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp` | FFMM `master` 10ea291f (2025-09-03) | **production path**: it writes the `*_results.txt` files of every FFMM measurement (`mlAnalysis = false` by default) |
| **FFMM Matlab path** | the same FFMM, function `RotatingCoilAnalyzerFuncML`, i.e. Matlab Coder output of `analysis/RotatingCoilAnalyzer/RotatingCoilAnalysisTurn.m` | same commit | selected only in a unit test (`mlAnalysis = true`); not used for production results |
| **SM18 legacy export** | the results files of the SM18 golden standard (2024-12) | — | behaves exactly like FFMM native (parity, `PARITY_REPORT.md` §1) |
| **MMM Python analyzer** | an independent Python analyzer from the MMM section, `rotcoil_lib.py` (DC and pulsed modes); local reference copy `golden_standards/mmm_python_analyzer/`, not distributed | local copy as of 2026-10 | second, independent implementation |
| **This package** | `rotating_coil_analyzer` | this repository | the subject of this documentation |

## 12.2 Stage-by-stage comparison

Notation: $f_n = 2\,\mathrm{FFT}(\psi)_n / N_s$; $C_n = f_n R_{ref}^{\,n-1}/\overline{k_n}$; $m$ main order; $H$ number of orders (15 in FFMM).

| Stage | Bottura | FFMM native | FFMM Matlab path | MMM Python analyzer | This package |
|---|---|---|---|---|---|
| `dit` current-ramp reweighting $w_k = \bar I / I_k$ | not in the note | only if $dI/dt > 0.1$ A/s **and** $\bar I > 10$ A (l. 117–124); skipped silently on NaN/inf weights | — | same signed thresholds (l. 384) | both ramp directions and current signs by default (`dit_signed=False`); `dit_signed=True` reproduces FFMM; `dit_applied` flag per turn |
| Drift + integration | AII.12–14 ($\Delta t$-weighted) | `cumsum(df − mean df) − mean(cumsum df)` (l. 127) | — | $\Delta t$-weighted, $\Delta t_0 = 0$ (l. 397, l. 294–315) | `legacy` = FFMM bit for bit; `weighted` = AII.12–14 with the correct first-increment duration |
| FFT, fold, $k_n$ | Eq. 15–18, AII.20–22 | identical | identical | identical | identical |
| Rotation reference | AIV.1–5: $\arg C_m$ wrapped to $\pm\pi/2$, divided by $m$ | same, always $\arg C_m$ (l. 158–165) | same | normal: same; **skew: rotates $C_m$ onto $\pm i$** (l. 741–760) | normal: same; skew (`skew_main`): $\arg(-i\,C_m)$, i.e. onto $\pm i$ (since 2026-10-30); turns within 0.05 rad of the wrap flagged |
| Rotation range | AIV.6: all orders | **$k = 1 \dots H-1$** (`for (k = 1; k < nrHarmonics; ++k)`, l. 170) | all $k = 1 \dots H$ | all orders | all orders by default; `legacy_rotate_excludes_last=True` reproduces FFMM native |
| Centre, $m \ge 2$ | AIII.4–5 | $z_R = -C_{m-1}/((m-1)C_m)$, absolute channel | same | same | same, with guards (below) |
| Centre, dipole | AIII.1–3 (16-pole polynomial or 20-pole) | linearised 20-pole on the **compensated** $C_{10}, C_{11}$ (l. 184–187) | **no centring for dipoles** | linearised 20-pole on the **absolute** $C_{10}, C_{11}$ (l. 444–472, 804) | as FFMM native (compensated), with guards and `diagnose_cel_fed()` |
| Feed-down | AIII.6 | binomial sum, applied for every $m$ including dipoles (l. 199–210) | **skipped for dipoles** | binomial sum | binomial sum; $|z_R| \ge 1$ reported; `max_zR` clamp |
| Normalisation | AIV.8–9 | $10^4/\mathrm{Re}\,C_m$, or $10^4/\mathrm{Im}\,C_m$ with `skw` (l. 212–227); no guard | $10^4/B_m$ for $n > m$ only (mixed record) | $10^4/B_{main}$; NaN only if $B_{main} = 0$ exactly (l. 488) | post-merge `safe_normalize_to_units`: NaN below `min_main_field`, `ok_main` flag; mixed Bottura §3.7 record |
| Compensated orders $1 \dots m$ in the output | — | kept | **overwritten by the bucking ratio** $|f^{abs}_k/f^{cmp}_k|$ | kept | kept; bucking diagnostics separately (`recommend_merge_choice`, `bucked_cmp_orders`) |
| Reported roll angle | $\alpha_m$ | $\alpha_m$ | $\alpha_m$ | $\alpha_m - 2\pi/N_s$ (l. 788) | $\alpha_m$ (minus `encoder_offset_rad` if given) |
| `absCalib` | — | multiplies the main field used for normalisation only | — | — | same as FFMM native (kept for parity) |
| External (third) coil | — | — | — | supported | read, not processed |
| Impedance gain $(Z_{coil}+Z_{inst})/Z_{inst}$ | — | — | — | applied | not applied (absorbed in $k_n$) |

## 12.3 Errors in the earlier implementations, and what this package does

Each entry gives the defect, where it is, its consequence, and the
treatment here. "Fixed here" means the package gives the correct result by
default. "Parity" means the old behaviour can still be reproduced on request,
so old results files can be checked bit for bit.

### E1 — FFMM native does not rotate the last harmonic
- **Where:** `for (size_t k = 1; k < nrHarmonics; ++k)` rotates orders 1–14 of 15.
- **Consequence:** order $H$ stays in the coil frame while all other orders
  are in the magnet frame. Its normal/skew split is wrong by the roll angle
  $H\alpha_m$. Measured: ~$1.8\times10^{-7}$ on order 15 of the Buckley degauss
  data. On the FCC-ee quadrupole, $b_{15}$ differs by $3\times10^{-6}$ units;
  with `legacy_rotate_excludes_last=True` the difference becomes $6\times10^{-18}$.
- **Here:** fixed (all orders rotated, AIV.6); parity flag
  `legacy_rotate_excludes_last=True`. The old documentation called this an
  "SM18-only" quirk; it is the current production behaviour.

### E2 — FFMM native gives meaningless units for skew magnets
- **Where:** the rotation reference is always $\arg C_m$ (l. 158), which turns
  $C_m$ **real**. With `skw`, normalisation then divides by
  $\mathrm{Im}\,C_m \approx 10^{-17}|C_m|$ (l. 217–219).
- **Consequence:** $b_n, a_n$ of a skew magnet analysed with `rot nor skw`
  come out ~$10^{16}$ units. The MMM Python analyzer does this correctly. The
  FFMM Matlab path has no skew option.
- **Here:** this package had the same defect until 2026-10-30. Fixed
  (BF-2026-10-30-01) with the MMM analyzer's convention: $C_m$ rotated onto the
  imaginary axis, units $= 10^4 C_n/A_m$.

### E3 — ramp correction only on rising ramps at positive current
- **Where:** FFMM native `crr > 0.1 && cm > 10` (l. 118); MMM analyzer `curr_AVG > 10.0 and RR > 0.1` (l. 384).
- **Consequence:** on falling ramps and at negative current, `dit` is
  silently not applied. The two branches of a hysteresis loop measured on
  ramps are then treated differently. FFMM also skips silently when a weight is NaN/inf.
- **Here:** fixed by default (`dit_signed=False`: $|dI/dt|$, $|\bar I|$); parity
  `dit_signed=True`. `dit_applied` shows per turn where it acted. (An earlier
  defect of this package, a negative weight for negative current, was fixed
  before 2026-10, see `correction_options_reference.md`.)

### E4 — centre location without guards
- **Where:** FFMM native and the MMM analyzer compute
  $z_R = -C_{n-1}/(k\,C_n)$ without checking for a zero, NaN or huge result.
  The MMM analyzer also needs $H \ge 11$ for dipoles (index error otherwise).
- **Consequence:** a vanishing denominator gives $z_R = \infty$/NaN and the
  feed-down propagates it to every order. A finite but unphysical $|z_R|$
  (weak $C_{11}$ at low field) silently corrupts every lower order. In
  FFMM native this applies to dipoles too, because feed-down is not skipped
  for $m = 1$.
- **Here:** denominators below `eps_main` give $z_R = 0$ and non-finite $z_R$ is
  zeroed. `max_zR` clamps and flags (`zR_clamped`). `diagnose_cel_fed()`
  returns SAFE/UNSAFE/MIXED. Since 2026-10-30, $|z_R| \ge 1$ (divergent series),
  a dipole `cel` with $H < 11$ and the per-turn `cel_applied` flags are reported.

### E5 — the dipole centre is computed three different ways
- **Where:** FFMM native uses the compensated $C_{10}, C_{11}$; the MMM analyzer
  uses the absolute ones; the FFMM Matlab path does not centre dipoles at all.
- **Consequence:** the same dipole measurement gives different centres (and,
  through feed-down, different harmonics) depending on the tool. None of them
  implements Bottura's preferred 16-pole polynomial (AIII.1–2).
- **Here:** FFMM native convention, documented as deviation 2 in chapter 10,
  with the guards of E4. Bottura recommends centring dipoles only at
  moderate-to-high field in steady state.

### E6 — normalisation by a vanishing main field
- **Where:** FFMM native divides by $\mathrm{Re}\,C_m$ (or $\mathrm{Im}$) with no
  threshold. The MMM analyzer returns NaN only when $B_{main}$ is exactly 0.
- **Consequence:** at zero current (remanent field, degauss steps) the units
  become arbitrarily large instead of "undefined".
- **Here:** `safe_normalize_to_units` returns NaN below `min_main_field`
  (default $10^{-4}$ T in `process_kn_pipeline`) and flags `ok_main`.

### E7 — $\Delta t$-weighted drift ignores the first sample
- **Where:** MMM analyzer `dT = np.diff(dtime, prepend=dtime[0])` gives
  $\Delta t_0 = 0$ (l. 397). The weighted mode of this package was transcribed
  from it and had the same defect.
- **Consequence:** sample 0 gets no drift correction and the offset is
  computed over $(N_s - 1)\Delta t$. Even with perfectly uniform sampling the
  result then differs from the correct one: $4\times10^{-5}$ relative on $f_3$
  (synthetic); on the SM18 data the integrated flux changes by up to
  $1.1\times10^{-6}$ for a flux range of $1.3\times10^{-3}$.
- **Here:** fixed (BF-2026-10-30-02). With uniform sampling the weighted mode
  now equals the legacy mode up to the discarded DC bin. FFMM has no weighted mode.

### E8 — bucking ratio written into the compensated harmonics
- **Where:** FFMM Matlab path, `C_cmp_rfn(k) = complex(BuckingRatio(k), BuckingRatio(k))` for $k \le m$.
- **Consequence:** in that path's output, the compensated "harmonics" $1 \dots m$
  are a quality ratio, not field. It does not affect the production (native) results.
- **Here:** not done; compensated harmonics are kept and the bucking quality is
  reported separately. Orders merged from a bucked channel are warned about
  (`bucked_cmp_orders`).

### E9 — reported roll angle convention (difference, not an error)
- The MMM analyzer subtracts $2\pi/N_s$ (6.1 mrad at $N_s = 1024$) from the
  reported angle. FFMM and this package do not.
- Which is right depends on whether an integrator increment $k$ ends at
  $\theta_k$ or at $\theta_{k+1}$. With increments ending at $\theta_k$ the
  cumulative sum needs no shift (verified by simulation here).
- Harmonic amplitudes are unaffected. Check the hardware convention before
  comparing absolute roll angles between the tools.

### E10 — harmless legacy details kept for parity
- FFMM's `− mean(cumsum(df))` term changes only the DC bin, which is discarded.
- `absCalib` scales only the main field used for normalisation.
- Both are reproduced exactly. In FFMM, `absolute_calibration` is hard-coded to 1.0
  in both callers (`src/core/CAS_RC.cpp:91`, `src/core/coilLevelingLDM.cpp:126`)
  and never changed, so every FFMM results file has `absCalib = 1`. No Parameters
  file examined carries the key. Leave it at 1.0.
- If it were ever different from 1, this convention would be inconsistent:
  the units $10^4 C_n/(\mathrm{absCalib}\,B_m)$ would scale by $1/\mathrm{absCalib}$,
  while a true gain correction of the coil leaves the ratio $C_n/C_m$ unchanged.
  The correct treatment then is to fold the factor into $k_n$ (scale both channels)
  instead.

## 12.4 Defects that were only in this package (fixed 2026-10-30)

The full list is in [`11_bugfix_diary.md`](11_bugfix_diary.md). The FFMM tools
take $R_{ref}$ and $m$ from their own configuration. The defects of this package's
**defaults** were its own: the silent 17 mm reference radius and the silent
$m = 1$ / $m = 2$ fallbacks (BF-05, BF-06), the GUI state defects (BF-11, BF-12),
and the reader memory use (BF-09).

## 12.5 Reproducing an FFMM results file exactly

To check an existing FFMM `*_results.txt` bit for bit, use FFMM native's
conventions:

```python
process_kn_pipeline(..., options=<Parameters magnetAnalyzer flags>, drift_mode="legacy",
                    dit_signed=True, legacy_rotate_excludes_last=True)
# with "nor" in options, compare the in-pipeline units of result.C_abs, or
# compare post-merge units (identical ratios; B_main kept in tesla)
```

For physics, use the defaults (rotate all orders, unsigned `dit`, post-merge
normalisation). They differ from FFMM only where FFMM is wrong (E1, E3) or
undefined (E4, E6). For a skew magnet, never compare with an FFMM run made with
`rot nor skw` (E2).

**Measured agreement on the bundled FCC-ee quadrupole:** $b_3 \dots b_6$ to
$\le 7\times10^{-11}$ units; $b_{15}$ to $6\times10^{-18}$ units with
`legacy_rotate_excludes_last=True`. Orders 7–14 differ by up to
$\sim10^{-5}$ units, growing ~×10 per order. That is below $10^{-9}$ of the main
field, and these values themselves are at the noise floor. The cause is
**not established**: it is not the rotation range, and not the precision of
the ASCII input, which carries 15 significant digits.
