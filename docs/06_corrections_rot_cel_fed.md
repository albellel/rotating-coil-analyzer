# 6. Post-calibration corrections: rotation, centre location, feed-down

Code: `analysis/kn_pipeline.compute_legacy_kn_per_turn` (steps after the
$k_n$ application), `_wrap_arg_to_pm_pi_over_2`. Bottura counterpart:
Appendix IV (rotation), Appendix III (centring, feed-down). Option guide with
failure modes: `notebooks/correction_options_reference.md`.

Fixed order (legacy FFMM ordering, enforced regardless of which options are
on):

```
dit → dri(+integration) → FFT → kn → [flip] → DB snapshot → [encoder offset]
    → rot → cel → [max_zR clamp] → fed → nor
```

`options` is a set of tokens among `{"dit","dri","rot","cel","fed","nor"}`.
The project standard is `("dri","rot","cel","fed")` with normalisation done
post-merge (chapter 7); `AnalysisProfile` defaults to `("dri","rot")`.

## 6.1 Rotation into the main-field frame — `rot` (AIV.1–6)

The reference is the **absolute** main harmonic after calibration,
$C_m^{abs}$. Bottura defines the main-field phase by
$e^{i\varphi_m} = (B_m - iA_m)/|C_m|$ (AIV.2–3), i.e. $\varphi_m = -\arg C_m$,
limits it to $[-\pi/2, \pi/2]$ (AIV.4), sets $\alpha_m = \varphi_m/m$ (AIV.5)
and rotates **all** orders by $C'_n = C_n e^{in\alpha_m}$ (AIV.6). The code:

$$
\phi = \operatorname{wrap}_{\pm\pi/2}\big(\arg C_m^{abs}\big), \qquad
\phi_{out} = \frac{\phi}{m}, \qquad
C_n \leftarrow C_n\,e^{-i n \phi_{out}}, \quad n = 1,\ldots,H,
$$

applied identically to `C_abs` and `C_cmp`. Since
$-\arg C_m$ wrapped to a symmetric interval equals minus the wrapped
$\arg C_m$, this is AIV.2–6 verbatim. After rotation $A_m = 0$ and
$B_m = \pm|C_m|$ — the sign can be **negative** when the wrap chose the
representative closest to "normal" (Bottura says so after AIV.6); see §6.4
for the knobs that make $B_1 > 0$ the convention of a campaign.

`phi_out_rad` is stored per turn; `phi_bad` flags turns where
$|C_m| < $ `eps_main` (default $10^{-20}$) or non-finite, for which
$\phi_{out} = 0$ and no rotation is effectively applied.

**`legacy_rotate_excludes_last`** (default `False`): when `True` the loop
stops at $n = H-1$, reproducing an off-by-one in the historical SM18 C++
export. This has **no theoretical basis** and exists only to reach
bit-for-bit parity with that export (`PARITY_REPORT.md` §1.2). Everything
else — Bottura AIV.6, the current FFMM C++ path, Pentella — rotates all
orders; use the default.

## 6.2 Centre location — `cel` (AIII.1–5)

The coil axis is generally not the magnetic axis; higher orders feed down
onto lower ones (Eq. 5). `cel` estimates the dimensionless offset
$z_R = \Delta z/R_{ref}$ **after** rotation.

**$2m$-pole, $m \ge 2$** (AIII.4, linearised in the off-centring), from the
**absolute** channel:

$$
z_R = -\frac{1}{m-1}\,\frac{C^{abs}_{m-1}}{C^{abs}_m}, \qquad
\Delta z = R_{ref}\,z_R .
$$

Both harmonics are strong (main field and its own feed-down), so this is
robust — "SAFE" for BTP8 and every quadrupole dataset tested.

**Dipole, $m = 1$** — the main text prescribes zeroing the 16-pole (AIII.1)
or alternatively the 20-pole (AIII.3) by solving a polynomial in $\Delta z$ and
picking the root that minimises the cost AIII.2, using compensated
harmonics. The code uses the **20-pole, linearised**, from the
**compensated** channel:

$$
z_R = -\frac{C^{cmp}_{10}}{10\,C^{cmp}_{11}}
$$

(first-order truncation of AIII.3: $C'_{10} \approx C_{10} + 10\,C_{11} z_R = 0$;
the binomial factor for $k=11$, $n=10$ is $10!/(9!\,1!) = 10$). This is the
legacy C++/FFMM choice and is parity-proven, but it rests on two weak
high-order signals: at low current or poor SNR it yields unphysical
$|z_R| \gtrsim 10^{-2}$ and a feed-down that corrupts every lower order.
Requires $H \ge 11$; otherwise $z_R = 0$. **Documented deviation 2**
(chapter 10).

Safeguards:

- `max_zR` (default `None`): turns with $|z_R| >$ `max_zR` get $z_R = 0$ and
  are flagged in `zR_clamped` (applied after `cel`, before `fed`);
- `diagnose_cel_fed(...)` runs the pipeline with and without `cel`+`fed` and
  returns `SAFE` / `UNSAFE` / `MIXED` with the per-turn $|z_R|$ and the
  $B_m$ comparison (`CelFedDiagnostic`). If `UNSAFE`, rerun with
  `("dri","rot")`. Bottura's own rule: centre only at moderate-to-high field
  and in steady state.

`z_m`, `x_m`, `y_m` (metres) are returned per turn; they are the magnetic
centre **relative to the coil rotation axis**, as in Bottura.

## 6.3 Feed-down correction — `fed` (Eq. 5 / AIII.6)

With $z_R$ known, both channels are re-expressed in the centred frame by the
translation law truncated at $H$:

$$
C'_n = \sum_{k=n}^{H} \binom{k-1}{n-1}\, z_R^{\,k-n}\, C_k , \qquad n = 1,\ldots,H,
$$

`math.comb(k, n)` on 0-based indices is exactly $\frac{(k-1)!}{(n-1)!(k-n)!}$
on 1-based orders. Exact, arbitrary-precision binomials (the C++ uses `int`;
no practical difference for $H \le 15$). Without `cel`, $z_R = 0$ and `fed`
is the identity.

## 6.4 Sign-convention knobs (not corrections of the physics)

| Knob | Where in the chain | Effect | When |
|---|---|---|---|
| `flip_signal_polarity` | after $k_n$, before the DB snapshot | $C_n \leftarrow -C_n$ for all $n$, both channels; $B_1$ sign flips, $b_n/a_n$ unchanged | genuine cable/coil polarity swap |
| `encoder_offset_rad` | after the DB snapshot, before `rot` | pre-rotation $C_n \leftarrow C_n e^{-i n \delta}$; final harmonics identical (the `rot` step compensates) but `phi_out` then reflects the magnet angle only and avoids wrap edge cases. At $\delta = \pi$ odd orders negate, even orders unchanged | known encoder index offset (all MC62 campaigns use $\pi$) |
| `skew_main` | normalisation | divide by $A_m$ instead of $B_m$ | skew magnets |
| `absCalib` | normalisation / `main_field` | scalar factor on the main field (legacy) | leave at 1.0 |

Diagnosing which knob applies: even-order sign flip with odd orders unchanged
⇒ encoder offset; all orders flipped ⇒ polarity. Full decision table in
`notebooks/correction_options_reference.md`, "Sign-Convention Parameters".
Consumers should state their chosen polarity convention in their campaign
`report.md` (hub §4).

## 6.5 In-pipeline normalisation — `nor`

If `"nor"` is in `options`, after `fed` both channels are scaled by
$10^4/B_m$ (or $10^4/A_m$ with `skew_main`), $B_m$ taken from the **absolute**
channel after all corrections (`main_field`). Turns with
$|B_m| \le$ `eps_main` become NaN. The project convention is to **omit**
`nor` and normalise after the merge with `safe_normalize_to_units`
(chapter 7), so that tesla and units are both available and the Bottura §3.7
mixed record can be written.
