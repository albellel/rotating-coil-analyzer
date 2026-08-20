# 3. Signal processing before the FFT

Code: `analysis/preprocess.py` (`di_dt_weights`, `apply_di_dt_to_channels`,
`integrate_to_flux`, `estimate_linear_slope_per_turn`, `provenance_columns`),
called from `analysis/kn_pipeline.compute_legacy_kn_per_turn`. Bottura
counterpart: AII.8–18. Narrative version: `notebooks/pipeline_reference.md`
§1–3.

All steps act **per turn** on `(n_turns, Ns)` arrays and never exchange
information between turns. Order inside the pipeline: `dit` → `dri`
(integration) → FFT.

## 3.1 The raw quantity is a flux increment, not a voltage

The FDI delivers, per sample $k$, the flux increment over the angular step,

$$
\Delta\psi_k = -\int_{t_k}^{t_{k+1}} V\,dt \quad [\mathrm{Vs}],
$$

already converted to Vs upstream (Bottura AII.8–10 are done in the
acquisition chain; the package applies no gain). The arrays `df_abs`,
`df_cmp` are these increments.

> **Corrected from the old LaTeX.** The LaTeX described the integration as a
> $\Delta t$-weighted cumulative sum of a *voltage*,
> $\Phi_k = \sum_i v_i (t_{i+1}-t_i)$, optionally disabled. That is not what
> the code does and would be wrong for increment data. The flux is the
> **running sum of the increments** (Bottura AII.17–18), with no $\Delta t$
> factor.

## 3.2 `dit` — current-ramp reweighting (optional, first)

A legacy FFMM step, **not in Bottura**. When the current ramps during a turn,
each increment is rescaled to the turn-mean current:

$$
\Delta\psi_k \leftarrow w_k\,\Delta\psi_k, \qquad w_k = \frac{\bar I}{I_k},
$$

with $\bar I$ the turn-mean current and $I_k$ the sample current. It is a
quasi-static, linear correction (field $\propto I$ within the turn); it does
**not** model eddy currents. Activation per turn (`di_dt_weights`):

| Mode | Condition | Use |
|---|---|---|
| `dit_signed=False` (default) | $\lvert dI/dt \rvert > 0.1$ A/s **and** $\lvert\bar I\rvert > 10$ A | both ramp directions, both polarities |
| `dit_signed=True` | $dI/dt > 0.1$ A/s **and** $\bar I > 10$ A | exact FFMM C++ parity (`crr > 0.1 && cm > 10`) |

$dI/dt$ is the per-turn least-squares slope of $I(t)$ on the measured time
(`estimate_linear_slope_per_turn`). Turns with non-finite weights, or with
$\min\lvert I_k\rvert \le 10^{-12}$ A, are left uncorrected and flagged
(`DiDtResult.applied`). On plateaus ($dI/dt \approx 0$) `dit` is a guaranteed
no-op.

> **Corrected from the old LaTeX.** The LaTeX gave
> $v_k^{corr} = v_k - \alpha\,dI/dt$ with an "empirical coupling coefficient
> $\alpha$". No such subtraction exists; `dit` is the multiplicative reweighting
> above, with no free parameter.

## 3.3 `dri` — drift (offset) correction and integration

Integrator/amplifier offset adds a constant spurious voltage, i.e. a linear
ramp to the running-sum flux. Bottura AII.12–14 (DC case: the flux must
return to itself after a full turn):

$$
V_{off} = -\frac{\sum_{k=1}^{N}\Delta\psi_k}{t_{N+1}}, \qquad
\Delta\psi_k \leftarrow \Delta\psi_k + V_{off}\,\Delta t_k
= \Delta\psi_k - \frac{\Delta t_k}{\sum_j \Delta t_j}\sum_{j}\Delta\psi_j .
$$

`integrate_to_flux(df, drift=True, drift_mode=…)` implements two variants:

**`drift_mode="legacy"`** (default; exact FFMM C++ expression, uniform $\Delta t$):

$$
\psi = \operatorname{cumsum}\!\big(\Delta\psi - \overline{\Delta\psi}\big)
      - \overline{\operatorname{cumsum}(\Delta\psi)} .
$$

The first term is AII.14 with equal intervals (the same $\sum\Delta\psi/N$
subtracted from every increment). The second subtracts the mean of the
**uncorrected** running sum — deliberately, to match the C++ line
`cumsum(df - mean(df)) - mean(cumsum(df))`. This constant only touches the DC
bin, which the FFT step discards, so the harmonics are identical to the
plain AII.14 result; keeping it gives bit-for-bit parity
(`PARITY_REPORT.md` §5.2).

**`drift_mode="weighted"`** (Bottura AII.12–14 with measured intervals):

$$
\Delta t_k = t_k - t_{k-1}\ (\Delta t_0 = 0), \qquad
\Delta\psi_k \leftarrow \Delta\psi_k - \frac{\sum_j \Delta\psi_j}{\sum_j \Delta t_j}\,\Delta t_k,
\qquad \psi = \operatorname{cumsum}(\Delta\psi).
$$

Requires `t_turns`; turns whose total time is non-positive or non-finite are
left uncorrected and flagged (`DriftResult.applied`, `offset_per_s`). This is
the mode appropriate when the rotation speed ripples within a turn; it is
implemented and unit-tested but **not yet validated end-to-end** on a speed-
ripple dataset (hub §4 open flag).

Without `"dri"` in `options`, $\psi = \operatorname{cumsum}(\Delta\psi)$ with
no correction.

> **Corrected from the old LaTeX.** The LaTeX defined drift correction as
> "subtracting the mean of the integrated flux over the turn",
> $\Phi^{corr}_k = \Phi_k - \langle\Phi\rangle$. That removes only the DC bin
> (irrelevant to the harmonics) and is **not** an offset correction. The
> actual correction removes the linear ramp through the increments, as above.

**When not to use it.** AII.12 assumes $\psi_{N+1} = \psi_1$, i.e. constant
field over the turn. During ramps the flux genuinely changes and the
"offset" estimate absorbs real signal (the $\partial C_n/\partial t$ term of
Eq. 14). Bottura's AC recipe drops the drift step and relies on
forward/backward averaging (AII.15), which this package does not implement —
so on ramps the analyst must judge whether `dri` is acceptable (the
per-turn linear trend is small when $\tau_{turn}\,\dot{B} \ll B$) and should
report the choice.

## 3.4 Why `cumsum` and not trapezoid

The global rule "integrate voltage with `cumulative_trapezoid`" applies to a
continuous **voltage** time series. FDI data are already integrated per
interval; accumulating increments is the exact reconstruction (Bottura note
§10, "already-integrated discrete increments" exception). Do not replace it.

## 3.5 Provenance

`provenance_columns()` produces per-turn boolean/float columns
(`preproc_di_dt_applied`, `preproc_dI_dt_A_per_s`, `preproc_drift_mode`,
`absolute_preproc_drift_offset_per_s`, …) that the GUI attaches to every
export; `format_preproc_tag()` encodes the same choices in file names
(e.g. `didt_off_m01_flux_dri_legacy_dc_off`).
