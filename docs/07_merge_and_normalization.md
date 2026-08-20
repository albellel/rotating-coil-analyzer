# 7. Abs/Cmp merge, recommendation diagnostics and normalisation to units

Code: `analysis/kn_pipeline.merge_coefficients`, `safe_normalize_to_units`;
`analysis/merge.recommend_merge_choice` (`MergeDiagnostics`);
`analysis/kn_bundle.MergeResult`; `analysis/utility_functions`
(`process_kn_pipeline`, `mixed_format_table`, `ba_table_from_C`,
`build_harmonic_rows`). Bottura counterpart: §3.7 (record of harmonics),
AII.23 (bucking ratio), AIV.8–9 (normalisation).

## 7.1 Why merge, and why never blindly

Bottura §3.7: the compensated signal contains ideally nothing below order
$m$, while the absolute signal is dominated by order $m$ whose leakage
pollutes the higher orders. The standard record therefore takes **absolute
harmonics up to $m$ and compensated from $m+1$**. In this package the merge
is an explicit, auditable, per-order channel selection — never implicit. Both
pre-merge arrays survive in `MergeResult.C_abs` / `C_cmp` and in the `*_db`
snapshots.

## 7.2 `merge_coefficients` — apply a policy

```python
C_merged, choice = merge_coefficients(C_abs=..., C_cmp=..., magnet_order=m,
                                      mode="abs_upto_m_cmp_above")
```

| `mode` | Orders from Abs | Orders from Cmp |
|---|---|---|
| `abs_all` | all | — |
| `cmp_all` | — (main still forced to Abs) | all $n \ne m$ |
| `abs_main_cmp_others` | $n = m$ | $n \ne m$ |
| `abs_upto_m_cmp_above` (**Bottura §3.7**, project default) | $n \le m$ | $n > m$ |
| `custom` | per `per_order_choice` (0 = abs, 1 = cmp) | … |

The main order $m$ is **always** taken from the absolute channel, in every
mode including `custom` (the compensated channel has bucked it away).
`choice` (length $H$, 0/1) is returned so the merge is reproducible; the GUI
stores it as `MergeResult.per_n_source_map` and exports it as
`merge_per_n_source_map` metadata.

> **Corrected from the old LaTeX.** The LaTeX said corrections (rotation,
> cel, feed-down) are applied "after calibration and optional merging". In
> the code all of them run **per channel inside** `compute_legacy_kn_per_turn`;
> the merge comes afterwards and only selects between already-corrected
> channels. Likewise the LaTeX's "merge diagnostics then user approval" is the
> GUI flow; the programmatic API (`process_kn_pipeline`) applies the chosen
> `merge_mode` directly.

## 7.3 `recommend_merge_choice` — diagnostic recommendation

Given post-$k_n$ (typically post-rotation) `C_abs`, `C_cmp` of shape
`(n_turns, H)`, for each order:

- robust per-channel noise $\sigma = \sqrt{\mathrm{MAD}(\mathrm{Re})^2 + \mathrm{MAD}(\mathrm{Im})^2}$
  across turns ($\mathrm{MAD}$ scaled by 1.4826);
- mismatch $= \operatorname{median}_{turns}\lvert C^{abs}_n - C^{cmp}_n\rvert$;
- choose Cmp if $\sigma_{cmp} < 0.90\,\sigma_{abs}$ (`prefer_cmp_if_better`),
  else Abs;
- if mismatch $> 50 \times \min(\sigma_{abs}, \sigma_{cmp})$
  (`mismatch_tol_rel`), fall back to Abs and set `FLAG_MISMATCH_LARGE`;
- non-finite channels → `FLAG_BAD_CHANNEL`; order $m$ → `FLAG_MAIN_FORCED_ABS`.

Returned as `MergeDiagnostics` (`noise_abs`, `noise_cmp`, `mismatch`,
`selected`, `flags`). The GUI Harmonic Merge tab shows this table and applies
the merge only after the user confirms. The noise comparison is the practical
stand-in for Bottura's bucking ratio $\beta_n = \Psi^{abs}_n/\Psi^{cmp}_n$
(AII.23) — the ratio itself is not computed as such.

## 7.4 `safe_normalize_to_units` — post-merge normalisation (AIV.8–9)

$$
c_n = 10^4\,\frac{C_n}{\mathrm{Re}(C_m)} \quad (\text{or } \mathrm{Im}\ \text{with } \texttt{skew\_main}),
$$

applied to **all** orders of the merged array; since $C_m$ comes from the
absolute channel after rotation, $b_m = 10^4$, $a_m = 0$. Turns with
$|B_m| \le$ `min_main_field` get NaN and `ok=False`. In the wrapper
`process_kn_pipeline` this threshold is `min_b1_T` (default $10^{-4}$ T, far
above the pipeline-internal `eps_main` $= 10^{-20}$), so "ok_main" means
"main field large enough for units to be meaningful", not merely non-zero.

## 7.5 The Bottura §3.7 record — `mixed_format_table`

Tesla for $n \le m$ (from `C_merged`, i.e. absolute), units for $n > m$
(from `C_units`, i.e. compensated under the default mode):

| Columns | Source |
|---|---|
| `B{n}_T`, `A{n}_T` for $n \le m$ | `C_merged` |
| `b{n}_units`, `a{n}_units` for $n > m$ | `C_units` |

plus, from `LegacyKnPerTurn` via `build_harmonic_rows`: `time_s`, `I_mean_A`,
`ok_main`, `phi_rad`, `x_mm`, `y_mm` — the centre coordinates and the field
angle that Bottura requires to be stored because centring and rotation cancel
information by construction. `ba_table_from_C` gives the all-tesla
`normal_B{n}` / `skew_A{n}` layout (legacy "DB" tables).

## 7.6 One-call wrapper

```python
result, C_merged, C_units, ok_main = process_kn_pipeline(
    flux_abs_turns, flux_cmp_turns, t_turns, I_turns,
    kn=kn, r_ref=R_REF, magnet_order=M,
    options=("dri", "rot", "cel", "fed"),     # never "nor" here
    drift_mode="legacy", min_b1_T=1e-4,
    merge_mode="abs_upto_m_cmp_above",
    dit_signed=False, max_zR=None,
    encoder_offset_rad=0.0, flip_signal_polarity=False,
    legacy_rotate_excludes_last=False,
)
```

`flux_*_turns` are the **incremental** signals despite the historical name.
Full worked recipe: `GUIDE.md` §3.

## 7.7 `MergeResult` — traceability

Frozen container with `C_merged`, `orders`, `per_n_source_map`,
`compensation_scheme`, `magnet_order`, `kn_provenance: KnBundle`,
`merge_mode`, timestamp, optional `diagnostics`, pre-merge `C_abs`/`C_cmp`,
and the derived `C_units`, `ok_main`, `I_mean_A` consumed by the Physics
Plots tab. `to_metadata_dict()` flattens merge + $k_n$ provenance for CSV
headers; `source_map_dataframe()` lists the per-order channel.
