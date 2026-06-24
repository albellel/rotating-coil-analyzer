# Buckley Steerer — Degauss FFMM Parity (metadata)

Metadata only — no physics results (those are produced by the notebooks and
their `csv/` + `figures/` outputs).

## Notebooks in this folder

- `parity_validation.ipynb` — batch FFMM golden parity over all 74 runs + degauss field decay (deterministic, executed).
- `gui_walkthrough.ipynb` — interactive GUI launcher; points the Catalog tab at this folder directly (the ingest layer reads the FFMM degauss format natively — `*_Parameters.txt` name and the H/V plateau naming) and walks through the 8 tabs (left unexecuted — it renders a live widget).

## Dataset

- **Source**: `golden_standards/degaussing_test/StandardDegauss/20260623_165352_Carbonara_test_Buckley-steerer/`
- **Project / magnet**: `Carbonara_test` / `Buckley_steerer` (combined H/V steerer, declared **Dipole**, normal)
- **Sequence**: degaussing staircase — alternating-polarity, decaying-amplitude
  set current applied first on the horizontal channel (runs `0–36`) then the
  vertical channel (runs `37–73`); each run is a measurement plateau.
- **Runs**: 74, each with `*_Main_raw_measurement_data.txt` (raw), `*_fluxes`
  (binary), and `*_Main_results.txt` (FFMM golden export).

The sibling `StandardDegauss/` sub-runs (`..._144731`, `..._150306`) and the
`SpiralDegauss/` folder share the same format; point `DATA` in the notebook at
any of them to re-validate.

## Acquisition / analyzer configuration (from `Parameters.txt`)

| Parameter | Value |
|-----------|-------|
| Reference radius `Rref` | 0.040 m |
| Magnet order `m` | 1 (dipole) |
| Samples per turn | 1024 |
| Turns analysed | 10 (raw file holds 20 physical turns; FFMM uses the first 10) |
| Encoder pulses | 4096 |
| Motor angular speed | −120 (≈2 rev/s) |
| Coil length | 1.2 m |
| Compensation scheme | **`A-C`** — active kn `kn.selection=fullfile` → `Kn_R45_N1_A_AC.txt` (abs coil A, compensated A−C; dipole bucking). The `A A-B-C+D` / `1.1-1.2-1.3+1.4` strings are from the *unused* `fullFolder`/`calc` paths. Label is metadata only (does not affect results). |
| Kn file | `Kn_values_Seg_Main.txt` (15 orders, 4 columns: AbsRe AbsIm CmpRe CmpIm) |
| Analyzer options | `dri rot` (abs/cel/fed/nor = false → output in Tesla) |

## Parity method (what `parity_validation.ipynb` does)

This analyzer is run with settings matching the FFMM golden export:

- `OPTIONS = ("dri", "rot")`, `magnet_order = 1`, `r_ref = 0.04`
- merge `abs_upto_m_cmp_above` (main from absolute channel, higher orders from compensated)
- `legacy_rotate_excludes_last = True` (FFMM skips the last harmonic in the rotation loop — a known C++ off-by-one)

For every run, the per-turn $B_n(T)$ / $A_n(T)$ for $n = 1\ldots15$ are compared
against the golden `*_Main_results.txt`. The notebook also demonstrates the
last-harmonic rotation convention difference (`rotate-all`, Bottura AIV.6, is
the physically correct default) and shows the degauss field decay and residual.

> **Current channel caveat.** The magnet ran at ≤10 A, but the recorded current
> column (col 4) is **not** the magnet drive — it is a near-constant reference
> reading ~1000 in its own raw units (std ~27), identical at +10 A, −10 A and
> 0 A, with no sign flip. The real per-run excitation is the **commanded set
> current** in the filename (bipolar, decaying 10→0 A), which is used for the
> physics axes; the field $B_1$ tracks it. Parity is unaffected (it compares
> harmonics and uses the same recorded column FFMM did). Also, each raw file has
> 20 turns but the current column is only populated for the **first 10**
> (turns 10–19 are NaN-current); the parity notebook uses the first 10, and the
> GUI loads all 20 (flux is valid throughout). Consequence for the GUI: the
> **Plateau Detection tab is meaningless here** (it would threshold the reference
> channel); the data is already segmented per run via `plateau_id`. The harmonics
> **never use the current** (only the `dit` di/dt correction would, and it is OFF
> — and it cannot even activate without a ramp), so $B_n/A_n$ and the parity are
> independent of this column.
>
> **Native ingest (no renaming/doctoring).** `find_parameters_txt` accepts
> `*_Parameters.txt`, and the shared `parse_plateau_filename` handles both the
> standard `_I_<cur>A_` and the H/V `..IH_..IV_..` layouts, so discovery + the
> plateau reader open this folder directly. The reader parses the signed
> commanded current from the filename into the **`plateau_I_hint`** column — the
> honest source of the true per-run current — while the measured `I` column is
> left untouched. The GUI's current-axis displays therefore still show the ~1000
> reference; use `plateau_I_hint` (or the parity notebook) for current-referenced
> results.

## Outputs

- `csv/parity_summary.csv` — per-run channel, currents, mean $B_1$, worst $|\Delta|$ vs golden
- `csv/parity_per_order.csv` — global max $|\Delta|$ per harmonic order
- `csv/rotation_offbyone_run0.csv` — per-order $|\Delta|$ for exclude-last vs rotate-all
- `csv/degauss_residual.csv` — remanent $B_1$ at the $I_\text{set}=0$ step per channel
- `figures/parity_overview.{svg,png}`, `figures/rotation_offbyone.{svg,png}`, `figures/degauss_field.{svg,png}`
