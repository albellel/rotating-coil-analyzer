# 2. Ingest, time policy and turn definition

Code: `rotating_coil_analyzer/ingest/` (`discovery.py`, `readers_streaming.py`,
`readers_plateau.py`, `channel_detect.py`), `models/` (`catalog.py`,
`frames.py`, `profile.py`), `analysis/turns.py`. Architecture diagram and data
models: `DOCUMENTATION.md` §2, §5, §6. Bottura counterpart: AII.1–11 (encoder
angles, pulse time, counts → flux).

## 2.1 Inputs

| Input | File | Reader |
|---|---|---|
| Streaming (continuous) acquisition | `*.bin` (little-endian float64 or float32; `*.txt`/`*.csv` also accepted) — columns `t, df_abs, df_cmp, [I0, I1, …]` | `StreamingReader` |
| Plateau (DC staircase) acquisition | one `*_raw_measurement_data.txt` per current level, whitespace-separated, no header | `PlateauReader` |
| Acquisition parameters | `Parameters.txt` or `<magnet>_<timestamp>_Parameters.txt` (key: value, with `TABLE{…}` payloads for the FDI map) | `MeasurementDiscovery` |
| Sensitivity $k_n$ | segment TXT (4 or 6 columns) **or** measurement-head geometry CSV | chapter 5 |

Plateau file names are parsed by `readers_plateau.parse_plateau_filename`,
shared by discovery and reader. Two layouts are recognised:

- standard staircase: `<base>_Run_<step>_I_<current>A_<seg>_raw_measurement_data.txt`;
- H/V steerer (e.g. FFMM degauss): `<base>_Run_<step>IH_<ih>IV_<iv>_<seg>_raw_measurement_data.txt`,
  where the reported current hint is the active channel ($I_H$ if non-zero,
  else $I_V$, signed).

`PlateauReaderConfig.filename_pattern` can supply an extra regex (named groups
`base`, `step`, `seg`, optionally `i`).

## 2.2 Discovery → `MeasurementCatalog`

`MeasurementDiscovery.build_catalog(folder)`:

1. finds the parameters file (up to two parent levels, exact `Parameters.txt`
   preferred over `*_Parameters.txt`);
2. parses `samples_per_turn`, `shaft_speed_rpm`, `magnet_order`, enabled
   apertures and the per-aperture **FDIs table** (segment label → absolute
   and compensated FDI channel index);
3. discovers one representative segment file per `(run_id, aperture, segment)`
   (streaming: `*corr_sigs*.bin`; plateau: one file per base+segment).

The catalog is a frozen dataclass. Parameters are **context**: they populate
`AnalysisProfile.from_catalog()` defaults and never silently modify data.
$R_{ref}$ has no source in `Parameters.txt` and must be given explicitly
(`from_catalog` falls back to a deliberately conservative 17 mm).

## 2.3 The no-synthetic-time rule

Time is a **measured** column. The package never creates, repairs,
interpolates, resamples or extrapolates time stamps. What the readers do:

**StreamingReader** (`_validate_candidate`, per candidate binary format):

- trims **trailing** rows with non-finite `t`/flux values (reported in
  `SegmentFrame.warnings`);
- trims to an integer number of turns;
- with `strict_time=True` (default) **rejects** the file/format if
  $\Delta t \le 0$ anywhere (time must be strictly increasing);
- checks the median $\Delta t$ against the nominal
  $\Delta t_{nom} = 60/(|v|\,N_s)$ from `shaft_speed_rpm` within `dt_rel_tol`
  (default 25 %) — this is also how the binary dtype / column count is
  inferred when several candidates fit the file size.

**PlateauReader** (`read`):

- reads every plateau file of the same base+segment, sorted by step;
- trims **each plateau independently** to whole turns, so no turn ever
  crosses a plateau boundary (`split_into_turns(strict_plateau_turns=True)`
  re-checks this invariant);
- concatenates and adds per-sample metadata `plateau_id`, `plateau_step`,
  `plateau_I_hint`, `sample_in_plateau`, `k` (a global ordering index —
  **not** time);
- keeps the raw `t` of each file, which may **reset** between plateaus and
  may contain NaN; no monotonicity is imposed across plateaus.

Channels are auto-assigned (`channel_detect.py`): the two flux columns are
ranked by `robust_range` ($p_{99.5} - p_{0.5}$), the larger being the absolute
coil; the current is the candidate with the largest robust range. An explicit
`ColumnMapping` overrides the heuristics.

> **Corrected from the old LaTeX.** The LaTeX stated that "if time stamps are
> non-finite or non-monotonic within a turn, the affected turn is rejected".
> That per-turn test does not exist. The code validates time **per file**
> (streaming, strict monotonicity or reject) and trims **tails**; it does not
> inspect each turn's time vector. Per-turn quality handling is the GUI
> Harmonics tab's explicit preview → apply step (tail trimming to full turns,
> first/last-turn dropping), and the pipeline itself flags — not drops —
> turns whose main harmonic is non-finite or below `eps_main`
> (`LegacyKnPerTurn.phi_bad`).

## 2.4 Turn definition

A turn is a block of exactly `samples_per_turn` ($N_s$) consecutive samples,
as fixed by the encoder in `Parameters.txt`. `analysis.turns.split_into_turns`
reshapes the `SegmentFrame` columns to `(n_turns, Ns)` arrays (`TurnBlock`)
— no search for encoder index pulses, no phase alignment between turns.

The angular grid is **implicit in the sample index**,

$$
\theta_k = \frac{2\pi k}{N_s}, \qquad k = 0,\ldots,N_s-1,
$$

which is Bottura's AII.1–2 ("triggers nominally equally spaced"). Deviations
of the instantaneous speed from uniform (AII.7) are *not* corrected; they are
visible through the measured `t` and show up as quality diagnostics
(`duration_s`, `dI_dt_A_per_s` in `LegacyKnPerTurn`).

Every turn is processed as an independent measurement: no averaging,
stitching or filtering across turns happens before the harmonics exist
(chapter 4); Bottura's forward/backward pairing (AII.15–16) is not
implemented (chapter 10, deviation 1).

## 2.5 Outputs of the ingest stage

`SegmentFrame` — frozen; `df` has float64 columns `t`, `df_abs`, `df_cmp`, `I`
(plus current candidates and plateau metadata), `len(df) == n_turns * Ns`,
`warnings` recording every trim. **No numerical transformation** of the
signal happens here: no integration, no drift correction, no FFT.
