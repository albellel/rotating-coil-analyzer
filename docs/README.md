# Rotating Coil Analyzer — Documentation

**Consolidated 2026-08-20.** This folder replaces the separate LaTeX/Overleaf
repository `RC-analyzer-documentation` (last edited 2026-01-22), which was
stale relative to the code and has been retired. Every formula and every
function/option name below was checked against the current source
(`rotating_coil_analyzer/analysis/*.py`, `ingest/`, `models/`). Where the old
LaTeX described behaviour that no longer exists (or never did), the text
follows the code and says so explicitly.

Formulas are written in LaTeX (`$…$`, `$$…$$`) so they copy directly into
Overleaf.

## One source of truth per topic

| Topic | Canonical document |
|---|---|
| Feature overview, install, GUI tab list, kn file format | [`../README.md`](../README.md) |
| Programmatic pipeline walkthrough (the recipe the magnet repos use) | [`../GUIDE.md`](../GUIDE.md) |
| Architecture, data models, full function reference | [`../DOCUMENTATION.md`](../DOCUMENTATION.md) |
| Golden-standard parity numbers (SM18, LIU BTP8, LEAR MC62) | [`../PARITY_REPORT.md`](../PARITY_REPORT.md) |
| Option-by-option guide, cel/fed failure modes, sign conventions | [`../rotating_coil_analyzer/notebooks/correction_options_reference.md`](../rotating_coil_analyzer/notebooks/correction_options_reference.md) |
| Stage-by-stage pipeline narrative (what/where/why) | [`../rotating_coil_analyzer/notebooks/pipeline_reference.md`](../rotating_coil_analyzer/notebooks/pipeline_reference.md) |
| Physics background (TF vs $L_d$, eddy settling, fringe $b_3$) | [`../rotating_coil_analyzer/notebooks/physics_reference.md`](../rotating_coil_analyzer/notebooks/physics_reference.md) |
| **Theory** (Bottura MTA-IN-97-007, every equation transcribed, numbered) | `bibliography-review/magnetic_measurements/bottura1997_standard_analysis_field_quality_LHC_harmonics_notes.md` (⭐⭐⭐ deep note, bib key `bottura1997`; §11 = equation → function map) |
| **Theory → code, with deviations** (this folder) | [`10_bottura_cross_reference.md`](10_bottura_cross_reference.md) |

The theory is **not re-derived here**. Chapters cite Bottura equation numbers
using the note's numbering (Eq. 1–22 main text, AI.*, AII.*, AIII.*, AIV.*)
and point to that note.

## Chapters

| # | File | Content |
|---|---|---|
| 1 | [`01_measurement_principle.md`](01_measurement_principle.md) | Complex multipoles, flux of a rotating coil, what the analyzer reconstructs |
| 2 | [`02_ingest_and_time_policy.md`](02_ingest_and_time_policy.md) | Input formats, discovery, readers, the no-synthetic-time rule, turn splitting |
| 3 | [`03_signal_processing.md`](03_signal_processing.md) | `dit` current-ramp reweighting, `dri` drift correction, integration to flux |
| 4 | [`04_fft_and_harmonics.md`](04_fft_and_harmonics.md) | Per-turn DFT, $2/N_s$ scaling, normal/skew, units |
| 5 | [`05_kn_calibration.md`](05_kn_calibration.md) | Sensitivity $k_n$: segment TXT files, head-geometry CSV, connections, `KnBundle` |
| 6 | [`06_corrections_rot_cel_fed.md`](06_corrections_rot_cel_fed.md) | Rotation, centre location, feed-down, sign-convention knobs |
| 7 | [`07_merge_and_normalization.md`](07_merge_and_normalization.md) | Abs/Cmp merge modes, recommendation diagnostics, normalization to units, Bottura §3.7 record |
| 8 | [`08_validation_and_regression.md`](08_validation_and_regression.md) | Test suite, golden standards, `golden_streaming.py`, how to diagnose a discrepancy |
| 9 | [`09_usage_examples.md`](09_usage_examples.md) | GUI walkthrough (8 tabs), precomputed $k_n$, $k_n$ from head CSV, parity configuration |
| 10 | [`10_bottura_cross_reference.md`](10_bottura_cross_reference.md) | Equation → function table and the **three documented deviations** |

## The three documented deviations from Bottura (summary)

1. **No forward/backward rotation averaging (AII.15–16).** Each rotation is
   analysed independently; the per-turn drift step carries the whole load.
2. **Dipole centring uses the 20-pole (AIII.3), linearised**, via the
   compensated $C_{10}, C_{11}$ — not the 16-pole polynomial + cost function
   (AIII.1–2) of the main text. Fragile at low field; see `diagnose_cel_fed()`.
3. **`legacy_rotate_excludes_last=True` is SM18-parity only.** It reproduces an
   off-by-one in the historical SM18 C++ rotation loop and has no basis in the
   note. The default `False` rotates all orders (AIV.6).

Details and the full table: [`10_bottura_cross_reference.md`](10_bottura_cross_reference.md).
