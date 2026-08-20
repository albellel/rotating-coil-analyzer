# 9. Usage examples

The canonical, fully worked programmatic recipe is **`GUIDE.md` §3** (the
path the magnet-analysis repos use). This chapter only maps the old LaTeX
examples onto the current code and keeps them short; it does not duplicate
the guide.

> **Corrected from the old LaTeX.** The LaTeX described a three-tab GUI
> ("Phase I / II / III") launched with `py -m rotating_coil_analyzer.gui.app`.
> The GUI is an **eight-tab ipywidgets application** built inside a Jupyter
> notebook with `build_gui()`; there is no `__main__` entry point.

## 9.1 Launching the GUI

```python
%matplotlib widget                      # ipympl, interactive zoom/pan

from rotating_coil_analyzer.gui.app import build_gui
gui = build_gui()
gui
```

| Tab | Purpose | Chapter |
|---|---|---|
| 0 Catalog | select folder → `MeasurementCatalog`; load one segment → `SegmentFrame`; preview | 2 |
| 1 Plateau Detection | (streaming) block-averaged range, three-rule plateau detection | `GUIDE.md` §3.5 |
| 2 Harmonics | preview/apply data-quality cuts (tail trim, first/last turns), `dit`/`dri` options, FFT, amplitude/phase per order and per plateau | 3, 4 |
| 3 Coil Calibration | load segment TXT **or** compute from head CSV; state the compensation scheme; export TXT → `KnBundle` | 5 |
| 4 Harmonic Merge | apply $k_n$, rotation, cel/fed; recommendation table; per-order Abs/Cmp choice; normalise; export CSV with full provenance → `MergeResult` | 6, 7 |
| 5 Raw Signal Plots | time-series exploration, decimation only | — |
| 6 Physics Plots | hysteresis $B$ vs $I$, transfer function $B_1/I$, $L_d = dB_1/dI$, eddy settling fits (fitting from `tools_for_data_analysis.fitting.eddy`) | `notebooks/physics_reference.md` |
| 7 Comparison | two exported CSVs, $\Delta$ with propagated $\sigma$ | — |

Walkthrough on a real dataset:
`notebooks/Buckley_steerer/2026-06-23_degauss_parity/gui_walkthrough.ipynb`.

## 9.2 Example 1 — standard analysis with a precomputed segment $k_n$

```python
from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_plateau import PlateauReader
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    process_kn_pipeline, build_harmonic_rows, diagnose_cel_fed)
import pandas as pd

cat = MeasurementDiscovery().build_catalog("path/to/measurement_folder")
run_id, ap, seg = cat.runs[0], cat.enabled_apertures[0], cat.segments[0].segment_id
seg_frame = PlateauReader().read(cat.segment_files[(run_id, ap, seg)],
                                 run_id=run_id, segment=seg,
                                 samples_per_turn=cat.samples_per_turn,
                                 aperture_id=ap, magnet_order=cat.magnet_order)
# StreamingReader.read additionally takes shaft_speed_rpm=cat.shaft_speed_rpm (dt check)
tb = split_into_turns(seg_frame)               # (n_turns, Ns) arrays

kn = load_segment_kn_txt("path/to/Kn_values_Seg_Main.txt")

diag = diagnose_cel_fed(tb.df_abs, tb.df_cmp, tb.t, tb.I,
                        kn=kn, r_ref=0.040, magnet_order=1, max_zR=0.01)
OPTIONS = ("dri", "rot", "cel", "fed") if diag.recommendation == "SAFE" else ("dri", "rot")

result, C_merged, C_units, ok_main = process_kn_pipeline(
    tb.df_abs, tb.df_cmp, tb.t, tb.I,
    kn=kn, r_ref=0.040, magnet_order=1, options=OPTIONS)

df = pd.DataFrame(build_harmonic_rows(result, C_merged, C_units, ok_main, 1,
                                      [{"plateau_id": int(p)} for p in tb.plateau_id]))
```

For streaming data replace `PlateauReader` by `StreamingReader` and select
plateau turns first (`GUIDE.md` §3.4–3.5).

## 9.3 Example 2 — $k_n$ from a measurement-head CSV

```python
from rotating_coil_analyzer.analysis.kn_head import (
    compute_head_kn_from_csv, compute_segment_kn_from_head, write_segment_kn_txt)
from rotating_coil_analyzer.analysis.kn_bundle import KnBundle

head = compute_head_kn_from_csv("path/to/head.csv",
                                warm_geometry=True,        # room-temperature dimensions
                                use_design_radius=True,    # fall back if calibrated radius missing
                                n_multipoles=15)

kn = compute_segment_kn_from_head(head,
                                  abs_connection="1.2",         # outer coil: absolute
                                  cmp_connection="1.1-1.3")     # A - C bucking

write_segment_kn_txt(kn, "segment_kn_from_head.txt")           # reusable in Example 1

bundle = KnBundle(kn=kn, source_type="head_csv", source_path=head.source_path,
                  timestamp=KnBundle.now_iso(),
                  head_abs_connection="1.2", head_cmp_connection="1.1-1.3",
                  head_warm_geometry=True, head_n_multipoles=15,
                  extra={"compensation_scheme": "A-C"})          # NOT inferable from the CSV
```

Coil orientations must be radial or tangential (chapter 5.4). Compare with a
legacy $k_n$ file: `tests/test_kn_head_csv_vs_reference.py`. Notebook:
`notebooks/tools/kn_from_mh_csv.ipynb`.

## 9.4 Example 3 — inspecting Abs vs Cmp before merging

```python
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice
from rotating_coil_analyzer.analysis.kn_pipeline import merge_coefficients, safe_normalize_to_units

choice, diag = recommend_merge_choice(C_abs=result.C_abs, C_cmp=result.C_cmp, magnet_order=1)
print(diag.noise_abs, diag.noise_cmp, diag.mismatch, diag.flags)

C_merged, choice = merge_coefficients(C_abs=result.C_abs, C_cmp=result.C_cmp,
                                      magnet_order=1, mode="custom", per_order_choice=choice)
C_units, ok = safe_normalize_to_units(C_merged, 1, min_main_field=1e-4)
```

## 9.5 Example 4 — reproducing a legacy export (parity configuration)

```python
res, Cm, Cu, ok = process_kn_pipeline(
    df_abs, df_cmp, t, I, kn=kn, r_ref=0.050, magnet_order=1,
    options=("dri", "rot"), drift_mode="legacy",
    dit_signed=True,                      # FFMM signed thresholds (only matters with "dit")
    legacy_rotate_excludes_last=True,     # ONLY for the historical SM18 export
)
```

Use `legacy_rotate_excludes_last=True` **only** for that SM18 reference; for
physics use the default (chapter 6.1). Full sweep tooling:
`validation/golden_streaming.py`; worked notebooks under
`notebooks/SM18/2024-12-04_parity/`, `notebooks/LIU_BTP8/2019-07-17/`,
`notebooks/Buckley_steerer/2026-06-23_degauss_parity/`.

## 9.6 Recommended workflow

1. Inspect the catalog and the reader warnings (trims) before anything else.
2. Run `diagnose_cel_fed` on the highest-current data; drop `cel`/`fed` if
   `UNSAFE` (dipoles at low field).
3. Never include `"nor"` in `options`; normalise post-merge.
4. Decide the merge from the diagnostics; keep both channels in the export.
5. State the sign-convention knobs used (`flip_signal_polarity`,
   `encoder_offset_rad`) in the campaign `report.md`.
6. Archive results together with the `AnalysisProfile` JSON and the $k_n$
   provenance.
