# Guided Example — Rotating Coil Analyzer

A hands-on walkthrough of the two ways to use this package:

1. **Interactive GUI** (Jupyter, ipywidgets) — point-and-click, good for exploration.
2. **Programmatic streaming pipeline** — the path the analysis repos
   (`mbb-data-analysis`, `mc62-data-analysis`, ...) actually use in their
   notebooks. This is the canonical, scriptable workflow.

Every code block below uses only the **current public API** (verified against
the source). Symbols are imported from their concrete submodules, which is the
stable import surface external notebooks bind to.

---

## 0. Install

```bash
pip install -e .          # editable install, pulls numpy/pandas/scipy/matplotlib/ipywidgets/ipympl
```

The package also depends on the companion library `tools-for-data-analysis`
(declared in `pyproject.toml`), which provides the **eddy-current models and
fitting** (`eddy_model`, `double_eddy_model`, `triple_eddy_model`,
`fit_eddy_per_run`, `EddyFitResult`, ...) and continuous-time signal helpers.
These used to live in `rotating_coil_analyzer` but were moved out; import them
from `tools_for_data_analysis.fitting.eddy`.

---

## 1. Core concepts (read this first)

| Concept | What it is |
|---------|-----------|
| **Turn** | One revolution of the coil = `Ns` samples (`samples_per_turn`, e.g. 1024). All analysis is per-turn. |
| **Incremental signal** (`df_abs`, `df_cmp`) | Flux *increments* per sample from the digital integrator (FDI). The pipeline integrates them with `cumsum` — this is the documented exception to the "use trapezoid" rule, because they are already differential. |
| **Abs / Cmp** | Absolute (single coil) and Compensated (bucked) flux channels. |
| **kn** (`SegmentKn`) | Complex calibration coefficients per harmonic order, from a TXT file or computed from measurement-head geometry. |
| **`C_n`** | Calibrated complex harmonic. `B_n = Re(C_n)` (normal), `A_n = Im(C_n)` (skew). |
| **units** | `b_n = 10^4 · C_n / C_main`. Tesla for `n ≤ m`, dimensionless "units" for `n > m`. |
| **`m` / `magnet_order`** | Main field order: 1 = dipole, 2 = quadrupole, 3 = sextupole. |
| **OPTIONS** | Pipeline steps: `dit` (di/dt), `dri` (drift+integrate), `rot` (rotation), `cel` (centre location), `fed` (feeddown). `nor` is **not** used here — normalisation is done post-merge by `safe_normalize_to_units`. |

**Hard constraints the library enforces** (don't fight them): no synthetic
time (time only comes from data; bad-time tails are dropped, never repaired),
no interpolation (downsampling is decimation only), and baseline subtraction is
a constant scalar only.

---

## 2. The GUI workflow (Jupyter)

```python
%matplotlib widget                          # interactive zoom/pan (needs ipympl)

from rotating_coil_analyzer.gui.app import build_gui
gui = build_gui()
gui                                          # renders the 8-tab GUI
```

Work left to right through the tabs:

| Tab | Do this |
|-----|---------|
| **0. Catalog** | Browse to a measurement folder, discover runs/segments, load one segment, preview the first turns. |
| **1. Plateau Detection** | (streaming data) detect current plateaus; result is shared with later tabs. |
| **2. Harmonics** | Preview the data-quality cuts, apply them, compute FFT harmonics, view amplitude-vs-current. |
| **3. Coil Calibration** | Load `kn` from TXT **or** compute it from a measurement-head CSV (you must state the compensation scheme — it is *not* in the CSV). |
| **4. Harmonic Merge** | Apply `kn`, pick Abs/Cmp per order, normalise, export traceable CSVs. |
| **5. Raw Signal Plots** | Read-only time-series exploration (decimation only). |
| **6. Physics Plots** | Hysteresis `B` vs `I`, transfer function `B1/I`, differential inductance `dB1/dI`, eddy-current settling fits. |
| **7. Comparison** | Overlay two exported CSVs and compute Δ / σ significance. |

The GUI is the quickest way to learn the data; the programmatic path below is
what you reach for once you want a reproducible notebook.

---

## 3. The programmatic streaming pipeline (canonical)

This mirrors exactly what the magnet-analysis repos do. The end-to-end shape is:

```
load kn ─► reshape current into turns ─► detect plateaus ─► classify turns
        ─► run kn pipeline on selected turns ─► build a tidy DataFrame
        ─► per-level statistics, hysteresis, eddy settling, diagnostics
```

### 3.1 Imports

```python
import numpy as np
import pandas as pd

from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
    process_kn_pipeline,
    build_harmonic_rows,
    plateau_summary,
    mad_sigma_clip,
    diagnose_cel_fed,
    diagnose_fdi_transitions,
)
from rotating_coil_analyzer.ingest.channel_detect import robust_range
# eddy fitting lives in the companion package:
from tools_for_data_analysis.fitting.eddy import eddy_model, fit_eddy_per_run
```

### 3.2 Configuration

```python
R_REF       = 0.020          # reference radius [m]
MAGNET_ORDER = 1             # dipole
Ns          = 1024           # samples per turn
N_BLOCKS    = 10             # for block-averaged current range
MIN_B1_T    = 1e-4           # below this, normalisation is flagged not-ok
OPTIONS     = ("dri", "rot", "cel", "fed")   # never include "nor"
PLATEAU_I_RANGE_MAX = 3.0    # A — max within-turn current variation on a plateau
```

### 3.3 Load calibration

```python
kn = load_segment_kn_txt("path/to/segment_Kn_values.txt")
# kn.orders -> [1..H], kn.kn_abs / kn.kn_cmp are complex (H,)
```

### 3.4 Get turn-shaped arrays

If you loaded a segment through the ingest layer, reshape it into turns:

```python
from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader
from rotating_coil_analyzer.analysis.turns import split_into_turns

seg = StreamingReader().read(path, run_id="run", segment="seg",
                             samples_per_turn=Ns, magnet_order=MAGNET_ORDER)
tb  = split_into_turns(seg)         # TurnBlock with (n_turns, Ns) arrays

flux_abs = tb.df_abs                # (n_turns, Ns) incremental absolute flux
flux_cmp = tb.df_cmp                # (n_turns, Ns) incremental compensated flux
t_all    = tb.t                     # (n_turns, Ns) measured time  — never synthetic
I_all    = tb.I                     # (n_turns, Ns) current
n_turns  = tb.n_turns
```

> If you already have `(n_turns, Ns)` numpy arrays from your own loader (as the
> magnet repos do), skip the reader and feed them in directly.

### 3.5 Detect plateaus and classify turns

```python
I_mean = I_all.mean(axis=1)
I_range, I_blocks = compute_block_averaged_range(I_all, Ns, N_BLOCKS)

info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
is_plateau = info["is_plateau"]      # all 3 rules: flat + starts-flat + ends-flat

# Label each plateau turn by current level (SPS thresholds by default;
# pass a custom dict for PS/PSB/LHC)
turn_label = np.array(["ramp"] * n_turns, dtype=object)
for j in range(n_turns):
    if is_plateau[j]:
        turn_label[j] = classify_current(I_mean[j])

# Contiguous groups of a given level (e.g. injection plateaus of >= 2 turns)
inj_groups = find_contiguous_groups(turn_label == "injection", min_length=2)

plateau_indices = np.where(is_plateau)[0]
```

### 3.6 Run the kn pipeline

`process_kn_pipeline` wraps `compute_legacy_kn_per_turn` → `merge_coefficients`
→ `safe_normalize_to_units` in one call. **The flux inputs are the incremental
signals** (the name `flux_*_turns` is historical) — integration happens inside
via the `dri` option.

```python
result, C_merged, C_units, ok_main = process_kn_pipeline(
    flux_abs_turns=flux_abs[plateau_indices],
    flux_cmp_turns=flux_cmp[plateau_indices],
    t_turns=t_all[plateau_indices],
    I_turns=I_all[plateau_indices],
    kn=kn,
    r_ref=R_REF,
    magnet_order=MAGNET_ORDER,
    options=OPTIONS,
    min_b1_T=MIN_B1_T,
)
# result   : LegacyKnPerTurn (per-turn complex harmonics + zR, phi, I_mean, ...)
# C_merged : (n_sel, H) complex, Tesla
# C_units  : (n_sel, H) complex, units (NaN where main field too weak)
# ok_main  : (n_sel,)  bool — True where |B_main| > min_b1_T
```

### 3.7 Build a tidy DataFrame

```python
extra = [
    {"global_turn": int(plateau_indices[k]),
     "label": str(turn_label[plateau_indices[k]])}
    for k in range(len(plateau_indices))
]
rows = build_harmonic_rows(result, C_merged, C_units, ok_main,
                           MAGNET_ORDER, extra)
df = pd.DataFrame(rows)
# columns: time_s, I_mean_A, ok_main, phi_rad, x_mm, y_mm,
#          B1_T, A1_T (n <= m, Tesla), b3_units, a3_units, ... (n > m, units),
#          plus your extra columns.
```

### 3.8 Clean and summarise

```python
df_clean, removed = mad_sigma_clip(df, col="b3_units", n_sigma=5, label_col="label")

# Per-level mean/std of B1, TF and all harmonics (needs run_id / turn_in_run /
# branch / I_nom columns — add them per your campaign's run structure):
summary = plateau_summary(df_clean, n_last=18, harmonics_range=range(2, 16))
```

### 3.9 Safety diagnostics (recommended)

`cel`/`fed` (centre location + feeddown) can be fragile for dipoles at low
field. Check before trusting them:

```python
diag = diagnose_cel_fed(
    flux_abs[plateau_indices], flux_cmp[plateau_indices],
    t_all[plateau_indices], I_all[plateau_indices],
    kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER, max_zR=0.01,
)
print(diag.recommendation, "—", diag.reason)   # SAFE / UNSAFE / MIXED
# If UNSAFE -> rerun the pipeline with OPTIONS = ("dri", "rot")
```

Detect FDI stuck-channel artefacts across plateau boundaries (they create fake
eddy transients):

```python
run_info = [{"run_id": g_i, "start": s, "end": e, "I_nom": I_mean[s]}
            for g_i, (s, e) in enumerate(inj_groups)]
checks = diagnose_fdi_transitions(flux_abs, I_mean, run_info)
for c in checks:
    if c.is_stuck:
        print(f"STUCK at run {c.run_before}->{c.run_after}: {c.reason}")
```

### 3.10 Eddy-current settling (from the companion package)

On a settled plateau the field relaxes as `B(t) = B_inf + A·exp(-t/tau)`. Fit it
per run to extract the time constant and the settled value:

```python
fit = fit_eddy_per_run(turn_index, b3_series)   # see tools_for_data_analysis docs
print(fit.B_inf, fit.A, fit.tau, fit.R2)
```

---

## 4. FFMM C++ golden-standard parity

To reproduce the legacy C++ analyzer bit-for-bit (used for validation), use the
restricted option set and the legacy rotation flag:

```python
res, Cm, Cu, ok = process_kn_pipeline(
    flux_abs, flux_cmp, t_all, I_all,
    kn=kn, r_ref=R_REF, magnet_order=MAGNET_ORDER,
    options=("dri", "rot"),
    min_b1_T=MIN_B1_T,
    legacy_rotate_excludes_last=True,   # SM18 C++ off-by-one in the rotation loop
)
```

`legacy_rotate_excludes_last=False` (the default) rotates **all** harmonics
`k=1..H` per Bottura Eq. AIV.6 and matches the modern FFMM C++ and Pentella
implementations. Only set it to `True` to reproduce the historical SM18 export.
See `PARITY_REPORT.md` for the validated tolerances.

---

## 5. Sign-convention knobs (when results look flipped)

| Symptom | Knob |
|---------|------|
| `B1` negative at positive current (cable/coil polarity inverted) | `flip_signal_polarity=True` |
| Even harmonics flipped, odd unchanged (encoder 180° offset) | `encoder_offset_rad=np.pi` |
| Suspect centre offsets blowing up `cel`/`fed` | `max_zR=<value>` (clamps `|zR|`, flags `result.zR_clamped`) |

These are passed straight through `process_kn_pipeline`. For the difference
between a polarity swap and an encoder offset, compare even/odd `b_n` signs
against a reference (see `notebooks/correction_options_reference.md`).

---

## 6. Where to go next

- `README.md` — feature overview and Kn file format.
- `DOCUMENTATION.md` — architecture and full function reference.
- `notebooks/pipeline_reference.md` — end-to-end pipeline explained (what each
  stage does, where it sits, and why): integrate-to-flux, `dit`, `dri` modes,
  `Kn`, `rot`, `cel`, `fed`, merge, `nor`.
- `notebooks/correction_options_reference.md` — option-by-option guide with
  `cel`/`fed` failure modes.
- `notebooks/SM18/2024-12-04_parity/` and `notebooks/LIU_BTP8/2019-07-17/` —
  worked parity-validation notebooks against the legacy analyzer.
- `theory/` — the Bottura and Marusov reference papers the algorithms follow.
