# 11. Bug-fix diary

A dated record of every defect found and fixed in this package, newest first.
Each entry says what was wrong, why, what changed, whether **earlier results**
are affected, and how the fix was verified. Add an entry for every fix, in the
same change as the fix.

Entry template:

```
### BF-YYYY-MM-DD-NN — short title
- Component: file(s)
- Symptom: what a user saw (numbers if possible)
- Cause: the actual defect
- Fix: what changed
- Earlier results affected: yes/no, which paths
- Verification: test name(s), dataset, numbers
```

---

## 2026-10-30 — deep review: physics fixes, safe defaults, external usability

**Scope check (applies to every entry below).** Every path that already worked
was proven unchanged. The old code (git `a484df0`) and the new code were run on
the same real datasets — SM18 streaming (`.bin`, 20 000 turns), Buckley degauss
staircase (1 421 turns), FCC-ee quadrupole (50 turns) — with eight option sets
(legacy `dri,rot,nor,cel,fed`; `dri,rot,cel,fed`; `dit` signed; `dit` +
`max_zR`; `legacy_rotate_excludes_last`; encoder offset $\pi$ + polarity flip;
no drift; `process_kn_pipeline` defaults). All $C_n$, centre, angle and
current outputs are **bit-identical**. The only differences are the two
intended ones (BF-…-01 skew with rotation, BF-…-02 weighted drift). The
streaming reader output is bit-identical on all five 1.17 GB SM18 segments.
The analysis repos that use this package (`mbb-data-analysis`,
`mc62-data-analysis`) call neither `skew_main` nor the weighted drift mode, so
their numbers do not change.

### BF-2026-10-30-01 — skew main harmonic with rotation gave ~1e16 units
- Component: `analysis/kn_pipeline.py` (`compute_legacy_kn_per_turn`), GUI checkbox "Skew main harmonic".
- Symptom: with `rot` and `skew_main=True`, normalised harmonics of a skew
  quadrupole came out $\sim 2.7\times10^{16}$ units (synthetic test).
- Cause: the rotation reference $\varphi = \mathrm{wrap}(\arg C_m)/m$ always turns
  $C_m$ onto the **real** axis; normalisation with `skew_main` then divides by
  $\mathrm{Im}(C_m) \approx 10^{-17}|C_m|$, which passes the $10^{-20}$ T threshold.
- Fix: for a skew main the reference is $\arg(-i\,C_m)$, so the rotation puts
  $C_m$ on the imaginary axis ($A_m$ carries the main field) and
  $c_n = 10^4\,C_n/A_m$. `process_kn_pipeline` gained a `skew_main` argument
  (default `False`, so existing calls are unchanged). Without `rot`, only the
  reported `phi_out_rad` changes meaning (angle from the skew axis).
- Earlier results affected: only runs with `skew_main=True` **and** `rot`,
  which were unusable before. None in the analysis repos.
- Verification: `test_review_fixes_2026_10.py::test_skew_main_with_rotation_*`.

### BF-2026-10-30-02 — weighted drift mode wrong even for uniform sampling
- Component: `analysis/preprocess.py` (`integrate_to_flux(drift_mode="weighted")`).
- Symptom: with perfectly uniform sampling the weighted mode (Bottura
  AII.12–14) differed from the legacy mode by $\sim4\times10^{-5}$ relative on
  $f_3$ (synthetic); on the SM18 data the integrated flux changed by up to
  $1.1\times10^{-6}$ for a flux range of $1.3\times10^{-3}$ (same units).
- Cause: the duration of the first increment of each turn was set to 0, so
  sample 0 got no drift correction and the total time was $(N_s-1)\Delta t$.
- Fix: the first increment spans the gap to the last sample of the previous
  turn when the turns are contiguous; otherwise (first turn, selected turns,
  plateau boundary) the turn's median measured interval. No timestamp is
  created. For uniform sampling weighted ≡ legacy up to the discarded DC bin.
- Earlier results affected: only `drift_mode="weighted"` (not used by any analysis repo).
- Verification: `test_preprocess.py::test_integrate_to_flux_weighted_equals_legacy_for_uniform_dt`,
  `::test_integrate_to_flux_weighted_uses_gap_to_previous_turn`.

### BF-2026-10-30-03 — dipole `cel` silently skipped with fewer than 11 orders
- Component: `kn_pipeline.py`.
- Symptom: `cel`/`fed` requested for a dipole with a kn file of $H<11$
  orders: centre stayed 0, no message.
- Cause: the dipole centring uses $C^{cmp}_{10}, C^{cmp}_{11}$; the branch
  for $H<11$ did nothing.
- Fix: new per-turn flags `LegacyKnPerTurn.cel_applied` (and `dit_applied`)
  and a diagnostic message in `LegacyKnPerTurn.warnings`. Numbers unchanged.
- Verification: `::test_dipole_cel_without_orders_10_11_is_reported`.

### BF-2026-10-30-04 — no guard against meaningless output
- Component: `kn_pipeline.py`, `utility_functions.process_kn_pipeline`.
- Symptom: the bundled FCC-ee quadrupole analysed with the then-default
  $m=1$ gave $B_{main}=1.2\times10^{9}$ T and a centre of $-18$ mm, silently.
- Cause: no plausibility checks. Three mechanisms: an order merged from a
  **bucked** Cmp channel ($|k_n^{cmp}|\approx10^{-18}$, noise divided by ~0);
  feed-down with $|z_R|\ge1$ (series diverges); a main angle within 0.05 rad of
  the $\pm\pi/2$ wrap boundary (sign of $B_m$ flips turn to turn: 93/200 turns in a
  test, mean $B_1$ 0.0035 T instead of 0.05 T; normalised units unaffected).
- Fix: diagnostics only, numbers unchanged: `LegacyKnPerTurn.warnings`,
  `phi_near_wrap`, `bucked_cmp_orders()`, and $|C_m|>20$ T flagged.
  `process_kn_pipeline` re-emits them as `UserWarning`; the GUI writes them
  to its log; `rca-analyze` prints them.
- Verification: `::test_wrap_boundary_flagged`, `::test_divergent_feeddown_and_implausible_field_reported`,
  `::test_bucked_cmp_orders_and_warning`.

### BF-2026-10-30-05 — reference radius never read; silent 17 mm
- Component: `ingest/discovery.py`, `models/catalog.py`, `models/profile.py`, GUI merge tab.
- Symptom: `AnalysisProfile.from_catalog` and the GUI used $R_{ref}=17$ mm
  whatever the file said (FCC-ee: 10 mm, Buckley: 40 mm). Every $b_n$ scales
  as $(R'/R)^{n-m}$: ×1.7 on $b_3$, ×8.4 on $b_6$ for FCC-ee.
- Cause: `Parameters.magnetAnalyzer.refRadius` was not parsed.
- Fix: parsed into `MeasurementCatalog.r_ref_m` (warning outside 1 mm – 0.5 m:
  coil-calibration folders store 1 m). `from_catalog` raises `ValueError` when
  neither the file nor the caller gives it.
- Earlier results affected: GUI/`from_catalog` users who kept the default.
  The analysis repos pass $R_{ref}$ explicitly (not affected).
- Verification: `test_analysis_profile.py::test_profile_from_catalog_requires_r_ref`,
  `::test_profile_from_catalog_takes_r_ref_and_skew_from_parameters`,
  `test_review_fixes_2026_10.py::test_discovery_reads_rref_type_skew_options`.

### BF-2026-10-30-06 — magnet order: silent and inconsistent fallbacks
- Component: discovery, profile, GUI harmonics/merge tabs.
- Symptom: `magnetOrder` is absent from almost every FFMM file; discovery and
  the Harmonics tab fell back to $m=1$, the Merge tab to $m=2$.
- Fix: $m$ is derived from `magnetInfo.magnetType` (Dipole 1, Quadrupole 2, …
  Dodecapole 6) when `magnetOrder` is absent; when both are absent the catalog
  keeps `None` with a `WARNING`, `from_catalog` raises, and the GUI refuses to
  apply kn until $m$ is entered. `magnetInfo.magnetNormSkew` fills `skew_main`;
  `magnetAnalyzer.{dit,dri,rot,cel,fed,nor}` fill `analyzer_options`.
- Verification: `::test_discovery_reads_rref_type_skew_options`, `::test_discovery_warns_on_missing_settings`,
  `test_analysis_profile.py::test_profile_from_catalog_none_magnet_order`.

### BF-2026-10-30-07 — silent samples-per-turn and shaft-speed defaults
- Component: `ingest/discovery.py`.
- Cause: missing keys fell back to 512 samples and 60 rpm without a word; newer
  FFMM files store the speed as `Measurement.MotorAngularSpeed`, not `Measurement.v`.
- Fix: `MotorAngularSpeed` is read as a fallback; both defaults now emit a `WARNING`.
  (The only binary golden dataset, SM18, carries `Measurement.v`: unchanged.)

### BF-2026-10-30-08 — ASCII streaming files not cut to whole turns
- Component: `ingest/readers_streaming.py`.
- Symptom: an ASCII (`.txt`/`.csv`) streaming file whose length is not a
  multiple of $N_s$ made `split_into_turns` raise; a wrong $N_s$ went unnoticed.
- Fix: trimmed to whole turns (as the binary path already did) and a timing
  check against the nominal speed (warning, not rejection).
- Verification: `::test_ascii_streaming_trimmed_to_whole_turns`.

### BF-2026-10-30-09 — streaming reader needed ~6× the file size in RAM
- Component: `ingest/readers_streaming.py` (`_infer_and_load`).
- Symptom: 1.17 GB SM18 file → 7.1 GB peak; an 11 GB file would need ~70 GB.
- Cause: format detection loaded the whole file once per dtype and made a
  float64 copy per column-count candidate.
- Fix: candidates are validated on read-only memory maps; only the selected
  layout is copied. Peak 4.2 GB (part of it reclaimable page cache), load
  7.6 s → 6.3 s. Output bit-identical on all five SM18 segments.

### BF-2026-10-30-10 — not installable or runnable from a fresh clone
- Component: `pyproject.toml`, `gui/physics_plots.py`, tests, notebooks, docs.
- Symptom: `pip install` failed (`tools-for-data-analysis` is not on PyPI);
  without it `build_gui()` failed and pytest stopped at collection; no
  license; `python-pptx`/`Pillow` undeclared; the `kn_from_mh_csv` tool notebook
  and one test read the untracked `golden_standards/`; `GUIDE.md` §3.4 called
  `StreamingReader().read()` without the required `shaft_speed_rpm` (TypeError).
- Fix: the companion package is the optional extra `[eddy]` and imported lazily
  in the Physics tab; extras `[pptx]`, `[dev]`; MIT `LICENSE`; the tool
  notebook uses the bundled head CSV; the eddy test skips without the extra;
  GUIDE fixed; command-line entry point `rca-analyze` with a bundled demo.
- Verification: fresh clone + clean virtual environment, see `README.md` → Installation.

### BF-2026-10-30-11 — GUI: settings, state and provenance (Harmonic Merge tab)
- Component: `gui/harmonic_merge.py`, `gui/app.py`.
- Symptoms and fixes:
  - $R_{ref}$ = 17 mm and $m$ = 2 were silent widget defaults. They now start at
    0, are pre-filled from `Parameters.txt` once per segment (and logged), and
    *Apply Kn* refuses while either is ≤ 0. `dit/dri/rot/cel/fed` are preset
    from Parameters; `nor` never is.
  - A typed $m$ was overwritten on every refresh. Pre-fill now happens only
    when the segment changes.
  - Changing $m$ between *Apply Kn* and *Apply Merge* mixed two orders in one
    result. The parameters used at *Apply Kn* are snapshotted, and the merge
    refuses if $m$, $R_{ref}$, skew or `absCalib` changed since.
  - A kn bundle loaded for one segment was applied silently to another. On a
    known segment mismatch *Apply Kn* refuses; loading a new segment clears
    the merge result and the plateau info.
  - Selecting the Ext channel filled it with Abs data while the provenance
    said Ext. The merge now refuses ("Ext channel not processed by the
    pipeline yet").
  - The outlier mask survived re-computation, so exports raised or silently
    dropped turns. All per-compute state is now reset.
  - The `n_last_turns` widget did nothing; it was removed.
  - Core diagnostics (BF-…-04) and bucked Cmp orders are now written to the log.
  - Exports gained the provenance columns `r_ref_mm`, `skew_main`, `abs_calib`
    and `pipeline_options`.
- Earlier results affected: GUI exports made with the old defaults. The
  physics call itself is unchanged: *Apply Kn* passes the same arguments as a
  direct call, and $C_{abs}$/$C_{cmp}$ are byte-identical on the Buckley degauss data.

### BF-2026-10-30-12 — GUI: Harmonics, Physics Plots, Comparison, app shell
- Harmonics tab:
  - The main order defaulted to 1 silently. It is now pre-filled from the
    segment/catalog, and compute refuses while it is unknown.
  - The raw-flux FFT coefficients (no kn, no $R_{ref}$) were labelled like
    calibrated $B_n/A_n$. Plots now say "raw flux coefficient (no kn, Wb)", and
    exports carry a `units_note` column. Column names are kept for compatibility.
- Physics Plots tab:
  - The eddy-fit import is lazy: the tab works without `tools-for-data-analysis`
    and the button says how to enable the fit.
  - The main field is $B_m$ (or $A_m$ for a skew main) instead of always the real part of $C_1$.
  - "Differential inductance" was renamed "differential transfer function dB/dI";
    it is $dB/dI$, not an inductance.
  - The branch (up/down) logic started from $I = 0$ and mislabelled bipolar
    runs. It now follows the signed current.
  - dB/dI mixed both hysteresis branches, so equal currents on the up and down
    branch gave $dI = 0$ and infinite points. It is now computed per branch,
    skipping repeated currents (checked: finite, 1 T/kA on a synthetic linear magnet).
- Comparison tab: it could not read the GUI's own merge exports (`mrg_`
  prefix) and was dipole-only (KeyError for $m = 2$). Both export layouts and
  any $m$ are now accepted; missing columns are listed in the error.
- App: tab-refresh errors were swallowed by `except: pass` and are now logged.
  Catalog `WARNING`s are shown in a red box after discovery.
- MAD outlier clip: the label said "b3" even when b3 did not exist. Labels are
  now correct, and the clip refuses when no harmonic is available.

### BF-2026-10-30-13 — documentation said FFMM rotates all orders (it does not)
- Component: `docs/06`, `docs/10` (deviation 3), `GUIDE.md`, `PARITY_REPORT.md` §5.4,
  `correction_options_reference.md` (D2, D7).
- Wrong statement: "the current FFMM C++ path rotates all orders; the off-by-one is
  only in the historical SM18 export", plus "FFMM skips feed-down for dipoles" and
  "FFMM overwrites the compensated orders with the bucking ratio".
- Truth (FFMM source `master` 10ea291f): the production native C++ path rotates
  orders 1..H-1 and applies centring and feed-down to dipoles. Skipping dipole
  cel/fed and the bucking-ratio overwrite belong to the Matlab path, which is used
  only in a unit test. Confirmed on data: FCC-ee $b_{15}$ agrees to
  $6\times10^{-18}$ units only with `legacy_rotate_excludes_last=True`.
- Fix: statements corrected; new chapter
  [`12_implementations_and_their_errors.md`](12_implementations_and_their_errors.md).
  Code unchanged.

### Known, deliberately not changed
- `absCalib` scales only the main field used for normalisation (legacy C++
  semantics), not the exported $B_n$ in tesla. Changing it would break FFMM
  parity; it is 1.0 in every Parameters file examined. Leave it at 1.0.
- FCC-ee: orders 7–14 of the bundled quadrupole sit at the numerical floor
  ($|b_n|\lesssim10^{-2}$ units) and agree with FFMM to $\lesssim10^{-5}$ units
  ($<10^{-9}$ of the main field). The cause is not established: it is not the
  rotation and not the 15-digit ASCII input. $b_{15}$ is FFMM's rotation
  off-by-one (agrees to $6\times10^{-18}$ with `legacy_rotate_excludes_last=True`),
  and $b_3$ agrees to $3\times10^{-13}$ units. Unchanged by these fixes.
- (Correction to an earlier draft of this entry: the FCC-ee residual was first
  attributed to ASCII input precision. That was checked and is wrong.)
