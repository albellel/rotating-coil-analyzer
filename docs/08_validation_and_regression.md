# 8. Validation, regression tests and traceability

Code: `rotating_coil_analyzer/tests/` (149 tests), `validation/golden_streaming.py`,
`golden_standards/`, `scripts/btp8_bruteforce_turns.py`. Numbers and
verdicts: `PARITY_REPORT.md` (canonical — not repeated here).

## 8.1 Principle

The analyzer is part of a metrological chain: a silent change of convention,
ordering or normalisation is a **trueness** error that cannot be detected
afterwards from the data. Hence: deterministic pipeline, explicit options,
immutable reference datasets, and the rule

> if a numerical change is intentional, a test must be updated explicitly;
> if it is not, a test must fail.

## 8.2 Unit and integration tests

```bash
python -m pytest rotating_coil_analyzer/tests/ -x -q        # 149 passed
```

| Area | Files |
|---|---|
| Ingest: discovery, readers, channel detection, time policy, plateau layouts (incl. H/V steerer) | `test_discovery.py`, `test_reader.py`, `test_channel_detect.py`, `test_time_policy.py`, `test_plateau.py`, `test_plateau_hv_format.py`, `test_preview.py` |
| Preprocessing: `dit`, `dri` legacy/weighted, integration, slope | `test_preprocess.py` |
| Turns and DFT correctness | `test_turns_fourier.py` |
| $k_n$ loading, synthetic end-to-end pipeline, head CSV vs reference $k_n$, compensation-scheme metadata | `test_kn_loader_and_merge.py`, `test_kn_synthetic_pipeline.py`, `test_kn_head_csv_vs_reference.py`, `test_mh_csv_compensation_scheme.py`, `test_kn_bundle.py` |
| Merge modes, recommendation, normalisation, safety guards (zero/NaN/weak main field, `max_zR`) | `test_harmonic_merge.py`, `test_safety_guards.py` |
| `AnalysisProfile` defaults, immutability, catalog integration | `test_analysis_profile.py` |
| Streaming utilities: plateau detection, `process_kn_pipeline`, FDI diagnostic, statistics | `test_streaming_utilities.py`, `test_new_utility_functions.py` |
| GUI widgets and events (debounce, plot lifecycle) | `test_gui_plot_widgets.py`, `test_plot_debounce.py`, `test_plot_clear_pattern.py` |

Synthetic data are used where an analytical expectation exists (a pure
$\cos n\theta$ flux must return order $n$ with the $2/N_s$ amplitude; a
known $z_R$ must be recovered by `cel`); real golden datasets for end-to-end
parity.

## 8.3 Golden reference datasets (`golden_standards/`)

Immutable; never regenerated; used only for comparison.

| Folder | Magnet / source | Reference output |
|---|---|---|
| `golden_standard_SM18_01` | SM18 dipole HCMCBXFB012, streaming, 5 segments | FFMM C++ per-turn `*_results_Ap_1_Seg_N.txt` |
| `golden_standard_01_LIU_BTP8` | LIU BTP8 quadrupole, plateau | legacy C++ `BTP8_*_results.txt` |
| `ffmm` | LEAR MC62 FFMM configuration + average results | `MC62_*_Average_results.txt` |
| `degaussing_test` | Buckley H/V steerer degauss staircase (74 runs) | FFMM `*_Main_results.txt` |
| `measurement_heads` | head geometry CSVs + reference $k_n$ | legacy $k_n$ TXT |
| `pentella_analyzer` | Pentella Python reference implementation | code, for cross-reading |
| `GSI_LDM`, `example_results` | additional reference material | — |

Outcomes (pointers): SM18 streaming $B_1$ max $|\Delta| \approx 1.8\times10^{-12}$ T
over 285 095 turns with `("dri","rot","nor","cel")`, `legacy_rotate_excludes_last=True`;
LIU BTP8 at the floating-point floor after brute-force recovery of the
undocumented turn selection (`scripts/btp8_bruteforce_turns.py`); MC62 limited
to ~µT by the reference's averaging window, not by the pipeline; Buckley
degauss worst $|\Delta| \approx 10^{-16}$ T. Details, tolerances and the
equation-by-equation proof: `PARITY_REPORT.md` §1–5, §8;
`notebooks/Buckley_steerer/2026-06-23_degauss_parity/report.md`.

> **Corrected from the old LaTeX.** The LaTeX listed "the MATLAB
> implementation" among the legacy references. The references actually used
> are the FFMM C++ analyzer (`MatlabAnalyzerRotCoil.cpp` is a C++ file despite
> its name), the historical SM18 C++ export, and the Pentella Python scripts.

## 8.4 `validation/golden_streaming.py`

`run_golden_folder(folder, config=GoldenRunConfig(...))`: discover the golden
streaming dataset (`_resolve_catalog_root`), auto-match the per-segment $k_n$
file and the C++ reference export, run the full per-turn pipeline, build the
canonical output table (`_build_output_table`, legacy column layout), and
compare per order against the reference with robust fixed-width parsing and
time alignment with positional fallback (`compare_units_table` →
`ComparisonSummary`). A CLI `main()` accepts the option set as CSV
(`_parse_options_csv`) so parameter sweeps (the way the SM18 configuration
was found) are scriptable.

## 8.5 Tolerances

Exact equality is not the criterion; agreement is assessed per order,

$$
\lvert B_n^{\mathrm{new}} - B_n^{\mathrm{ref}}\rvert \le \varepsilon_B, \qquad
\lvert A_n^{\mathrm{new}} - A_n^{\mathrm{ref}}\rvert \le \varepsilon_A ,
$$

with $\varepsilon$ set in the test or notebook from the float64 rounding floor
of the reference file format (e.g. `%Le` long double in the C++ $k_n$ files),
the compensated-channel sensitivity and the physical relevance of the order.
Where the reference itself is an average over an undocumented window (MC62),
the comparison is at the plateau-average level and the tolerance is the
reference's own scatter.

## 8.6 Traceability of results

Every export carries: pipeline options and drift mode (`provenance_columns`,
`format_preproc_tag`), $k_n$ source and connections (`KnBundle`), merge mode
and per-order source map (`MergeResult`), $R_{ref}$, $m$, and the
`AnalysisProfile` as JSON (`to_dict`). A table or plot can be regenerated
exactly from its metadata.

## 8.7 Diagnosing a discrepancy

Because every stage is explicit and ordered, localise rather than tune:

1. compare the `*_db` snapshots (after $k_n$, before any correction) —
   disagreement here is ingest, `dit`, `dri` or $k_n$;
2. disable the merge (`abs_all` / `cmp_all`) and compare channels separately;
3. toggle `rot`, then `cel`+`fed`, then `nor` one at a time;
4. check the sign-convention knobs (chapter 6.4): even-order flip ⇒ encoder
   offset, all-order flip ⇒ polarity, last-order-only ⇒
   `legacy_rotate_excludes_last`;
5. for plateau references, suspect the turn selection / averaging window of
   the reference before the equations (`PARITY_REPORT.md` §2.3).
