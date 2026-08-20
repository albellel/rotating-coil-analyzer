# 5. Coil sensitivity $k_n$: sources, head geometry, channels

Code: `analysis/kn_pipeline.py` (`SegmentKn`, `load_segment_kn_txt`),
`analysis/kn_head.py` (`CoilGeom`, `HeadKnData`, `compute_head_kn_from_csv`,
`parse_connection`, `compute_segment_kn_from_head`, `write_segment_kn_txt`),
`analysis/kn_bundle.py` (`KnBundle`). Bottura counterpart: Eq. 13, 19–22;
finite-winding generalisation: Deniau MTA-IN-98-021 (`deniau1998`,
`deniau1998b`).

## 5.1 Definition used by the code

The calibrated harmonic is (chapter 4.3)

$$
C_n = \frac{R_{ref}^{\,n-1}}{\overline{k_n}}\,f_n ,
$$

so $k_n$ plays the role of Bottura's complex sensitivity $\kappa_n$
(Eq. 19, units $\mathrm{m}^{n+1}$), up to the conjugation convention of the
legacy files. $k_n$ encodes the effective turns $\times$ length, the winding
radii (so the $r^{n-1}$ growth with order), the coil orientation (radial:
real; tangential: a phase $e^{-i n\pi/2}$), and the electrical connection of
the coils in a channel. $R_{ref}$ is **not** part of $k_n$ — it is applied
separately, once.

> **Corrected from the old LaTeX.** The LaTeX listed "the reference radius
> used for normalisation" among the quantities $k_n$ encapsulates. In this
> package $k_n$ is $R_{ref}$-independent; the same segment TXT is valid for
> any $R_{ref}$.

## 5.2 Two channels, two sensitivities

The head is read through two channels with independent sensitivities:

- **absolute** (`kn_abs`): a single coil (or a sum), sensitive to the main
  field; used for $n \le m$ and for the rotation reference;
- **compensated / bucked** (`kn_cmp`): coils combined with signs/gains so that
  the main order (and usually $m-1$) cancels; used for $n > m$ (Bottura
  Eq. 22, §3.7).

An optional third **external** channel (`kn_ext`) is read from 6-column files
and carried in `SegmentKn`/`KnBundle`, but `compute_legacy_kn_per_turn`
processes only abs and cmp (gap vs Pentella, `DOCUMENTATION.md` §13.2 F).
The two channels stay separate through calibration, rotation, centring and
feed-down; they are combined only by the explicit merge step (chapter 7).

The **compensation scheme label** (e.g. `"A-C"`, `"ABCD"`) is metadata
(`KnBundle.extra["compensation_scheme"]`, `MergeResult.compensation_scheme`).
It is not inferable from a head CSV and does not influence the numbers.

## 5.3 Source 1 — segment $k_n$ TXT file

`load_segment_kn_txt(path)` reads whitespace-separated rows, one per order
$n = 1,\ldots,H$:

```
Abs_Re  Abs_Im  Cmp_Re  Cmp_Im  [Ext_Re  Ext_Im]
```

Lines starting with `#` are comments; non-finite values raise. This is the
format the legacy C++ analyzer and FFMM write (`Kn_values_Seg_<seg>.txt`), and
the one to use when parity with a legacy result is required.

## 5.4 Source 2 — measurement-head geometry CSV

`compute_head_kn_from_csv(csv_path, warm_geometry=True, n_multipoles=15,
use_design_radius=True, strict_header=True)` mirrors the legacy C++
`loadHeadKn` / `calculateHeadKn`. The CSV must carry the CERN head header
(`LEGACY_HEAD_HEADER`: Measurement Head, Array/Coil Position, Number of
Turns, Coil Inner Width/Length [m], Winding Thickness [m], Magnetic Surface
[m], Radius (calibrated) [m], Alpha/Beta/Tilt [rad], Radius (design) [m],
Magnetic Surface (Single Coil calibration) [m], Z position …).

Per coil (`CoilGeom`: $N_t$, $W_{in}$, $L_{in}$, $T$, $S$, $r_0$, $\alpha$,
$\beta$, $\phi$, $p_Z$):

1. **Geometry scaling.** Warm: $W_{in} \times 0.999999$, others unchanged.
   Cold: $W_{in}, L_{in}, r_0, p_Z \times 0.997$ and $S \times 0.995$ (thermal
   contraction factors hard-coded as in the C++).
2. **Radius.** The calibrated radius is used; if missing and
   `use_design_radius=True`, the design radius. Missing magnetic surface
   falls back to the single-coil calibration surface.
3. **Effective winding width and length** from the magnetic surface,
   $w = \tfrac12\big[(W_{in}-L_{in}) + \sqrt{(W_{in}-L_{in})^2 + 4S/N_t}\big]$,
   $L = \tfrac12\big[(L_{in}-W_{in}) + \sqrt{(L_{in}-W_{in})^2 + 4S/N_t}\big]$.
4. **Filament positions** $z_0 = r_0 e^{i\alpha}$, $z_{1,2} = z_0 \mp
   \tfrac{w}{2}e^{i\phi}$, with the finite-winding offsets
   $\Delta Z = \tfrac12(w - W_{in}) + i\,\tfrac{T}{2}$ and the tilt-rotated
   $Z_{a1,2} = z_{1,2}e^{-i\phi}$.
5. **Finite-winding factor** $\xi_n$ (`_csi_n`, built on the legacy
   `_gamma_function`) — the Deniau correction for a rectangular winding
   cross-section; for a thin coil it reduces to $z_2^{\,n} - z_1^{\,n}$
   (Bottura's $\chi_n$, Eq. 13).
6. **Orientation factor.** $\alpha \approx 0$ or $\pi$ (radial): $1$;
   $\alpha \approx \pm\pi/2$ (tangential): $e^{-i n \pi/2}$; anything else
   raises ("not tangential or radial") — the automatic path does not handle
   arbitrary orientations.
7. **Sensitivity** $k_n = \dfrac{N_t L}{n}\,\xi_n\,\times(\text{orientation})$,
   $n = 1,\ldots,$ `n_multipoles` — Bottura Eq. 19 with $\chi_n \to \xi_n$.

Result: `HeadKnData.kn_by_index[(array_pos, coil_pos)]`, plus the magnetic
length and $Z$ position per coil.

### Connections

`parse_connection("1.1-1.3+2*1.2")` → list of (coefficient, `(array, coil)`)
terms; `compute_segment_kn_from_head(head, abs_connection=…,
cmp_connection=…, ext_connection=None)` forms each channel as the linear
combination $k_n^{ch} = \sum_s g_s\,k_n^{(s)}$ — Bottura Eq. 22 with the gains
$g_s$ being the connection coefficients. `write_segment_kn_txt` exports the
result in the TXT format of §5.3 so it can be reused or compared with a
legacy file (`tests/test_kn_head_csv_vs_reference.py` does exactly this
against `golden_standards/measurement_heads/`).

## 5.5 Provenance — `KnBundle`

Whatever the source, the Coil Calibration tab (and the programmatic API) wraps
the `SegmentKn` in a frozen `KnBundle` with `source_type`
(`"segment_txt"` | `"head_csv"`), `source_path`, ISO timestamp, segment and
aperture ids, the connection strings, `head_warm_geometry`,
`head_n_multipoles` and free `extra` metadata. `to_metadata_dict()` flattens
this into CSV-header/JSON keys (`kn_source_type`, `kn_head_abs_connection`,
…) so every exported harmonic table states which calibration produced it.

## 5.6 Magnet parameters used in this project

| Magnet | $R_{ref}$ | $m$ | $N_s$ | Source of $k_n$ |
|---|---|---|---|---|
| SM18 (HCMCBXFB012) | 50 mm | 1 | 512 | segment TXT from the golden folder |
| LIU BTP8 | 59 mm | 2 | 512 | segment TXT |
| LEAR MC62 | 33 mm | 1 | 1024 / 512 | segment TXT |
| SPS MBB | 20 mm | 1 | 1024 | per-segment TXT from the session directory |
| Buckley steerer (degauss) | 40 mm | 1 | 1024 | `Kn_values_Seg_Main.txt` (A–C bucked) |

(Campaign-specific values; the physics lives in the respective analysis repos.)
