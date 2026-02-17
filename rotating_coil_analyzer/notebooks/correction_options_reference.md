# Correction Options Reference -- Rotating Coil Analysis Pipeline

## Pipeline Order (fixed, regardless of which options are enabled)

```
Raw df  ──►  dit  ──►  dri  ──►  FFT  ──►  Kn  ──►  rot  ──►  cel  ──►  fed  ──►  nor
             [1]       [2]       [3]       [4]       [5]       [6]       [7]       [8]
```

Steps [3] and [4] (FFT and Kn calibration) always run. The rest are optional.

---

## Standard Configuration

```python
OPTIONS = ("dri", "rot", "cel", "fed")
```

This is the correct default for **quadrupoles and higher** at any temperature
and any measurement mode.

For **dipoles** (m=1), this is also the default, but be aware that `cel`/`fed`
can produce corrupt data when the compensated-channel SNR is poor (low current,
small coils, Central PCB). See the detailed cel/fed section below. If you see
wildly wrong B_main values, try `OPTIONS = ("dri", "rot")` to disable cel/fed.

Temperature (warm vs cold) affects the Kn calibration file, not the options.

---

## Option-by-Option Guide

### `dri` -- Drift Correction + Integration

**What it does:** Removes integrator DC offset from the incremental signal,
then integrates (cumsum) to obtain flux per turn.

**Why always enable:** Integrator electronics always drift. Without correction,
the integrated flux accumulates a linear ramp that contaminates the DC
and first-harmonic content. The drift is a measurement artifact, not physics.

**Modes:**
- `legacy` (default): `df0 = df - mean(df); flux = cumsum(df0) - mean(cumsum(df))`.
  Matches the C++ analyzer exactly.
- `weighted` (Bottura AII.14): uses measured dt per sample. More accurate when sample
  timing is non-uniform, but rarely needed for encoder-triggered systems.

**Safe to enable always?** Yes.

---

### `rot` -- Rotation (Phase Alignment)

**What it does:** Rotates all harmonic phases so that the main field component
is purely real (Bn) and the skew component is purely imaginary (An).

**Why always enable:** The coil angular position relative to the magnet is never
exactly zero. Without rotation, the reported Bn and An are arbitrary mixtures
of the true normal and skew components.

**Formula:** `C_n <- C_n * exp(-i * n * phi)` where `phi = angle(C_m) / m`,
wrapped to `[-pi/2, +pi/2]` before dividing by m.

**Safe to enable always?** Yes.

---

### `cel` -- Centre Location

**What it does:** Computes the transverse offset (dx, dy) of the magnetic
centre from the coil measurement centre, expressed as a dimensionless
complex number `zR = z / R_ref`.

**cel alone does NOT modify harmonic data.** It only computes zR as a
diagnostic. The harmonics are only changed if `fed` is also enabled.

**Formulas (depend on magnet order):**

#### Quadrupole and higher (m >= 2): ROBUST

```
zR = -C_abs[m-1] / ((m-1) * C_abs[m])
```

Uses **absolute-channel** harmonics at orders m-1 and m. For a quadrupole
(m=2), this is `zR = -C_abs[1] / C_abs[2]`, i.e., the dipole component
divided by the quadrupole main field. Both are strong physical signals in
the absolute channel (typical SNR > 100). The dipole component C_1 is a
real physical signal -- it appears when the coil is off-centre in a
quadrupole field.

**Reliable whenever the main field is above noise** (i.e., not at zero current).

#### Dipole (m = 1): FRAGILE

```
zR = -C_cmp[10] / (10 * C_cmp[11])
```

Uses **compensated-channel** harmonics at orders n=10 and n=11. This is a
legacy CERN convention (Bottura AIII.4) and is fundamentally fragile because:

1. **Compensated channel suppresses the main field by design.** The whole
   point of the compensated coil is to cancel the dipole field (~2.3e15
   suppression ratio). What remains is weak residual signal.
2. **n=10 and n=11 are high-order harmonics.** Coil sensitivity (Kn) drops
   rapidly with harmonic order. For MC62: k10 ~ 4.6e-13, k11 ~ 1.9e-14.
3. **Result: pure noise at low current or with small coils.** For the MC62
   Central PCB (small coil, k1_abs = 0.032), the compensated SNR is ~3 --
   dividing two noise values produces random garbage.

**Reliable ONLY when the compensated channel has sufficient SNR at n=10,11.**
In practice, this requires large coils at high current with high FDI gain.

---

### `fed` -- Feeddown Correction

**What it does:** Corrects all harmonics for the coil being displaced from the
magnetic centre, using the binomial expansion:

```
C'_n = sum_{k=n}^{H-1} C(k,n) * zR^{k-n} * C_k
```

When the coil is off-centre, higher-order multipoles "feed down" into
lower-order ones. This correction undoes that contamination.

**The amplification danger:** Binomial coefficients grow fast. For H=15
harmonics, `C(14,7) = 3432`. If `|zR| = 0.5` (garbage from noisy cel),
the amplification factor is `3432 * 0.5^7 ~ 27x` -- high-harmonic noise
gets amplified ~27x into lower harmonics, corrupting the entire spectrum.

**When is fed safe?**

| Magnet order | cel formula | Signal source | fed safety |
|-------------|------------|---------------|------------|
| m >= 2 (quad+) | C_abs[m-1] / C_abs[m] | Absolute channel, strong | Safe at any current above noise floor |
| m = 1 (dipole) | C_cmp[10] / C_cmp[11] | Compensated channel, weak | **Dangerous** -- only safe at high current with high-SNR compensated channel |

**Rule of thumb:** If you are measuring a **dipole**, be cautious with `fed`.
If you are measuring a **quadrupole or higher**, `fed` is safe.

---

### `dit` -- di/dt Current-Ramp Correction

**What it does:** Reweights incremental flux samples to compensate for a changing
current during rotation: `w_k = I_mean / I_k`.

**When it matters:**
- Plateau turns with constant current: dI/dt ~ 0, thresholds never fire,
  correction is identity. **dit has zero effect.**
- Ramp turns with changing current: dI/dt != 0, correction improves accuracy.

**Thresholds (a turn is corrected only if both are met):**
- `signed=False` (default): `|dI/dt| > 0.1 A/s` AND `|mean(I)| > 10 A`
- `signed=True` (FFMM C++ native): `dI/dt > 0.1 A/s` AND `mean(I) > 10 A`
  (ascending positive ramps only)

**When to enable:**
- Plateau-only analysis: **not needed** (no effect).
- Per-turn analysis including ramps: **recommended**.
- FFMM parity: match whatever FFMM used, with `dit_signed=True` for C++ native.

**Safe to enable always?** Yes -- on plateau turns it's a no-op. Including it
when not needed has no cost and no risk.

---

### `nor` -- Normalization (In-Pipeline)

**What it does:** Divides ALL harmonics (n=1 through H) by the main field
component, producing dimensionless "units" (1 unit = 10^-4 of B_main):

```
C_n <- C_n * 10000 / Re(C_m)
```

**IMPORTANT: Do NOT include `nor` in the Python OPTIONS tuple.**

The Python pipeline handles normalization differently and more conveniently
than the C++ pipeline:

The C++ `nor` and the Python post-merge normalization produce mathematically
identical harmonic ratios. The difference is how B_main is handled:

| | C++ `nor` (in-pipeline) | Python (post-merge, no `nor` in OPTIONS) |
|-|------------------------|------------------------------------------|
| What gets normalized | ALL harmonics n=1..H | ALL harmonics n=1..H |
| B_main after normalization | 10000 (dimensionless) -- Tesla value lost | `C_units` has 10000, but `C_merged` retains Tesla |
| Output format | Everything in units | **Mixed**: n <= m in Tesla, n > m in units |

The Python wrapper `process_kn_pipeline()` implements the standard
Bottura Section 3.7 convention automatically:

1. Runs the pipeline **without** `nor` -- harmonics stay in Tesla.
2. Merges absolute/compensated channels -> `C_merged` (in Tesla).
3. Calls `safe_normalize_to_units()` -> `C_units` (in units).
4. `build_harmonic_rows()` selects from **both** arrays:
   - n <= m: **B_main in Tesla** (from `C_merged`)
   - n > m: **bn/an in units** (from `C_units`)

This is not mathematically better -- both produce correct harmonic ratios.
The practical advantage of the Python approach is that B_main stays in Tesla
throughout, which is needed for transfer function curves, hysteresis analysis,
quench detection, and magnet acceptance. With C++ `nor`, B_main becomes 10000
by definition and you must recover the Tesla value by dividing back.

The Bottura Section 3.7 "record" format uses this mixed convention
(Tesla for n <= m, units for n > m) as the standard way to report harmonics.

If you include `nor` in OPTIONS, the pipeline normalizes in-place, and then
`safe_normalize_to_units()` normalizes again -- **double normalization, wrong
results.** This is why `nor` must stay out of OPTIONS.

The only exception is the SM18 golden standard parity notebook, which sets
`nor_was_checked=True` to match the legacy SM18 workflow where everything
is exported in units.

---

## cel/fed Failure Modes

### The Core Problem

cel computes `zR = (numerator) / (denominator)`. When the denominator is
small (weak main field or poor SNR), the division amplifies noise into a
random large zR. fed then applies the binomial expansion with this garbage
zR, corrupting every harmonic.

### Dipole vs Quadrupole: Why It Matters

The cel formula changes with magnet order, and this changes the failure mode:

**Quadrupole (m=2):** `zR = -C_abs[1] / C_abs[2]`
- Denominator = main quadrupole field (absolute channel) -- strong signal
- Numerator = dipole component (absolute channel) -- physical, measurable
- **Fails only at very low current** (when C_2 drops below noise floor)
- Typical failure threshold: |I| < ~5-10 A (depends on magnet)

**Dipole (m=1):** `zR = -C_cmp[10] / (10 * C_cmp[11])`
- Denominator = 22-pole harmonic in **compensated** channel -- tiny signal
- Numerator = 20-pole harmonic in **compensated** channel -- equally tiny
- **Fails at moderate current** (because compensated n=10,11 are always weak)
- Typical failure threshold: |I| < ~50-100 A for Integral PCB, **always fails**
  for Central PCB with small coils (SNR ~ 3)

This is a fundamental asymmetry: quadrupole cel uses the strong absolute
channel, while dipole cel uses the weak compensated channel at very high
harmonic orders. Higher-order magnets (m >= 2) all use the absolute channel
and are similarly robust.

### When cel/fed Produces Corrupt Data

| Scenario | Magnet | Root cause | Typical symptom |
|----------|--------|-----------|-----------------|
| Low current, Central PCB | Dipole | Compensated SNR ~ 3, n=10,11 are noise | B1 = +34 T, +3065 T, -5.5e6 T |
| Low current, Integral PCB | Dipole | Compensated n=10,11 at sensitivity limit | Moderate corruption of higher harmonics |
| Very low current | Quadrupole | Absolute C_m near noise floor | Noisy harmonics (less severe) |
| End segments (multi-segment) | Any | Fringe field only ~5 mT at full current | Large ppb ratios |
| ADC glitches | Any | Corrupt sample propagates through FFT | Outlier turns |

### Real Examples from This Project

1. **MC62 Central PCB** (all dipole tests): compensated SNR ~ 3 (k1_abs = 0.032,
   suppression ~ 2.3e15). At low current, cel produces |zR| > 0.5, and fed
   amplifies this into B1 values of thousands of Tesla.

2. **SM18 end segments** (segments 1 and 5, dipole): fringe field ~5 mT at
   1740 A. Absolute agreement is machine-precision, but ppb ratios are large.

3. **SPS MBB CS/NCS** (dipole): outlier turns at specific current levels show
   extreme harmonic values -- ADC glitches amplified through cel/fed.

4. **BTP8** (quadrupole): cel/fed works correctly at all current levels because
   it uses the robust absolute-channel formula.

### Current Protection (Insufficient)

- `eps_main = 1e-20`: gates cel, but this is extremely permissive -- any
  nonzero signal passes.
- Non-finite zR → 0: catches NaN/inf, but not large finite garbage.
- **No max_zR clamp in production code.** A `max_zR = 0.01` (0.33 mm offset
  for R_ref = 33 mm) was designed but never integrated into the pipeline.

Both the Python and C++ (FFMM) pipelines produce identical corrupt values --
this is a fundamental algorithm limitation at low SNR, not a bug.

### `max_zR` Guard

The pipeline now supports a `max_zR` parameter in `compute_legacy_kn_per_turn()`
and `process_kn_pipeline()`.  When set, turns where `|zR| > max_zR` are flagged
in `result.zR_clamped` (boolean array) and their `zR` is set to 0 before
feeddown.  This prevents garbage offsets from corrupting harmonics.

**Default:** `max_zR=None` (no clamping — backward compatible).

**Recommended value:** `max_zR=0.01` (1% of R_ref, ~0.33 mm for a 33 mm coil).

### `diagnose_cel_fed()` — Interactive Diagnostic

The `diagnose_cel_fed()` function in `utility_functions.py` runs the pipeline
twice (with and without cel/fed), analyses the `|zR|` distribution, and
returns a structured `CelFedDiagnostic` result with:

- Per-turn `|zR|` values
- Count and percentage of suspect turns
- B_main comparison (with vs without fed)
- Full pipeline results from both runs
- A recommendation: `"SAFE"`, `"UNSAFE"`, or `"MIXED"`

**Recommendation logic:**
- `"SAFE"`: all turns have `|zR| <= max_zR` — cel/fed is reliable
- `"UNSAFE"`: >50% of turns have `|zR| > max_zR` — skip cel/fed
- `"MIXED"`: some turns are suspect — use `max_zR` clamp

**Usage in notebooks:**

All analysis notebooks run `diagnose_cel_fed()` early (before the main
pipeline) and conditionally strip `cel`/`fed` from `OPTIONS` if the
diagnostic says `UNSAFE`.  This ensures that the pipeline always runs
with the appropriate correction set, without manual intervention.

```python
from rotating_coil_analyzer.analysis.utility_functions import diagnose_cel_fed

# --- Run diagnostic on highest-current data ---
diag = diagnose_cel_fed(
    flux_abs_turns, flux_cmp_turns, t_turns, I_turns,
    kn=kn, r_ref=R_REF, magnet_order=1,
)

print(f"Recommendation: {diag.recommendation}")
print(f"Reason: {diag.reason}")
print(f"Suspect turns: {diag.n_suspect}/{diag.n_total}")

# --- Act on diagnostic: disable cel/fed if unsafe ---
if diag.recommendation == "UNSAFE":
    OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
    print(f"  -> cel/fed disabled, OPTIONS = {OPTIONS}")
else:
    print(f"  -> cel/fed safe, keeping OPTIONS = {OPTIONS}")
```

For notebooks with multiple PCB segments (Integral + Central), the
diagnostic runs on both and disables cel/fed if *either* is unsafe:

```python
diag_int = _cel_fed_check(runs_integral, kn_integral, "Integral")
diag_cen = _cel_fed_check(runs_central,  kn_central,  "Central")

if diag_int.recommendation == "UNSAFE" or diag_cen.recommendation == "UNSAFE":
    OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
```

**Note:** The config cell always starts with the full set
`OPTIONS = ("dri", "rot", "cel", "fed")`.  The diagnostic cell then
strips cel/fed at runtime if needed.  This way, re-running the notebook
on different data (e.g., a quadrupole) will automatically keep cel/fed
when the diagnostic says SAFE.

---

## Quick Reference: Which Options by Magnet Type

### Quadrupole and Higher (m >= 2)

```python
OPTIONS = ("dri", "rot", "cel", "fed")   # always safe
```

cel/fed is robust because it uses absolute-channel harmonics with high SNR.
Add `"dit"` only when analyzing ramp turns.

### Dipole (m = 1)

```python
# Start with full OPTIONS -- let the diagnostic decide at runtime
OPTIONS = ("dri", "rot", "cel", "fed")

# diagnose_cel_fed() will strip cel/fed if UNSAFE (see diagnostic section)
```

cel/fed is fragile for dipoles because it relies on compensated-channel
n=10 and n=11, which are weak high-order harmonics. In practice:

- **Integral PCB at |I| > ~50 A**: usually safe (Kn sensitivity sufficient)
- **Integral PCB at |I| < ~50 A**: may produce noisy offsets
- **Central PCB (small coils)**: often unreliable at any current (SNR ~ 3)

When cel/fed corrupts dipole data, you see wildly wrong B_main values
(orders of magnitude too large). All notebooks now run `diagnose_cel_fed()`
and conditionally disable cel/fed when the diagnostic says `UNSAFE`.

### Summary Table

| Measurement | Magnet | OPTIONS | dit? | cel/fed safe? |
|-------------|--------|---------|------|---------------|
| Plateau, quad+ | m >= 2 | `dri rot cel fed` | No effect | Yes |
| Plateau, dipole, high I | m = 1 | `dri rot cel fed` | No effect | Usually yes (Integral PCB) |
| Plateau, dipole, low I | m = 1 | `dri rot` | No effect | Disable fed if corrupt |
| Plateau, dipole, Central PCB | m = 1 | `dri rot` | No effect | Disable fed (low SNR) |
| Streaming + ramps | Any | `dri rot cel fed dit` | Yes | Depends on above |
| FFMM parity | Any | Match FFMM | Match FFMM | Match FFMM |

### Never varies by:
- **Temperature** (warm vs cold) -- affects Kn calibration file, not options
- **Coil geometry** -- affects Kn values, not options
- **Rotation speed** -- no impact on correction logic

---

## FFMM Configurations Used in This Project

| Magnet | Type | FFMM Options | dit | fed | Notes |
|--------|------|-------------|-----|-----|-------|
| BTP8 | Quadrupole | `dri rot nor cel fed` | OFF | ON | Stop-and-measure, dit N/A |
| SM18 HCMCBXFB012 | Dipole (cold) | `dri rot nor cel dit` | ON | OFF | dit never fires (constant I) |
| MC62 (tests 01-02) | Dipole (warm) | `dri rot nor cel fed` | OFF | ON | Plateau-averaged comparison |
| MC62 (test 03, 2Hz) | Dipole (warm) | `dri rot nor cel fed dit` | ON | ON | Streaming with ramps |

---

## Benchmark Comparison: Bottura Theory, Pentella Analyzer, FFMM

This section documents a systematic stage-by-stage comparison of our pipeline
against the three reference implementations:

- **Bottura**: L. Bottura, *Standard Analysis Procedures for Field Quality
  Measurement of the LHC Magnets - Part I: Harmonics*, MTA-IN-97-007 (1997,
  rev. 2000).  The theoretical reference.
- **Pentella**: `golden_standards/pentella_analyzer/rotcoil_lib.py`.
  A modern Python library with DC and pulsed modes.
- **FFMM**: `golden_standards/ffmm/` (C++ core + Matlab prototype).
  The CERN production analyzer.

### Stage-by-Stage Agreement

Every core algorithm is consistent across all four implementations.

#### 1. FFT Normalization

All four use the same one-sided DFT convention:

```
f_n = (2 / Ns) * FFT(flux)[n]      n = 1 .. H
```

Drop DC (index 0).  Factor of 2 folds the negative-frequency half of the
symmetric real-signal spectrum.  This follows Bottura Eq. AII.20 / AI.10.

#### 2. Kn Calibration (Conjugate-Reciprocal)

```
C_n = f_n * Rref^(n-1) / conj(kn_n)
```

- `conj(kn)` preserves the complex phase relationship between the coil
  sensitivity and the measured flux.
- `Rref^(n-1)` converts from unit-radius calibration to the physical
  reference radius.
- Bottura writes `C_n = Xi_n / (kappa_n * Rref^(n-1))`, but his `kappa_n`
  is a real scalar defined differently (it already absorbs the coil length L
  and geometric factor chi_n).  When complex kn values are used (as in all
  modern implementations), the conjugate-reciprocal form is the correct
  generalization.

| Implementation | Formula | Matches? |
|---------------|---------|----------|
| Bottura Eq. AII.22 | `Xi_n / (kappa_n * Rref^(n-1))` | Equivalent (real kappa) |
| Pentella | `Rref^(n-1) * f_n / conj(kn)` | Identical |
| FFMM (C++ & Matlab) | `(1/conj(Kn)) * Rref^(n-1) * F_n` | Identical |
| This code | `f_n * (Rref^idx / conj(kn))` | Identical |

#### 3. Drift Correction

Two modes implemented, both validated:

| Mode | Formula | Matches |
|------|---------|---------|
| `legacy` | `cumsum(df - mean(df)) - mean(cumsum(df))` | FFMM C++ line ~127 |
| `weighted` | `offset = sum(df)/sum(dt); df -= offset*dt; cumsum` | Bottura Eq. AII.14, Pentella |

Default is `legacy` for FFMM parity.  For encoder-triggered systems with
uniform timing, both modes give equivalent results.  The `weighted` mode
(Bottura AII.14) is theoretically better when sample timing is non-uniform.

#### 4. Rotation Phase Extraction

```
phi_m = angle(C_m)                        (Bottura AIV.2-3)
if phi_m >  pi/2:  phi_m -= pi            (Bottura AIV.4)
if phi_m < -pi/2:  phi_m += pi
alpha_m = phi_m / m                       (Bottura AIV.5)
```

Identical in all four implementations.  The wrapping to [-pi/2, pi/2]
resolves the m-fold rotational symmetry ambiguity.

#### 5. Rotation Application

```
C'_k = exp(-j * k * alpha_m) * C_k       (Bottura AIV.6)
```

Applied to **all** harmonics k = 1 .. H by default.

| Implementation | Range | Notes |
|---------------|-------|-------|
| Bottura AIV.6 | All orders | Theory |
| Pentella | k = 1 .. N | All harmonics |
| FFMM Matlab | k = 1 .. 15 | All harmonics |
| FFMM C++ | k = 1 .. 14 | Off-by-one (`k < nrHarmonics` with nrH=15) |
| This code (default) | k = 1 .. H | `legacy_rotate_excludes_last=False` |
| This code (SM18 parity) | k = 1 .. H-1 | `legacy_rotate_excludes_last=True` |

The FFMM C++ off-by-one is harmless in practice (harmonic H is near
Nyquist, negligible amplitude), but the `True` option exists for
exact SM18 parity when needed.

#### 6. Center Localization (CEL)

**Quadrupole and higher (m >= 2):**

```
zR = -C_abs[m-1] / ((m-1) * C_abs[m])
```

Uses the **absolute** channel, robust.  The denominator factor is
`(m-1)`, the binomial coefficient C(m-1, m-2) = m-1.

**Dipole (m = 1):**

```
zR = -C_cmp[10] / (10 * C_cmp[11])
```

Uses the **compensated** channel, fragile at low SNR (see cel/fed
section above).  The denominator factor 10 = C(10, 9) = 10.

Both formulas are the linear approximation of Bottura AIII.1-4.
All four implementations agree.

#### 7. Feeddown Correction (FED)

```
C'_n = sum_{k=n}^{H-1} C(k,n) * zR^{k-n} * C_k
```

where `C(k,n) = k! / (n! * (k-n)!)` is the binomial coefficient
with **0-indexed** k and n (array indices).  This equals `C(p-1, q-1)`
with 1-indexed harmonic orders p, q -- i.e., the Taylor expansion
coefficients of `(z + dz)^{p-1}` for field `sum C_p * (z/Rref)^{p-1}`.

Applied to **both** C_abs and C_cmp using the same zR from CEL.
All four implementations agree on the binomial formula.

#### 8. Normalization

```
scale = 10000 / Re(C_m)              (Bottura AIV.8-9)
C_n_units = C_n * scale              for all n
```

- `Re(C_m)` for normal magnets, `Im(C_m)` for skew magnets.
- All four implementations agree.
- Our pipeline applies normalization **post-merge** via
  `safe_normalize_to_units()`, not in-pipeline (see `nor` section above).

#### 9. DIT (di/dt) Correction

```
w_k = I_mean / I_k                   weight per sample
df_corrected = df * w
```

Activation thresholds:

| Mode | Slope threshold | Current threshold |
|------|----------------|------------------|
| `signed=True` (FFMM C++) | `dI/dt > 0.1` | `mean(I) > 10` |
| `signed=False` (our default) | `\|dI/dt\| > 0.1` | `\|mean(I)\| > 10` |
| Pentella | `RR > 0.1` and `curr_AVG > 10` | Same as `signed=True` |

The weight formula `I_mean / I_k` is the same in all implementations.
The `signed=False` mode extends it to negative-current ramps by using
absolute-value thresholds.

---

### Deliberate Differences from Benchmarks

These are intentional design decisions, not bugs.

#### D1. No in-pipeline `nor` (normalization)

Our pipeline runs **without** `nor` in OPTIONS.  Normalization happens
post-merge via `safe_normalize_to_units()`.  This preserves B_main in
Tesla throughout and implements the Bottura Section 3.7 "record" format
(Tesla for n <= m, units for n > m).  See the `nor` section above.

**FFMM** and **Pentella** normalize in-pipeline (`nor` in options).  The
harmonic ratios are mathematically identical; the difference is B_main
retention.

#### D2. Feeddown allowed for dipoles (with diagnostic guard)

**FFMM C++** skips feeddown entirely when `MagOrder == 1`.  Our code
applies feeddown if requested, but `diagnose_cel_fed()` flags unsafe
turns and notebooks auto-disable cel/fed when the diagnostic says
`UNSAFE`.  This is more flexible: dipole feeddown IS valid when
compensated SNR is sufficient (large coils, high current).

#### D3. No impedance gain correction in pipeline

**Pentella** applies a circuit impedance correction before harmonic
analysis: `gain = (Z_coil + Z_inst) / Z_inst`.  This compensates for
the voltage-divider effect of the coil impedance against the instrument
input impedance (typically 400 kOhm).

Our pipeline does **not** include this correction because:
- The kn calibration values can absorb the impedance factor
- The correction is hardware-specific (depends on cabling)
- For high-impedance instruments (Z_inst >> Z_coil), the gain is ~1

If needed for a specific measurement setup, the correction can be
applied during data ingestion (before the pipeline).

#### D4. No encoder-step phase offset

**Pentella** subtracts `2*pi/encStep` from the computed roll angle after
rotation, compensating for a half-sample (or one-sample) phase offset
in the FFT grid.  This is small (~6 mrad for 1024 encoder steps) and
affects only the **reported roll angle**, not harmonic amplitudes.

Neither Bottura nor FFMM include this correction.  Our code follows
the FFMM convention (no offset).

#### D5. No revolve_z transformation

**Pentella** offers a `rev` option that flips signs of specific harmonic
components (even b_n, odd a_n) for special cross-coil geometries.
This is not needed for standard rotating-coil measurements and is
not implemented.

#### D6. No external (Ext) coil channel

**Pentella** supports a 3-coil system (Abs, Cmp, Ext) where the external
coil provides an independent gradient reference.  Our pipeline handles
the standard 2-channel case (Abs + Cmp), which covers all measurements
in this project.

#### D7. No bucking ratio overwrite

**FFMM** overwrites C_cmp harmonics 1..m with the bucking ratio
`|FFT_abs[k] / FFT_cmp[k]|` after analysis.  This is a diagnostic
metric (compensation quality), not a harmonic correction.  Our code
preserves the original C_cmp values and computes merge diagnostics
separately via `recommend_merge_choice()`.

---

### Bug Fixes Triggered by This Comparison

#### BF1. DIT unsigned weight formula (fixed)

**Before:** `w_k = |I_mean| / I_k` (unsigned mode).  Produced negative
weights when current was negative (descending ramps), because only the
numerator had the absolute value.

**After:** `w_k = I_mean / I_k` (same formula for both modes).  The
signed vs unsigned distinction is **only** in the activation thresholds,
not in the weight formula.  This matches the physics: the ratio
`I_mean / I_k` is always positive when all samples have the same sign
(guaranteed by the `min(|I|) > eps` check).

**Impact:** Only affected unsigned mode with negative currents.  No
golden standard tests use this path (FFMM uses signed mode, Pentella
uses positive-only thresholds).  A regression test was added:
`test_di_dt_weights_negative_current_gives_positive_weights()`.

---

### Verification Summary

| Pipeline Stage | Bottura | Pentella | FFMM | This Code |
|---------------|---------|----------|------|-----------|
| FFT: `2*FFT/N` | Eq. AII.20 | `2*fft/N` | `2*fft/N` | `2*fft/N` |
| Kn: `f/conj(kn)*R^(n-1)` | Eq. AII.22 | `conj(kn)` | `conj(kn)` | `conj(kn)` |
| Drift (legacy) | -- | -- | C++ line ~127 | Matches C++ |
| Drift (weighted) | Eq. AII.14 | `sum(df)/sum(dT)*dT` | -- | Matches Bottura |
| Phase wrap [-pi/2,pi/2] | Eq. AIV.4 | Same | Same | Same |
| Rotation: `exp(-j*k*phi)` | Eq. AIV.6 | All k | All k (Matlab) | All k (default) |
| CEL (m>=2): `C[m-1]/((m-1)*C[m])` | AIII.4 | Same | Same | Same |
| CEL (dipole): `C10/(10*C11)` | AIII.3 | Same | Same | Same |
| Feeddown: `binom(k,n)*zR^(k-n)` | AIII.6 | Same | Same | Same |
| Normalization: `10000/B_main` | Eq. AIV.8 | Same | Same | Same |
| DIT: `I_mean / I_k` | -- | Same | Same | Same |
