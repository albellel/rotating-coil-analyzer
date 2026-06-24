# Analysis Pipeline — End to End: What, Where, and Why

This is the **conceptual companion** to the two lookup references:

- [`correction_options_reference.md`](correction_options_reference.md) — option-by-option
  knobs, `cel`/`fed` failure modes, sign conventions, and the Bottura/FFMM/Pentella benchmark.
- [`physics_reference.md`](physics_reference.md) — transfer function, eddy currents,
  NMR comparison, fringe-field sextupole, integrated field.

Here we follow the signal from the raw acquisition all the way to the reported
harmonics, explaining **what each stage does, where it sits in the chain, and
why it must sit there**. Formulas follow the implementation in
`analysis/kn_pipeline.py` and `analysis/preprocess.py`.

---

## 0. The physical principle

A rotating coil spins about an axis inside the magnet. As it turns through angle
$\theta$ it links a magnetic flux $\Phi(\theta)$. For a 2-D multipole field the
flux is a sum of harmonics,

$$
\Phi(\theta) \;=\; \mathrm{Re} \left\{ \sum_{n\geq 1} S_n\, C_n \, e^{\,i n \theta} \right\},
$$

where $C_n = B_n + i A_n$ is the complex multipole coefficient we want
($n=1$ dipole, $2$ quadrupole, $3$ sextupole, …), $B_n$ is the *normal* and
$A_n$ the *skew* component, and $S_n$ is the coil's geometric sensitivity to
order $n$. **The whole pipeline is one idea**: recover $\Phi(\theta)$, take its
Fourier transform, and turn each Fourier coefficient into a physical $C_n$.

The reason the Fourier transform works is that the harmonics of $\Phi(\theta)$
are in one-to-one correspondence with the multipoles: the $n$-th Fourier
coefficient of the flux is proportional to $C_n$.

---

## 1. What the hardware gives you — and the integrate-to-flux step

The digital fluxmeter (FDI) does **not** output $\Phi(\theta)$. It outputs flux
**increments** between consecutive angular samples:

$$
\mathrm{d}\!f[k] \;\approx\; \Phi(\theta_k) - \Phi(\theta_{k-1}).
$$

So the very first thing the analysis must do is **reconstruct the flux by
accumulating the increments** (the GUI tick *"Integrate differential signal to
flux"*):

$$
\Phi[k] \;=\; \sum_{j \le k} \mathrm{d}\!f[j] \;=\; \texttt{cumsum}(\mathrm{d}\!f).
$$

### Why `cumsum` and not trapezoidal integration

The project rule is *"always integrate continuous signals with scipy
trapezoid."* That rule is for raw **voltage** $v(t)$, where the flux is a true
time-integral $\Phi=\int v\,\mathrm{d}t$ and the trapezoid rule matters at the
$10^{-4}$ level. Here the FDI has **already performed that integral in
hardware**: each sample *is* a $\Delta\Phi$. Reconstructing $\Phi$ from
increments is an exact cumulative **sum**, not a quadrature — using trapezoid
here would be wrong (it would average adjacent increments). This is the one
documented exception to the trapezoid rule, and it is why this step is a plain
`cumsum`.

This step is **mandatory** (default ON). Skipping it would FFT the increments
instead of the flux and yield the wrong spectrum (each harmonic would be
multiplied by a spurious $\sim i n$ factor).

---

## 2. The chain, in order

```
 raw increments  df_abs, df_cmp   (+ measured t, I per turn)
   │
 [1] di/dt correction        (dit)   — on the increments, BEFORE integrating
   │
 [2] integrate to flux  +  drift correction   (dri)        — cumsum
   │
 [3] FFT          f_n = 2·FFT(Φ)/Ns ,  drop DC, keep n = 1..H
   │
 [4] apply Kn     C_n = f_n / conj(k_n) · Rref^(n-1)
   │
 [5] rotation                (rot)   — phase reference from the CALIBRATED main
   │
 [6] centre location         (cel)   — magnetic-centre offset zR
   │
 [7] feed-down                (fed)   — re-expand harmonics about that centre
   │
 [8] merge abs / cmp                  — best channel per order
   │
 [9] normalise to "units"    (nor)   — ratio to the main field   (LAST)
```

Steps [3] and [4] always run; the rest are optional tokens in
`OPTIONS = (...)`. The order is **fixed** — it is not a free choice, because
each stage consumes the previous stage's output (see the rationale boxes below).

### Where each stage lives in the GUI

| Stage | GUI location |
|-------|--------------|
| [1] di/dt, [2] integrate + drift | **Harmonics tab** tick-boxes *and* inside the **Harmonic Merge** tab |
| [3] FFT | both tabs (Harmonics shows the raw spectrum) |
| [4] Kn, [5] rot, [6] cel, [7] fed, [8] merge, [9] nor | **Harmonic Merge tab** (`OPTIONS` + merge preset) |

The **Harmonics tab stops at [3]** — it shows the *raw, un-calibrated* spectrum
(no Kn), which is why it does not depend on Coil Calibration. The full
calibrated chain [4]–[9] runs in the **Harmonic Merge tab**.

---

## 3. Stage by stage

### [1] `dit` — di/dt (current-ramp) correction *(first, before integrating)*

**What.** Reweights each increment by the ratio of the turn-mean current to the
instantaneous current,

$$
w_k = \frac{I_\text{mean}}{I_k}, \qquad \mathrm{d}\!f_k \leftarrow w_k\,\mathrm{d}\!f_k .
$$

**Why.** One revolution takes a finite time ($\sim$0.5 s here). If the magnet
**current is ramping** during that turn, the field is a moving target: the flux
changes partly because the coil rotated and partly because $I$ changed. That
contaminates every harmonic. The reweight rescales each sample to the flux it
*would* have had at the turn-mean current, removing the **quasi-static,
current-proportional** part of the within-turn change.

**Activation (a turn is "on a ramp").** $|\mathrm{d}I/\mathrm{d}t| > 0.1$ A/s and
$|\overline{I}| > 10$ A. On a plateau $I$ is constant $\Rightarrow w_k \approx 1$
and `dit` is a **no-op**. The signed variant (`dit_signed=True`) uses the FFMM
C++ activation ($\mathrm{d}I/\mathrm{d}t>0.1$, $\overline{I}>10$, ascending only).

**Limit.** It removes only the *linear* current effect. It does **not** remove
eddy currents — those lag the drive and carry their own harmonic content (see
[`physics_reference.md`](physics_reference.md) §2).

**When OFF.** Plateau/staircase data and FFMM parity here use only `dri`+`rot`,
so `dit` is OFF.

> **Why first?** The reweight acts on the *increments*, and it must happen
> before they are summed into flux — once summed, you can no longer reweight
> individual samples by their current.

### [2] `dri` — drift correction (and the integration itself)

The integration ([cumsum](#1-what-the-hardware-gives-you--and-the-integrate-to-flux-step))
always happens. `dri` adds **drift correction** on top.

**What/why.** The FDI integrator carries a tiny DC offset per increment.
Negligible per sample, but `cumsum` turns a constant bias into a **linear ramp
in the flux** — a fake baseline that leaks directly into the low-order
harmonics (above all $n=1$). Drift correction removes the per-turn offset so the
integrated flux **closes** over one revolution, as a periodic field requires.
It is a **single constant per turn** — never a polynomial detrend (that is a
hard project rule).

**The two drift modes — the actual difference.**

- **Legacy (C++) — uniform $\Delta\theta$.** Assumes equal time spacing of the
  angular samples:
  $$
  \Phi \;=\; \texttt{cumsum}\!\big(\mathrm{d}\!f - \overline{\mathrm{d}\!f}\big)
            \;-\; \overline{\texttt{cumsum}(\mathrm{d}\!f)} .
  $$
  Subtracting $\overline{\mathrm{d}\!f}$ forces the increments to sum to zero
  (the flux closes); the second term re-centres the flux. A subtlety
  reproduced for bit-exact C++ parity: the re-centring uses the mean of the
  **original** cumsum (before drift removal), not of the corrected one. Use this
  for **FFMM parity** and clean/uniform sampling.

- **Weighted (Bottura) — $\Delta t$-weighted.** Uses the **measured** time to
  spread the offset in proportion to each sample's duration:
  $$
  \mathrm{d}\!f_k \;\leftarrow\; \mathrm{d}\!f_k
     - \frac{\sum_j \mathrm{d}\!f_j}{\sum_j \Delta t_j}\,\Delta t_k,
  \qquad \Phi = \texttt{cumsum}(\mathrm{d}\!f).
  $$
  This is the right choice when the samples are **not** equally spaced in time
  (encoder jitter, variable shaft speed) — it puts the correct amount of
  correction on each sample.

**Rule of thumb.** *Legacy* for C++ parity / uniform sampling; *Weighted* when
$\Delta t$ visibly varies within a turn. (This dataset → Legacy.)

> **Why second?** Drift is a property of the *integrated* signal (it only
> appears once you `cumsum`), so it is bound to the integration step and must
> come after `dit` (which needs the raw increments).

### [3] FFT — extract the harmonics

$$
f_n \;=\; \frac{2}{N_s}\,\mathrm{FFT}(\Phi)\big|_n, \qquad n = 1,\ldots,H,
$$

dropping the DC ($n=0$) term. The factor $2/N_s$ is the standard single-sided
amplitude normalisation. This converts the angular flux profile $\Phi(\theta)$
into complex harmonic coefficients. The FFT is defined on the **sample/angle
index**, not on time — consistent with the project's *no synthetic time* rule
(time is never used to build the angular grid).

### [4] Kn — calibration to a physical multipole

$$
C_n \;=\; \frac{f_n}{\overline{k_n}}\; R_\text{ref}^{\,n-1},
$$

where $k_n$ is the coil's complex **sensitivity** to harmonic $n$ (from geometry
/ calibration), $\overline{k_n}$ its conjugate, and $R_\text{ref}$ the reference
radius (here $0.04$ m). Dividing by $k_n$ turns "flux the coil saw" into "field
multipole that produced it"; the $R_\text{ref}^{\,n-1}$ factor references every
order to a common radius. This is done **separately for the absolute and
compensated channels** — each has its own $k_n$. (For an A-C compensated dipole,
$|k_n^\text{cmp}|$ at $n=1$ is tiny because the wiring bucks the dipole — see
the Buckley degauss notes.)

> **Why here?** Everything downstream (rotation reference, centre, feed-down,
> normalisation) is defined in terms of the **physical** field, so calibration
> must precede them.

### [5] `rot` — rotation (phase alignment) *(after Kn)*

The coil begins each acquisition at an **arbitrary azimuth**, so all measured
phases carry an unknown offset $\varphi$. Rotation removes it. It reads the
reference angle from the **main harmonic's phase**,

$$
\varphi \;=\; \frac{1}{m}\,\mathrm{wrap}_{[-\pi/2,\,\pi/2]}\!\big(\arg C_m\big),
$$

and rotates every order:

$$
C_k \;\leftarrow\; C_k \, e^{-\,i k \varphi}, \qquad k = 1,\ldots,H .
$$

This aligns the result to the magnet's own symmetry axis, so that
$B_n=\mathrm{Re}\,C_n$ (normal) and $A_n=\mathrm{Im}\,C_n$ (skew) are physically
meaningful instead of depending on where the coil happened to start.

> **Why after Kn?** $k_n$ is complex and shifts the phase, so the *magnet's*
> physical angle lives in the **calibrated** coefficient $C_m$, not in the raw
> FFT coefficient. Taking the reference before Kn would fold the coil's
> calibration phase into $\varphi$.

Applied to all orders $k=1..H$ (Bottura Eq. AIV.6). The legacy SM18/FFMM C++ has
an off-by-one that **skips the last order** ($k=H$); set
`legacy_rotate_excludes_last=True` only to reproduce that export. See
[`correction_options_reference.md`](correction_options_reference.md) (`rot`
section) for the encoder-offset interaction and the $\pm\pi/2$ wrap caveat.

### [6] `cel` — centre location

The multipole expansion is taken about the coil's **rotation** axis, which is
not exactly the **magnetic** centre. `cel` estimates the (complex, dimensionless)
offset $z_R$ from the harmonics:

$$
\text{quadrupole+}\;(m\ge 2):\quad z_R = -\frac{C_{m-1}}{(m-1)\,C_m},
\qquad
\text{dipole}\;(m=1):\quad z_R = -\frac{C_{10}^\text{cmp}}{10\,C_{11}^\text{cmp}} ,
$$

with the physical centre $z = R_\text{ref}\,z_R$, $x=\mathrm{Re}\,z$,
$y=\mathrm{Im}\,z$. The dipole branch leans on **weak, high-order compensated**
terms ($n=10,11$), which is why it is fragile at low field / poor SNR. The
`max_zR` clamp and `diagnose_cel_fed()` guard against this — see the dedicated
cel/fed failure-mode section in
[`correction_options_reference.md`](correction_options_reference.md).

### [7] `fed` — feed-down *(needs cel)*

Once the centre offset $z_R$ is known, feed-down **re-expands** the harmonics
about the true magnetic centre:

$$
C_n' \;=\; \sum_{k=n}^{H} \binom{k}{n}\, z_R^{\,k-n}\, C_k .
$$

This removes "feed-down" contamination, where a higher multipole measured
**off-centre** masquerades as a lower order. It requires `cel` first (it needs
$z_R$). Because it builds on the fragile dipole `cel`, `fed` inherits that
fragility for $m=1$.

> **Why [6] then [7]?** You must locate the centre before you can re-expand
> about it. Both are OFF for the FFMM golden on the Buckley dataset.

### [8] Merge abs / cmp — best of both channels

The **compensated** channel has the main field bucked away, so it has far better
signal-to-noise on the **small** harmonics; but the **main** field itself must
come from the **absolute** channel (the compensated one discarded it). The merge
therefore takes **main from Abs, higher orders from Cmp**
(`abs_main_cmp_others` / `abs_upto_m_cmp_above`). The per-order choice is
recorded for traceability.

### [9] `nor` — normalisation to "units" *(last, after merge)*

"Units" expresses each harmonic as a **ratio** to the main field,

$$
b_n + i a_n \;=\; 10^{4}\,\frac{C_n}{C_\text{main}},
$$

with $C_\text{main}=\mathrm{Re}\,C_m$ (or $\mathrm{Im}\,C_m$ for a skew magnet).
It is applied **last**, and **after the merge**, for two reasons:

1. It needs the **final, fully-corrected** harmonics (post rot/cel/fed) so the
   ratio reflects the true field.
2. It must divide by the **single merged** main field. Normalising before the
   merge would scale Abs and Cmp each by their own main — inconsistent. After
   merging there is one agreed main field to normalise everything to.

Normalisation is a pure presentation rescaling (Tesla $\to$ dimensionless), so
doing it last keeps all the physics in the Tesla coefficients. In this analyzer
the wrapper keeps `C_merged` in Tesla and produces `C_units` separately; the
export then uses the **Bottura §3.7 mixed format**: Tesla for $n\le m$, units
for $n>m$. When the FFMM golden is generated with `nor` OFF (as on the Buckley
degauss data), the output stays in Tesla and the parity test compares
$B_n/A_n$ in Tesla directly.

---

## 4. The ordering logic in one paragraph

Increments → (optionally undo the within-turn current ramp, `dit`) → sum to
flux while removing the integrator baseline so the loop closes (`dri`) → FFT to
get harmonics → calibrate to the real field (`Kn`) → align the phase using the
**calibrated** main (`rot`) → locate the magnetic centre (`cel`) → re-expand
about it (`fed`) → combine each channel's best part (merge) → finally express as
ratios to the main field (`nor`). Each step consumes the previous step's output,
which is exactly why the order is fixed — and why normalisation is the **last**
thing, never the first.

---

## 5. See also

- Option knobs, `cel`/`fed` failure modes, sign conventions, FFMM/Pentella
  benchmark: [`correction_options_reference.md`](correction_options_reference.md)
- Transfer function, eddy/multi-$\tau$ settling, NMR, fringe sextupole:
  [`physics_reference.md`](physics_reference.md)
- Hands-on API walkthrough: [`../../GUIDE.md`](../../GUIDE.md)
- Theory source: Bottura, *Standard Analysis Procedures for Field Quality
  Measurement of the LHC Magnets — Part I: Harmonics* (in `theory/`).
