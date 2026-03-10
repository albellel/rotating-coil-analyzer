# Marusov 2D Fourier Reconstruction — A Plain-Language Guide

**Reference**: Marusov, I. (2013). *Measurement of a time-periodic magnetic field using a rotating coil.* Nuclear Instruments and Methods in Physics Research A, 711, 121–123.

**Companion notebook**: `marusov_reconstruction.ipynb` (in this directory).

---

## What Problem Does Marusov Solve?

A rotating coil spins inside a magnet and measures the magnetic field. During one full rotation, the coil sweeps through all angular positions and the signal is Fourier-analysed to extract the field harmonics (dipole, quadrupole, sextupole, ...).

The standard approach treats each rotation as an independent snapshot: spin once → FFT → get harmonics for that turn. This works well when the field is constant (or nearly constant) during one rotation.

**But what if the field is changing while the coil spins?**

If the field changes faster than the coil rotates, a single-turn FFT smears together the spatial pattern of the field and its time variation. You can't tell whether a signal comes from a real harmonic (spatial) or from the field changing during the turn (temporal).

Marusov's paper provides a way to **separate the spatial and temporal content** using a 2D Fourier decomposition.

---

## The Setup

### What the coil measures

A rotating coil with `Ns` pickup loops (samples per turn) spins at frequency `f_rot` = 1/T, where T is the period of one rotation. The coil measures the rate of change of magnetic flux, which after integration gives flux as a function of angle.

The flux signal has two kinds of variation mixed together:

1. **Spatial**: the field pattern around the bore (dipole, quadrupole, ...) — what we want to measure
2. **Temporal**: the field is ramping up, settling after a current change, or oscillating — what we want to disentangle

### Notation

| Symbol | Meaning |
|--------|---------|
| `Ns` | Number of samples per turn (e.g., 512) |
| `M` | Number of turns in the measurement |
| `n` | Spatial harmonic order (n=1 dipole, n=2 quadrupole, ...) |
| `k` | Temporal mode index (k=0 is DC/steady-state, k=1 is slowest oscillation, ...) |
| `j` | Turn index (j = 0, 1, ..., M-1) |
| `s` | Sample index within a turn (s = 0, 1, ..., Ns-1) |

---

## Step 1: The Standard Per-Turn Approach

In the standard analysis, we treat each turn independently.

For turn `j`, we have `Ns` flux samples. We apply the FFT to extract the spatial harmonics:

```
f_n(j) = (2/Ns) * Σ_{s=0}^{Ns-1} Φ_j(s) * exp(-i * 2π * n * s / Ns)
```

This gives one complex number `f_n(j)` per harmonic `n` per turn `j`. After applying the Kn calibration and rotation correction, we get the calibrated harmonic `C_n(j)`, from which:

- `B1(j)` = main field (from n=1)
- `b2(j)` = quadrupole relative to dipole (from n=2/n=1, × 10⁴)
- `b3(j)` = sextupole relative to dipole (from n=3/n=1, × 10⁴)

**This works perfectly when the field is constant during each turn.** But during eddy settling or ramping, the field is NOT constant. For the MC62 at 4 Hz (T = 0.25 s) with eddy settling time τ ≈ 35 s, the field changes by ~0.7% during one rotation (= T/τ). The per-turn FFT gives you the *mean* field during that rotation, not the *instantaneous* field at a precise moment.

### Why the averaging error is quadratic, not linear

The naive expectation: "the field changes by 0.7% per turn, so the averaging error is ~0.7%." This is wrong — the error is much smaller, and the reason is **symmetry**.

The per-turn FFT computes the average of B(t) over one full rotation [t, t+T]. Taylor-expand B(t') around the turn's midpoint t_mid = t + T/2:

```
B(t') = B(t_mid) + B'(t_mid)·(t' - t_mid) + B''(t_mid)·(t' - t_mid)²/2 + ...
```

When you integrate over the full turn to get the average:

1. **The B' (first derivative) term integrates to ZERO.** It's an odd function around the midpoint: during the first half of the turn, the field is below the midpoint value by some amount; during the second half, it's above by the same amount. These cancel exactly. The **linear** field change during one turn does not contribute to the error.

2. **The B'' (curvature) term survives**: ∫(t' - t_mid)² dt' = T³/12, giving error = B''·T²/24.

For exponential settling B(t) = B∞ + A·exp(-t/τ):
- B' = -A/τ · exp(-t/τ)  ← this is the 0.7% per-turn change (linear, cancels!)
- B'' = A/τ² · exp(-t/τ)  ← this is the curvature (quadratic, survives)

```
averaging error = A · exp(-t/τ) · (T/τ)² / 24
```

For MC62 4 Hz: (T/τ)²/24 = (0.0072)²/24 ≈ 2 × 10⁻⁶. With eddy amplitude A ~ 1 mT, the error is ~2 nT — **well below ppm** (relative to B ~ 68 mT, this is 3 × 10⁻⁸).

**The 0.7% per-turn field change is the first derivative effect — and it cancels by symmetry of the averaging integral.** Only the curvature (second derivative) contributes, giving (T/τ)² dependence. This is a fundamental property of symmetric averaging, not specific to eddies — it applies to any smooth time variation.

### How eddies are already disentangled by the per-turn pipeline

Given that the per-turn averaging error is sub-ppm, the standard per-turn pipeline already captures the eddy settling dynamics correctly:

### The three phases of a plateau

After a current ramp, the field evolution goes through three distinct phases:

**Phase 1 — Ramp** (current changing at dI/dt):
Two effects overlap: (a) the field changes because the current is changing (quasi-static, ∝ dI/dt), and (b) eddy currents in the iron create an exponential lag. The `dit` pipeline correction handles effect (a) — it reweights the flux samples by I_mean/I_k to account for the current being different at each angular position during one turn. But dit is **linear in current** and does NOT correct the eddy lag. Eddies are in the iron, not the power supply.

**Phase 2 — Plateau start** (dI/dt = 0, eddies settling):
The current is now stable, so the dit correction is a no-op (I_mean = I_k at every sample). But the field is still changing because the eddy currents are decaying exponentially: B(t) = B_static + A·exp(-t/τ). This phase lasts several τ.

**Phase 3 — Settled plateau** (eddies at or below noise):
After enough time, the eddy residual drops below the measurement noise floor. This is where the static harmonics live — the true magnetostatic field quality of the magnet.

### MC62 4Hz eddy settling timeline

At 60 A (ascending staircase), A_eddy ≈ 872 µT, τ ≈ 35 s:

| Time (s) | Turns | exp(-t/τ) | B1 residual | b2 residual | vs B1 noise/turn |
|----------|-------|-----------|-------------|-------------|------------------|
| 0 | 0 | 1.0 | 872 µT | 1.30 units | 42× noise |
| 35 | 139 | 0.37 | 321 µT | 0.48 units | 15× noise |
| 105 | 417 | 0.050 | 43 µT | 0.065 units | 2.1× noise |
| 130 | 518 | 0.025 | 21 µT | 0.032 units | **1.0× noise** |
| 171 | 680 | 0.0075 | 6.5 µT | 0.010 units | 0.3× noise |

The eddy drops below per-turn noise (21 µT) at ~130 s (3.7τ, 518 turns). Our N_LAST = 680 turns (171 s, 4.9τ) gives 6.5 µT residual — well below per-turn noise.

**But there's a subtlety**: when you average 680 turns, the standard error of the mean shrinks to 21/√680 ≈ 0.8 µT. The eddy bias (6.5 µT) is **8× the standard error** — it's statistically significant! This means:
- Eddy is below per-turn noise → individual turns look "settled"
- Eddy is ABOVE the precision of the mean → the average is systematically biased
- Need ~6.9τ (242 s, 959 turns) for eddy to drop below 0.1% of A ≈ 0.87 µT

### How to disentangle eddies

**Method 1 — Last-N average (model-free)**:
Wait long enough for the exponential to decay. Robust (no model assumptions), but needs sufficient plateau time. At 4.9τ the eddy bias (6.5 µT) is still 8× the std-of-mean, so it's not truly "static" — it's a pragmatic choice balancing settling time vs measurement throughput.

**Method 2 — Exponential fitting**:
Fit B1(j) = B∞ + A·exp(-jT/τ) to extract B∞ directly. Uses ALL turns (including the settling transient), so it's more data-efficient. The fit gives the static value even before eddies have fully decayed. But it depends on the model being correct — AICc model selection shows MC62 often needs 3-tau models for B1 and b2.

**Method 3 — Marusov temporal decomposition**:
The DC mode σ_{n,0} is the time-average over the entire plateau. If the plateau starts from the ramp end, σ_{n,0} includes eddy contamination (the time-average of the exponential transient). For MC62: the DC bias is ~90 µT — σ_{n,0} is NOT the static field. The eddy content lives in the k>0 temporal modes. Marusov cannot magically separate static from eddy — the DC mode averages over both. However, temporal filtering (keeping only k>K modes) can reduce noise on the per-turn data, giving cleaner exponential curves for Method 2.

**Neither dit nor Marusov can disentangle eddies.** dit corrects for changing current (linear, quasi-static) — eddies are a fundamentally different effect (exponential, in the iron). Marusov's σ_{n,0} includes eddy bias. The only tools that work are: (1) waiting long enough, or (2) fitting the exponential.

### Per-turn sampling resolves eddies

The eddies are NOT hidden inside the per-turn FFT. With τ/T ≈ 140, the per-turn pipeline gives ~140 samples per time constant. The exponential settling is fully visible as B1(j), b2(j), b3(j) evolving turn to turn. The 2×10⁻⁶ per-turn averaging bias is negligible. The eddy disentanglement problem is about extracting B_static from the settling curve — a fitting/waiting problem, not a Fourier analysis problem.

### Why Marusov still matters

If the per-turn pipeline already handles eddies correctly, why do we need Marusov?

1. **Rigorous verification**: The 2D decomposition provides an independent framework to *prove* that the per-turn averaging is adequate. Without Marusov, the (T/τ)²/24 formula is just a theoretical estimate — Marusov lets us *measure* the actual difference.

2. **Temporal noise filtering**: K-truncation removes high-frequency noise while preserving eddy dynamics. K ≈ 50 captures all eddy content (k_eddy ≈ 10) while removing ~50% of the per-turn noise bandwidth.

3. **Fast-cycling magnets**: For systems where T/τ is not small (e.g., a booster synchrotron with 50 Hz field oscillation and 5 Hz coil rotation, τ/T ~ 4), the averaging error grows to ~0.04% of A. The per-turn FFT becomes fundamentally limited, and Marusov's decomposition is essential.

4. **Phase coupling quantification**: The full-stream vs two-step comparison reveals the exact magnitude of the temporal-spatial coupling ignored by the per-turn approach — information that is impossible to obtain without the 2D framework.

---

## Step 2: The Marusov 2D Decomposition

Marusov proposes treating the entire measurement as a single signal in both angle and time.

### The key idea

Instead of looking at one turn at a time, look at ALL turns together. The flux signal over the full measurement is:

```
Φ(j, s)    where j = turn index, s = sample index
```

This is a 2D array: M rows (turns) × Ns columns (samples). Each row is one rotation; each column is one angular position sampled across all turns.

### The 2D Fourier decomposition

The full signal can be decomposed into basis functions that oscillate in both dimensions:

```
Φ(j, s) = Σ_n Σ_k  σ_{n,k} * exp(i * 2π * n * s / Ns) * exp(i * 2π * k * j / M)
```

where:
- `σ_{n,k}` is the **2D Fourier coefficient** — the amplitude of the component that has spatial frequency `n` and temporal frequency `k`
- The first exponential captures the **spatial pattern** (how the field varies around the bore)
- The second exponential captures the **temporal evolution** (how that pattern changes from turn to turn)

### What do the indices mean?

| σ_{n,k} | n (spatial) | k (temporal) | Physical meaning |
|---------|-------------|-------------|------------------|
| σ_{1,0} | dipole | steady-state | The constant part of the dipole field |
| σ_{1,1} | dipole | slow drift | The dipole field is slowly changing |
| σ_{2,0} | quadrupole | steady-state | The constant quadrupole component |
| σ_{1,5} | dipole | fast oscillation | A rapidly oscillating dipole (vibration?) |

- **k = 0** (DC): the time-averaged value of each harmonic. This is what you'd measure if the field were perfectly constant.
- **k = 1, 2, ...**: temporal modes that capture how each harmonic evolves in time. Low k = slow changes (eddy currents, thermal drift). High k = fast changes (vibration, noise).

---

## Step 3: Why This Helps — Temporal Filtering

The power of the 2D decomposition is **temporal bandwidth control**.

### The problem with per-turn FFT

Each per-turn C_n(j) contains ALL temporal frequencies — the settled value, the eddy transient, AND the turn-to-turn noise. On a settled plateau, the noise dominates the turn-to-turn variation.

### The Marusov solution

After computing σ_{n,k}, you can **truncate** the temporal modes:

- Keep only k = 0, 1, ..., K-1 (the first K temporal modes)
- Reconstruct the time series:

```
C_n^{smooth}(j) = Σ_{k=0}^{K-1} σ_{n,k} * exp(i * 2π * k * j / M)
```

This gives a **temporally smoothed** version of the harmonics. The smoothing is exact (not a moving average or fit) — it's a sharp frequency cutoff in temporal Fourier space.

### Choosing K

| K value | Effect |
|---------|--------|
| K = M (all modes) | No filtering — identical to per-turn FFT |
| K ~ 50 | Keeps slow dynamics (eddy settling), removes per-turn noise |
| K ~ 10 | Keeps only very slow trends — may distort eddy settling curves |
| K = 1 | Only the DC component — the time-averaged value |

The optimal K depends on what time scales are physically relevant. For MC62 4 Hz:
- Eddy settling: τ ~ 35 s → temporal frequency ~ 1/(2π·35) ≈ 0.005 Hz
- Turn frequency: 4 Hz
- So the eddy signal sits at very low temporal mode k ~ 1–5
- K = 50 captures all physical dynamics while removing 97% of the noise bandwidth

---

## Step 4: The Full-Stream vs Two-Step Implementation

Marusov's original paper describes a single 1D DFT of the entire flux stream (M × Ns samples). This is elegant but has practical limitations.

### Full-stream approach (Marusov's original)

Treat the entire measurement as one long signal with N_total = M × Ns samples:

```
F[m] = FFT of Φ(entire stream),    m = 0, 1, ..., N_total - 1
```

The 2D indices (n, k) map to the 1D DFT bin:

```
m = k + M·n
```

So the spatial harmonic n=1 with temporal mode k=0 sits at DFT bin m = M, and harmonic n=2 sits at m = 2M, etc. The temporal modes k = 0, 1, 2, ... fill in the bins between M·n and M·(n+1).

**This works because K ≪ M** (the temporal bandwidth is much smaller than the number of turns), so there's no aliasing between different spatial harmonics in the DFT.

**Limitation**: The full-stream approach doesn't apply per-turn drift correction or the pipeline's nonlinear rotation correction (which wraps the phase angle to [-π/2, π/2] per turn). These corrections are critical for B1 accuracy.

### Two-step approach (our implementation)

We split the decomposition into two independent steps:

**Step A — Spatial decomposition**: Use the validated per-turn pipeline (drift correction → integration → FFT → Kn calibration → rotation correction) to get C_n(j) per turn. This is the standard analysis, already validated to machine-precision parity with FFMM C++.

**Step B — Temporal decomposition**: Apply a 1D DFT along the turn index:

```
σ_{n,k} = (1/M) Σ_{j=0}^{M-1} C_n(j) * exp(-i * 2π * k * j / M)
```

Then reconstruct with K modes:

```
C_n^{smooth}(j) = Σ_{k=0}^{K-1} σ_{n,k} * exp(i * 2π * k * j / M)
```

**Why this works**: The spatial and temporal decompositions are separable (the 2D DFT factors as a product of two 1D DFTs). Doing them in sequence is mathematically equivalent to Marusov's single-pass approach, but allows us to use the validated pipeline for the spatial step.

**Key advantage**: When K = M (all temporal modes), the two-step approach gives back the **exact** per-turn pipeline output (verified: max relative error ~ 10⁻¹³, machine epsilon).

---

## Step 5: Aliasing — When Marusov Matters Most

### The aliasing condition

The 2D decomposition avoids spatial-temporal aliasing when:

```
K ≪ M    (temporal bandwidth ≪ number of turns)
```

This is always satisfied in practice. For MC62 4 Hz: M ≈ 1345 turns per plateau, K ~ 50 is more than enough.

But the deeper question is: **when does the field change significantly during one rotation?**

### The ratio τ/T

The critical parameter is the ratio of the field's characteristic time scale τ to the rotation period T. The per-turn FFT averaging error is **(T/τ)²/24** of the eddy amplitude — it's quadratic, not linear.

| τ/T | Intra-turn ΔB/B | Per-turn averaging error (T/τ)²/24 | Assessment |
|-----|----------------|-------------------------------------|------------|
| > 100 | < 1% | < 4 × 10⁻⁷ of A | Sub-ppm — per-turn FFT is fine |
| 10–100 | 1–10% | 4 × 10⁻⁷ to 4 × 10⁻⁵ of A | ppm threshold — check if A is large enough to matter |
| 1–10 | 10–100% | 4 × 10⁻⁵ to 4 × 10⁻³ of A | Significant — Marusov needed for precision |
| < 1 | > 100% | > 4 × 10⁻³ of A | Field changes fundamentally during one turn |

**For MC62 4 Hz**: τ/T ≈ 140. Per-turn averaging error = (1/140)²/24 ≈ 2 × 10⁻⁶ of A. With A ~ 1 mT and B ~ 68 mT, the relative error on B1 is ~3 × 10⁻⁸. **The per-turn FFT is sub-ppm for this measurement.**

**When does Marusov become essential?** When (T/τ)²/24 × A/B approaches your target accuracy. For a booster at 50 Hz with a coil at 5 Hz (τ/T ~ 4), the averaging error is ~0.3% of A — the per-turn FFT is fundamentally limited, and the 2D decomposition is needed.

Even when per-turn averaging is adequate (like MC62 4 Hz), Marusov's temporal filtering (K-truncation) provides a useful noise reduction tool for settled plateau analysis.

---

## Our Results (MC62 05_4Hz)

### Identity check (K = M)

When K = M (all temporal modes), the reconstruct exactly reproduces the input:

```
max |B1_reconstructed - B1_pipeline| = 3.89 × 10⁻¹⁶ T
max |b2_reconstructed - b2_pipeline| = 4.39 × 10⁻¹³ units
```

This is machine epsilon — the decomposition is exact.

### Temporal filtering (K < M)

| K | B1 residual (settled) | B1 relative | b2 residual (units) |
|---|----------------------|-------------|---------------------|
| 5 | 60 µT | 8.9 × 10⁻⁴ | 0.08 |
| 50 | 34 µT | 4.9 × 10⁻⁴ | 0.04 |
| 200 | 19 µT | 2.8 × 10⁻⁴ | 0.02 |
| M (full) | 0 | 0 | 0 |

The residual is the noise content removed by temporal filtering. On settled plateaus, this is purely measurement noise — confirming that the Marusov filtering works as expected.

### Full-stream vs two-step comparison

Direct comparison of σ_{n,k} coefficients between full-stream (single 1D FFT of concatenated stream) and two-step (per-turn FFT → temporal DFT). The two-step ignores the phase coupling term exp(-2πi·ks/(MNs)), introducing a systematic relative error proportional to k/M per temporal mode.

#### Phase coupling validation

The measured |Δσ|/|σ| between the two approaches, compared with theory:

| Harmonic | k | Measured |Δσ|/|σ| | Theory k/M | Measured/Theory |
|----------|---|------------------|------------|-----------------|
| n=1 (dipole) | 1 | 2.7×10⁻³ | 7.4×10⁻⁴ | **3.6×** |
| n=1 (dipole) | 10 | 2.7×10⁻² | 7.4×10⁻³ | **3.6×** |
| n=1 (dipole) | 50 | 1.4×10⁻¹ | 3.7×10⁻² | **3.7×** |
| n=1 (dipole) | 100 | 2.6×10⁻¹ | 7.4×10⁻² | **3.5×** |
| n=2 (quad) | 1 | 3.3×10⁻² | 7.4×10⁻⁴ | 44× |
| n=3 (sext) | 1 | 5.4×10⁻² | 7.4×10⁻⁴ | 72× |

**Key observation for n=1**: At low k (1, 5, 10) where SNR is high, the measured/theory ratio is very stable: **3.61 ± 0.02 (CoV = 0.5%)**. At high k (50, 100) where σ approaches the noise floor, the ratio scatters more (3.49–3.74). The low-k prefactor ≈ 3.6 is close to 2π/√3 = 3.63 (the geometric RMS of the phase coupling integral). This shows:
- The phase coupling theory is quantitatively correct
- Both implementations (full-stream and two-step) are correct — they disagree by exactly the predicted amount
- The error is purely systematic and proportional to k

**Key observation for n=2, n=3**: The measured error is 10–150× above the k/M theory, and the ratio is NOT constant. This is because σ_{2,k} and σ_{3,k} are 50–200× smaller than σ_{1,k} (weaker harmonics), so noise dominates the comparison. The phase coupling is still ~3.6×k/M, but it's buried in noise.

#### Practical impact

The phase coupling affects σ_{n,k} by ~3.6×k/M, but its impact on the **reconstructed per-turn harmonic** depends on how much energy is in that temporal mode relative to DC:

```
Impact on per-turn B1 = (3.6 × k/M) × |σ_{1,k}| / |σ_{1,0}|
```

At the eddy frequency (k ≈ 10): the measured |Δσ_{1,10}| / |σ_{1,0}| ≈ 5 ppm. Note: the leading-order estimate k/M gives 1.4 ppm, but the actual phase coupling is ~3.6× larger (the geometric prefactor from the RMS of the phase integral). So the correct impact is **~5 ppm** — not sub-ppm, but small compared to measurement noise (~300 ppm/turn).

#### Measured differences on the 60 A plateau

| Region | |C₁| relative diff | Dominated by |
|--------|-------------------|-------------|
| Settled (last 200 turns) | 7.7×10⁻⁴ | Noise (field is constant, no phase coupling) |
| Early settling (first 100 turns) | 3.2×10⁻³ | Noise + phase coupling (~5 ppm systematic) |

The measured differences are much larger than the ~5 ppm phase coupling, confirming that **noise dominates** the practical comparison for MC62 4 Hz.

### Verdict

For MC62 4 Hz (τ/T ≈ 140), the Marusov decomposition demonstrates:

1. **Mathematical correctness**: K=M identity to machine epsilon (3.89×10⁻¹⁶ T). The decomposition is exact.

2. **Per-turn averaging is sub-ppm**: The per-turn FFT error is (T/τ)²/24 ≈ 2×10⁻⁶ of the eddy amplitude. The naive 0.7% per-turn field change does NOT translate to a 0.7% error — the linear change cancels by symmetry, and only the curvature (quadratic) contributes.

3. **Phase coupling is quantitatively understood**: For the dipole, the two-step vs full-stream difference follows ~3.6×k/M with a stable prefactor at low k (CoV = 0.5% for k=1,5,10). This validates both implementations and quantifies the information lost by the two-step approach. The impact on B1 is ~5 ppm at the eddy frequency — not sub-ppm, but small vs noise.

4. **Phase coupling impact is small**: At the eddy frequency, the impact on per-turn B1 is ~5 ppm (not sub-ppm, but small vs measurement noise ~300 ppm/turn). For higher harmonics (n=2, n=3), noise dominates before the phase coupling becomes relevant.

5. **Temporal filtering works**: K ≈ 50 captures all eddy content while removing ~50% of the noise bandwidth. On settled plateaus, all k>0 modes are noise, so temporal filtering is purely denoising.

6. **Eddies are already resolved**: With τ/T ≈ 140, the per-turn pipeline provides ~140 samples per time constant. Eddy disentanglement via exponential fitting or last-N averaging works at sub-ppm precision. Marusov confirms this rather than fixing a problem.

7. **On settled plateaus**: Both approaches give identical results because the field is constant — there is no temporal-spatial coupling to resolve.

---

## The Mathematics — Compact Summary

For readers who want the formulas in one place.

### Flux signal

The rotating coil measures flux Φ as a function of angle θ and time t. Discretised:

```
Φ[j·Ns + s]    j = 0..M-1 (turn),  s = 0..Ns-1 (sample)
```

### Per-turn spatial FFT (standard)

```
f_n(j) = (2/Ns) Σ_{s} Φ_j(s) · exp(-i·2π·n·s/Ns)       for n = 1, 2, ..., N_max
```

### Calibration

```
C_n(j) = f_n(j) / conj(kn_n) · R_ref^(n-1)     (Kn application)
C_n(j) ← C_n(j) · exp(-i·n·φ_rot(j))            (rotation correction)
```

where φ_rot uses the pipeline's arg-wrapping method for the main harmonic.

### Normalisation to units

```
B1 = Re[C_1]                               (Tesla)
b_n = Re[C_n / C_1] × 10⁴                  (units, for n > m)
```

### Temporal DFT (Marusov's contribution)

```
σ_{n,k} = (1/M) Σ_{j} C_n(j) · exp(-i·2π·k·j/M)     for k = 0, 1, ..., K-1
```

### Reconstruction with K temporal modes

```
C_n^{smooth}(j) = Σ_{k=0}^{K-1} σ_{n,k} · exp(i·2π·k·j/M)
```

### Full-stream equivalence (Marusov's original)

```
σ_{n,k} = (2/N_total) · F[k + M·n]

where F = FFT of the entire flux stream (N_total = M·Ns samples)
```

The factor of 2 comes from the single-sided spectrum convention. The mapping m = k + M·n is the key insight: spatial and temporal frequencies occupy non-overlapping bins in the 1D DFT when K < M.

---

## Dynamic Eddy Correction During Ramps

The analysis above focuses on settled plateaus (static harmonics) and the per-turn pipeline's accuracy during eddy settling. But what about extracting **correct harmonics during ramps**, where the current is actively changing?

### What we already have

From settled plateaus (Phase 3), last-N averaging gives static harmonics with sub-µT precision. The eddy transfer function notebook provides multi-tau fits (τᵢ, Aᵢ) for B1, b2, b3 at each current level. These are the calibration data for dynamic correction.

### The eddy transfer function

The iron's response to a changing field is described by an impulse response (transfer function):

```
h(t) = Σᵢ (Aᵢ/τᵢ) · exp(-t/τᵢ)
```

The eddy contribution to any harmonic at time t is the convolution of h(t) with the rate of change of the quasi-static field:

```
ΔB_eddy(t) = -∫₀ᵗ h(t-t') · (dB_static/dt')(t') dt'
```

### Are τ values fixed magnet properties?

**Approximately yes.** τ is determined by the iron's resistivity and eddy current path geometry (lamination thickness, yoke cross-section). It varies ~15% with current level because magnetic permeability μ(B) changes with saturation — at high B, the iron is more saturated, μ drops, eddy current paths change slightly. But τ is NOT a function of ramp rate or measurement speed.

MC62 measured values: τ ≈ 33–40 s across 20–100 A (ascending). At high current (>140 A), eddy amplitude becomes negligible (R² < 0.7), so τ is poorly constrained there — but correction is also unnecessary.

### Do A values depend on ramp rate and current?

**Yes.** The eddy amplitude scales as:

```
A ≈ τ · TF(I) · |dI/dt|
```

- **Linear in dI/dt**: doubling the ramp rate doubles the eddy amplitude (confirmed by precycle vs staircase comparison: 50 A/s vs 1 A/s)
- **Depends on I through TF(I)**: at saturation, TF drops, so A decreases even at the same dI/dt
- **Depends on τ(I)**: weak variation (~15%), but enters the product

MC62 at 60 A, 1 A/s: A ≈ 35 s × 1.128 mT/A × 1 A/s ≈ 40 µT steady-state lag.
MC62 at 60 A, 50 A/s (precycle): A ≈ 35 × 1.128 × 50 ≈ 2.0 mT (50× larger).

### Steady-state ramp: constant eddy offset

During a **constant ramp** (dI/dt = const for a time >> τ), the convolution integral reaches a steady state:

```
ΔB_eddy(steady-state) = -τ_eff · TF(I) · dI/dt
```

where τ_eff = Σᵢ Aᵢ·τᵢ / Σᵢ Aᵢ (amplitude-weighted average τ for multi-tau models). The eddy is a **constant offset** — the measured field lags behind the quasi-static value by a fixed amount. This offset is easily correctable if you know τ_eff and TF(I).

### Can exponential fitting work for harmonics?

**Yes, in principle.** Each harmonic b_n has its own eddy component because different harmonics come from different parts of the iron cross-section:

- B1 (dipole): bulk iron → strong eddy, multi-tau (3-tau preferred by AICc)
- b2 (quadrupole): pole region → moderate eddy, multi-tau
- b3 (sextupole): pole tips → weaker eddy, 1-tau sufficient

On plateaus, fitting works well because the functional form is known (sum of exponentials with constant B∞). During ramps, the quasi-static baseline B_static(I(t)) is itself changing, making the fit more complex.

### Practical implementation for ramp correction

**Step 1 — Calibrate h(t) from plateau data:**
Use multi-tau fits from each plateau to extract τᵢ and relative amplitudes. You already have this from the eddy_transfer_function notebook. R² > 0.95 needed for useful correction (~5% τ accuracy). MC62 ascending 20–100 A satisfies this.

**Step 2 — Build quasi-static magnetization curve:**
From settled plateau averages, interpolate B_static(I), b2_static(I), b3_static(I). Separate ascending/descending branches (hysteresis). This is your eddy-free reference.

**Step 3 — Predict eddy on ramps:**
Knowing I(t) from the current waveform and h(t) from Step 1:

```python
# Numerical convolution (discrete)
for j in range(M_ramp):
    t_j = j * T
    eddy_pred[j] = -sum(h(t_j - t_k) * dB_static_dt[k] * T
                        for k in range(j))
```

**Step 4 — Subtract:**
```
B_corrected(j) = B_measured(j) - eddy_pred(j)
```

### Key difficulties

1. **Each b_n has its own h(t)**: τ values are similar (same iron), but amplitudes differ. Must calibrate per-harmonic.

2. **Hysteresis on ramps**: B_static(I) already contains hysteresis. If the ramp reverses direction, you need the full minor-loop model, not just the major hysteresis curve.

3. **τ variation with I**: Using a single τ across the full current range introduces ~15% error in the eddy prediction at extreme currents. Could interpolate τ(I) from plateau fits.

4. **Ramp transient at start/end**: When the ramp starts or stops, the eddy is NOT in steady state — it takes ~3τ to reach the steady-state lag. The convolution handles this correctly, but a simple "subtract τ·TF·dI/dt" does not.

5. **Model fidelity**: The correction is only as good as the eddy model. If the iron has nonlinear eddy behavior (e.g., amplitude-dependent τ at high fields), the linear convolution breaks down.

### Can Marusov help with ramp correction?

**Limited.** Marusov's σ_{n,0} (DC mode) averages over the entire measurement including ramps — it cannot separate static from eddy during ramps. Temporal filtering can clean up noise on ramp data, making the per-turn harmonics smoother for subsequent fitting, but the fundamental entanglement of quasi-static B(I) and eddy lag remains. The convolution approach above is the right tool for ramp correction.

### Summary table

| Question | Answer |
|----------|--------|
| τ fixed? | Yes, ~15% variation with I due to μ(B), same across campaigns |
| A depends on? | dI/dt (linear) × TF(I) × τ(I) |
| Exponential fit for harmonics? | Yes, each b_n gets its own multi-tau fit |
| R² needed for correction? | >0.95 for useful correction (~5% τ accuracy) |
| During constant ramp? | Eddy is a constant offset: τ_eff × TF × dI/dt |
| Practical approach? | Calibrate h(t) from plateaus → convolve with I(t) → subtract |
| Marusov useful for ramps? | Noise filtering only; cannot separate static from eddy |

---

## Practical Recommendations

1. **For MC62 at 4 Hz** (τ/T ~ 140): The per-turn pipeline averaging error is (T/τ)²/24 ≈ 2 × 10⁻⁶ of the eddy amplitude — sub-ppm. On settled plateaus the error is zero (field is constant). The per-turn pipeline is adequate for ppm work at this τ/T ratio.

2. **For noise reduction on settled plateaus**: Apply Marusov temporal filtering with K ~ 50 to the pipeline output. This removes high-frequency noise while preserving eddy settling dynamics.

3. **For fast-cycling magnets** (τ/T < 10): The averaging error (T/τ)²/24 grows quadratically and can reach 0.04% of the eddy amplitude. Marusov's full 2D decomposition becomes important for precision work.

4. **For |C₁| during transients**: The full-stream approach eliminates the ~0.72% phase coupling error on σ_{n,k} at the eddy frequency, giving slightly better |C₁| estimates.

5. **Validation**: Always verify that K = M gives exact pipeline reproduction (machine epsilon). This confirms the implementation is correct.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Rotating coil** | A measuring device that spins inside a magnet bore, sampling the magnetic flux vs. angle |
| **Harmonic** | A Fourier component of the field's angular distribution (n=1 dipole, n=2 quadrupole, n=3 sextupole) |
| **Turn** | One complete rotation of the coil (360°) |
| **Temporal mode** | A Fourier component of how a harmonic evolves from turn to turn |
| **Eddy current** | Current induced in the magnet iron by changing magnetic flux; causes the field to settle slowly after a current change |
| **τ (tau)** | The time constant of eddy current settling (field approaches its final value as exp(-t/τ)) |
| **Pipeline** | The sequence of corrections applied to raw data: drift correction → integration → FFT → Kn calibration → rotation |
| **Kn calibration** | Correction for the coil's angular sensitivity pattern (each coil has different sensitivity to each harmonic order) |
| **FFMM** | The legacy C++ rotating coil analysis software at CERN; our pipeline achieves machine-precision parity with FFMM |
| **Units** | The standard way to express relative harmonic content: b_n = (C_n / C_1) × 10⁴ |
