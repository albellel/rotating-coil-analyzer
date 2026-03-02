# Physics Reference -- Rotating Coil Measurements

## 1. Transfer Function L vs Differential Inductance Ld

### Simple Explanation (no formulas)

**L (transfer function)** answers the question: "How much total field do I get
per amp of current?"  Think of it as your average fuel economy over an entire
road trip -- it includes all the easy highway kilometres and the slow city ones.

**Ld (differential inductance)** answers: "If I add one more amp right now,
how much MORE field do I get?"  Think of it as your instantaneous fuel economy
at this exact moment.

In a magnet with iron yoke, at **low current** the iron is unsaturated.
Every amp you add produces the same amount of extra field.  The trip so far
has been all highway.  L and Ld are equal and roughly constant.

At **high current**, the iron saturates.  Adding one more amp now gives LESS
extra field than before, because the iron can no longer contribute as much.
But L = B/I still remembers all the easy field you accumulated at low current.
Like your trip average still being high because of the long highway stretch,
even though you are now stuck in traffic.

So at saturation:
- **L stays relatively large** (it averages over the whole history, including
  the unsaturated region).
- **Ld drops** (only the current incremental response matters, and that
  response is weaker because the iron is saturated).
- **Ld < L** is the hallmark of saturation.

**Important:** Ld does NOT diverge at saturation -- it *decreases*.  The
incremental permeability drops toward mu_0 (air-like) as the iron saturates.
In the extreme saturation limit, Ld approaches the air-only transfer function
(the magnet behaves as if the iron weren't there for incremental changes).

The ratio Ld/L < 1 quantifies how far into saturation the magnet is:
- Ld/L ~ 1.0 : linear regime (unsaturated)
- Ld/L ~ 0.95 : mild saturation
- Ld/L ~ 0.7 : strong saturation

### Everyday Analogies

**Analogy 1 -- Savings account:**
You deposit 100 euros every month into a savings account.  In the first years,
the bank gives a generous 5% bonus on each deposit (like unsaturated iron
amplifying the field).  After 10 years, the bank changes policy: new deposits
get only 1% bonus (like iron saturating).

- **L = balance / total_deposited**: still high because all those early
  generous bonuses are already in your account.
- **Ld = bonus on last deposit**: low, only 1%.
- Ld < L because the past was more generous than the present.

**Analogy 2 -- Filling a sponge:**
You pour water onto a dry sponge.  At first it absorbs eagerly -- every ml
you pour goes straight in.  As the sponge fills, it absorbs less and less;
water starts running off.

- **L = total_absorbed / total_poured**: still decent because the early
  pouring was very efficient.
- **Ld = how much the next ml gets absorbed**: low, because the sponge is
  nearly full.
- The sponge is the iron yoke.  "Full" = saturated.

**Analogy 3 -- Hiring employees:**
A startup hires 10 people.  Each new hire adds a lot of productivity (like
each amp adding a lot of field in unsaturated iron).  At 200 employees,
offices are full, bureaucracy grows -- each new hire adds less marginal
productivity.

- **L = total_output / total_headcount**: high (averages over the early
  productive hires).
- **Ld = productivity of the next hire**: low (diminishing returns).

### Numerical Worked Examples

#### Example 1: Linear magnet (no saturation)

Imagine a magnet with perfectly linear iron (constant mu_r):

```
I (A)     B1 (T)    L = B1/I (T/kA)    Ld = dB1/dI (T/kA)
  0       0.000     --                  0.400
100       0.040     0.400               0.400
200       0.080     0.400               0.400
500       0.200     0.400               0.400
1000      0.400     0.400               0.400
2000      0.800     0.400               0.400
5000      2.000     0.400               0.400
```

L and Ld are identical everywhere.  The B(I) curve is a straight line.
Every amp gives the same 0.400 mT of extra field.

#### Example 2: Magnet with saturation (typical SPS MBB-like)

Now the iron saturates above ~3000 A.  The field still increases, but slower:

```
I (A)     B1 (T)    L = B1/I (T/kA)    Ld = dB1/dI (T/kA)    Ld/L
  0       0.000     --                  0.400                  --
100       0.040     0.400               0.400                  1.000
301       0.116     0.385               0.385                  1.000
1000      0.380     0.380               0.380                  1.000
2000      0.750     0.375               0.370                  0.987
3000      1.100     0.367               0.340                  0.927
4000      1.420     0.355               0.300                  0.845
4815      1.680     0.349               0.250                  0.716
5000      1.720     0.344               0.200                  0.581
6000      1.880     0.313               0.120                  0.383
```

Key observations:
- Up to ~2000 A: L ~ Ld ~ 0.375-0.400.  Iron is linear.
- At 4815 A: L = 0.349 but Ld = 0.250.  Iron is moderately saturated.
  Each new amp gives only 250 mT/kA of extra field, but the average over
  the whole range is 349 mT/kA because of the easy early amps.
- At 6000 A: L = 0.313 but Ld = 0.120.  Heavily saturated.  The magnet
  barely responds to current increments.
- The Ld/L ratio drops from 1.0 to 0.38 -- severe saturation.

#### Example 3: Reading L and Ld from two-plateau data

Our MBB measurement has two current levels:

```
Point 1 (injection):  I = 301 A,   B1 = 0.116 T
Point 2 (flat-top):   I = 4815 A,  B1 = 1.782 T
```

Step-by-step:

```
L_app at injection = 0.116 / 0.301 = 0.3854 T/kA
L_app at flat-top  = 1.782 / 4.815 = 0.3701 T/kA
```

L_app dropped from 0.385 to 0.370 -- a 4% decrease.  Mild saturation.

```
Ld = (1.782 - 0.116) / (4.815 - 0.301) = 1.666 / 4.514 = 0.3691 T/kA
```

The three quantities in order:

```
L_app(injection) = 0.385 > L_app(flat-top) = 0.370 > Ld = 0.369 T/kA
```

Ld/L_app(flat-top) = 0.997.  The MBB at SPS operating current (4815 A) is
barely into saturation -- almost perfectly linear.  This is consistent with
the MBB design: the iron gap is large, so saturation effects are weak up to
the maximum SPS operating current.

#### Example 4: Heavily-saturated magnet (LHC main dipole at 12 kA)

An LHC main dipole (MB) operates up to ~12 kA.  Illustrative numbers:

```
L_app at 1000 A  = 8.33 T/kA (injection, linear iron)
L_app at 12000 A = 6.92 T/kA (flat-top, saturated iron)
Ld at 12 kA      ≈ 5.0 T/kA  (heavily saturated)

Ld/L = 5.0/6.92 = 0.72  -- significant saturation
```

This means each extra kA at 12 kA only gives 5.0 T/kA of field, compared
to 8.33 T/kA at low current.  The iron contributes 40% less per amp than
in the linear regime.

### Formal Definitions and Derivation

The **magnetic transfer function** (often loosely called "inductance" in
accelerator parlance) relates the dipole field B1 to the excitation current I.

#### Transfer function (apparent inductance)

```
L_app(I) = B1(I) / I      [T/A  or equivalently  T/kA]
```

This is the slope of the line from the origin to the point (I, B1) on the
magnetisation curve.  It represents the average field production efficiency
over the range [0, I].

#### Differential inductance

```
L_d(I) = dB1/dI |_I       [T/A  or equivalently  T/kA]
```

This is the local slope of the B1(I) curve at current I.  It represents the
instantaneous (incremental) field production efficiency.

#### Physical model

For a dipole with iron yoke:

```
B1(I) = mu_0 * N * I / g_eff + B_iron(I)
        \___ air gap term ___/  \_ iron _/
```

where N is the number of conductor turns, g_eff is the effective gap, and
B_iron(I) is the iron contribution (saturates at high I).

In the **linear regime** (mu_r = const):

```
B1 = (mu_0 * mu_r_eff * N / g_eff) * I  =  c * I

L_app = c           (constant)
L_d   = c           (constant)
L_app = L_d         (equal)
```

In the **saturation regime**, B_iron(I) flattens out:

```
dB1/dI = mu_0 * N / g_eff + dB_iron/dI
                             \___ small (iron saturated) ___/

So L_d ≈ mu_0 * N / g_eff   (approaches the "air-only" slope)
```

Meanwhile, L_app = B1/I still includes the large B_iron accumulated before
saturation, so L_app > L_d.

#### Relationship between L_app and L_d

From the product rule:

```
L_d = dB1/dI = d(L_app * I)/dI = L_app + I * dL_app/dI
```

Since dL_app/dI < 0 in the saturation regime (L_app decreases with I):

```
L_d = L_app + I * (negative number) < L_app
```

This proves L_d < L_app whenever the magnet enters saturation.

#### Practical computation from staircase data

With two current plateaus (e.g. injection at I_inj and flat-top at I_fh):

```
L_app(I_inj) = B1(I_inj) / I_inj
L_app(I_fh)  = B1(I_fh) / I_fh
L_d          ≈ [B1(I_fh) - B1(I_inj)] / [I_fh - I_inj]    (finite difference)
```

The finite-difference Ld is an approximation to the true differential
inductance, averaged over the interval [I_inj, I_fh].  For well-separated
plateaus (e.g., 301 A to 4815 A), it captures the bulk saturation behaviour.

#### SPS MBB example (2026-02-25 2Hz, CS segment)

```
Injection (301 A):  B1 = 0.116 T,  L_app = 0.386 T/kA
Flat-high (4815 A): B1 = 1.782 T,  L_app = 0.370 T/kA
Differential:       Ld = (1.782 - 0.116) / (4.815 - 0.301) = 0.369 T/kA
```

L_app(injection) > L_app(flat-high) > Ld : classic saturation signature.
The ratio Ld / L_app(flat-high) = 0.369/0.370 = 0.997, indicating only
very mild saturation for the MBB at 4815 A (the MBB iron is far from
full saturation at SPS operating currents).

---

## 2. Eddy Currents and Multi-Tau Settling

### Physical origin

When the excitation current changes (dI/dt != 0), the changing magnetic flux
induces eddy currents in the magnet yoke (and beam screen, collars, etc.).
These eddy currents create their own magnetic field that opposes the change
(Lenz's law), causing the total field to lag behind the current.

When the current stops changing (e.g., reaches an injection plateau), the
eddy currents have no driving source and decay exponentially.  The field
then "settles" toward its DC (eddy-free) equilibrium value.

### Single-exponential model

```
B(t) = B_inf + A * exp(-t / tau)
```

- **B_inf**: asymptotic (eddy-free) field value
- **A**: eddy-current amplitude at t=0 (beginning of plateau)
- **tau**: time constant of the decay

For laminated yoke (thin iron sheets of thickness d):

```
tau = mu_0 * mu_r * sigma * d^2 / pi^2
```

where mu_r is the relative permeability (current-dependent) and sigma is
the electrical conductivity.  Key consequence: **tau decreases at high
current** because mu_r drops with saturation.

### Multi-exponential models

Real magnets have multiple conducting components, each with their own
time constant:

```
Two-tau:   B(t) = B_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2)
Three-tau: B(t) = B_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2) + A3*exp(-t/tau3)
```

**Physical interpretation:**
- **tau1 (fast, ~0.1-1 s)**: thin components -- beam screen, thin laminations,
  end spacers
- **tau2 (medium, ~1-10 s)**: main yoke laminations (dominant contribution)
- **tau3 (slow, ~10-100 s)**: thick components -- end plates, collars, bus bars,
  support structure

Not all components are always visible.  A single-tau fit suffices when one
process dominates.  The improvement criterion for adding a second tau is
typically delta_R^2 > 0.01 (if the R^2 improves by less than 1%, the extra
complexity is not justified).

### What do outliers at the beginning of a plateau mean?

The first few turns after a current ramp ends often show points that are
"completely out of the trend" compared to the exponential decay.  These
are NOT random measurement errors -- they are real physics from several
overlapping effects:

1. **Ramp-to-plateau transition**: The current doesn't switch instantly
   from ramp to flat.  During the brief deceleration of dI/dt, the field
   measurement captures a mix of ramp dynamics and settling dynamics.  The
   single-exponential model assumes an instantaneous step, which is not
   exactly true.

2. **Fast eddy component**: If there is a fast tau1 << tau2 (e.g., 0.2 s
   vs 1.6 s), the fast component has already decayed significantly by the
   time the first full turn completes (0.5 s at 2 Hz).  The single-exp fit
   with tau ~ 1.6 s cannot capture this fast transient, so the early points
   deviate.  A two-tau fit would capture them.

3. **Magnetic aftereffect (accommodation)**: After a field change, domain
   walls continue to creep on a logarithmic time scale.  This produces a
   brief burst of extra field change in the first ~0.1-1 s that doesn't
   follow exponential decay.

4. **"Stuck field" effect**: In thick yoke iron, the magnetic diffusion
   time can be long.  The surface field changes quickly but the bulk iron
   takes longer to respond.  The first few turns may see the surface-dominated
   response before the bulk catches up.

The current MBB 2Hz analysis trims these early turns automatically via the
`N_LAST_TURNS_INJ = 18` settling window (only the last 18 turns of each
injection plateau are used for harmonic averaging).

### The 3-tau and 5-tau settling criteria

After n time constants, the exponential residual is:

| n*tau | Residual exp(-n) | % of amplitude remaining |
|-------|-----------------|-------------------------|
| 1 tau | 0.368           | 36.8%                   |
| 2 tau | 0.135           | 13.5%                   |
| 3 tau | 0.050           | 5.0%                    |
| 4 tau | 0.018           | 1.8%                    |
| 5 tau | 0.0067          | 0.67%                   |

**"3 tau = eddies practically gone"**: After 3*tau, the eddy contribution
is only 5% of its initial amplitude.  For the MBB (A ~ 1.4 units for b3),
this means a residual of ~0.07 units -- below the turn-to-turn scatter
(~0.15 units).  This is the "engineering" criterion.

**"5 tau = eddies negligible"**: After 5*tau, the residual is 0.67% of A,
i.e., ~0.009 units for MBB b3.  This is well below any measurement noise.
This is the "precision" criterion used for metrology.

**Which to use depends on your accuracy requirement:**
- Machine operation (FGC tables, optics): 3*tau is usually sufficient
  (residual < measurement noise)
- Magnetic measurement reports (EDMS): 5*tau preferred (residual negligible
  compared to systematic uncertainties)
- Reference magnet calibration: 5-7*tau (push below all error sources)

For the MBB 2Hz data: tau ~ 1.6 s, turns are 0.5 s apart.
- 3*tau = 4.8 s = ~10 turns from ramp end
- 5*tau = 8.0 s = ~16 turns from ramp end
- N_LAST = 18 turns from the END of a ~54-turn plateau, meaning we discard
  ~36 turns (18 s = 11*tau).  This is very conservative.

### For multi-exponential decays

When there are multiple time constants, the "eddies gone" criterion must
use the LONGEST tau:

```
Two-tau: B(t) = B_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2)

Wait time for < 0.7% total residual:
   t_settle = max(5*tau1, 5*tau2)
   (in practice just 5*tau_slow, since the fast component is long gone)
```

If tau_slow = 10 s and tau_fast = 1.6 s, you need to wait 50 s (not 8 s)
for the slow component to settle.  This is why identifying whether there
are multiple time constants matters -- the single-tau fit might give
tau = 1.6 s, but a hidden slow component with tau = 10 s and amplitude
A2 = 0.3 units would still bias the "settled" value by 0.3 * exp(-8/10) =
0.13 units after 5 * 1.6 s.

---

## 3. B1 from Rotating Coil vs NMR

### What the rotating coil measures

The rotating coil measures the **flux linkage** as the coil rotates.  After
FFT and kn calibration, B1_T = Re(C_1) is the n=1 (dipole) Fourier
coefficient of the transverse field, **averaged over the coil's active
length** and evaluated at the coil's axial position within the magnet.

For a coil of active length L_coil positioned in the uniform-field region
of a magnet of magnetic length L_mag:

```
B1_coil = (1/L_coil) * integral_{z_start}^{z_start + L_coil} B1(z) dz
```

If the coil is in the main body where the field is uniform (dB1/dz ~ 0),
then B1_coil = B1_center -- the coil value matches the NMR point measurement.

If the coil is near the magnet end (fringe region), the field drops off
rapidly with z, and B1_coil << B1_center.

### SPS MBB coil segments

The MBB rotating coil has two segments:
- **NCS (Non-Connection Side)**: one end of the shaft
- **CS (Connection Side)**: the other end of the shaft

Each segment has its own set of windings (absolute + compensated) with
identical kn calibration (same coil geometry).  The two segments measure
the field at different longitudinal positions within the ~6.26 m MBB dipole.

### Segment labelling swap in the 2026-02-25 2Hz campaign

**IMPORTANT: In the 2026-02-25 2Hz campaign, the DAQ segment labels were
swapped relative to the physical coil positions.**

From the 2026-02-25 2Hz campaign at 4815 A:

```
DAQ label "NCS": B1 = -0.291 T,  TF = 0.065 T/kA  <- FRINGE FIELD
DAQ label "CS":  B1 = -1.782 T,  TF = 0.370 T/kA  <- MAIN BODY
```

The CS transfer function (0.370 T/kA) matches the expected main-body value
and agrees with NMR measurements (116 mT at 301 A -> TF = 0.385 T/kA).
The NCS transfer function (0.065 T/kA) is about 6x smaller -- characteristic
of the fringe field at the magnet end.

**Conclusion:** What the DAQ calls "CS" is actually the segment inside the
magnet (physically the non-connection side).  What the DAQ calls "NCS" is
the fringe-field segment (physically the connection side, sticking out of
the magnet end).

**Cross-check with the 2026-02-06 campaign:** The earlier campaign (1 Hz)
analysed only the NCS segment and obtained B1 = 1.794 T at 4815 A
(TF = 0.373 T/kA) -- matching the 2Hz "CS" value.  This confirms the swap
happened between the two campaigns.

**Action taken:** The analysis notebooks (generated by `generate_notebooks.py`)
now set `is_fringe=True` for DAQ-label "NCS" and `is_fringe=False` for
DAQ-label "CS" in the 2Hz configs.  This means all single-segment analyses
(eddy currents, harmonics, inductance) use the CS segment -- the one actually
inside the magnet.

**Fix for future measurements:** Swap the NCS/CS DAQ channel assignments so
that "NCS" corresponds to the main-body segment again.

### Comparison with NMR reference values

NMR measures the field at a single point at the magnet centre.  Expected
comparison (using CS = main body for the 2Hz campaign):

| Current | NMR (centre) | Coil CS (avg over 0.47 m) | Ratio |
|---------|-------------|--------------------------|-------|
| 301 A   | ~116 mT     | 116.3 mT                 | ~1.00 |
| ~4800 A | ~1.7-1.8 T  | 1.782 T                  | ~1.00 |
| 5000 A  | ~2.02 T     | (not measured)            | --    |

The coil and NMR agree closely because the coil (0.47 m) is short compared
to the uniform-field region of the MBB (~5+ m), so the length-averaged and
point values are essentially identical in the main body.

---

## 4. Why b3 (Sextupole) Is Strong in the Fringe Field

### Physical picture

In the **main body** of a well-designed dipole magnet (like the SPS MBB),
the field is extremely uniform at injection current.  The iron yoke and pole
geometry are optimised so that higher-order multipoles are negligible:

```
Main body at injection:  b2 ~ 0 units,  b3 ~ -0.2 units  (design target: zero)
```

At the **magnet ends** (fringe region), the field drops from its full value
inside the magnet to zero outside over a distance comparable to the gap height.
This field rolloff is highly nonlinear in the transverse plane (x, y):

- At the centre of the aperture, the field drops smoothly along z.
- Near the pole tips, the field drops more steeply because the iron ends
  abruptly.
- The net effect is a z-dependent transverse field shape that is NOT a pure
  dipole -- it contains significant sextupole (n=3) and higher-order components.

### Why sextupole specifically?

The fringe field of a dipole has a characteristic multipole signature
dictated by symmetry:

1. **Dipole symmetry (midplane symmetry)**: The MBB has top-bottom mirror
   symmetry, so only "allowed" multipoles appear: n = 1, 3, 5, 7, ...
   (b1, b3, b5, b7, ...).  Even-order harmonics (b2, b4, ...) are
   "forbidden" by symmetry and remain small even in the fringe.

2. **Sextupole (n=3) dominates** because it is the lowest allowed
   higher-order harmonic.  The amplitude of multipoles generally decreases
   with order (roughly as (r/R_ref)^n), so n=3 is the first and largest
   deviation from a pure dipole.

3. **Quantitatively for MBB fringe**:
   ```
   b3_fringe ~ +5 units      (strong, measurable)
   b5_fringe ~ +0.5 units    (smaller)
   b7_fringe ~ +0.05 units   (negligible)
   b2_fringe ~ 0 units       (forbidden by symmetry)
   ```

### Why eddy current settling is visible in b3 fringe but not b3 main body

Eddy currents change the effective multipole content because different parts
of the yoke (pole tips vs return yoke) have different eddy current
distributions and time constants.  The transient multipole content is:

```
b_n(t) = b_n_DC + delta_b_n * exp(-t/tau_n)
```

For b3:
- **Fringe region**: b3_DC ~ +5 units, delta_b3 ~ 1-2 units.  The eddy
  amplitude is ~30% of the DC value -- clearly visible as exponential settling.
- **Main body**: b3_DC ~ -0.2 units, delta_b3 ~ 0.01-0.05 units.  The eddy
  amplitude is comparable to the turn-to-turn measurement noise (~0.15 units).
  No exponential trend is discernible -- the fit returns R² < 0.1.

This is why b3 eddy fits work in the fringe (R² ~ 0.97) but fail in the main
body (R² ~ 0.08): there is simply no significant sextupole component to
settle in the uniform-field region.

### Implications for measurement practice

- **For harmonic analysis**: Always use the main-body segment.  The fringe
  b3 is a geometric artefact of the magnet ends, not a property of the field
  that the beam experiences over the full magnetic length.

- **For eddy current studies**: The fringe segment provides better signal-
  to-noise for studying eddy dynamics (especially b3 settling), even though
  its absolute multipole values are not representative of the beam region.

- **For accelerator optics**: The integrated b3 over the full magnetic
  length (both body and fringes) is what matters.  The fringe contribution
  partly cancels between entry and exit ends.
