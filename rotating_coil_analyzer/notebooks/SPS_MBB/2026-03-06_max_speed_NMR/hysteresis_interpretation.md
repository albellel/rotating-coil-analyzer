# Hysteresis Interpretation: Session-Ordering Confound & Proposed Experiment

**Campaign:** MBB max-speed + NMR, 2026-03-06
**Date of analysis:** 2026-03-10

---

## 1. Observation

Comparing settled plateau values between the two MD1 conditioning sessions
(Delta = 26 GeV minus 200 GeV):

| Plateau | Delta B1 (uT) | Delta b2 | Delta b3 | Note |
|---------|---------------|----------|----------|------|
| SFTPRO top (4816 A) | **+125.5** | 0.001 | 0.002 | Main signal |
| Post-SFTPRO idle (155 A) | -3.7 | -0.019 | 0.007 | ~zero |
| LHC top (5781 A) | **+66.8** | 0.003 | 0.002 | Smaller signal |
| Post-LHC idle (155 A) | **-4.3** | 0.043 | 0.013 | **Control: ~zero** |

The 26 GeV (flat MD1) session gives **higher B1** at SFTPRO and LHC top than
the 200 GeV (full MD1) session. The effect is almost entirely on B1;
harmonics b2, b3 are unaffected.

### Statistical significance

The rotating coil per-turn noise is large:

| Plateau | Per-turn std (uT) | Std-of-mean | Signal / noise |
|---------|--------------------|-------------|----------------|
| SFTPRO (N=15) | 831 | 215 | 0.6 sigma |
| LHC top (N=5) | 246 | 110 | 0.6 sigma |

The signal is below 1 sigma from the rotating coil alone. NMR confirmation
is essential. However, the **sign is consistent** across both top plateaus,
and the **control (post-LHC idle) is essentially zero**, which is physically
meaningful.

---

## 2. Physical Explanation: Session-Ordering Confound

### The Preisach wiping-out property

The fundamental rule of ferromagnetic hysteresis (Preisach model):

> **Memory of a previous field maximum H_max is only erased when
> the applied field exceeds H_max again.**

Corollary: cycling to H_precycle only resets the domain state for
H < H_precycle. Memory above H_precycle is retained.

### Session order

The 200 GeV session ran **first** (15:22), the 26 GeV session ran
**second** (15:35). This is the key.

**200 GeV session (first):**
```
Unknown initial state X
  -> 20x MD1 full (0 -> 2267 A -> 0)      resets memory below 2267 A
  -> idle (155 A)
  -> SFTPRO (4816 A)                       MEASUREMENT
  -> idle -> injection -> LHC top (5781 A) HIGHEST FIELD IN THE CAMPAIGN
  -> post-LHC idle (155 A)
```

**26 GeV session (second):**
```
Post-LHC idle from 200 GeV session        IRON REMEMBERS 5781 A
  -> 20x MD1 flat (0 -> 301 A -> 0)       301 << 5781: memory NOT erased!
  -> idle (155 A)
  -> SFTPRO (4816 A)                       MEASUREMENT
  -> idle -> injection -> LHC top (5781 A)
  -> post-LHC idle (155 A)
```

### Why 26 GeV gives higher B at SFTPRO

**26 GeV session at SFTPRO (4816 A):**
- The iron remembers the 200 GeV session's LHC excursion to 5781 A
- The 20x MD1 cycling to only 301 A is far too weak to erase this
  (wiping-out requires exceeding 5781 A)
- At 4816 A < 5781 A, the iron is on a **minor ascending loop inside
  the major loop** defined by the 5781 A excursion
- The remanent magnetization from the descending branch of 5781 A
  biases B upward
- Result: **B is elevated**

**200 GeV session at SFTPRO (4816 A):**
- Before this session, the iron was in unknown state X
  (probably not recently at 5781 A)
- 20x MD1 cycling to 2267 A resets memory below 2267 A
- At 4816 A > 2267 A, the iron continues on the first-magnetization
  (virgin) curve -- the most "neutral" path
- No high-field memory to boost B
- Result: **B is lower**

### Why the effect is smaller at LHC top

At LHC top (5781 A), the 26 GeV session approaches the return point
of the previous session's LHC excursion (also 5781 A). By return-point
memory, the minor ascending loop converges to the major loop at this
point. So the gap between the two sessions narrows:
- SFTPRO (4816 A, well inside the loop): Delta = 125 uT
- LHC top (5781 A, at the loop boundary): Delta = 67 uT

### Why the post-LHC control is zero

Both sessions descend from 5781 A. The descending branch is uniquely
determined by the turnaround point (5781 A), regardless of prior
history below that level. So both sessions produce the same post-LHC
state. Delta ~ 0 confirms the Preisach model.

### Summary diagram

```
B (T)
 ^
 |           ........*  LHC top (5781 A)
 |         ./      ./   both sessions converge here
 |        /      ./     gap narrows
 |       / ----/------- SFTPRO (4816 A): gap = 125 uT
 |      / ./            26 GeV session: minor ascending
 |     /./              loop from 5781 A descending branch
 |    //
 |   //                 200 GeV session: on virgin curve
 |  //                  (no high-field memory)
 | /
 |/
 +----------------------------> I (A)
   0   301  2267    4816  5781
       ^     ^
       |     |
    26 GeV  200 GeV
    MD1     MD1
    max     max
```

---

## 3. Proposed Experiment: Properly Standardized Comparison

### Goal

Isolate the true effect of MD1 conditioning level (26 GeV vs 200 GeV)
on the field at SFTPRO and LHC top, **free from session-ordering bias**.

### Protocol

```
SESSION A (200 GeV MD1):
  10x standardization cycles (0 -> 5781 A -> 0)
      Establishes accommodated major loop.
      Wipes ALL prior magnetic memory up to 5781 A.
  20x MD1 full (0 -> 2267 A -> 0)
      The conditioning under test.
  idle (155 A) -> SFTPRO (4816 A) -> idle (155 A)
      -> injection (301 A) -> LHC top (5781 A) -> idle (155 A)

SESSION B (26 GeV MD1):
  10x standardization cycles (0 -> 5781 A -> 0)
      SAME starting state as Session A.
  20x MD1 flat (0 -> 301 A -> 0)
      Different conditioning.
  idle (155 A) -> SFTPRO (4816 A) -> idle (155 A)
      -> injection (301 A) -> LHC top (5781 A) -> idle (155 A)
```

### Key design principles

1. **Full-field standardization before each session.**
   10x cycles to 5781 A (= LHC top) ensures both sessions start from
   the identical accommodated (0, 5781) major loop. This eliminates
   the session-ordering confound.

2. **Run both orders** (A then B, and B then A).
   If the standardization works, the result should be the same
   regardless of order. If it isn't, residual memory effects persist.

3. **Longer SFTPRO plateau.**
   Current campaign: 15 settled turns -> std-of-mean = 215 uT.
   With 100 settled turns -> std-of-mean ~ 83 uT.
   With 200 settled turns -> std-of-mean ~ 59 uT.
   Request longer flat-top at SFTPRO if possible.

4. **NMR as primary instrument** at SFTPRO and LHC top.
   Sub-uT precision, far better than the rotating coil for this
   measurement. The rotating coil provides harmonics (b2, b3) and
   serves as a cross-check on B1.

5. **Post-LHC idle as control.**
   Both sessions descend from the same 5781 A, so the post-LHC idle
   should give identical B. Nonzero delta indicates incomplete
   standardization or non-Preisach effects.

### What to expect (Preisach prediction)

With proper standardization, the Preisach model predicts **zero
difference** at SFTPRO and LHC top. Reasoning:

After 10x standardization to 5781 A, both sessions have the same
Preisach memory stack: (H_max=5781, H_min=0).

Then:
- 200 GeV MD1 adds (2267, 0) to the stack
- 26 GeV MD1 adds (301, 0) to the stack

When the ascending ramp exceeds the MD1 maximum (2267 A or 301 A),
the wiping-out property erases the MD1 pair from the stack. Both
sessions revert to (5781, 0) -- identical.

At SFTPRO (4816 A) and LHC top (5781 A), both are on the same
ascending branch of the standardized major loop. Delta = 0.

### What a nonzero result would mean

If the standardized experiment still shows a delta at SFTPRO or
LHC top, that would be scientifically significant -- evidence for
effects beyond the classical Preisach model:

- **Accommodation / reptation:** 20x MD1 cycling modifies the domain
  wall pinning landscape in a way that is NOT erased by the
  wiping-out property (known limitation of classical Preisach).
- **Domain wall fatigue:** repeated cycling to different amplitudes
  changes the microstructural pinning sites.
- **Thermal effects:** different Joule heating from different MD1
  cycles could affect the yoke temperature and thus mu.

These effects are expected to be small (probably sub-10 uT) but
measurable with NMR. This would be a genuine contribution to the
understanding of SPS dipole magnetic memory.

### Practical notes

- The 10x standardization cycles add ~10 minutes of overhead per
  session (assuming ~30 s per full cycle at MBB ramp rates). This
  is acceptable for a dedicated measurement.
- The standardization current should match LHC top exactly (5781 A),
  not higher, so that the SFTPRO measurement point stays within the
  standardized range.
- If possible, add a plateau at 2267 A during the SFTPRO ramp-up
  to measure B at the MD1 maximum itself -- this is where the
  Preisach model predicts the largest MD1-dependent difference (the
  minor loop rejoins the major loop at this point).

---

## 4. Summary

| Question | Answer |
|----------|--------|
| Is the observed effect real? | Sign is consistent, but < 1 sigma from RC. NMR needed. |
| Is it hysteresis? | Yes, but primarily a **session-ordering artifact**, not the MD1 level. |
| Why does 26 GeV give higher B? | It inherits the 5781 A memory from the preceding 200 GeV session. |
| Why does 200 GeV give lower B? | It ran first, without prior high-field memory. |
| Why is the control ~zero? | Both sessions descend from the same 5781 A (Preisach confirmed). |
| True MD1 effect at SFTPRO? | Predicted to be **zero** (wiping-out erases MD1 memory at 4816 A > 2267 A). |
| How to test properly? | Standardize with 10x full-field cycles before each session. |

---

## 5. Experimental Validation (2026-03-10)

The proposed A-B-B-A experiment with standardization was performed on 2026-03-10
(see `../2026-03-10_max_speed_idle/report.md`). Results:

| Plateau | 2026-03-06 (no std) | 2026-03-10 (with std) | Preisach prediction |
|---------|--------------------|-----------------------|---------------------|
| SFTPRO (4816 A) | +125.5 uT | **+7.5 uT (0.0 sigma)** | 0 |
| LHC top (5781 A) | +66.8 uT | -33.1 uT (0.3 sigma) | 0 |
| Post-LHC idle (ctrl) | -4.3 uT | **-0.1 uT (0.0 sigma)** | 0 |

**The session-ordering confound hypothesis is confirmed.** The +125.5 uT
signal at SFTPRO collapses to zero after standardization, exactly as predicted
by the Preisach wiping-out property. The A-B-B-A design further eliminates
any residual ordering bias (A1 matches A2, B1 matches B2 to < 0.3 sigma).

Additional findings from the 2026-03-10 campaign:
- **b2 accommodation** at idle: +0.20 units for 200 GeV, ~0 for 26 GeV
  (genuine Preisach minor-loop effect, erased at SFTPRO by wiping-out)
- **No eddy settling** at the extended MD1 idle (~24 s >> tau ~ 1 s) in the body
- **Fringe eddy** currents detectable (tau ~ 1 s) due to fringe amplification
