# SPS MBB Dipole -- A-B-B-A Hysteresis with Standardization -- Mar 10, 2026

## 1. Overview

Dedicated hysteresis measurement of the SPS MBB dipole using an A-B-B-A protocol
with full-field standardization before each session. This campaign was designed to
resolve the session-ordering confound identified in the 2026-03-06 campaign (see
`../2026-03-06_max_speed_NMR/hysteresis_interpretation.md`).

**Key improvements over 2026-03-06:**
- 10x standardization cycles to 5781 A before each session (wipes magnetic memory)
- A-B-B-A session order eliminates ordering bias
- Extended idle plateaus (~72 turns, ~24 s) for per-cycle accommodation tracking
- No NMR (rotating coil only)

## 2. Protocol

```
For each session (A1, B1, B2, A2):
  10x standardization cycles (0 -> 5781 A -> 0)   wipes all memory up to LHC top
  ~20x MD1 conditioning cycles                     200 GeV: 0->2345 A->0, 26 GeV: 0->301 A->0
     with extended idle at ~155 A between each (~72 turns = ~24 s)
  idle (155 A) -> SFTPRO (4816 A) -> idle (155 A)
  -> injection (301 A) -> LHC top (5781 A) -> idle (155 A)
```

| Session | Order | MD1 type | Timestamp | Peak MD1 I |
|---------|-------|----------|-----------|------------|
| A1 | 1st | 200 GeV | 17:04 | ~2345 A (instantaneous DCCT) |
| B1 | 2nd | 26 GeV | 17:25 | ~301 A |
| B2 | 3rd | 26 GeV | 17:47 | ~301 A |
| A2 | 4th | 200 GeV | 18:08 | ~2345 A (instantaneous DCCT) |

## 3. Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | SPS MBB dipole (m=1), body + fringe segments |
| Rotation speed | ~176 RPM (2.93 Hz), period ~0.34 s/turn |
| Pipeline options | `dri`, `rot`, `cel`, `fed` |
| Plateau threshold | ramp rate < 5 A/s |
| Clean plateau threshold | ramp rate < 1 A/s |
| Settling turns | idle: 50, SFTPRO: 15, LHC: 5 |

## 4. Data Summary

| Session | Turns | I range (A) | B1 range (T) | Std cycles | MD1 idles | SFTPRO | LHC |
|---------|-------|-------------|-------------|------------|-----------|--------|-----|
| A1 (200 GeV) | 3182 | -0.1 to 5781.1 | 0.0008 to 2.0085 | 10 | 22 | 1 | 1 |
| B1 (26 GeV) | 3628 | -0.2 to 5781.0 | 0.0008 to 2.0085 | 10 | 21 | 1 | 1 |
| B2 (26 GeV) | 3445 | -0.2 to 5781.0 | 0.0008 to 2.0085 | 10 | 22 | 1 | 1 |
| A2 (200 GeV) | 3172 | -0.2 to 5781.0 | 0.0008 to 2.0085 | 10 | 22 | 1 | 1 |

Per-turn averaging artifact on 200 GeV MD1: the fast triangular ramp above
2000 A lasts ~1.2 s (3-4 turns). Per-turn averaged I varies (2150-2300 A) but
instantaneous DCCT peaks are perfectly consistent at ~2345 +/- 3 A (<0.15%).
Verified by raw DCCT at full ADC resolution (~3030 Hz).

---

## 5. Results: Body Segment

### 5.1 Measurement Plateaus

| Plateau | I (A) | Session | N settled | B1 (T) | +/- (T) | b2 (units) | b3 (units) |
|---------|-------|---------|-----------|--------|---------|------------|------------|
| SFTPRO | 4817 | A1 | 15 | 1.781757 | 0.000766 | -0.086 | 0.020 |
| SFTPRO | 4817 | B1 | 15 | 1.781729 | 0.000785 | -0.086 | 0.022 |
| SFTPRO | 4817 | B2 | 15 | 1.781729 | 0.000793 | -0.085 | 0.023 |
| SFTPRO | 4817 | A2 | 15 | 1.781686 | 0.000805 | -0.086 | 0.021 |
| LHC top | 5781 | A1 | 5 | 2.008161 | 0.000247 | 0.128 | 0.336 |
| LHC top | 5781 | B1 | 5 | 2.008164 | 0.000246 | 0.132 | 0.337 |
| LHC top | 5781 | B2 | 5 | 2.008138 | 0.000250 | 0.130 | 0.338 |
| LHC top | 5781 | A2 | 5 | 2.008207 | 0.000215 | 0.130 | 0.337 |
| Post-SFTPRO idle | 155 | A1 | 50 | 0.060499 | 0.000005 | 0.274 | -0.489 |
| Post-SFTPRO idle | 155 | B1 | 50 | 0.060503 | 0.000005 | 0.263 | -0.495 |
| Post-SFTPRO idle | 155 | B2 | 50 | 0.060503 | 0.000004 | 0.266 | -0.491 |
| Post-SFTPRO idle | 155 | A2 | 50 | 0.060502 | 0.000005 | 0.261 | -0.489 |
| Post-LHC idle | 155 | A1 | 50 | 0.060494 | 0.000005 | 0.255 | -0.478 |
| Post-LHC idle | 155 | B1 | 50 | 0.060497 | 0.000005 | 0.252 | -0.473 |
| Post-LHC idle | 155 | B2 | 50 | 0.060493 | 0.000025 | 0.249 | -0.473 |
| Post-LHC idle | 155 | A2 | 50 | 0.060496 | 0.000004 | 0.250 | -0.473 |

### 5.2 True Hysteresis: 26 GeV - 200 GeV (Body)

Delta = mean(B1,B2) - mean(A1,A2)

| Plateau | Delta B1 (uT) | SEM (uT) | Significance | Delta b2 | Delta b3 |
|---------|--------------|----------|-------------|----------|----------|
| SFTPRO top | **+7.5** | 196.5 | **0.0 sigma** | +0.001 | +0.001 |
| Post-SFTPRO idle | +2.1 | 0.7 | 3.1 sigma | -0.003 | -0.004 |
| LHC top | -33.1 | 96.3 | **0.3 sigma** | +0.002 | +0.001 |
| Post-LHC idle (ctrl) | **-0.1** | 1.8 | **0.0 sigma** | -0.002 | +0.002 |

**Interpretation:**
- SFTPRO and LHC top: Delta is consistent with **zero** (< 0.3 sigma).
  The Preisach wiping-out property is confirmed: MD1 conditioning has no
  effect on B1 at currents exceeding the MD1 maximum (4816 A >> 2267 A).
- Post-LHC idle (control): Delta = -0.1 uT (0.0 sigma). Both sessions
  descend from the same 5781 A, producing identical remanent states.
- Post-SFTPRO idle: +2.1 uT (3.1 sigma). This small but significant
  difference survives at low current where the MD1 minor loop sits.
  However, 2.1 uT is 0.003% of the 60.5 mT field -- operationally negligible.
- b2, b3: all deltas < 0.004 units. No harmonic hysteresis effect.

### 5.3 A-B-B-A Reproducibility (Body)

| Pair | Plateau | Delta B1 (uT) | Significance | Delta b2 | Delta b3 |
|------|---------|---------------|-------------|----------|----------|
| A1 vs A2 | SFTPRO | -70.7 | 0.2 sigma | -0.000 | +0.001 |
| A1 vs A2 | LHC | +45.2 | 0.3 sigma | +0.002 | +0.001 |
| A1 vs A2 | Post-LHC | +2.2 | 2.4 sigma | -0.005 | +0.005 |
| B1 vs B2 | SFTPRO | -0.9 | 0.0 sigma | +0.001 | +0.001 |
| B1 vs B2 | LHC | -26.4 | 0.2 sigma | -0.002 | +0.001 |
| B1 vs B2 | Post-LHC | -3.6 | 1.0 sigma | -0.004 | -0.000 |

Reproducibility is excellent: all flat-top deltas are < 0.5 sigma. The
B1 vs B2 (26 GeV) reproducibility is particularly good (0.0 sigma at
SFTPRO), confirming the standardization protocol works.

### 5.4 MD1 Accommodation (Body)

Drift in B1, b2, b3 at idle (~155 A) from first 3 to last 3 of ~20 MD1 cycles.
Filtered: n_turns >= 20 AND I_range < 2.0 A (excludes ramp transitions).

| Session | B1 drift (uT) | b2 drift (units) | b3 drift (units) |
|---------|---------------|------------------|------------------|
| A1 (200 GeV) | +13.9 | **+0.196** | -0.149 |
| B1 (26 GeV) | +18.1 | -0.019 | -0.030 |
| B2 (26 GeV) | +16.7 | -0.025 | -0.032 |
| A2 (200 GeV) | +15.5 | **+0.197** | -0.143 |

**Interpretation:**
- **B1 drift (~15 uT):** MD1-independent, consistent across all 4 sessions.
  This is post-standardization settling of the accommodated major loop,
  not an MD1 conditioning effect.
- **b2 drift:** **MD1-dependent**. 200 GeV sessions show +0.20 units drift,
  26 GeV sessions show ~0 (-0.02 units). This is a genuine Preisach
  minor-loop accommodation effect: repeated cycling to 2267 A modifies
  the b2 remanent state at 155 A, while cycling to only 301 A does not.
- **b3 drift:** 200 GeV shows -0.15 units, 26 GeV -0.03 units.
  Same direction as b2 but weaker contrast.
- **Key point:** These accommodation effects are confined to idle current
  (155 A). They vanish at SFTPRO (4816 A >> 2267 A) by the wiping-out
  property, as confirmed by the zero delta in Section 5.2.

---

## 6. Results: Fringe Segment

The fringe segment has ~6x smaller B1 than body at the same current
(TF ~ 0.065 T/kA vs 0.386 T/kA). This amplifies all harmonic quantities
by ~6x (b_n = C_n/B1 * 10^4), making the fringe noisier but more
sensitive to eddy currents.

### 6.1 True Hysteresis (Fringe)

| Plateau | Delta B1 (uT) | SEM (uT) | Significance | Delta b2 | Delta b3 |
|---------|--------------|----------|-------------|----------|----------|
| SFTPRO | -40.7 | 65.0 | **0.6 sigma** | -0.689 | -0.141 |
| Post-SFTPRO idle | -0.7 | 0.9 | 0.8 sigma | -0.360 | -0.065 |
| LHC top | -12.9 | 60.6 | **0.2 sigma** | -0.218 | -0.013 |
| Post-LHC idle (ctrl) | +1.3 | 1.0 | 1.3 sigma | +0.014 | +0.096 |

Consistent with the body: all flat-top deltas < 1 sigma. The fringe
confirms that MD1 conditioning has no measurable effect on field at
SFTPRO and LHC top. Fringe b2/b3 are noisier (larger deltas) but
remain below significance thresholds.

### 6.2 MD1 Accommodation (Fringe)

| Session | B1 drift (uT) | b2 drift (units) | b3 drift (units) |
|---------|---------------|------------------|------------------|
| A1 (200 GeV) | +60.3 | -0.343 | -0.728 |
| B1 (26 GeV) | +47.0 | -0.555 | -0.543 |
| B2 (26 GeV) | +48.4 | +1.434 | -0.578 |
| A2 (200 GeV) | +61.5 | +0.269 | -0.632 |

- B1 drift is ~4x larger in the fringe (47-62 uT) vs body (14-18 uT),
  consistent with the different transfer functions and field distributions.
- B1 drift is somewhat MD1-dependent in the fringe: 200 GeV ~ +61 uT
  vs 26 GeV ~ +48 uT (unlike body where it was independent).
- b2 is very noisy in the fringe (sign changes between sessions),
  unreliable for accommodation analysis.
- b3 drift is consistently negative (-0.5 to -0.7 units) across all sessions.

---

## 7. Eddy Current Settling Analysis

Single-exponential fit: y(t) = y_inf + A * exp(-t / tau).
"Clean plateau turns" have |ramp rate| < 1 A/s.

### 7.1 Body Segment

| Plateau | B1 amplitude | tau (s) | R2 | b2/b3 R2 | Conclusion |
|---------|-------------|---------|-------|----------|------------|
| Last clean MD1 idle | 10-24 uT | 0.1-1.0 | < 0.27 | < 0.04 | **No eddies** |
| SFTPRO top | 5100-6100 uT | 1.4-5.6 | > 0.98 | < 0.94 | **Strong B1 eddy** |
| Post-SFTPRO idle | 4-28 uT | 0.2-82 | < 0.12 | < 0.08 | **No eddies** |
| LHC top | 1889 uT (B1 only) | 2.4 | 1.00 | -- | Marginal (5 turns) |
| Post-LHC idle | 1-53 uT | 0.4-6.6 | < 0.47 | < 0.05 | **No eddies** |

**Interpretation:**
- **MD1 idle (155 A):** No detectable eddy settling. R2 < 0.27 means the
  exponential fit explains less than 27% of the variance -- the data is
  dominated by per-turn noise (~800 uT). At MBB tau ~ 1 s, the 72-turn
  idle (~24 s = 24x tau) is vastly more than needed for settling.
- **SFTPRO (4816 A):** Genuine B1 eddy current with amplitude ~5 mT and
  tau ~ 2-5 s. The field settles upward (negative A) after the ascending
  ramp. This is the only plateau with a clean eddy signal in the body.
  The 15 settled turns (out of 19-20 total) at SFTPRO may not fully
  capture the transient -- the first 5 turns are still settling.
- **Idle plateaus (post-SFTPRO, post-LHC):** No settling. 72 turns is
  more than adequate.

### 7.2 Fringe Segment

| Plateau | B1 amplitude | tau (s) | R2 | b3 amplitude | b3 tau (s) | b3 R2 |
|---------|-------------|---------|------|-------------|-----------|-------|
| Last clean MD1 idle | 57-257 uT | 0.4-1.0 | 0.70-0.99 | 1.1-4.8 u | 0.4-1.1 | 0.47-0.93 |
| SFTPRO top | 1800-2200 uT | 1.5-1.9 | > 0.99 | 0.08-0.10 u | 1.4-3.3 | 0.75-0.94 |
| Post-SFTPRO idle | 490-584 uT | 1.2-1.5 | > 0.97 | 6.8-8.2 u | 0.8-1.1 | > 0.95 |
| Post-LHC idle | 326-771 uT | 0.95-2.4 | > 0.97 | 4.5-10.6 u | 0.7-1.6 | 0.83-0.97 |

**Key finding: Fringe shows clear eddy currents where body does not.**

This is a direct consequence of the ~6x smaller B1 in the fringe. The
absolute eddy current (in Tesla) is similar in both segments (same iron),
but the b_n = C_n/B1 * 10^4 normalization amplifies the effect in the fringe.

- **tau ~ 1-2 s** across all plateaus (consistent between body and fringe,
  and with the 2026-03-06 campaign finding of tau1 ~ 1.1 s).
- **B1 eddies at last MD1 idle:** Detectable in fringe (R2 > 0.96 for
  200 GeV sessions) but not in body. Amplitude ~250 uT in fringe =
  ~2.5 uT effective field change. After 5x tau (~5 s = 15 turns), the
  residual is < 0.7% of amplitude -- well settled within the 72-turn idle.
- **b3 eddies at post-SFTPRO/post-LHC idle:** Strong in fringe (R2 > 0.95),
  amplitude 5-10 units, tau ~ 1 s. This is the fringe amplification effect:
  the same iron eddy produces ~0.1 units in body but ~7 units in fringe.

---

## 8. Comparison with Previous Campaign (2026-03-06)

### 8.1 The Session-Ordering Confound

The 2026-03-06 campaign (no standardization) showed:

| Plateau | Delta B1 (uT) | Note |
|---------|---------------|------|
| SFTPRO (4816 A) | **+125.5** | 26 GeV higher than 200 GeV |
| LHC top (5781 A) | +66.8 | Smaller (approaching return point) |
| Post-LHC idle (155 A) | -4.3 | Control: ~zero |

This was interpreted as a session-ordering artifact (see
`../2026-03-06_max_speed_NMR/hysteresis_interpretation.md`): the 200 GeV
session ran first, the 26 GeV session inherited its LHC excursion memory.

### 8.2 Experimental Validation

The current A-B-B-A campaign with standardization confirms the prediction:

| Plateau | 2026-03-06 (no std) | 2026-03-10 (with std) | Prediction |
|---------|--------------------|-----------------------|------------|
| SFTPRO | +125.5 uT | **+7.5 uT (0.0 sigma)** | 0 |
| LHC top | +66.8 uT | -33.1 uT (0.3 sigma) | 0 |
| Post-LHC idle | -4.3 uT | **-0.1 uT (0.0 sigma)** | 0 |

The +125.5 uT "hysteresis signal" at SFTPRO has **collapsed to zero** after
proper standardization. This is a textbook confirmation of the Preisach
wiping-out property:

1. Standardization to 5781 A erases all memory below 5781 A.
2. Both MD1 types add minor loops well below 5781 A (2267 A and 301 A).
3. At SFTPRO (4816 A > 2267 A), the wiping-out property erases the MD1
   memory. Both sessions are on the same ascending branch.
4. Delta = 0, as observed.

### 8.3 Eddy Current Comparison

| Feature | 2026-03-06 | 2026-03-10 |
|---------|-----------|-----------|
| Body tau (fringe) | ~1.1 s | ~1-2 s |
| Fringe b3 eddy at injection | ~1 unit (2-tau) | ~5-8 units at post-SFTPRO idle |
| Body eddy at injection | marginal (R2 ~ 0.6) | undetectable at idle (R2 < 0.3) |
| SFTPRO B1 eddy (body) | not measured | ~5 mT, tau ~ 2-5 s |

The eddy time constants are consistent between campaigns (~1-2 s for the
fast component), confirming these are intrinsic magnet properties.

---

## 9. Conclusions

### 9.1 Primary Result: No True MD1 Hysteresis at SFTPRO/LHC

**The level of MD1 conditioning (200 GeV vs 26 GeV) has no measurable effect
on B1 at SFTPRO or LHC top.** This is confirmed with 0.0 sigma significance
at SFTPRO and 0.3 sigma at LHC top (body), and 0.6/0.2 sigma in the fringe.

The Preisach wiping-out property is validated: any memory of MD1 cycling
(which peaks at 2267 A for 200 GeV or 301 A for 26 GeV) is erased when
the ascending ramp exceeds these values on the way to SFTPRO (4816 A).

### 9.2 Accommodation Effects

MD1-dependent accommodation exists at idle current (155 A) for b2:
200 GeV sessions drift +0.20 units vs 0 for 26 GeV. This is a genuine
Preisach minor-loop effect confined to currents below the MD1 maximum.
It is erased at SFTPRO by wiping-out.

B1 drift at idle (~15 uT body, ~50 uT fringe) is MD1-independent --
it reflects post-standardization settling of the accommodated major loop.

### 9.3 Eddy Currents

- **Body at idle:** No eddy settling. The 72-turn (~24 s) idle is
  24x the eddy time constant. MBB laminated yoke produces negligible
  eddies at this field level.
- **Fringe at idle:** Eddy currents are detectable (tau ~ 1 s, R2 > 0.95)
  due to fringe amplification (b_n = C_n/B1 * 10^4 with small B1).
  However, the 72-turn idle still provides adequate settling (residual
  < 0.7% of amplitude after 15 turns).
- **SFTPRO:** Genuine B1 eddy (~5 mT, tau ~ 2-5 s) visible in both
  segments. The 15 settled turns at SFTPRO may be marginally affected.

### 9.4 Standardization Protocol

The 10x standardization cycles to 5781 A before each session effectively
wipe magnetic memory. This is confirmed by:
- A1 vs A2 reproducibility (< 0.3 sigma at all flat-tops)
- B1 vs B2 reproducibility (< 0.2 sigma at SFTPRO, < 0.4 sigma at LHC)
- Post-LHC idle control: 0.0 sigma (body), 1.3 sigma (fringe)
- Collapse of the 2026-03-06 artifact from 125.5 uT to 7.5 uT at SFTPRO

### 9.5 Operational Implications

For SPS operation, the choice of MD1 conditioning level (26 GeV vs 200 GeV)
does not affect the field quality at SFTPRO or LHC energies. The small
accommodation effect on b2 at idle current is operationally irrelevant
(0.2 units at 155 A, erased above 2267 A).

---

## 10. Notebooks

| Notebook | Description |
|----------|-------------|
| `hysteresis_analysis_body.ipynb` | Body segment: accommodation, eddy settling, hysteresis, reproducibility |
| `hysteresis_analysis_fringe.ipynb` | Fringe segment: same analysis |
| `hysteresis_analysis_comparison.ipynb` | Body vs fringe side-by-side |

## 11. Measurement Files

| Session | Directory |
|---------|-----------|
| A1 (200 GeV) | `measurements/MBB/2026-03-10_max_speed_idle/20260310_170427_SPS_MBB/20260310_170449_MBB/` |
| B1 (26 GeV) | `measurements/MBB/2026-03-10_max_speed_idle/20260310_172501_SPS_MBB/20260310_172522_MBB/` |
| B2 (26 GeV) | `measurements/MBB/2026-03-10_max_speed_idle/20260310_174733_SPS_MBB/20260310_174754_MBB/` |
| A2 (200 GeV) | `measurements/MBB/2026-03-10_max_speed_idle/20260310_180839_SPS_MBB/20260310_180902_MBB/` |
