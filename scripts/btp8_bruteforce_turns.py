"""Brute-force turn-selection recovery for BTP8 golden standard.

For each of the 37 runs (~14 turns each, 6 in the reference), exhaustively
try all C(14,6) = 3003 ordered combinations.  The combination that matches
the reference at machine precision IS the one the C++ analyzer selected.

SM18 streaming already proved all equations are identical at ~1e-12 T level.
So the correct BTP8 combination should also give machine-precision match.
"""

from __future__ import annotations

import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rotating_coil_analyzer.analysis.kn_pipeline import (
    compute_legacy_kn_per_turn,
    load_segment_kn_txt,
    merge_coefficients,
)

# ── Configuration ──────────────────────────────────────────────────────
DATASET = REPO / "golden_standards/golden_standard_01_LIU_BTP8/Integral/20190717_161332_LIU"
KN_PATH = REPO / "golden_standards/golden_standard_01_LIU_BTP8/COIL_PCB/Kn_R45_PCB_N1_0001_A_ABCD.txt"

MAGNET_ORDER = 2
R_REF_M = 0.059
SAMPLES_PER_TURN = 512
SHAFT_SPEED_RPM = 60
OPTIONS = ("dri", "rot", "cel", "fed")  # same as reference

# ── Helpers ────────────────────────────────────────────────────────────
def parse_btp8_flux(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 2], data[:, 1]   # df_abs, df_cmp, encoder

def parse_btp8_current(path):
    return np.loadtxt(path)

def encoder_to_time(enc, rpm=SHAFT_SPEED_RPM, res=40000):
    return enc / (rpm * res / 60.0)

def extract_current(path):
    m = re.search(r"I_(\d+)A", path.name)
    return int(m.group(1)) if m else 0


# ── Load reference ─────────────────────────────────────────────────────
ref_path = next(
    f for f in DATASET.glob("*results*.txt")
    if "Average" not in f.name and "Parameters" not in f.name
)
ref_df = pd.read_csv(ref_path, sep="\t")
print(f"Reference: {ref_path.name}  ({len(ref_df)} turns)")

# Find column names
I_col = next(c for c in ref_df.columns if "I(A)" in c)

# Build comparison columns
REF_COLS = []
COMP_COLS = []

# B1 (T), A1 (T), B2 (T) — Tesla for n <= m
for n in range(1, MAGNET_ORDER + 1):
    for component in ("B", "A"):
        for pat in (f"{component}{n} (T)", f"{component}{n}(T)"):
            if pat in ref_df.columns:
                REF_COLS.append(pat)
                COMP_COLS.append(f"{component}{n}_T")
                break

# b3..b15, a3..a15 — units for n > m
for n in range(MAGNET_ORDER + 1, 16):
    for component in ("b", "a"):
        for pat in (f"{component}{n} (units)", f"{component}{n}(units)"):
            if pat in ref_df.columns:
                REF_COLS.append(pat)
                COMP_COLS.append(f"{component}{n}_units")
                break

print(f"Using {len(REF_COLS)} comparison channels per turn")

# ── Detect run boundaries in reference ──────────────────────────────────
ref_I = ref_df[I_col].values.astype(float)
ref_run_starts = [0]
for i in range(1, len(ref_I)):
    if abs(ref_I[i] - ref_I[i - 1]) > 2.0:
        ref_run_starts.append(i)
ref_run_starts.append(len(ref_df))
n_ref_runs = len(ref_run_starts) - 1

# ── Load Kn ─────────────────────────────────────────────────────────────
kn = load_segment_kn_txt(KN_PATH)

# ── Discover flux/current file pairs ────────────────────────────────────
flux_files = sorted(DATASET.glob("*_fluxes_Ascii.txt"))
current_files = sorted(DATASET.glob("*Run*_current.txt"))
assert len(flux_files) == len(current_files) == n_ref_runs, (
    f"File count mismatch: {len(flux_files)} flux, {len(current_files)} current, {n_ref_runs} ref runs"
)

# ── Process all runs & brute-force ──────────────────────────────────────
print(f"\n{'Run':>4s} {'I (A)':>8s} {'turns':>6s} {'combos':>8s} "
      f"{'best_turns':>22s} {'total|diff|':>14s} {'max|diff|':>14s} "
      f"{'2nd_score':>14s} {'gap':>10s}")
print("-" * 120)

all_results = []
t0 = time.time()

for run_id in range(n_ref_runs):
    # Reference slice for this run
    rs = ref_run_starts[run_id]
    re_ = ref_run_starts[run_id + 1]
    n_ref = re_ - rs
    I_level = ref_I[rs]

    ref_mat = ref_df.iloc[rs:re_][REF_COLS].values.astype(np.float64)  # (n_ref, n_ch)

    # Process raw data
    fp, cp = flux_files[run_id], current_files[run_id]
    df_abs, df_cmp, encoder = parse_btp8_flux(fp)
    current = parse_btp8_current(cp)
    time_arr = encoder_to_time(encoder)

    n_flux = len(df_abs)
    if len(current) != n_flux:
        idx = np.linspace(0, len(current) - 1, n_flux).astype(int)
        current = current[idx]

    n_turns = n_flux // SAMPLES_PER_TURN
    n_samp = n_turns * SAMPLES_PER_TURN
    shape = (n_turns, SAMPLES_PER_TURN)

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs[:n_samp].reshape(shape),
        df_cmp_turns=df_cmp[:n_samp].reshape(shape),
        t_turns=time_arr[:n_samp].reshape(shape),
        I_turns=current[:n_samp].reshape(shape),
        kn=kn,
        Rref_m=R_REF_M,
        magnet_order=MAGNET_ORDER,
        options=OPTIONS,
        legacy_rotate_excludes_last=False,
    )

    C_merged, _ = merge_coefficients(
        C_abs=result.C_abs, C_cmp=result.C_cmp,
        magnet_order=MAGNET_ORDER, mode="abs_upto_m_cmp_above",
    )

    # Build computed matrix (n_turns, n_ch) with same column layout
    comp_mat = np.empty((n_turns, len(COMP_COLS)), dtype=np.float64)
    for t in range(n_turns):
        Bm = C_merged[t, MAGNET_ORDER - 1].real
        col_idx = 0
        for i, n in enumerate(result.orders):
            if n > 15:
                break
            C = C_merged[t, i]
            if n <= MAGNET_ORDER:
                # B_n and A_n in Tesla
                bn_key = f"B{n}_T"
                an_key = f"A{n}_T"
                if bn_key in COMP_COLS:
                    j = COMP_COLS.index(bn_key)
                    comp_mat[t, j] = C.real
                if an_key in COMP_COLS:
                    j = COMP_COLS.index(an_key)
                    comp_mat[t, j] = C.imag
            else:
                # b_n and a_n in units
                bn_key = f"b{n}_units"
                an_key = f"a{n}_units"
                if bn_key in COMP_COLS:
                    j = COMP_COLS.index(bn_key)
                    if abs(Bm) > 1e-30:
                        comp_mat[t, j] = C.real / Bm * 10000.0
                    else:
                        comp_mat[t, j] = np.nan
                if an_key in COMP_COLS:
                    j = COMP_COLS.index(an_key)
                    if abs(Bm) > 1e-30:
                        comp_mat[t, j] = C.imag / Bm * 10000.0
                    else:
                        comp_mat[t, j] = np.nan

    # ── Brute-force all C(n_turns, n_ref) combinations ──
    best_combo = None
    best_score = np.inf
    second_score = np.inf

    for combo in combinations(range(n_turns), n_ref):
        selected = comp_mat[list(combo)]
        score = np.nansum(np.abs(selected - ref_mat))
        if score < best_score:
            second_score = best_score
            best_score = score
            best_combo = combo
        elif score < second_score:
            second_score = score

    # Per-turn max |diff| for the winning combo
    selected_best = comp_mat[list(best_combo)]
    per_turn_max = np.nanmax(np.abs(selected_best - ref_mat), axis=1)
    max_diff = np.nanmax(per_turn_max)

    gap = second_score / best_score if best_score > 0 else np.inf

    result_dict = {
        "run_id": run_id,
        "I_A": I_level,
        "n_turns_raw": n_turns,
        "n_ref": n_ref,
        "best_turns": list(best_combo),
        "total_abs_diff": best_score,
        "max_abs_diff": max_diff,
        "second_score": second_score,
        "gap_ratio": gap,
    }
    all_results.append(result_dict)

    elapsed = time.time() - t0
    print(f"{run_id:4d} {I_level:+8.1f} {n_turns:6d} {len(list(combinations(range(n_turns), n_ref))):8d} "
          f"{str(list(best_combo)):>22s} {best_score:14.3e} {max_diff:14.3e} "
          f"{second_score:14.3e} {gap:10.1f}x  [{elapsed:.1f}s]")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

df = pd.DataFrame(all_results)
print(f"\nTotal runs       : {len(df)}")
print(f"Total turns ref  : {df['n_ref'].sum()}")
print(f"Total |diff|     : min={df['total_abs_diff'].min():.3e}, "
      f"max={df['total_abs_diff'].max():.3e}, median={df['total_abs_diff'].median():.3e}")
print(f"Max |diff|       : min={df['max_abs_diff'].min():.3e}, "
      f"max={df['max_abs_diff'].max():.3e}, median={df['max_abs_diff'].median():.3e}")
print(f"Gap (2nd/1st)    : min={df['gap_ratio'].min():.1f}x, "
      f"max={df['gap_ratio'].max():.1f}x, median={df['gap_ratio'].median():.1f}x")

# Check for machine precision
MACHINE_THRESH = 1e-8  # generous threshold
n_machine = (df["max_abs_diff"] < MACHINE_THRESH).sum()
print(f"\nRuns at machine precision (max|diff| < {MACHINE_THRESH}): {n_machine} / {len(df)}")

# Show the turn selection pattern
print(f"\nTurn selections by run:")
standard_14 = [0, 1, 2, 3, 4, 13]
standard_15 = [0, 1, 2, 3, 4, 14]
for _, r in df.iterrows():
    turns = r["best_turns"]
    marker = ""
    if turns == standard_14 or turns == standard_15:
        marker = "  (standard: first-5 + last)"
    print(f"  Run {int(r['run_id']):2d} (I={r['I_A']:+7.1f}A, {int(r['n_turns_raw'])} raw): "
          f"{turns}  max|diff|={r['max_abs_diff']:.3e}{marker}")

# Count standard vs non-standard
n_std = sum(1 for _, r in df.iterrows()
            if r["best_turns"] == standard_14 or r["best_turns"] == standard_15)
print(f"\nStandard pattern (first-5 + last): {n_std} / {len(df)}")
print(f"Non-standard: {len(df) - n_std} / {len(df)}")

# Machine-precision verdict
print("\n" + "=" * 80)
if n_machine == len(df):
    print("VERDICT: ALL runs match at machine precision.")
    print("The turn selection has been uniquely determined for every run.")
    print("Equations are identical to machine precision (float64).")
else:
    n_good = (df["max_abs_diff"] < 1e-6).sum()
    n_close = (df["max_abs_diff"] < 1e-3).sum()
    print(f"VERDICT: {n_machine}/{len(df)} at machine precision, "
          f"{n_good}/{len(df)} at sub-ppm, {n_close}/{len(df)} at sub-unit.")
print("=" * 80)
