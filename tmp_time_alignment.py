import sys
sys.path.insert(0, r"C:\Users\albellel\python-projects\rotating-coil-analyzer")

import numpy as np
import pandas as pd
from pathlib import Path
from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader

BASE = "HCMCBXFB012-E9000006_20241204_150437_stair_step_outer_negative"
MEAS_DIR = Path(r"C:\Users\albellel\python-projects\rotating-coil-analyzer\golden_standards\golden_standard_SM18_01") / BASE / "aperture1"

# Read binary for Segment 3
reader = StreamingReader()
frame = reader.read(
    str(MEAS_DIR / f"{BASE}_corr_sigs_Ap_1_Seg3.bin"),
    run_id="golden", segment="3",
    samples_per_turn=512, shaft_speed_rpm=60,
)
print(f"Binary: {frame.n_turns} turns, {len(frame.df)} samples")
print(f"Warnings: {frame.warnings}")

Ns = 512
nt = frame.n_turns

# Per-turn time: take mean time of each turn
t_all = frame.df["t"].values
I_all = frame.df["I"].values
t_turns = t_all.reshape(nt, Ns)
I_turns = I_all.reshape(nt, Ns)
t_mid = np.mean(t_turns, axis=1)  # midpoint time per turn
I_mean = np.mean(I_turns, axis=1)  # mean current per turn

# Read reference
ref = pd.read_csv(str(MEAS_DIR / f"{BASE}_results_Ap_1_Seg_3.txt"), sep="\t")
print(f"\nReference: {len(ref)} turns")
ref_t = ref["Time(s)"].values
ref_I = ref["I(A)"].values

# Time comparison
print(f"\n--- TIME COMPARISON ---")
print(f"Binary t_mid[0:5]:  {t_mid[0:5]}")
print(f"Reference t[0:5]:   {ref_t[0:5]}")
print(f"Binary t_mid[-5:]:  {t_mid[-5:]}")
print(f"Reference t[-5:]:   {ref_t[-5:]}")

# Try different offsets
for offset in range(0, 6):
    n_cmp = min(len(ref_t), nt - offset)
    dt = t_mid[offset:offset+n_cmp] - ref_t[:n_cmp]
    print(f"\nOffset={offset}: dt_mean={np.mean(dt):.6f}, dt_std={np.std(dt):.6f}, dt_max={np.max(np.abs(dt)):.6f}")
    # Also check current alignment
    dI = I_mean[offset:offset+n_cmp] - ref_I[:n_cmp]
    print(f"  dI_mean={np.mean(dI):.6f}, dI_max={np.max(np.abs(dI)):.6f}")

# Current profile summary: where are plateaus and ramps?
print(f"\n--- CURRENT PROFILE ---")
print(f"Binary I_mean[0:20]:  {I_mean[0:20]}")
print(f"Reference I[0:20]:    {ref_I[0:20]}")

# Find where current changes significantly (ramp detection)
dI_dt = np.diff(I_mean)
ramp_mask = np.abs(dI_dt) > 0.1  # more than 0.1 A/turn change
ramp_starts = np.where(np.diff(ramp_mask.astype(int)) == 1)[0]
ramp_ends = np.where(np.diff(ramp_mask.astype(int)) == -1)[0]
print(f"\nNumber of ramp transitions: {len(ramp_starts)}")
for i, (s, e) in enumerate(zip(ramp_starts[:10], ramp_ends[:10])):
    print(f"  Ramp {i+1}: turns {s}-{e}, I from {I_mean[s]:.1f} to {I_mean[e]:.1f} A")

# Check where first 2000 turns sit in current profile
print(f"\n--- FIRST 2000 TURNS ---")
print(f"I range: [{I_mean[:2000].min():.4f}, {I_mean[:2000].max():.4f}] A")
print(f"Mean I: {I_mean[:2000].mean():.4f} A")

# Check turns around 2000-5000
print(f"\n--- TURNS 2000-5000 ---")
print(f"I at turn 2000: {I_mean[2000]:.4f} A")
print(f"I at turn 3000: {I_mean[3000]:.4f} A")
print(f"I at turn 4000: {I_mean[4000]:.4f} A")
print(f"I at turn 5000: {I_mean[5000]:.4f} A")

del frame
