"""
Generate a PDF report for the MBB hysteresis analysis.
Figures + summary table only (no code).

Usage:
    python generate_hysteresis_report.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[4]
MEAS = REPO / "measurements"
OUT_PDF = Path(__file__).resolve().parent / "hysteresis_report.pdf"

# ── Configuration ──────────────────────────────────────────────────────
SESSIONS = {
    "200 GeV": {
        "desc": "Full MD1 (peak ~2267 A)",
        "results_body": "MBB/2026-03-06_max_speed_NMR/20260306_152236_SPS_MBB/20260306_152257_MBB/20260306_152257_MBB_Run_00_I_100.00A_body_results.txt",
        "nmr_h5": "MBB/2026-03-06_max_speed_NMR/20260306_152447_TestCaylarTeslameterNMR20_ md1full_.h5",
    },
    "26 GeV": {
        "desc": "Flattened MD1 (injection only, ~301 A)",
        "results_body": "MBB/2026-03-06_max_speed_NMR/20260306_153553_SPS_MBB/20260306_153614_MBB/20260306_153614_MBB_Run_00_I_100.00A_body_results.txt",
        "nmr_h5": "MBB/2026-03-06_max_speed_NMR/20260306_153650_TestCaylarTeslameterNMR20_ md1flat_.h5",
    },
}
SESSION_NAMES = list(SESSIONS.keys())

I_BANDS = {
    "idle": (145, 170), "injection": (290, 315),
    "sftpro_top": (4800, 4830), "lhc_top": (5770, 5790),
}
N_SETTLE = {"idle": 18, "injection": 18, "sftpro_top": 15, "lhc_top": 5}
RAMPRATE_THRESHOLD = 5.0  # plateau noise floor ~2.5 A/s peak, ramps jump to 100+ A/s
MD1_SKIP_FIRST = 1
NMR_SETTLE_S = 1.0
HALL_LHC_MIN = 1.95

COLORS = {"200 GeV": "tab:blue", "26 GeV": "tab:orange"}
PLATEAU_COLORS = {
    "idle": "tab:cyan", "injection": "tab:green",
    "sftpro_top": "tab:red", "lhc_top": "tab:purple",
}
COLOR_MAP = {"ramp": "tab:red", "transient": "tab:orange", "settled": "tab:green"}
LABEL_MAP = {"ramp": "Ramp", "transient": "Plateau (excluded)",
             "settled": "Plateau (settled, used)"}

plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})


# ── Helper functions ───────────────────────────────────────────────────
def load_turns(cfg):
    fpath = MEAS / cfg["results_body"]
    df = pd.read_csv(fpath, sep="\t")
    df = df.rename(columns={
        "Time(s)": "t_s", "Duration(s)": "dur_s", "I(A)": "I_A",
        "Ramprate(A/s)": "rr",
        "B_main(T)": "B1_T", "A_main(T)": "A1_T",
        "b2(Units)": "b2", "a2(Units)": "a2",
        "b3(Units)": "b3", "a3(Units)": "a3",
        "b5(Units)": "b5", "a5(Units)": "a5",
    })
    df["turn"] = np.arange(len(df))
    df["t_end"] = df["t_s"] + df["dur_s"]
    # Encoder offset correction (equivalent to encoder_offset_rad = -pi/2)
    df["B1_T"] = -df["B1_T"]
    if "A1_T" in df.columns:
        df["A1_T"] = -df["A1_T"]
    for n in range(2, 16):
        if n % 2 == 0:
            for prefix in ["b", "a"]:
                for col in [f"{prefix}{n}", f"{prefix}{n}(Units)"]:
                    if col in df.columns:
                        df[col] = -df[col]
    return df


def classify_and_group(df, rr_thresh=RAMPRATE_THRESHOLD, min_consecutive=3):
    I = df["I_A"].values
    rr = df["rr"].values
    on_plateau = np.abs(rr) < rr_thresh
    band = np.full(len(I), "", dtype=object)
    for bname, (lo, hi) in I_BANDS.items():
        mask = (I >= lo) & (I <= hi) & on_plateau
        band[mask] = bname
    groups = []
    i = 0
    while i < len(I):
        if band[i] != "":
            label = band[i]; start = i
            while i < len(I) and band[i] == label:
                i += 1
            if i - start >= min_consecutive:
                groups.append({"label": label, "start": start, "end": i,
                               "n_turns": i - start, "I_mean": I[start:i].mean(),
                               "t_start": df["t_s"].iloc[start],
                               "t_end": df["t_s"].iloc[i - 1]})
        else:
            i += 1
    return groups


def assign_cycles(groups):
    sft = next((g for g in groups if g["label"] == "sftpro_top"), None)
    lhc = next((g for g in groups if g["label"] == "lhc_top"), None)
    t_sft = sft["t_start"] if sft else 1e9
    t_lhc = lhc["t_start"] if lhc else 1e9
    md1_inj_idx = 0
    for g in groups:
        t, lab = g["t_start"], g["label"]
        if t < t_sft:
            if lab == "injection":
                md1_inj_idx += 1; g["cycle"] = "MD1"; g["cycle_idx"] = md1_inj_idx
            elif lab == "idle":
                g["cycle"] = "pre-SFTPRO"; g["cycle_idx"] = 0
            else:
                g["cycle"] = "MD1"; g["cycle_idx"] = 0
        elif lab == "sftpro_top":
            g["cycle"] = "SFTPRO"; g["cycle_idx"] = 0
        elif t > t_sft and t < t_lhc:
            g["cycle"] = "SFTPRO\u2192LHC" if lab == "idle" else ("LHC" if lab == "injection" else "transition")
            g["cycle_idx"] = 0
        elif lab == "lhc_top":
            g["cycle"] = "LHC"; g["cycle_idx"] = 0
        else:
            g["cycle"] = "post-LHC"; g["cycle_idx"] = 0
    return groups


def classify_turns(df, grps):
    status = np.full(len(df), "ramp", dtype=object)
    for g in grps:
        n_s = min(N_SETTLE.get(g["label"], g["n_turns"]), g["n_turns"])
        status[g["start"]:g["end"] - n_s] = "transient"
        status[g["end"] - n_s:g["end"]] = "settled"
    return status


def get_group(groups, cycle, label):
    for g in groups:
        if g["cycle"] == cycle and g["label"] == label:
            return g
    return None


def get_settled_slice(df, group, n_settle):
    n = min(n_settle, group["n_turns"])
    return df.iloc[group["end"] - n : group["end"]]


def settled_stats(df, group, n_settle):
    n = min(n_settle, group["n_turns"])
    sub = df.iloc[group["end"] - n : group["end"]]
    out = {"n_turns": n}
    for col in ["I_A", "B1_T", "b2", "b3"]:
        if col in sub.columns:
            out[f"{col}_mean"] = sub[col].mean()
            out[f"{col}_std"] = sub[col].std()
    return out


def draw_turn_bars(ax, df, status, idx=None):
    if idx is None:
        idx = range(len(df))
    drawn = set()
    for i in idx:
        s = status[i]; kw = {}
        if s not in drawn:
            kw["label"] = LABEL_MAP[s]; drawn.add(s)
        ax.plot([df.t_s.iloc[i], df.t_end.iloc[i]],
                [df.I_A.iloc[i], df.I_A.iloc[i]],
                linewidth=2.5, color=COLOR_MAP[s], alpha=0.85,
                solid_capstyle="butt", zorder=2, **kw)


def compute_nmr_colors(ds, settle_s):
    lock = ds["lock"]; t_s = ds["t_ms"] / 1000.0; n = len(lock)
    tsl = np.zeros(n); lock_start = 0.0; in_lock = False
    for i in range(n):
        if lock[i] == 1:
            if not in_lock:
                in_lock = True; lock_start = t_s[i]
            tsl[i] = t_s[i] - lock_start
        else:
            in_lock = False
    return lock == 0, (lock == 1) & (tsl < settle_s), (lock == 1) & (tsl >= settle_s), t_s


def get_nmr_settled(nmr_ds, plat_name):
    _, _, mask_s, t_s = compute_nmr_colors(nmr_ds, NMR_SETTLE_S)
    lock = nmr_ds["lock"]; hall = nmr_ds["hall"]; nmr_abs = np.abs(nmr_ds["nmr"])
    diff = np.diff(lock)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if lock[-1] == 1:
        ends = np.append(ends, len(lock))
    for s, e in zip(starts, ends):
        if t_s[e - 1] - t_s[s] < 0.5:
            continue
        p = "LHC top" if hall[s:e].mean() > HALL_LHC_MIN else "SFTPRO top"
        if p == plat_name:
            sm = (t_s[s:e] - t_s[s]) >= NMR_SETTLE_S
            vals = nmr_abs[s:e][sm]
            if len(vals) > 0:
                return vals.mean(), vals.std(), len(vals)
    return np.nan, np.nan, 0


# ── Load data ──────────────────────────────────────────────────────────
print("Loading data...")
turns = {}
all_groups = {}
for name, cfg in SESSIONS.items():
    turns[name] = load_turns(cfg)
    grps = assign_cycles(classify_and_group(turns[name]))
    all_groups[name] = grps

nmr_data = {}
for name, cfg in SESSIONS.items():
    with h5py.File(MEAS / cfg["nmr_h5"], "r") as f:
        data = f["RawData/SFTPRO-NMR"][:]
        start_s = f["RawData/CaylarStartTime(s)"][0, 0]
    nmr_data[name] = {"t_ms": data[:, 0], "nmr": data[:, 1],
                       "hall": data[:, 2], "lock": data[:, 3].astype(int),
                       "start_s": start_s}

print("Generating PDF report...")

# ── PDF generation ─────────────────────────────────────────────────────
with PdfPages(str(OUT_PDF)) as pdf:

    # ── Page 1: Title ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.65, "Hysteresis Analysis", ha="center", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.55, "Effect of MD1 Conditioning on SFTPRO & LHC Cycles",
             ha="center", fontsize=18)
    fig.text(0.5, 0.42, "MBB max-speed + NMR campaign  \u2014  2026-03-06",
             ha="center", fontsize=14, color="grey")
    fig.text(0.5, 0.28,
             "200 GeV session: 20\u00d7 full MD1 (peak ~2267 A) \u2192 SFTPRO \u2192 LHC\n"
             "26 GeV session:  20\u00d7 flat MD1 (peak ~301 A)  \u2192 SFTPRO \u2192 LHC\n\n"
             "Body segment  \u2022  Encoder offset corrected  \u2022  Plateau by |ramp rate| < 5 A/s",
             ha="center", fontsize=11, linespacing=1.8)
    fig.text(0.5, 0.05, "Generated automatically from hysteresis_analysis.ipynb",
             ha="center", fontsize=8, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 2: Current profile (full) ─────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=False)
    for ax, name in zip(axes, SESSION_NAMES):
        df = turns[name]
        ax.plot(df.t_s, df.I_A, "-", lw=0.5, color=COLORS[name])
        ax.set_ylabel("I (A)")
        ax.set_title(f"{name} \u2014 {SESSIONS[name]['desc']}")
        for bname, (lo, hi) in I_BANDS.items():
            ax.axhspan(lo, hi, alpha=0.08, color="grey")
            ax.text(df.t_s.iloc[0] + 2, (lo + hi) / 2, bname, fontsize=7,
                    va="center", color="grey")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Current Profile \u2014 Both Sessions", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 3: Turn map (full) ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=False)
    for ax, name in zip(axes, SESSION_NAMES):
        status = classify_turns(turns[name], all_groups[name])
        draw_turn_bars(ax, turns[name], status)
        ax.set_ylabel("I (A)")
        ax.set_title(f"{name} \u2014 {SESSIONS[name]['desc']}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Current Profile with Turn Classification", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 4: Turn map (SFTPRO+LHC zoom) ────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=False)
    for ax, name in zip(axes, SESSION_NAMES):
        df = turns[name]; grps = all_groups[name]
        status = classify_turns(df, grps)
        non_md1 = [g for g in grps if g["cycle"] not in ("MD1",)]
        if not non_md1:
            continue
        t_lo = non_md1[0]["t_start"] - 10
        idx_zoom = np.where(df.t_s.values >= t_lo)[0]
        draw_turn_bars(ax, df, status, idx_zoom)
        ax.set_ylabel("I (A)")
        ax.set_title(f"{name} \u2014 SFTPRO + LHC region")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SFTPRO + LHC Region \u2014 Turn Classification (zoom)", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 5: Transfer Function (settled turns) ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5))
    for name in SESSION_NAMES:
        df = turns[name]
        status = classify_turns(df, all_groups[name])
        mask = status == "settled"
        sub = df[mask].copy()
        sub = sub[sub.I_A > 100]  # exclude idle (TF = B1/I diverges)
        sub["TF"] = sub.B1_T / sub.I_A * 1000  # T/kA
        axes[0].plot(sub.t_s, sub.TF, ".", ms=3, color=COLORS[name], alpha=0.6, label=name)
        axes[1].plot(sub.I_A, sub.TF, ".", ms=3, color=COLORS[name], alpha=0.6, label=name)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("TF = B1/I (T/kA)")
    axes[0].set_title("Transfer Function vs Time"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("I (A)"); axes[1].set_ylabel("TF = B1/I (T/kA)")
    axes[1].set_title("Transfer Function vs Current"); axes[1].legend(fontsize=8)
    fig.suptitle("Transfer Function \u2014 Settled Turns Only (body)", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 6: MD1 convergence ────────────────────────────────────────
    md1_conv = {}
    for name in SESSION_NAMES:
        md1_grps = [g for g in all_groups[name]
                    if g["cycle"] == "MD1" and g["label"] == "injection"]
        rows = []
        for g in md1_grps:
            s = settled_stats(turns[name], g, N_SETTLE["injection"])
            s["cycle_idx"] = g["cycle_idx"]; rows.append(s)
        md1_conv[name] = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.5))
    for col, ylabel, ax in zip(["B1_T", "b2", "b3"],
                                ["B1 (T)", "b2 (units)", "b3 (units)"], axes):
        for name in SESSION_NAMES:
            dc = md1_conv[name]
            dc = dc[dc.cycle_idx > MD1_SKIP_FIRST]
            ax.errorbar(dc.cycle_idx, dc[f"{col}_mean"], yerr=dc[f"{col}_std"],
                        fmt="o-", ms=3, capsize=2, color=COLORS[name], alpha=0.8, label=name)
        ax.set_xlabel("MD1 cycle #"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at injection (~301 A)")
        if col != "B1_T":
            ax.axhline(0, color="grey", lw=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("MD1 Minor Loop Convergence \u2014 Injection Plateau (body, cycle 1 excluded)",
                 fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 6: SFTPRO top turn-by-turn ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.5))
    for col, ylabel, ax in zip(["B1_T", "b2", "b3"],
                                ["B1 (T)", "b2 (units)", "b3 (units)"], axes):
        for name in SESSION_NAMES:
            g = get_group(all_groups[name], "SFTPRO", "sftpro_top")
            sub = turns[name].iloc[g["start"]:g["end"]]
            lt = sub.t_s.values - sub.t_s.values[0]
            ax.plot(lt, sub[col], "o-", ms=3, color=COLORS[name], alpha=0.7, label=name)
        ax.set_xlabel("Time on plateau (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"SFTPRO top \u2014 {ylabel}")
        ax.legend(fontsize=8)
    fig.suptitle("SFTPRO Top Plateau \u2014 Turn-by-Turn (body)", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 7: Post-SFTPRO idle turn-by-turn ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.5))
    for col, ylabel, ax in zip(["B1_T", "b2", "b3"],
                                ["B1 (T)", "b2 (units)", "b3 (units)"], axes):
        for name in SESSION_NAMES:
            g = get_group(all_groups[name], "SFTPRO\u2192LHC", "idle")
            if g is None:
                continue
            sub = turns[name].iloc[g["start"]:g["end"]]
            lt = sub.t_s.values - sub.t_s.values[0]
            ax.plot(lt, sub[col], "o-", ms=2, color=COLORS[name], alpha=0.7, label=name)
        ax.set_xlabel("Time on plateau (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"Post-SFTPRO idle \u2014 {ylabel}")
        ax.legend(fontsize=8)
    fig.suptitle("Post-SFTPRO Idle Plateau \u2014 Turn-by-Turn (body)", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 8: LHC top turn-by-turn ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.5))
    for col, ylabel, ax in zip(["B1_T", "b2", "b3"],
                                ["B1 (T)", "b2 (units)", "b3 (units)"], axes):
        for name in SESSION_NAMES:
            g = get_group(all_groups[name], "LHC", "lhc_top")
            if g is None:
                continue
            sub = turns[name].iloc[g["start"]:g["end"]]
            lt = sub.t_s.values - sub.t_s.values[0]
            ax.plot(lt, sub[col], "o-", ms=4, color=COLORS[name], alpha=0.7, label=name)
        ax.set_xlabel("Time on plateau (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"LHC top \u2014 {ylabel}")
        ax.legend(fontsize=8)
    fig.suptitle("LHC Top Plateau \u2014 Turn-by-Turn (body)", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 9: NMR 3-colour timeline ─────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    for ax, name in zip(axes, SESSION_NAMES):
        ds = nmr_data[name]
        mu, me, ms, t_s = compute_nmr_colors(ds, NMR_SETTLE_S)
        nmr_abs = np.abs(ds["nmr"])
        ax.plot(t_s[mu], nmr_abs[mu], ".", ms=1, color="0.70", alpha=0.4,
                label="unlocked", rasterized=True)
        ax.plot(t_s[me], nmr_abs[me], ".", ms=2.5, color="tab:orange", alpha=0.7,
                label=f"locked < {NMR_SETTLE_S:.0f} s")
        ax.plot(t_s[ms], nmr_abs[ms], ".", ms=2.5, color="tab:green", alpha=0.8,
                label=f"locked \u2265 {NMR_SETTLE_S:.0f} s (settled)")
        ax.set_ylabel("|NMR| (T)"); ax.set_title(f"{name} \u2014 NMR")
        ax.legend(fontsize=8, markerscale=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("NMR Lock Status \u2014 Grey=unlocked, Orange=early, Green=settled",
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 10: NMR zoom on lock periods ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    for row, name in enumerate(SESSION_NAMES):
        ds = nmr_data[name]
        _, _, _, t_s = compute_nmr_colors(ds, NMR_SETTLE_S)
        nmr_abs = np.abs(ds["nmr"]); lock = ds["lock"]
        diff = np.diff(lock)
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if lock[-1] == 1:
            ends = np.append(ends, len(lock))
        durations = sorted([(t_s[min(e-1, len(t_s)-1)] - t_s[s], s, e)
                            for s, e in zip(starts, ends)], key=lambda x: -x[0])
        for ci, (dur, s, e) in enumerate(durations[:2]):
            ax = axes[row, ci]
            tl = t_s[s:e] - t_s[s]; nl = nmr_abs[s:e]
            m_e = tl < NMR_SETTLE_S; m_s = tl >= NMR_SETTLE_S
            ax.plot(tl[m_e], nl[m_e], ".", ms=3, color="tab:orange", alpha=0.7,
                    label=f"< {NMR_SETTLE_S:.0f} s")
            ax.plot(tl[m_s], nl[m_s], ".", ms=3, color="tab:green", alpha=0.8,
                    label=f"\u2265 {NMR_SETTLE_S:.0f} s")
            ax.axvline(NMR_SETTLE_S, color="grey", ls="--", lw=0.8, alpha=0.5)
            hall_mean = ds["hall"][s:e].mean()
            plat = "LHC top" if hall_mean > HALL_LHC_MIN else "SFTPRO top"
            if m_s.sum() > 0:
                sm = nl[m_s].mean(); ss = nl[m_s].std()
                ax.axhline(sm, color="tab:green", ls=":", lw=0.8)
                ax.text(0.98, 0.05, f"settled: {sm:.6f} \u00b1 {ss:.6f} T",
                        transform=ax.transAxes, ha="right", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_title(f"{name} \u2014 {plat} ({dur:.1f} s lock)", fontsize=10)
            ax.set_xlabel("Time since lock (s)"); ax.set_ylabel("|NMR| (T)")
            ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("NMR Lock Periods \u2014 Zoom", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

    # ── Page 11: Summary table ─────────────────────────────────────────
    plateaus = [
        ("SFTPRO top\n(4816 A)",        "SFTPRO",       "sftpro_top"),
        ("SFTPRO\u2192LHC idle\n(155 A)", "SFTPRO\u2192LHC", "idle"),
        ("LHC injection\n(302 A)",       "LHC",          "injection"),
        ("LHC top\n(5781 A)",           "LHC",          "lhc_top"),
        ("Post-LHC idle\n(155 A)",      "post-LHC",     "idle"),
    ]

    # Gather data
    table_data = []  # each row: [plateau, 200 B1, 26 B1, dB1, 200 b2, 26 b2, db2, 200 b3, 26 b3, db3]
    for plat_label, cycle, label in plateaus:
        row = [plat_label]
        vals = {}
        for sname in SESSION_NAMES:
            g = get_group(all_groups[sname], cycle, label)
            if g is None:
                vals[sname] = {"B1_T": np.nan, "b2": np.nan, "b3": np.nan}
                continue
            sub = get_settled_slice(turns[sname], g, N_SETTLE.get(label, 18))
            vals[sname] = {c: sub[c].mean() for c in ["B1_T", "b2", "b3"]}

        v200 = vals.get("200 GeV", {"B1_T": np.nan, "b2": np.nan, "b3": np.nan})
        v26 = vals.get("26 GeV", {"B1_T": np.nan, "b2": np.nan, "b3": np.nan})

        for qty, fmt200, fmtd in [("B1_T", ".6f", ".1f"), ("b2", ".3f", ".3f"), ("b3", ".3f", ".3f")]:
            row.append(format(v200[qty], fmt200))
            row.append(format(v26[qty], fmt200))
            d = v26[qty] - v200[qty]
            if qty == "B1_T":
                row.append(format(d * 1e6, fmtd))  # in uT
            else:
                row.append(format(d, fmtd))
        table_data.append(row)

    # Also add NMR rows for top plateaus
    for plat_label, nmr_plat in [("SFTPRO top\n(NMR)", "SFTPRO top"),
                                  ("LHC top\n(NMR)", "LHC top")]:
        row = [plat_label]
        n200, _, _ = get_nmr_settled(nmr_data["200 GeV"], nmr_plat)
        n26, _, _ = get_nmr_settled(nmr_data["26 GeV"], nmr_plat)
        row.append(format(n200, ".6f"))
        row.append(format(n26, ".6f"))
        row.append(format((n26 - n200) * 1e6, ".1f"))
        row.extend(["\u2014"] * 6)  # no b2/b3 for NMR
        table_data.append(row)

    col_labels = [
        "Plateau",
        "200 GeV\nB1 (T)", "26 GeV\nB1 (T)", "\u0394 B1\n(\u00b5T)",
        "200 GeV\nb2 (units)", "26 GeV\nb2 (units)", "\u0394 b2\n(units)",
        "200 GeV\nb3 (units)", "26 GeV\nb3 (units)", "\u0394 b3\n(units)",
    ]

    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.95, "Hysteresis Summary: \u0394 = 26 GeV (flat MD1) \u2212 200 GeV (full MD1)",
            ha="center", va="top", fontsize=14, fontweight="bold",
            transform=ax.transAxes)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)

    # Highlight delta columns
    for i in range(1, len(table_data) + 1):
        for j in [3, 6, 9]:  # delta columns
            cell = table[i, j]
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")

    # Colour plateau column
    for i in range(1, len(table_data) + 1):
        table[i, 0].set_text_props(fontsize=7, ha="left")
        table[i, 0].set_facecolor("#f8f8f8")

    # NMR rows in light blue
    for i in range(len(plateaus) + 1, len(table_data) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#e8f4fd")

    ax.text(0.5, 0.08,
            "Body segment only  \u2022  Encoder offset corrected (equiv. to \u2212\u03c0/2)  \u2022  "
            "Plateau: |ramp rate| < 5 A/s\n"
            "NMR: settled values \u2265 1 s after lock acquisition",
            ha="center", va="bottom", fontsize=8, color="grey",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 12: Bar chart of deltas ──────────────────────────────────
    plat_names = [p[0].replace("\n", " ") for p in plateaus]
    x = np.arange(len(plat_names))

    deltas = {"B1": [], "b2": [], "b3": []}
    for _, cycle, label in plateaus:
        vals = {}
        for sname in SESSION_NAMES:
            g = get_group(all_groups[sname], cycle, label)
            if g is None:
                vals[sname] = {"B1_T": np.nan, "b2": np.nan, "b3": np.nan}
                continue
            sub = get_settled_slice(turns[sname], g, N_SETTLE.get(label, 18))
            vals[sname] = {c: sub[c].mean() for c in ["B1_T", "b2", "b3"]}
        v200 = vals.get("200 GeV", {"B1_T": np.nan, "b2": np.nan, "b3": np.nan})
        v26 = vals.get("26 GeV", {"B1_T": np.nan, "b2": np.nan, "b3": np.nan})
        deltas["B1"].append((v26["B1_T"] - v200["B1_T"]) * 1e6)
        deltas["b2"].append(v26["b2"] - v200["b2"])
        deltas["b3"].append(v26["b3"] - v200["b3"])

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 5))
    for ax, (qty, unit) in zip(axes, [("B1", "\u00b5T"), ("b2", "units"), ("b3", "units")]):
        vals = np.array(deltas[qty])
        colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
        ax.bar(x, vals, color=colors, alpha=0.7, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(plat_names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(f"\u0394 {qty} ({unit})")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"\u0394 {qty}  (26 GeV \u2212 200 GeV)")
    fig.suptitle("Static Hysteresis Impact \u2014 Bar Chart", fontsize=13, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

print(f"\nPDF report saved to: {OUT_PDF}")
