"""Unified notebook generator for MBB and MC62 rotating coil analyses.

Replaces three separate scripts:
  - generate_mbb_2hz_notebooks.py  (MBB NCS+CS, 2 Hz sessions)
  - generate_mbb_notebooks.py      (MBB NCS-only sessions)
  - update_mc62_notebooks.py        (MC62 notebook updates)

Generates ALL analysis and comparison notebooks from measurement configs.

Usage:
    python scripts/generate_notebooks.py --all
    python scripts/generate_notebooks.py --mbb
    python scripts/generate_notebooks.py --mc62
    python scripts/generate_notebooks.py MBB_2Hz_200GeV
"""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path


# ================================================================
# Notebook helpers
# ================================================================

def make_cell(cell_type, cell_id, source_lines):
    """Create an ipynb cell dict."""
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source_lines,
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def md(cell_id, text):
    """Create a markdown cell from multiline text."""
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("markdown", cell_id, source)


def code(cell_id, text):
    """Create a code cell from multiline text."""
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", cell_id, source)


def write_notebook(path, cells):
    """Write a notebook to disk."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.1"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Wrote {path} ({len(cells)} cells)")


# ================================================================
# Configuration dataclasses
# ================================================================

@dataclass
class SegmentConfig:
    """Per-segment configuration."""
    name: str                      # "NCS", "CS", "Integral", "Central"
    kn_path: str                   # relative to measurements/
    merge_mode: str = "abs_upto_m_cmp_above"
    data_path: str | None = None   # per-segment file/bin path within session
    is_fringe: bool = False


@dataclass
class MeasurementConfig:
    """Full measurement analysis configuration."""
    # Identity
    title: str
    magnet_family: str             # "MBB" | "MC62"
    notebook_path: str             # output notebook path (relative to repo root)
    output_csv_dir: str            # CSV export dir (relative to output/)

    # Magnet physics
    magnet_order: int              # 1=dipole
    r_ref: float                   # reference radius [m]
    l_coil: float                  # coil length [m]
    samples_per_turn: int

    # Segments + data loading
    segments: list                 # list[SegmentConfig]
    data_loader: str               # "text_streaming" | "binary_streaming" | "file_discovery"
    session: str = ""              # measurement session path rel. to measurements/
    meas_subdir: str = ""          # measurement subdirectory within session (text_streaming)
    run_dir_rel: str = ""          # run directory rel. to measurements/ (file_discovery)

    # Pipeline
    options: tuple = ("dri", "rot", "cel", "fed")
    encoder_offset_rad: float = 0.0
    flip_signal_polarity: bool = False
    drift_mode: str = "legacy"
    min_b1_T: float = 1e-4

    # Plateau detection (streaming)
    plateau_i_range_max: float = 2.5
    plateau_n_blocks: int = 10
    plateau_min_length: int = 50
    plateau_merge_gap: int = 0

    # Averaging
    n_last_turns: int = 170
    n_last_turns_high: int | None = None  # flat-top (None = use all)
    n_skip_end: int = 0
    n_sigma_clip: float = 5.0
    rpm: float = 120.0

    # Optional sections
    has_precycle: bool = False
    has_fdi: bool = True
    has_allturn: bool = True
    has_ffmm: bool = False
    has_eddy: bool = True
    has_inductance: bool = True

    # FFMM config
    ffmm_r_ref: float | None = None
    ffmm_options: tuple = ("dri", "rot")
    ffmm_rotate_excludes_last: bool = True

    # Labels
    energy_label: str = ""
    min_injection_turns: int = 5

    @property
    def segment_names(self):
        return [s.name for s in self.segments]

    @property
    def t_per_turn(self):
        return 60.0 / self.rpm

    @property
    def main_segment(self):
        """First non-fringe segment (for single-segment analysis)."""
        for s in self.segments:
            if not s.is_fringe:
                return s.name
        return self.segments[0].name


@dataclass
class ComparisonConfig:
    """Comparison notebook configuration."""
    title: str
    notebook_path: str
    magnet_family: str             # "MBB" or "MC62"
    segments: list                 # list of segment names
    datasets: list                 # list of {"name": str, "csv_dir": str}
    output_csv_dir: str = ""
    n_last_turns: int = 170


# ================================================================
# Section-builder helpers
# ================================================================

def _seg_configs_repr(cfg):
    """Generate SEGMENT_CONFIGS + SEGMENTS Python code for the notebook."""
    lines = ["SEGMENT_CONFIGS = ["]
    for s in cfg.segments:
        lines.append(
            f'    {{"name": "{s.name}", "kn_path": "{s.kn_path}", '
            f'"merge_mode": "{s.merge_mode}", "is_fringe": {s.is_fringe}}},'
        )
    lines.append("]")
    lines.append('SEGMENTS = [s["name"] for s in SEGMENT_CONFIGS]')
    return "\n".join(lines)


def _kn_paths_repr(cfg):
    """Generate KN_PATHS dict literal."""
    items = ", ".join(f'"{s.name}": "{s.kn_path}"' for s in cfg.segments)
    return f"KN_PATHS = {{{items}}}"


def _roman(n):
    """Convert integer to Roman numeral."""
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = ""
    for val, sym in vals:
        while n >= val:
            result += sym
            n -= val
    return result


def _categorize_section(name):
    """Assign a section to a TOC part."""
    if any(kw in name for kw in [
        "Configuration", "Kn", "Loading", "Discovery", "Raw Signal",
        "cel/fed", "Plateau Det", "FDI", "Precycle", "Current Prof", "Channel",
    ]):
        return "Setup & Data Quality"
    if any(kw in name for kw in ["Process", "All-Turn", "FFMM", "Plateau Qual"]):
        return "Pipeline Processing"
    if any(kw in name for kw in ["Main Field", "b2", "b3", "Higher", "Spectrum", "Multipole"]):
        return "Harmonic Analysis"
    if any(kw in name for kw in ["Transfer", "Inductance"]):
        return "Transfer Function & Inductance"
    if any(kw in name for kw in ["Settling", "Exponential", "Bias", "N_LAST", "Double"]):
        return "Eddy Current & Settling"
    return "Summary"


def _list_streaming_sections(cfg):
    """List section names for a streaming analysis notebook."""
    sections = [
        "Configuration & Imports",
        "Kn Calibration",
        "Data Loading & Channel Detection",
        "Raw Signals Overview",
        "cel/fed Safety Diagnostic",
        "Plateau Detection & Turn Classification",
    ]
    if cfg.has_fdi:
        sections.append("FDI Stuck-Channel Diagnostic")
    if cfg.has_precycle:
        sections.append("Precycle Identification")
    sections.append("Process Plateau Turns")
    if cfg.has_allturn:
        sections.append("All-Turn Harmonics vs Time")
    if cfg.has_ffmm:
        sections.append("FFMM Golden Standard Validation")
    sections.extend([
        "Main Field (B1)", "b2 (Quadrupole)", "b3 (Sextupole)",
        "Higher Harmonics Overview", "Multipole Spectrum",
        "Transfer Function B1/I",
    ])
    if cfg.has_inductance:
        sections.append("Apparent vs Differential Inductance")
    if cfg.has_eddy:
        sections.extend([
            "Eddy Current Settling Analysis", "Exponential Fits",
            "Settling Bias Analysis", "N_LAST Sensitivity Study",
        ])
    sections.extend([
        "Comprehensive Statistics Table",
        "Analysis Choices Summary",
        "CSV Export",
    ])
    return sections


def _list_file_discovery_sections(cfg):
    """List section names for a file-discovery analysis notebook."""
    sections = [
        "Configuration & Imports",
        "Kn Calibration",
        "Run Discovery & Data Loading",
        "Current Profile",
        "cel/fed Safety Diagnostic",
        "Pipeline Processing",
        "Plateau Quality",
        "Main Field (B1)", "b2 (Quadrupole)", "b3 (Sextupole)",
        "Higher Harmonics Overview", "Multipole Spectrum",
        "Transfer Function B1/I",
    ]
    if cfg.has_inductance:
        sections.append("Apparent vs Differential Inductance")
    if cfg.has_eddy:
        sections.extend([
            "Eddy Current Settling Analysis", "Exponential Fits",
            "Settling Bias Analysis", "N_LAST Sensitivity Study",
        ])
    sections.extend([
        "Comprehensive Statistics Table",
        "Analysis Choices Summary",
        "CSV Export",
    ])
    return sections


# ================================================================
# Section builders -- Title & TOC
# ================================================================

def section_title_toc(cfg, section_names):
    """Generate title + table of contents."""
    seg_desc = ", ".join(
        f"{s.name}{' (fringe)' if s.is_fringe else ''}" for s in cfg.segments
    )
    if cfg.magnet_family == "MBB":
        header = (
            f"# SPS MBB Dipole -- Comprehensive Analysis"
            + (f" ({cfg.energy_label})" if cfg.energy_label else "")
            + f"\n\n**Measurement session:** `{cfg.session}`"
            + f"\n**Segments:** {seg_desc}"
            + "\n**Magnet:** MBB (normal dipole, m=1)"
        )
    else:
        header = (
            f"# {cfg.title}"
            + f"\n\n**Segments:** {seg_desc}"
            + f"\n**Magnet order:** {cfg.magnet_order}"
            + f"\n**R_ref:** {cfg.r_ref} m"
        )

    # Build TOC grouped by part
    parts = []
    current_part = None
    for i, name in enumerate(section_names, 1):
        part = _categorize_section(name)
        if part != current_part:
            current_part = part
            parts.append((part, []))
        parts[-1][1].append((i, name))

    toc_lines = []
    for pi, (part_title, secs) in enumerate(parts, 1):
        toc_lines.append(f"\n### Part {_roman(pi)}: {part_title}")
        toc_lines.append("| # | Section |")
        toc_lines.append("|---|---------|")
        for num, name in secs:
            toc_lines.append(f"| {num} | {name} |")

    return [md("title", header + "\n".join(toc_lines))]


# ================================================================
# Section builders -- Configuration & Imports
# ================================================================

def section_config_imports(cfg, n):
    """Generate config constants + imports cells."""
    cells = []
    cells.append(md(f"s{n}-hdr", f"---\n## {n}. Configuration & Imports"))

    # --- Build config code ---
    lines = ["# === CONFIGURATION ===", _seg_configs_repr(cfg), ""]

    if cfg.data_loader == "text_streaming":
        lines += [
            f'SESSION = "{cfg.session}"',
            f'MEAS_SUBDIR = "{cfg.meas_subdir}"',
            _kn_paths_repr(cfg), "",
            f"MAGNET_ORDER = {cfg.magnet_order}",
            f"R_REF = {cfg.r_ref}",
            f"L_COIL = {cfg.l_coil}",
            f"SAMPLES_PER_TURN = {cfg.samples_per_turn}", "",
            f"OPTIONS = {cfg.options}",
            f"MIN_B1_T = {cfg.min_b1_T}",
            f"PLATEAU_I_RANGE_MAX = {cfg.plateau_i_range_max}",
            f"N_BLOCKS = {cfg.plateau_n_blocks}", "",
            f"N_LAST_TURNS_INJ = {cfg.n_last_turns}",
            f"N_LAST_TURNS_HIGH = {repr(cfg.n_last_turns_high)}", "",
            f"N_SIGMA_CLIP = {cfg.n_sigma_clip}",
            f"MIN_INJECTION_TURNS = {cfg.min_injection_turns}",
        ]
    elif cfg.data_loader == "binary_streaming":
        # Build DATA_PATHS from segment configs
        dp_items = ", ".join(
            f'"{s.name}": "{s.data_path}"' for s in cfg.segments if s.data_path
        )
        lines += [
            f'SESSION = "{cfg.session}"',
            f"DATA_PATHS_REL = {{{dp_items}}}",
            _kn_paths_repr(cfg), "",
            f"MAGNET_ORDER = {cfg.magnet_order}",
            f"R_REF = {cfg.r_ref}",
            f"SAMPLES_PER_TURN = {cfg.samples_per_turn}", "",
            f"OPTIONS = {cfg.options}",
            f"ENCODER_OFFSET_RAD = {cfg.encoder_offset_rad}",
            f"MIN_B1_T = {cfg.min_b1_T}",
            f"RPM = {cfg.rpm}",
            "T_PER_TURN = 60.0 / RPM", "",
            f"PLATEAU_I_RANGE_MAX = {cfg.plateau_i_range_max}",
            f"N_BLOCKS = {cfg.plateau_n_blocks}",
            f"PLATEAU_MIN_LENGTH = {cfg.plateau_min_length}",
            f"PLATEAU_MERGE_GAP = {cfg.plateau_merge_gap}", "",
            f"N_LAST_TURNS = {cfg.n_last_turns}",
            f"N_SKIP_END = {cfg.n_skip_end}",
            f"N_SIGMA_CLIP = {cfg.n_sigma_clip}",
        ]
    else:  # file_discovery
        lines += [
            f'RUN_DIR_REL = "{cfg.run_dir_rel}"',
            _kn_paths_repr(cfg), "",
            f"MAGNET_ORDER = {cfg.magnet_order}",
            f"R_REF = {cfg.r_ref}",
            f"SAMPLES_PER_TURN = {cfg.samples_per_turn}", "",
            f"OPTIONS = {cfg.options}",
            f"ENCODER_OFFSET_RAD = {cfg.encoder_offset_rad}",
            f"MIN_B1_T = {cfg.min_b1_T}",
            "T_PER_TURN = 1.0  # 1 Hz rotation", "",
            f"N_LAST_TURNS = {cfg.n_last_turns}",
            f"N_SKIP_END = {cfg.n_skip_end}",
            f"N_SIGMA_CLIP = {cfg.n_sigma_clip}",
        ]

    # Print summary
    lines += [
        "", f'print("{cfg.title}")', 'print("=" * 60)',
        'print(f"  Segments      : {SEGMENTS}")',
        'print(f"  Magnet order  : {MAGNET_ORDER}")',
        'print(f"  R_ref         : {R_REF} m")',
        'print(f"  Samples/turn  : {SAMPLES_PER_TURN}")',
        'print(f"  Options       : {OPTIONS}")',
    ]
    cells.append(code(f"s{n}-config", "\n".join(lines)))

    # --- Build imports cell ---
    imp = [
        "import sys", "from pathlib import Path",
        "import numpy as np", "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from matplotlib.patches import Patch",
        "from scipy.optimize import curve_fit", "",
        "%matplotlib widget",
        "plt.rcParams.update({",
        '    "figure.figsize": (14, 5),',
        '    "axes.grid": True,',
        '    "grid.alpha": 0.3,',
        '    "figure.dpi": 100,',
        "})", "",
        'REPO_ROOT = Path(".").resolve()',
        "while REPO_ROOT != REPO_ROOT.parent:",
        '    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists():',
        "        break",
        "    REPO_ROOT = REPO_ROOT.parent",
        "if str(REPO_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(REPO_ROOT))", "",
        "from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt",
        "from rotating_coil_analyzer.analysis.utility_functions import (",
        "    process_kn_pipeline,",
        "    build_harmonic_rows,",
        "    diagnose_cel_fed,",
        "    mad_sigma_clip,",
        "    eddy_model,",
        "    fit_eddy_per_run,",
        "    plateau_summary,",
    ]

    if cfg.data_loader in ("text_streaming", "binary_streaming"):
        imp += [
            "    compute_block_averaged_range,",
            "    detect_plateau_turns,",
            "    classify_current,",
            "    find_contiguous_groups,",
            "    diagnose_fdi_transitions,",
        ]
    if cfg.data_loader == "file_discovery":
        imp += [
            "    discover_runs,",
        ]

    imp += [
        ")",
        "from rotating_coil_analyzer.ingest.channel_detect import robust_range",
    ]

    if cfg.data_loader == "text_streaming":
        imp.insert(2, "import re")

    # Session/directory setup
    imp.append("")
    if cfg.data_loader == "text_streaming":
        imp += [
            'SESSION_DIR = REPO_ROOT / "measurements" / SESSION',
            "RUN_DIR = SESSION_DIR / MEAS_SUBDIR",
        ]
    elif cfg.data_loader == "binary_streaming":
        imp += [
            'SESSION_DIR = REPO_ROOT / "measurements" / SESSION',
            "BIN_PATHS = {seg: SESSION_DIR / rel for seg, rel in DATA_PATHS_REL.items()}",
        ]
    else:
        imp += [
            'RUN_DIR = REPO_ROOT / "measurements" / RUN_DIR_REL',
        ]

    # Kn path setup
    imp += [
        "",
        "KN = {}",
        "for _seg_name, _kn_rel in KN_PATHS.items():",
        '    _kp = REPO_ROOT / "measurements" / _kn_rel',
        "    assert _kp.exists(), f\"Kn file not found: {_kp}\"",
        "    KN[_seg_name] = load_segment_kn_txt(str(_kp))",
        "",
        'print(f"Repo root : {REPO_ROOT}")',
        'print(f"Kn loaded : {list(KN.keys())}")',
        'print("Imports ready.")',
    ]

    cells.append(code(f"s{n}-imports", "\n".join(imp)))
    return cells


# ================================================================
# Section builders -- Kn Calibration
# ================================================================

def section_kn(cfg, n):
    """Generate Kn display cell."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Kn Calibration")]
    cells.append(code(f"s{n}-kn", """\
for seg_name, kn_seg in KN.items():
    H = len(kn_seg.orders)
    print(f"\\n{seg_name}: {H} harmonics")
    print(f"  Orders: {list(kn_seg.orders)}")
    kn_abs_n1 = abs(kn_seg.kn_abs[0])
    kn_cmp_n1 = abs(kn_seg.kn_cmp[0])
    ratio = kn_abs_n1 / max(kn_cmp_n1, 1e-30)
    print(f"  |Kn_abs(n=1)| = {kn_abs_n1:.6e}")
    print(f"  |Kn_cmp(n=1)| = {kn_cmp_n1:.6e}")
    print(f"  Abs/Cmp ratio (n=1): {ratio:.0f}x")

# Use first segment's Kn for harmonic count
_first_seg = SEGMENTS[0]
H = len(KN[_first_seg].orders)
Ns = SAMPLES_PER_TURN
m = MAGNET_ORDER
print(f"\\nH={H}, Ns={Ns}, m={m}")"""))
    return cells


# ================================================================
# Section builders -- Data Loading
# ================================================================

def section_load_text_streaming(cfg, n):
    """Data loading for MBB text streaming files."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Data Loading & Channel Detection\n\n"
                f"Load raw measurement data for all segments.")]
    cells.append(code(f"s{n}-load", """\
FILE_PAT = re.compile(
    r"Run_(\\d+)_I_([\\d.]+)A_(N?CS)_raw_measurement_data\\.txt$"
)

data = {}
for seg in SEGMENTS:
    seg_files = [
        f for f in sorted(RUN_DIR.iterdir())
        if FILE_PAT.search(f.name) and FILE_PAT.search(f.name).group(3) == seg
    ]
    assert seg_files, f"No {seg} raw files in {RUN_DIR}"
    raw_file = seg_files[0]

    raw = np.loadtxt(raw_file)
    n_turns = raw.shape[0] // Ns
    n_keep = n_turns * Ns
    ncols = raw.shape[1]

    t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
    flux_col1 = raw[:n_keep, 1].reshape(n_turns, Ns)
    flux_col2 = raw[:n_keep, 2].reshape(n_turns, Ns)
    I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

    # Auto-detect channel swap
    I_mean_quick = I_all.mean(axis=1)
    best_turn = np.argmax(np.abs(I_mean_quick))
    r1 = robust_range(raw[best_turn * Ns:(best_turn + 1) * Ns, 1])
    r2 = robust_range(raw[best_turn * Ns:(best_turn + 1) * Ns, 2])
    swap = r2 > r1

    if swap:
        flux_abs_all, flux_cmp_all = flux_col2, flux_col1
    else:
        flux_abs_all, flux_cmp_all = flux_col1, flux_col2

    data[seg] = {
        "raw_file": raw_file, "n_turns": n_turns,
        "t_all": t_all, "flux_abs": flux_abs_all, "flux_cmp": flux_cmp_all,
        "I_all": I_all, "swap": swap, "r1": r1, "r2": r2,
    }
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe_tag = " [FRINGE FIELD]" if _scfg["is_fringe"] else ""
    print(f"\\n{seg}{fringe_tag}: {raw_file.name}")
    print(f"  Shape: {raw.shape} -> {n_turns} turns, {ncols} columns")
    print(f"  Time span: {raw[-1,0] - raw[0,0]:.1f} s ({(raw[-1,0] - raw[0,0])/60:.1f} min)")
    print(f"  Flux swap: {swap}  (abs range={max(r1,r2):.4e}, cmp range={min(r1,r2):.4e})")"""))
    return cells


def section_load_binary_streaming(cfg, n):
    """Data loading for MC62 binary streaming files."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Data Loading\n\n"
                "Load binary measurement data for all segments.")]
    cells.append(code(f"s{n}-load", """\
data = {}
for seg in SEGMENTS:
    bin_path = BIN_PATHS[seg]
    assert bin_path.exists(), f"Binary file not found: {bin_path}"
    raw = np.fromfile(str(bin_path), dtype="<f8")
    n_cols = 4
    n_samples = len(raw) // n_cols
    raw = raw.reshape(n_samples, n_cols)
    n_turns = n_samples // Ns
    n_keep = n_turns * Ns

    t_all = raw[:n_keep, 0].reshape(n_turns, Ns)
    flux_abs_all = raw[:n_keep, 1].reshape(n_turns, Ns)
    flux_cmp_all = raw[:n_keep, 2].reshape(n_turns, Ns)
    I_all = raw[:n_keep, 3].reshape(n_turns, Ns)

    data[seg] = {
        "raw_file": bin_path, "n_turns": n_turns,
        "t_all": t_all, "flux_abs": flux_abs_all, "flux_cmp": flux_cmp_all,
        "I_all": I_all, "swap": False,
    }
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe_tag = " [FRINGE]" if _scfg["is_fringe"] else ""
    print(f"\\n{seg}{fringe_tag}: {bin_path.name}")
    print(f"  Shape: ({n_samples}, {n_cols}) -> {n_turns} turns x {Ns} samples")"""))
    return cells


def section_load_file_discovery(cfg, n):
    """Run discovery and data loading for file-based MC62 measurements."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Run Discovery & Data Loading\n\n"
                "Discover and load individual run files for all segments.")]
    cells.append(code(f"s{n}-discover", """\
runs = {}
for seg in SEGMENTS:
    runs[seg] = discover_runs(RUN_DIR, seg)
    # Classify ascending / descending branch
    for i, r in enumerate(runs[seg]):
        if i == 0 or r["I_nom"] >= runs[seg][i - 1]["I_nom"]:
            r["branch"] = "ascending"
        else:
            r["branch"] = "descending"
        if i > 0 and abs(r["I_nom"] - runs[seg][i - 1]["I_nom"]) < 1.0:
            r["branch"] = runs[seg][i - 1]["branch"]
    print(f"{seg}: {len(runs[seg])} runs discovered")
    for r in runs[seg][:3]:
        print(f"  Run {r['run_id']}: I_nom={r['I_nom']:.1f} A, {r['branch']}")
    if len(runs[seg]) > 3:
        print(f"  ... ({len(runs[seg]) - 3} more)")"""))
    return cells


# ================================================================
# Section builders -- Raw Signals & Diagnostics
# ================================================================

def section_raw_signals(cfg, n):
    """Raw flux and current signal overview plots."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Raw Signals Overview")]
    cells.append(code(f"s{n}-raw", """\
fig, axes = plt.subplots(len(SEGMENTS), 3, figsize=(18, 5 * len(SEGMENTS)), sharex="col")
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    d = data[seg]
    n_keep = d["n_turns"] * Ns
    x = np.arange(n_keep)
    axes[i, 0].plot(x, d["flux_abs"].ravel(), linewidth=0.2, color="steelblue")
    axes[i, 0].set_ylabel(f"Flux abs ({seg})")
    axes[i, 0].set_title(f"Absolute flux -- {seg}")
    axes[i, 1].plot(x, d["flux_cmp"].ravel(), linewidth=0.2, color="teal")
    axes[i, 1].set_ylabel(f"Flux cmp ({seg})")
    axes[i, 1].set_title(f"Compensated flux -- {seg}")
    axes[i, 2].plot(x, d["I_all"].ravel(), linewidth=0.2, color="tab:orange")
    axes[i, 2].set_ylabel(f"Current ({seg})")
    axes[i, 2].set_title(f"Current -- {seg}")

axes[-1, 0].set_xlabel("Sample index")
axes[-1, 1].set_xlabel("Sample index")
axes[-1, 2].set_xlabel("Sample index")
fig.suptitle("Raw signals", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))
    return cells


def section_celfed(cfg, n):
    """cel/fed safety diagnostic."""
    main_seg = cfg.main_segment
    cells = [md(f"s{n}-hdr", f"---\n## {n}. cel/fed Safety Diagnostic\n\n"
                f"Run `diagnose_cel_fed()` on {main_seg} high-current turns.")]

    # For streaming: use data[seg], for file_discovery: use df[seg] later
    if cfg.data_loader == "file_discovery":
        celfed_code = f"""\
# Run cel/fed diagnostic per segment
for seg in SEGMENTS:
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    kn_seg = KN[seg]
    # Use first few runs at highest current
    _runs_sorted = sorted(runs[seg], key=lambda r: abs(r["I_nom"]), reverse=True)
    _hi_runs = _runs_sorted[:min(3, len(_runs_sorted))]
    if not _hi_runs:
        print(f"{{seg}}: no runs for cel/fed diagnostic")
        continue
    # Load turns from high-current runs
    _flux_abs, _flux_cmp, _t, _I = [], [], [], []
    for r in _hi_runs:
        raw = np.loadtxt(r["file"])
        _nt = raw.shape[0] // Ns
        _nk = _nt * Ns
        _flux_abs.append(raw[:_nk, 1].reshape(_nt, Ns))
        _flux_cmp.append(raw[:_nk, 2].reshape(_nt, Ns))
        _t.append(raw[:_nk, 0].reshape(_nt, Ns))
        _I.append(raw[:_nk, 3].reshape(_nt, Ns))
    _fa = np.concatenate(_flux_abs)[:100]
    _fc = np.concatenate(_flux_cmp)[:100]
    _tt = np.concatenate(_t)[:100]
    _ii = np.concatenate(_I)[:100]
    diag = diagnose_cel_fed(_fa, _fc, _tt, _ii, kn=kn_seg, r_ref=R_REF, magnet_order=MAGNET_ORDER)
    print(f"{{seg}} cel/fed: {{diag.recommendation}} -- {{diag.reason}}")
    if diag.recommendation == "UNSAFE":
        OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
        print(f"  -> cel/fed disabled, OPTIONS = {{OPTIONS}}")"""
    else:
        celfed_code = f"""\
# Diagnostic on main segment (not fringe)
_main_seg = "{main_seg}"
d = data[_main_seg]
I_mean = d["I_all"].mean(axis=1)
hi_mask = np.abs(I_mean) > np.percentile(np.abs(I_mean), 90)
if hi_mask.sum() < 5:
    hi_mask = np.abs(I_mean) > np.median(np.abs(I_mean))

n_diag = min(100, int(hi_mask.sum()))
if n_diag == 0:
    print(f"No high-I turns in {{_main_seg}} for cel/fed diagnostic -- skipping")
else:
    hi_idx = np.where(hi_mask)[0][:n_diag]

    diag = diagnose_cel_fed(
        d["flux_abs"][hi_idx], d["flux_cmp"][hi_idx],
        d["t_all"][hi_idx], d["I_all"][hi_idx],
        kn=KN[_main_seg], r_ref=R_REF, magnet_order=MAGNET_ORDER,
    )
    print(f"cel/fed diagnostic ({{n_diag}} {{_main_seg}} high-I turns):")
    print(f"  Recommendation: {{diag.recommendation}}")
    print(f"  {{diag.reason}}")
    Bd = np.max(np.abs(diag.B_main_with_fed - diag.B_main_without_fed))
    print(f"  B_main max |diff|: {{Bd:.4e}} T")

    if diag.recommendation == "UNSAFE":
        OPTIONS = tuple(o for o in OPTIONS if o not in ("cel", "fed"))
        print(f"  -> cel/fed disabled, OPTIONS = {{OPTIONS}}")
    else:
        print(f"  -> cel/fed safe, keeping OPTIONS = {{OPTIONS}}")"""

    cells.append(code(f"s{n}-celfed", celfed_code))
    return cells


def section_plateau_detection(cfg, n):
    """Plateau detection and turn classification for streaming data."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Plateau Detection & Turn Classification")]

    if cfg.data_loader == "binary_streaming":
        # MC62 staircase: block-averaged range + contiguous groups + run map
        cells.append(code(f"s{n}-plateau", f"""\
label_colors = {{"ascending": "tab:blue", "descending": "tab:red"}}

for seg in SEGMENTS:
    d = data[seg]
    I_mean = d["I_all"].mean(axis=1)
    t_mean = d["t_all"].mean(axis=1)
    I_range, I_blocks = compute_block_averaged_range(d["I_all"], Ns, N_BLOCKS)

    plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
    is_plateau = plateau_info["is_plateau"]

    # Find contiguous plateau groups
    groups = find_contiguous_groups(is_plateau, min_length=PLATEAU_MIN_LENGTH)

    # Merge groups separated by small gaps
    if PLATEAU_MERGE_GAP > 0 and len(groups) > 1:
        merged = [groups[0]]
        for gs, ge in groups[1:]:
            prev_e = merged[-1][1]
            if gs - prev_e <= PLATEAU_MERGE_GAP:
                merged[-1] = (merged[-1][0], ge)
            else:
                merged.append((gs, ge))
        groups = merged

    # Build turn_run_map and classify branch
    turn_run_map = np.full(d["n_turns"], -1, dtype=int)
    run_info = []
    for gi, (gs, ge) in enumerate(groups):
        turn_run_map[gs:ge+1] = gi
        I_nom = float(np.median(I_mean[gs:ge+1]))
        branch = "ascending" if gi == 0 or I_nom >= run_info[-1]["I_nom"] else "descending"
        if gi > 0 and abs(I_nom - run_info[-1]["I_nom"]) < 1.0:
            branch = run_info[-1]["branch"]
        run_info.append({{"run_id": gi, "start": gs, "end": ge,
                         "I_nom": I_nom, "branch": branch, "n_turns": ge - gs + 1}})

    d.update({{
        "I_mean": I_mean, "t_mean": t_mean, "I_range": I_range,
        "is_plateau": is_plateau, "groups": groups,
        "turn_run_map": turn_run_map, "run_info": run_info,
    }})
    print(f"\\n{{seg}}: {{is_plateau.sum()}} plateau turns, {{len(groups)}} groups")
    for ri in run_info[:5]:
        print(f"  Run {{ri['run_id']}}: I={{ri['I_nom']:+.1f}} A, {{ri['branch']}}, {{ri['n_turns']}} turns")
    if len(run_info) > 5:
        print(f"  ... ({{len(run_info) - 5}} more)")

# Plot
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5), sharey=True)
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    d = data[seg]
    ax.plot(d["t_mean"], d["I_mean"], ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
    for ri in d["run_info"]:
        gs, ge = ri["start"], ri["end"]
        col = label_colors.get(ri["branch"], "tab:purple")
        ax.scatter(d["t_mean"][gs:ge+1], d["I_mean"][gs:ge+1], s=6, color=col, zorder=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("I (A)")
    ax.set_title(f"Plateau Detection -- {{seg}}")
fig.suptitle("Current Profile & Plateau Detection", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))
    else:
        # MBB supercycle: classify into injection/flat-high
        cells.append(code(f"s{n}-plateau", """\
label_colors = {"injection": "tab:green", "flat-mid": "tab:purple", "flat-high": "tab:blue"}

for seg in SEGMENTS:
    d = data[seg]
    I_mean = d["I_all"].mean(axis=1)
    t_mean = d["t_all"].mean(axis=1)
    I_range, I_blocks = compute_block_averaged_range(d["I_all"], Ns, N_BLOCKS)

    plateau_info = detect_plateau_turns(I_blocks, I_mean, I_range, PLATEAU_I_RANGE_MAX)
    is_plateau = plateau_info["is_plateau"]

    turn_label = np.array(["ramp"] * d["n_turns"], dtype=object)
    for j in range(d["n_turns"]):
        if is_plateau[j]:
            turn_label[j] = classify_current(I_mean[j])

    inj_groups = find_contiguous_groups(turn_label == "injection", min_length=2)
    fh_groups = find_contiguous_groups(turn_label == "flat-high", min_length=2)

    d.update({
        "I_mean": I_mean, "t_mean": t_mean, "I_range": I_range,
        "is_plateau": is_plateau, "turn_label": turn_label,
        "inj_groups": inj_groups, "fh_groups": fh_groups,
    })

    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe = " [FRINGE]" if _scfg["is_fringe"] else ""
    print(f"\\n{seg}{fringe}: {is_plateau.sum()} plateau, "
          f"{len(inj_groups)} inj groups, {len(fh_groups)} flat-high groups")
    for lab in ["injection", "flat-mid", "flat-high"]:
        mask = turn_label == lab
        if mask.sum() > 0:
            print(f"  {lab:12s}: {mask.sum():4d} turns, I = {I_mean[mask].mean():.1f} +/- {I_mean[mask].std():.1f} A")
    print(f"  {'ramp':12s}: {(turn_label == 'ramp').sum():4d} turns")

# Plot
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5), sharey=True)
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    d = data[seg]
    ax.plot(d["t_mean"], d["I_mean"], ".-", markersize=1, linewidth=0.3, color="lightgrey", zorder=0)
    for lab, col in label_colors.items():
        mask = d["turn_label"] == lab
        idx = np.where(mask)[0]
        if len(idx) > 0:
            ax.scatter(d["t_mean"][idx], d["I_mean"][idx], s=6, color=col, zorder=2, label=lab)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("I (A)")
    ax.set_title(f"Plateau Detection -- {seg}"); ax.legend(fontsize=9)
fig.suptitle("Current Profile & Plateau Detection", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))
    return cells


def section_precycle_id(cfg, n):
    """Precycle identification for MC62 streaming measurements."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Precycle Identification\n\n"
                "Separate precycle from staircase based on initial plateau pattern.")]
    cells.append(code(f"s{n}-precycle", """\
for seg in SEGMENTS:
    d = data[seg]
    run_info = d["run_info"]
    # Find first 0 A plateau with >= 500 turns -> staircase start
    staircase_start_run = 0
    for ri in run_info:
        if abs(ri["I_nom"]) < 1.0 and ri["n_turns"] >= 500:
            staircase_start_run = ri["run_id"]
            break
    # Check if pre-start groups are systematic steps (no real precycle)
    pre_runs = [ri for ri in run_info if ri["run_id"] < staircase_start_run]
    if pre_runs:
        steps = [abs(pre_runs[i+1]["I_nom"] - pre_runs[i]["I_nom"]) for i in range(len(pre_runs)-1)]
        if steps and max(steps) < 50:
            print(f"{seg}: pre-start groups appear to be systematic 20 A steps, not precycle")
            staircase_start_run = 0
    d["staircase_start_run"] = staircase_start_run
    n_pre = sum(1 for ri in run_info if ri["run_id"] < staircase_start_run)
    print(f"{seg}: staircase starts at run {staircase_start_run} ({n_pre} precycle runs)")"""))
    return cells


def section_fdi(cfg, n):
    """FDI stuck-channel diagnostic."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. FDI Stuck-Channel Diagnostic\n\n"
                "Check whether the FDI responds to current changes between plateau groups.")]

    if cfg.data_loader == "binary_streaming":
        fdi_code = """\
for seg in SEGMENTS:
    d = data[seg]
    run_info = d["run_info"]
    staircase_start = d.get("staircase_start_run", 0)
    _groups = [ri for ri in run_info if ri["run_id"] >= staircase_start]
    if len(_groups) < 2:
        print(f"{seg}: fewer than 2 groups, skipping FDI check")
        continue

    flux_turns = d["flux_abs"].mean(axis=1)
    checks = diagnose_fdi_transitions(
        flux_turns, d["I_mean"], _groups,
        stuck_threshold=0.3, partial_threshold=0.7, min_delta_I=5.0,
    )
    n_ok = sum(1 for c in checks if c.severity == "OK")
    n_stuck = sum(1 for c in checks if c.severity == "STUCK")
    print(f"\\n{seg}: {len(checks)} transitions, OK={n_ok}, STUCK={n_stuck}")
    for c in checks:
        if c.severity != "OK":
            print(f"  ! Run {c.run_before}->{c.run_after}: {c.severity} -- {c.reason}")
    if n_stuck > 0:
        print(f"  WARNING: {n_stuck} stuck transitions!")
    else:
        print(f"  All transitions OK.")"""
    else:
        fdi_code = """\
for seg in SEGMENTS:
    d = data[seg]
    # Build run_info from contiguous plateau groups
    all_groups = []
    for lab_name in ["injection", "flat-mid", "flat-high"]:
        groups = find_contiguous_groups(d["turn_label"] == lab_name, min_length=2)
        for gs, ge in groups:
            all_groups.append({"start": gs, "end": ge,
                               "I_nom": float(d["I_mean"][gs:ge+1].mean())})
    all_groups.sort(key=lambda x: x["start"])
    for i, g in enumerate(all_groups):
        g["run_id"] = i

    if len(all_groups) < 2:
        print(f"{seg}: fewer than 2 plateau groups, skipping FDI check")
        continue

    flux_turns = d["flux_abs"].mean(axis=1)
    checks = diagnose_fdi_transitions(
        flux_turns, d["I_mean"], all_groups,
        stuck_threshold=0.3, partial_threshold=0.7, min_delta_I=5.0,
    )
    n_ok = sum(1 for c in checks if c.severity == "OK")
    n_stuck = sum(1 for c in checks if c.severity == "STUCK")
    print(f"\\n{seg}: {len(checks)} transitions, OK={n_ok}, STUCK={n_stuck}")
    for c in checks:
        if c.severity != "OK":
            print(f"  ! Run {c.run_before}->{c.run_after}: {c.severity} -- {c.reason}")
    if n_stuck > 0:
        print(f"  WARNING: {n_stuck} stuck transitions!")
    else:
        print(f"  All transitions OK.")"""

    cells.append(code(f"s{n}-fdi", fdi_code))
    return cells


def section_current_profile(cfg, n):
    """Current profile plot for file-discovery measurements."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Current Profile\n\n"
                "Timeline showing all discovered runs.")]
    cells.append(code(f"s{n}-profile", """\
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax, seg in zip(axes, SEGMENTS):
    for r in runs[seg]:
        color = "tab:blue" if r["branch"] == "ascending" else "tab:red"
        ax.barh(r["I_nom"], 1, left=r["run_id"], height=20, color=color, alpha=0.7)
    ax.set_xlabel("Run index"); ax.set_ylabel("I_nom (A)")
    ax.set_title(f"Current Profile -- {seg}")
fig.suptitle("Run Discovery", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))
    return cells

# ================================================================
# Section builders -- Pipeline Processing
# ================================================================

def section_pipeline_streaming(cfg, n):
    """Process plateau turns through the Kn pipeline (streaming data)."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Process Plateau Turns\n\n"
                "Re-process plateau turns with the full pipeline. "
                "Group by supercycle/run, apply settling window and MAD sigma-clip.")]

    if cfg.data_loader == "binary_streaming":
        # MC62 staircase: process per-run
        pipeline_code = f"""\
results = {{}}

for seg in SEGMENTS:
    d = data[seg]
    run_info = d["run_info"]
    staircase_start = d.get("staircase_start_run", 0)
    kn_seg = KN[seg]
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    merge_mode = _scfg["merge_mode"]

    all_rows = []
    for ri in run_info:
        if ri["run_id"] < staircase_start:
            continue
        gs, ge = ri["start"], ri["end"]
        idx = np.arange(gs, ge + 1)
        result, C_merged, C_units, ok_main = process_kn_pipeline(
            flux_abs_turns=d["flux_abs"][idx],
            flux_cmp_turns=d["flux_cmp"][idx],
            t_turns=d["t_all"][idx],
            I_turns=d["I_all"][idx],
            kn=kn_seg, r_ref=R_REF, magnet_order=m,
            options=OPTIONS, min_b1_T=MIN_B1_T,
            encoder_offset_rad=ENCODER_OFFSET_RAD,
            merge_mode=merge_mode,
        )
        extra = [
            {{"global_turn": int(idx[t]), "run_id": ri["run_id"],
              "I_nom": ri["I_nom"], "branch": ri["branch"],
              "turn_in_run": t, "segment": seg}}
            for t in range(len(idx))
        ]
        rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
        all_rows.extend(rows)

    df_seg = pd.DataFrame(all_rows)
    # Apply N_LAST and sigma clip
    if N_SKIP_END > 0:
        summ_seg = plateau_summary(df_seg, N_LAST_TURNS, n_skip_end=N_SKIP_END)
    else:
        summ_seg = plateau_summary(df_seg, N_LAST_TURNS)
    n_before = len(summ_seg)
    summ_seg, clip_info = mad_sigma_clip(summ_seg, "B1_mean", N_SIGMA_CLIP, label_col="branch")
    n_clipped = n_before - len(summ_seg)
    if n_clipped > 0:
        print(f"  {{seg}}: sigma clip removed {{n_clipped}} rows ({{clip_info}})")

    # Transfer function
    df_seg["TF_TperkA"] = df_seg["B1_T"] / (df_seg["I_mean_A"] / 1000.0)
    if "I_mean_A" in summ_seg.columns:
        summ_seg["TF_TperkA"] = summ_seg["B1_mean"] / (summ_seg["I_nom"] / 1000.0)

    results[seg] = {{"df": df_seg, "summ": summ_seg}}
    print(f"{{seg}}: {{len(df_seg)}} all turns, {{len(summ_seg)}} summary rows")"""
    else:
        # MBB supercycle: process all plateau turns, group by SC
        pipeline_code = f"""\
ANALYSIS_LABELS = {{"injection", "flat-mid", "flat-high"}}

results = {{}}

for seg in SEGMENTS:
    d = data[seg]
    turn_label = d["turn_label"]
    kn_seg = KN[seg]

    is_analysis = np.array([l in ANALYSIS_LABELS for l in turn_label])
    plateau_indices = np.where(is_analysis)[0]
    print(f"\\n{{seg}}: processing {{len(plateau_indices)}} plateau turns (OPTIONS={{OPTIONS}})")

    result, C_merged, C_units, ok_main = process_kn_pipeline(
        flux_abs_turns=d["flux_abs"][plateau_indices],
        flux_cmp_turns=d["flux_cmp"][plateau_indices],
        t_turns=d["t_all"][plateau_indices],
        I_turns=d["I_all"][plateau_indices],
        kn=kn_seg, r_ref=R_REF, magnet_order=m,
        options=OPTIONS, min_b1_T=MIN_B1_T,
    )

    extra = [
        {{"global_turn": int(plateau_indices[t]),
          "label": str(turn_label[plateau_indices[t]]),
          "I_range_A": float(d["I_range"][plateau_indices[t]]),
          "segment": seg}}
        for t in range(len(plateau_indices))
    ]

    rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
    df = pd.DataFrame(rows)

    # Group by supercycle
    df["sc_idx"] = -1
    settled_idx = []

    for gi, (gs, ge) in enumerate(d["inj_groups"]):
        group_globals = set(range(gs, ge + 1))
        gmask = df["global_turn"].isin(group_globals) & (df["label"] == "injection")
        df.loc[gmask, "sc_idx"] = gi
        group_rows = df.index[gmask]
        if N_LAST_TURNS_INJ is not None and len(group_rows) > N_LAST_TURNS_INJ:
            settled_idx.extend(group_rows[-N_LAST_TURNS_INJ:])
        else:
            settled_idx.extend(group_rows)

    for gi, (gs, ge) in enumerate(d["fh_groups"]):
        group_globals = set(range(gs, ge + 1))
        gmask = df["global_turn"].isin(group_globals) & (df["label"] == "flat-high")
        df.loc[gmask, "sc_idx"] = gi
        group_rows = df.index[gmask]
        if N_LAST_TURNS_HIGH is not None and len(group_rows) > N_LAST_TURNS_HIGH:
            settled_idx.extend(group_rows[-N_LAST_TURNS_HIGH:])
        else:
            settled_idx.extend(group_rows)

    df_settled = df.loc[sorted(settled_idx)].copy()

    n_before = len(df_settled)
    df_settled, clip_info = mad_sigma_clip(df_settled, "B1_T", N_SIGMA_CLIP, label_col="label")
    n_clipped = n_before - len(df_settled)
    if n_clipped > 0:
        print(f"  Sigma clip ({{N_SIGMA_CLIP}} MAD sigma): removed {{n_clipped}} turns ({{clip_info}})")

    df["TF_TperkA"] = df["B1_T"] / (df["I_mean_A"] / 1000.0)
    df_settled["TF_TperkA"] = df_settled["B1_T"] / (df_settled["I_mean_A"] / 1000.0)

    results[seg] = {{"df": df, "df_settled": df_settled}}

    print(f"  {{seg}}: {{len(df)}} all plateau, {{len(df_settled)}} settled")
    for lab in ["injection", "flat-high"]:
        n_all = len(df[df["label"] == lab])
        n_set = len(df_settled[df_settled["label"] == lab])
        print(f"    {{lab:12s}}: {{n_all}} -> {{n_set}}")"""

    cells.append(code(f"s{n}-pipeline", pipeline_code))
    return cells


def section_pipeline_file_discovery(cfg, n):
    """Process per-run data through pipeline (file-discovery)."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Pipeline Processing\n\n"
                "Process each run through the Kn pipeline.")]
    cells.append(code(f"s{n}-pipeline", f"""\
df = {{}}
for seg in SEGMENTS:
    kn_seg = KN[seg]
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    merge_mode = _scfg["merge_mode"]
    all_rows = []

    for r in runs[seg]:
        raw = np.loadtxt(r["file"])
        _nt = raw.shape[0] // Ns
        _nk = _nt * Ns

        _t = raw[:_nk, 0].reshape(_nt, Ns)
        _fa = raw[:_nk, 1].reshape(_nt, Ns)
        _fc = raw[:_nk, 2].reshape(_nt, Ns)
        _I = raw[:_nk, 3].reshape(_nt, Ns)

        result, C_merged, C_units, ok_main = process_kn_pipeline(
            flux_abs_turns=_fa, flux_cmp_turns=_fc,
            t_turns=_t, I_turns=_I,
            kn=kn_seg, r_ref=R_REF, magnet_order=m,
            options=OPTIONS, min_b1_T=MIN_B1_T,
            encoder_offset_rad=ENCODER_OFFSET_RAD,
            merge_mode=merge_mode,
        )
        extra = [
            {{"run_id": r["run_id"], "I_nom": r["I_nom"], "branch": r["branch"],
              "turn_in_run": t, "segment": seg}}
            for t in range(_nt)
        ]
        rows = build_harmonic_rows(result, C_merged, C_units, ok_main, m, extra)
        all_rows.extend(rows)

    df[seg] = pd.DataFrame(all_rows)
    print(f"{{seg}}: {{len(df[seg])}} turns from {{len(runs[seg])}} runs")"""))
    return cells


def section_plateau_quality(cfg, n):
    """Compute plateau summary for file-discovery data."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Plateau Quality\n\n"
                "Compute per-run averages using the last N turns (settled).")]
    skip_end = cfg.n_skip_end
    cells.append(code(f"s{n}-quality", f"""\
summ = {{}}
for seg in SEGMENTS:
    summ[seg] = plateau_summary(df[seg], N_LAST_TURNS, n_skip_end=N_SKIP_END)
    n_before = len(summ[seg])
    summ[seg], clip_info = mad_sigma_clip(summ[seg], "B1_mean", N_SIGMA_CLIP, label_col="branch")
    n_clipped = n_before - len(summ[seg])
    if n_clipped > 0:
        print(f"  {{seg}}: sigma clip removed {{n_clipped}} rows ({{clip_info}})")
    summ[seg]["TF_TperkA"] = summ[seg]["B1_mean"] / (summ[seg]["I_nom"] / 1000.0)
    print(f"{{seg}}: {{len(summ[seg])}} summary rows")
    print(summ[seg][["run_id", "I_nom", "branch", "B1_mean", "b2_units_mean", "b3_units_mean"]].head(10).to_string(index=False))"""))
    return cells


def section_allturn(cfg, n):
    """All-turn harmonics vs time (streaming only)."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. All-Turn Harmonics vs Time\n\n"
                "Process all turns (including ramps) to show B1, b2, b3 evolution.")]
    cells.append(code(f"s{n}-allturn", """\
all_turn_dfs = {}

for seg in SEGMENTS:
    d = data[seg]
    kn_seg = KN[seg]

    result_all, C_merged_all, C_units_all, ok_main_all = process_kn_pipeline(
        flux_abs_turns=d["flux_abs"], flux_cmp_turns=d["flux_cmp"],
        t_turns=d["t_all"], I_turns=d["I_all"],
        kn=kn_seg, r_ref=R_REF, magnet_order=m,
        options=OPTIONS, min_b1_T=MIN_B1_T,
    )

    extra_all = [{"global_turn": int(i), "segment": seg} for i in range(d["n_turns"])]
    rows_all = build_harmonic_rows(result_all, C_merged_all, C_units_all, ok_main_all, m, extra_all)
    df_all = pd.DataFrame(rows_all)
    df_all["t_mean_s"] = d["t_mean"]
    all_turn_dfs[seg] = df_all
    print(f"{seg}: {d['n_turns']} all-turns processed, ok_main={ok_main_all.sum()}")

# Plot B1, b2, b3 vs time
fig, axes = plt.subplots(3, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 12))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df_all = all_turn_dfs[seg]
    ok = df_all["ok_main"]
    for ax_idx, (col, ylabel) in enumerate([("B1_T", "B1 (T)"), ("b2_units", "b2 (units)"), ("b3_units", "b3 (units)")]):
        ax = axes[ax_idx, j]
        ax.scatter(df_all.loc[ok, "t_mean_s"], df_all.loc[ok, col],
                   s=4, alpha=0.3, color="steelblue", zorder=1)
        ax.set_ylabel(ylabel)
        if ax_idx == 0:
            ax.set_title(f"All-turn evolution -- {seg}")
        if ax_idx == 2:
            ax.set_xlabel("Time (s)")

fig.suptitle("All-Turn Harmonics vs Time", fontsize=14, y=1.01)
plt.tight_layout(); plt.show()"""))
    return cells


def section_ffmm(cfg, n):
    """FFMM golden standard validation."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. FFMM Golden Standard Validation\n\n"
                "Compare pipeline output against FFMM per-turn and average results.")]
    # Generate FFMM code based on config
    ffmm_opts = repr(cfg.ffmm_options)
    ffmm_rel = str(cfg.ffmm_rotate_excludes_last)
    r_ref_str = repr(cfg.ffmm_r_ref) if cfg.ffmm_r_ref else "R_REF"

    cells.append(code(f"s{n}-ffmm", f"""\
OPTIONS_FFMM = {ffmm_opts}
FFMM_ROTATE_EXCLUDES_LAST = {ffmm_rel}

print("=" * 70)
print("FFMM GOLDEN STANDARD COMPARISON")
print(f"FFMM pipeline options: {{OPTIONS_FFMM}}")
print(f"legacy_rotate_excludes_last = {{FFMM_ROTATE_EXCLUDES_LAST}}")
print("=" * 70)

# This section expects FFMM result files alongside the raw data.
# Adjust paths as needed for your measurement.
print("(FFMM validation section -- configure paths for your measurement)")"""))
    return cells


# ================================================================
# Section builders -- Harmonic Analysis
# ================================================================

def _harmonic_section(cfg, n, harmonic_name, harmonic_col, ylabel, title_desc):
    """Generic harmonic analysis section (B1/b2/b3)."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. {title_desc}")]

    if cfg.data_loader in ("text_streaming",):
        # MBB streaming: scatter + per-SC errorbar
        cells.append(code(f"s{n}-plot", f"""\
fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

for j, seg in enumerate(SEGMENTS):
    df = results[seg]["df"]
    df_settled = results[seg]["df_settled"]
    ok = df["ok_main"]
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe = " [fringe]" if _scfg["is_fringe"] else ""

    ax = axes[0, j]
    ax.scatter(df.loc[ok, "I_mean_A"], df.loc[ok, "{harmonic_col}"], s=8, alpha=0.5, color="steelblue")
    {"" if harmonic_col == "B1_T" else 'ax.axhline(0, color="grey", linewidth=0.5)'}
    ax.set_xlabel("I (A)"); ax.set_ylabel("{ylabel}")
    ax.set_title(f"{harmonic_name} vs current -- {{seg}}{{fringe}}")

    ax = axes[1, j]
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["{harmonic_col}"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{{marker}}-", markersize=4, capsize=2, color=col, label=lab)
    {"" if harmonic_col == "B1_T" else 'ax.axhline(0, color="grey", linewidth=0.5)'}
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("{ylabel}")
    ax.set_title(f"{harmonic_name} per supercycle (settled) -- {{seg}}{{fringe}}"); ax.legend(fontsize=9)

fig.suptitle("{title_desc}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Statistics
print("\\n{harmonic_name} per operating point (settled turns):")
for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            print(f"  {{seg}} {{lab:12s}}: N={{len(sub):4d}}, mean={{sub['{harmonic_col}'].mean():+.6f}}, std={{sub['{harmonic_col}'].std():.6f}}")"""))
    else:
        # Binary streaming or file-discovery: scatter only
        data_var = "df" if cfg.data_loader == "file_discovery" else "results"
        if cfg.data_loader == "file_discovery":
            cells.append(code(f"s{n}-plot", f"""\
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax_idx, seg in enumerate(SEGMENTS):
    _df = df[seg]
    ok = _df["ok_main"]
    ax = axes[ax_idx]
    for branch, col in [("ascending", "tab:blue"), ("descending", "tab:red")]:
        mask = ok & (_df["branch"] == branch)
        ax.scatter(_df.loc[mask, "I_nom"], _df.loc[mask, "{harmonic_col}"],
                   s=8, alpha=0.5, color=col, label=branch)
    {"" if harmonic_col == "B1_T" else 'ax.axhline(0, color="grey", linewidth=0.5)'}
    ax.set_xlabel("I (A)"); ax.set_ylabel("{ylabel}")
    ax.set_title(f"{harmonic_name} -- {{seg}}"); ax.legend(fontsize=9)
fig.suptitle("{title_desc}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))
        else:
            # binary_streaming (MC62 staircase with run_info)
            cells.append(code(f"s{n}-plot", f"""\
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax_idx, seg in enumerate(SEGMENTS):
    _df = results[seg]["df"]
    ok = _df["ok_main"]
    for branch, col in [("ascending", "tab:blue"), ("descending", "tab:red")]:
        mask = ok & (_df["branch"] == branch)
        ax = axes[ax_idx]
        ax.scatter(_df.loc[mask, "I_nom"], _df.loc[mask, "{harmonic_col}"],
                   s=8, alpha=0.5, color=col, label=branch)
    {"" if harmonic_col == "B1_T" else 'ax.axhline(0, color="grey", linewidth=0.5)'}
    axes[ax_idx].set_xlabel("I (A)"); axes[ax_idx].set_ylabel("{ylabel}")
    axes[ax_idx].set_title(f"{harmonic_name} -- {{seg}}"); axes[ax_idx].legend(fontsize=9)
fig.suptitle("{title_desc}", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))

    return cells


def section_b1(cfg, n):
    return _harmonic_section(cfg, n, "B1", "B1_T", "B1 (T)", "Main Field (B1)")


def section_b2(cfg, n):
    return _harmonic_section(cfg, n, "b2", "b2_units", "b2 (units)",
                             "b2 (Quadrupole) -- first allowed harmonic error")


def section_b3(cfg, n):
    return _harmonic_section(cfg, n, "b3", "b3_units", "b3 (units)",
                             "b3 (Sextupole) -- first non-allowed harmonic")


def section_higher_harmonics(cfg, n):
    """Higher harmonics overview: b4..bH, a2..aH."""
    main_seg = cfg.main_segment
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Higher Harmonics Overview\n\n"
                f"Statistics for all harmonics at key operating points ({main_seg}).")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-higher", f"""\
seg = "{main_seg}"
df_settled = results[seg]["df_settled"]
ok = df_settled["ok_main"]

for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok]
    if len(sub) == 0:
        continue
    print(f"\\n=== {{lab.upper()}} (N={{len(sub)}} settled turns, {{seg}}) ===")
    print(f"  {{'n':>3s}} {{'bn mean':>10s}} {{'bn std':>10s}} {{'an mean':>10s}} {{'an std':>10s}}")
    print("  " + "-" * 50)
    for nn in range(2, H + 1):
        bn_col = f"b{{nn}}_units"
        an_col = f"a{{nn}}_units"
        if bn_col in sub.columns:
            bn_m, bn_s = sub[bn_col].mean(), sub[bn_col].std()
            an_m, an_s = sub[an_col].mean(), sub[an_col].std()
            flag = " *" if abs(bn_m) > 2 * bn_s and abs(bn_m) > 0.5 else ""
            print(f"  {{nn:3d}} {{bn_m:+10.4f}} {{bn_s:10.4f}} {{an_m:+10.4f}} {{an_s:10.4f}}{{flag}}")
    print("  (* = |mean| > 2*std and |mean| > 0.5 units)")"""))
    else:
        # File-discovery or binary: use summ[seg]
        data_source = "summ" if cfg.data_loader == "file_discovery" else "results"
        cells.append(code(f"s{n}-higher", f"""\
seg = "{main_seg}"
if "{cfg.data_loader}" == "file_discovery":
    _peak_mask = summ[seg]["I_nom"].abs() == summ[seg]["I_nom"].abs().max()
    _peak = summ[seg][_peak_mask]
    _bn_cols = sorted([c for c in summ[seg].columns if c.endswith("_mean")
                       and (c.startswith("b") or c.startswith("a"))
                       and c not in ("b2_units_mean", "b3_units_mean")],
                      key=lambda c: (c[0], int(c.split("_")[0][1:])))
    if _peak.empty or not _bn_cols:
        print("No peak-current data or harmonic columns found.")
    else:
        rows_tbl = []
        for col in _bn_cols:
            base = col.replace("_mean", "")
            std_col = base + "_std"
            mean_val = _peak[col].values[0]
            std_val = _peak[std_col].values[0] if std_col in _peak.columns else float("nan")
            rows_tbl.append({{"Harmonic": base, "Mean [units]": mean_val, "Std [units]": std_val}})
        _htable = pd.DataFrame(rows_tbl)
        print(f"Higher harmonics at peak |I| ({{seg}}):")
        print(_htable.to_string(index=False, float_format="%.3f"))
else:
    _df = results[seg]["df"]
    _summ = results[seg]["summ"]
    if not _summ.empty:
        _peak_mask = _summ["I_nom"].abs() == _summ["I_nom"].abs().max()
        _peak = _summ[_peak_mask]
        print(f"Higher harmonics at peak |I| ({{seg}}):")
        _bn_cols = sorted([c for c in _summ.columns if ("_mean" in c)
                           and (c.startswith("b") or c.startswith("a"))
                           and c not in ("b2_units_mean", "b3_units_mean", "B1_mean")],
                          key=lambda c: (c[0], int(c.split("_")[0][1:])))
        for col in _bn_cols[:20]:
            vals = _peak[col].values
            print(f"  {{col}}: {{vals[0]:.4f}}" if len(vals) > 0 else f"  {{col}}: --")"""))
    return cells


def section_spectrum(cfg, n):
    """Multipole spectrum (bar chart)."""
    main_seg = cfg.main_segment
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Multipole Spectrum\n\n"
                "Bar charts of normal (bn) and skew (an) harmonics.")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-spectrum", f"""\
seg = "{main_seg}"
df_settled = results[seg]["df_settled"]
ok = df_settled["ok_main"]

operating_points = {{}}
for lab in ["injection", "flat-high"]:
    sub = df_settled[(df_settled["label"] == lab) & ok]
    if len(sub) > 0:
        operating_points[lab] = sub

n_ops = len(operating_points)
if n_ops > 0:
    fig, axes = plt.subplots(n_ops, 2, figsize=(16, 5 * n_ops))
    if n_ops == 1:
        axes = axes[np.newaxis, :]

    for i, (lab, sub) in enumerate(operating_points.items()):
        orders = list(range(2, H + 1))
        bn_means = [sub[f"b{{nn}}_units"].mean() for nn in orders]
        an_means = [sub[f"a{{nn}}_units"].mean() for nn in orders]
        x = np.arange(len(orders))
        w = 0.35

        ax = axes[i, 0]
        ax.bar(x - w/2, bn_means, w, label="bn", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, an_means, w, label="an", color="tab:orange", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(orders)
        ax.set_xlabel("n"); ax.set_ylabel("Units")
        ax.set_title(f"Spectrum -- {{lab}} (linear)"); ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.bar(x - w/2, np.abs(bn_means), w, label="|bn|", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, np.abs(an_means), w, label="|an|", color="tab:orange", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(orders)
        ax.set_xlabel("n"); ax.set_ylabel("|Units|")
        ax.set_title(f"Spectrum -- {{lab}} (log)"); ax.legend(fontsize=8)

    fig.suptitle("Multipole Spectrum", fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No operating points with data for spectrum plot.")"""))
    else:
        cells.append(code(f"s{n}-spectrum", f"""\
seg = "{main_seg}"
# Detect available harmonic orders from columns
if "{cfg.data_loader}" == "file_discovery":
    _cols = df[seg].columns
else:
    _cols = results[seg]["df"].columns
bn_cols = [c for c in _cols if c.startswith("b") and c.endswith("_units") and c != "b1_units"]
orders = sorted([int(c.replace("b", "").replace("_units", "")) for c in bn_cols])

if orders:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(orders))
    w = 0.35

    if "{cfg.data_loader}" == "file_discovery":
        _src = df[seg][df[seg]["ok_main"]]
    else:
        _src = results[seg]["df"][results[seg]["df"]["ok_main"]]

    bn_means = [_src[f"b{{nn}}_units"].mean() for nn in orders]
    an_means = [_src[f"a{{nn}}_units"].mean() for nn in orders]

    ax = axes[0]
    ax.bar(x - w/2, bn_means, w, label="bn", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, an_means, w, label="an", color="tab:orange", alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(orders)
    ax.set_xlabel("n"); ax.set_ylabel("Units")
    ax.set_title("Spectrum (linear)"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - w/2, np.abs(bn_means), w, label="|bn|", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, np.abs(an_means), w, label="|an|", color="tab:orange", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(orders)
    ax.set_xlabel("n"); ax.set_ylabel("|Units|")
    ax.set_title("Spectrum (log)"); ax.legend(fontsize=8)

    fig.suptitle("Multipole Spectrum", fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No harmonic columns found for spectrum plot.")"""))
    return cells

# ================================================================
# Section builders -- Transfer Function & Inductance
# ================================================================

def section_tf(cfg, n):
    """Transfer function B1/I."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Transfer Function B1/I\n\n"
                "TF = B1 / I (units: T/kA).")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-tf", """\
fig, axes = plt.subplots(2, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 10))
if len(SEGMENTS) == 1:
    axes = axes[:, np.newaxis]

tf_summary = {}
for j, seg in enumerate(SEGMENTS):
    df_settled = results[seg]["df_settled"]
    ok = df_settled["ok_main"]
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe = " [fringe]" if _scfg["is_fringe"] else ""

    ax = axes[0, j]
    sub_ok = df_settled[ok]
    ax.scatter(sub_ok["I_mean_A"], sub_ok["TF_TperkA"], s=8, alpha=0.5, color="steelblue")
    ax.set_xlabel("I (A)"); ax.set_ylabel("TF = B1/I (T/kA)")
    ax.set_title(f"TF vs current -- {seg}{fringe}")

    ax = axes[1, j]
    ds_tf = {}
    for lab, col, marker in [("injection", "tab:green", "o"), ("flat-high", "tab:blue", "s")]:
        sub = df_settled[(df_settled["label"] == lab) & ok]
        if len(sub) > 0:
            sc_avg = sub.groupby("sc_idx")["TF_TperkA"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt=f"{marker}-", markersize=4, capsize=2, color=col, label=lab)
            ds_tf[lab] = sc_avg
    ax.set_xlabel("Supercycle index"); ax.set_ylabel("TF (T/kA)")
    ax.set_title(f"TF per supercycle -- {seg}{fringe}"); ax.legend(fontsize=9)
    tf_summary[seg] = ds_tf

fig.suptitle("Transfer Function B1/I", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))
    else:
        cells.append(code(f"s{n}-tf", f"""\
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(8 * len(SEGMENTS), 5))
if len(SEGMENTS) == 1:
    axes = [axes]
for ax_idx, seg in enumerate(SEGMENTS):
    if "{cfg.data_loader}" == "file_discovery":
        _df = df[seg][df[seg]["ok_main"]]
    else:
        _df = results[seg]["df"][results[seg]["df"]["ok_main"]]
    _df_tf = _df.copy()
    _df_tf["TF"] = _df_tf["B1_T"] / (_df_tf["I_mean_A"] / 1000.0)
    for branch, col in [("ascending", "tab:blue"), ("descending", "tab:red")]:
        mask = _df_tf["branch"] == branch
        axes[ax_idx].scatter(_df_tf.loc[mask, "I_nom"].abs(), _df_tf.loc[mask, "TF"],
                             s=8, alpha=0.5, color=col, label=branch)
    axes[ax_idx].set_xlabel("|I| (A)"); axes[ax_idx].set_ylabel("TF (T/kA)")
    axes[ax_idx].set_title(f"TF -- {{seg}}"); axes[ax_idx].legend(fontsize=9)
fig.suptitle("Transfer Function B1/I", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()"""))
    return cells


def section_inductance(cfg, n):
    """Apparent vs differential inductance."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Apparent vs Differential Inductance\n\n"
                "**L_app** = B1/I, **L_d** = dB1/dI (from paired current levels).")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-inductance", """\
ld_results = {}
for seg in SEGMENTS:
    df_settled = results[seg]["df_settled"]
    ok = df_settled["ok_main"]
    df_inj = df_settled[(df_settled["label"] == "injection") & ok]
    df_fh = df_settled[(df_settled["label"] == "flat-high") & ok]

    if len(df_inj) == 0 or len(df_fh) == 0:
        ld_results[seg] = pd.DataFrame()
        continue

    inj_avg = df_inj.groupby("sc_idx").agg(
        B1_inj=("B1_T", "mean"), I_inj=("I_mean_A", "mean")).reset_index()
    fh_avg = df_fh.groupby("sc_idx").agg(
        B1_fh=("B1_T", "mean"), I_fh=("I_mean_A", "mean")).reset_index()

    merged = inj_avg.merge(fh_avg, on="sc_idx", how="inner")
    if len(merged) == 0:
        ld_results[seg] = pd.DataFrame()
        continue

    merged["Ld_TperkA"] = (merged["B1_fh"] - merged["B1_inj"]) / ((merged["I_fh"] - merged["I_inj"]) / 1000.0)
    ld_results[seg] = merged
    print(f"{seg}: {len(merged)} SC pairs, "
          f"Ld = {merged['Ld_TperkA'].mean():.4f} +/- {merged['Ld_TperkA'].std():.4f} T/kA")

print("\\nSaturation check (Ld < L_app(FT) => saturated):")
for seg in SEGMENTS:
    m_df = ld_results.get(seg, pd.DataFrame())
    if len(m_df) == 0:
        continue
    Ld_mean = m_df["Ld_TperkA"].mean()
    fh = results[seg]["df_settled"]
    fh_ok = fh[(fh["label"] == "flat-high") & fh["ok_main"]]
    if len(fh_ok) > 0:
        Lapp_fh = fh_ok["TF_TperkA"].mean()
        ratio = Ld_mean / Lapp_fh
        verdict = "SATURATED" if ratio < 0.99 else "LINEAR"
        print(f"  {seg}: Ld={Ld_mean:.4f}, L_app(FT)={Lapp_fh:.4f}, ratio={ratio:.4f} -> {verdict}")"""))
    else:
        cells.append(code(f"s{n}-inductance", """\
print("Inductance analysis for staircase data:")
for seg in SEGMENTS:
    if "file_discovery" == \"""" + cfg.data_loader + """\": _s = summ[seg]
    else: _s = results[seg]["summ"]
    if _s.empty:
        continue
    _s_ok = _s[_s["B1_mean"].notna()].copy()
    _s_ok["TF"] = _s_ok["B1_mean"] / (_s_ok["I_nom"] / 1000.0)
    for branch in ["ascending", "descending"]:
        sub = _s_ok[_s_ok["branch"] == branch].sort_values("I_nom")
        if len(sub) >= 2:
            # Compute Ld between consecutive steps
            dB = np.diff(sub["B1_mean"].values)
            dI = np.diff(sub["I_nom"].values) / 1000.0
            Ld = dB / dI
            valid = np.abs(dI) > 0.001
            if valid.any():
                print(f"  {seg} {branch}: Ld range = {Ld[valid].min():.4f} .. {Ld[valid].max():.4f} T/kA")"""))
    return cells


# ================================================================
# Section builders -- Eddy Current & Settling
# ================================================================

def section_eddy_config(cfg, n):
    """Eddy current analysis config cell."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Eddy Current Settling Analysis\n\n"
                "Turn-by-turn B1 for all runs. Eddy currents cause exponential decay.")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-eddy-cfg", """\
# Build per-supercycle injection data
eddy_data = {}

for seg in SEGMENTS:
    d = data[seg]
    df = results[seg]["df"]
    inj = df[df["label"] == "injection"].copy()
    if len(inj) == 0:
        eddy_data[seg] = pd.DataFrame()
        continue

    inj["t_mean_s"] = d["t_mean"][inj["global_turn"].values]
    for sc_id in inj["sc_idx"].unique():
        if sc_id < 0: continue
        mask = inj["sc_idx"] == sc_id
        t0 = inj.loc[mask, "t_mean_s"].min()
        inj.loc[mask, "t_since_inj_start"] = inj.loc[mask, "t_mean_s"] - t0

    for sc_id in inj["sc_idx"].unique():
        if sc_id < 0: continue
        mask = inj["sc_idx"] == sc_id
        inj.loc[mask, "turn_in_group"] = np.arange(mask.sum())

    eddy_data[seg] = inj
    print(f"{seg}: {len(inj)} injection turns across {inj['sc_idx'].nunique()} supercycles")"""))
    else:
        is_streaming = cfg.data_loader == "binary_streaming"
        t_per_turn_code = "T_PER_TURN" if is_streaming else "T_PER_TURN"
        data_ref = "results[_eddy_seg][\"df\"]" if is_streaming else "df[_eddy_seg]"
        cells.append(code(f"s{n}-eddy-cfg", f"""\
from rotating_coil_analyzer.analysis.utility_functions import fit_eddy_per_run

MIN_I_FOR_FIT = 10.0
{"" if is_streaming else "T_PER_TURN = 1.0  # 1 Hz rotation"}

# Use first segment for eddy analysis
_eddy_seg = SEGMENTS[0]
df_eddy = {data_ref}.copy()
print(f"Eddy analysis: {{len(df_eddy)}} turns, {{df_eddy['run_id'].nunique()}} runs")"""))

    return cells


def section_eddy_raw(cfg):
    """Raw settling curve plots (subsection, no section number)."""
    cells = [md("eddy-hdr-raw", "### Raw B1 settling curves")]

    if cfg.data_loader == "text_streaming":
        cells.append(code("eddy-raw", """\
fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    inj = eddy_data[seg]
    for col_idx, (col, ylabel) in enumerate([("B1_T", "B1 (T)"), ("b3_units", "b3 (units)")]):
        ax = axes[i, col_idx]
        if len(inj) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        sc_ids = sorted([s for s in inj["sc_idx"].unique() if s >= 0])
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(sc_ids), 1)))
        for k, sc_id in enumerate(sc_ids):
            sub = inj[inj["sc_idx"] == sc_id]
            ax.plot(sub["t_since_inj_start"], sub[col], ".-",
                    markersize=4, linewidth=0.8, alpha=0.7, color=cmap[k % len(cmap)])
        ax.set_xlabel("t - t_inj_start (s)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} settling -- {seg}")

fig.suptitle("Injection Settling -- Supercycle Overlay", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))
    else:
        cells.append(code("eddy-raw", """\
_I_noms = sorted(df_eddy["I_nom"].unique())
if len(_I_noms) == 0:
    print("No eddy data to plot")
else:
    cmap = plt.cm.coolwarm
    _norm_c = plt.Normalize(min(_I_noms), max(_I_noms))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for sign_idx, (sign_label, sign_cond) in enumerate([("Positive current", lambda x: x > 0), ("Negative current", lambda x: x < 0)]):
        ax = axes[sign_idx]
        for run_id in sorted(df_eddy["run_id"].unique()):
            rdf = df_eddy[(df_eddy["run_id"] == run_id) & df_eddy["ok_main"]]
            if rdf.empty: continue
            I_nom = rdf["I_nom"].iloc[0]
            if not sign_cond(I_nom): continue
            ax.plot(rdf["turn_in_run"], rdf["B1_T"], lw=0.5,
                    color=cmap(_norm_c(I_nom)), alpha=0.7)
        ax.set_ylabel("B1 [T]"); ax.set_title(sign_label)
    axes[-1].set_xlabel("Turn in run")
    fig.suptitle("Raw settling curves -- B1 vs turn number", y=1.01)
    fig.tight_layout(); plt.show()"""))
    return cells


def section_eddy_fits(cfg, n):
    """Exponential fits for eddy current settling."""
    cells = [md(f"s{n}-hdr-fits", f"---\n## {n}. Exponential Fits\n\n"
                "Fit single-exponential eddy model per run/supercycle.")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-fits", """\
def fit_supercycle(df_sc):
    t = df_sc["t_since_inj_start"].values
    b3 = df_sc["b3_units"].values
    ok = np.isfinite(b3) & np.isfinite(t)
    t, b3 = t[ok], b3[ok]
    if len(t) < MIN_INJECTION_TURNS:
        return None
    try:
        popt, pcov = curve_fit(
            eddy_model, t, b3,
            p0=[b3[-1], b3[0] - b3[-1], max(t[-1] / 3, 1.0)],
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 1000]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        b3_pred = eddy_model(t, *popt)
        ss_res = np.sum((b3 - b3_pred) ** 2)
        ss_tot = np.sum((b3 - b3.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"b3_inf": popt[0], "A": popt[1], "tau": popt[2],
                "b3_inf_err": perr[0], "A_err": perr[1], "tau_err": perr[2],
                "r2": r2, "n_turns": len(t)}
    except (RuntimeError, ValueError):
        return None

fit_results = {}
df_fits = {}
for seg in SEGMENTS:
    inj = eddy_data[seg]
    fits = []
    if len(inj) == 0:
        fit_results[seg] = fits
        df_fits[seg] = pd.DataFrame()
        continue
    for sc_id in sorted(inj["sc_idx"].unique()):
        if sc_id < 0: continue
        result = fit_supercycle(inj[inj["sc_idx"] == sc_id])
        if result is not None:
            result["supercycle_id"] = sc_id
            fits.append(result)
    fit_results[seg] = fits
    df_fits[seg] = pd.DataFrame(fits)
    print(f"{seg}: {len(fits)} supercycles fitted")

for seg, df_f in df_fits.items():
    if len(df_f) == 0: continue
    print(f"\\n{seg}: tau mean={df_f['tau'].mean():.2f} s, R2 mean={df_f['r2'].mean():.3f}")"""))
    else:
        cells.append(code(f"s{n}-fits", """\
_fit_results = []
for run_id in sorted(df_eddy["run_id"].unique()):
    rdf = df_eddy[(df_eddy["run_id"] == run_id) & df_eddy["ok_main"]].sort_values("turn_in_run")
    if rdf.empty: continue
    I_nom = rdf["I_nom"].iloc[0]
    if abs(I_nom) < MIN_I_FOR_FIT: continue
    branch = rdf["branch"].iloc[0]
    I_mean = rdf["I_mean"].values if "I_mean" in rdf.columns else None
    res = fit_eddy_per_run(
        turns=rdf["turn_in_run"].values.astype(float),
        B1=rdf["B1_T"].values,
        run_id=run_id, I_nom=I_nom, branch=branch, I_mean=I_mean,
    )
    _fit_results.append(res)

_cols = ["run_id", "I_nom", "branch", "B_inf", "A", "tau", "tau_err",
         "tau_s", "tau_err_s", "r2", "n_turns", "quality", "reason"]
if _fit_results:
    df_fits_all = pd.DataFrame([
        {"run_id": r.run_id, "I_nom": r.I_nom, "branch": r.branch,
         "B_inf": r.B_inf, "A": r.A, "tau": r.tau, "tau_err": r.tau_err,
         "tau_s": r.tau * T_PER_TURN, "tau_err_s": r.tau_err * T_PER_TURN,
         "r2": r.r2, "n_turns": r.n_turns, "quality": r.quality, "reason": r.reason}
        for r in _fit_results
    ])
else:
    df_fits_all = pd.DataFrame(columns=_cols)

df_fits = df_fits_all[df_fits_all["quality"] == "GOOD"].copy()
print(f"Fits: {len(df_fits)} GOOD / {len(df_fits_all)} total")
if len(df_fits_all) > 0:
    print(f"{'Run':>4s} {'I [A]':>8s} {'tau [s]':>8s} {'R2':>8s} {'Quality':>12s}")
    print("-" * 50)
    for _, r in df_fits_all.iterrows():
        print(f"{r['run_id']:4.0f} {r['I_nom']:+8.1f} {r['tau_s']:8.2f} {r['r2']:8.4f} {r['quality']:>12s}")
else:
    print("No eddy fits produced (no qualifying runs)")"""))
    return cells


def section_eddy_tau(cfg):
    """Tau vs current plot (subsection, no section number)."""
    cells = [md("eddy-hdr-tau", "### Tau vs Current")]

    if cfg.data_loader == "text_streaming":
        cells.append(code("eddy-tau", """\
print("Tau statistics:")
for seg in SEGMENTS:
    df_f = df_fits.get(seg, pd.DataFrame())
    if len(df_f) == 0: continue
    tau_v = df_f["tau"].values
    print(f"  {seg}: N={len(df_f)}, mean={tau_v.mean():.2f} s, std={tau_v.std():.2f} s")"""))
    else:
        cells.append(code("eddy-tau", """\
_branch_colors = {"ascending": "tab:blue", "descending": "tab:red"}
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for branch, col in _branch_colors.items():
    mask = df_fits["branch"] == branch
    ax.scatter(df_fits.loc[mask, "I_nom"].abs(), df_fits.loc[mask, "tau_s"],
               c=col, label=branch, s=30, alpha=0.7)
ax.set_xlabel("|I| [A]"); ax.set_ylabel("tau [s]")
ax.set_title("Settling time constant vs current"); ax.legend()
plt.tight_layout(); plt.show()

if len(df_fits) > 0:
    print(f"Tau range: {df_fits['tau_s'].min():.1f} -- {df_fits['tau_s'].max():.1f} s")"""))
    return cells


def section_eddy_bias(cfg, n):
    """Settling bias analysis."""
    cells = [md(f"s{n}-hdr-bias", f"---\n## {n}. Settling Bias Analysis\n\n"
                "How b2/b3 averages change with averaging window.")]

    if cfg.data_loader == "text_streaming":
        main_seg = cfg.main_segment
        cells.append(code(f"s{n}-bias", f"""\
seg = "{main_seg}"
inj = eddy_data[seg]
if len(inj) > 0 and "turn_in_group" in inj.columns:
    sc_ids = sorted([s for s in inj["sc_idx"].unique() if s >= 0])
    max_turns = inj.groupby("sc_idx").size().min()
    n_last_values = list(range(1, max_turns + 1))

    bias_b3, bias_b2 = [], []
    for n_last in n_last_values:
        b3_m, b2_m = [], []
        for sc_id in sc_ids:
            sub = inj[inj["sc_idx"] == sc_id].sort_values("turn_in_group")
            tail = sub.tail(n_last)
            if len(tail) > 0 and tail["ok_main"].any():
                ok_t = tail[tail["ok_main"]]
                b3_m.append(ok_t["b3_units"].mean())
                b2_m.append(ok_t["b2_units"].mean())
        bias_b3.append(np.mean(b3_m) if b3_m else np.nan)
        bias_b2.append(np.mean(b2_m) if b2_m else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(n_last_values, bias_b3, "o-", markersize=4, color="tab:blue")
    axes[0].set_xlabel("N_LAST"); axes[0].set_ylabel("b3 mean (units)")
    axes[0].set_title("b3 bias vs averaging window")
    axes[1].plot(n_last_values, bias_b2, "o-", markersize=4, color="tab:orange")
    axes[1].set_xlabel("N_LAST"); axes[1].set_ylabel("b2 mean (units)")
    axes[1].set_title("b2 bias vs averaging window")
    fig.suptitle("Settling Bias Analysis", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for bias analysis.")"""))
    else:
        cells.append(code(f"s{n}-bias", """\
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
_show_runs = [r for _, r in df_fits.iterrows()]
_show_runs = _show_runs[::max(1, len(_show_runs) // 8)]

for quantity, col_name, ax, ylabel in [
    ("B1", "B1_T", axes[0], "B1 deviation [1e-4 rel.]"),
    ("b2", "b2_units", axes[1], "b2 deviation [units]"),
    ("b3", "b3_units", axes[2], "b3 deviation [units]"),
]:
    for row in _show_runs:
        run_id = row["run_id"]
        rdf = df_eddy[(df_eddy["run_id"] == run_id) & df_eddy["ok_main"]].sort_values("turn_in_run")
        if rdf.empty or col_name not in rdf.columns: continue
        vals = rdf[col_name].values
        ref = vals[-N_LAST_TURNS:].mean() if len(vals) >= N_LAST_TURNS else vals[-10:].mean()
        if quantity == "B1":
            dev = (vals - ref) / abs(ref) * 1e4 if abs(ref) > 1e-9 else vals - ref
        else:
            dev = vals - ref
        ax.plot(rdf["turn_in_run"].values, dev, lw=0.5, alpha=0.6)
    ax.set_ylabel(ylabel); ax.axhline(0, color="k", lw=0.5)
axes[-1].set_xlabel("Turn in run")
fig.suptitle("Settling bias", y=1.01); fig.tight_layout(); plt.show()"""))
    return cells


def section_eddy_nlast(cfg, n):
    """N_LAST sensitivity study."""
    cells = [md(f"s{n}-hdr-nlast", f"---\n## {n}. N_LAST Sensitivity Study\n\n"
                "Scan N_LAST and show convergence.")]

    if cfg.data_loader == "text_streaming":
        main_seg = cfg.main_segment
        cells.append(code(f"s{n}-nlast", f"""\
seg = "{main_seg}"
d = data[seg]
_df = results[seg]["df"]
inj_all = _df[_df["label"] == "injection"].copy()

if len(inj_all) > 0:
    turns_per_sc = inj_all.groupby("sc_idx").size()
    max_n_last = int(turns_per_sc.min())
    n_last_scan = list(range(1, max_n_last + 1))

    scan_results = {{"B1_T": [], "b2_units": [], "b3_units": []}}
    for n_last in n_last_scan:
        settled_idx = []
        for sc_id in inj_all["sc_idx"].unique():
            if sc_id < 0: continue
            group_rows = inj_all.index[inj_all["sc_idx"] == sc_id]
            if len(group_rows) > n_last:
                settled_idx.extend(group_rows[-n_last:])
            else:
                settled_idx.extend(group_rows)
        sub = inj_all.loc[sorted(settled_idx)]
        sub_ok = sub[sub["ok_main"]]
        for col in ["B1_T", "b2_units", "b3_units"]:
            scan_results[col].append(sub_ok[col].mean() if len(sub_ok) > 0 else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, ylabel, color) in zip(axes, [
            ("B1_T", "B1 (T)", "steelblue"),
            ("b2_units", "b2 (units)", "tab:orange"),
            ("b3_units", "b3 (units)", "tab:green")]):
        ax.plot(n_last_scan, scan_results[col], "o-", markersize=3, color=color)
        ax.axvline(N_LAST_TURNS_INJ, color="red", linestyle="--", linewidth=1,
                    label=f"N_LAST={{N_LAST_TURNS_INJ}}")
        ax.set_xlabel("N_LAST"); ax.set_ylabel(ylabel)
        ax.set_title(f"{{ylabel.split()[0]}} vs N_LAST"); ax.legend(fontsize=8)
    fig.suptitle("N_LAST Sensitivity", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No injection data for N_LAST sensitivity.")"""))
    else:
        is_streaming = cfg.data_loader == "binary_streaming"
        n_last_range = "np.arange(50, 701, 10)" if is_streaming else "np.arange(20, 331, 5)"
        cells.append(code(f"s{n}-nlast", f"""\
_n_last_values = {n_last_range}
_study_runs = df_fits["run_id"].unique()

_results_sweep = []
for _nl in _n_last_values:
    _B1_errs = []
    for run_id in _study_runs:
        rdf = df_eddy[(df_eddy["run_id"] == run_id) & df_eddy["ok_main"]].sort_values("turn_in_run")
        if len(rdf) < _nl + 20: continue
        B1_true = rdf["B1_T"].values[-20:].mean()
        B1_est = rdf["B1_T"].values[-_nl:].mean()
        _B1_errs.append((B1_est - B1_true) / abs(B1_true) * 1e4)
    if _B1_errs:
        _results_sweep.append({{"N_LAST": _nl, "B1_bias": np.mean(_B1_errs), "B1_std": np.std(_B1_errs)}})

_df_sweep = pd.DataFrame(_results_sweep)
if not _df_sweep.empty:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.fill_between(_df_sweep["N_LAST"],
                    _df_sweep["B1_bias"] - _df_sweep["B1_std"],
                    _df_sweep["B1_bias"] + _df_sweep["B1_std"], alpha=0.2, color="tab:blue")
    ax.plot(_df_sweep["N_LAST"], _df_sweep["B1_bias"], ".-", color="tab:blue")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(N_LAST_TURNS, color="red", ls="--", lw=1, label=f"N_LAST = {{N_LAST_TURNS}}")
    ax.set_xlabel("N_LAST"); ax.set_ylabel("B1 bias [1e-4 rel.]")
    ax.set_title("B1 systematic bias vs averaging window"); ax.legend()
    plt.tight_layout(); plt.show()"""))
    return cells


# ================================================================
# Section builders -- Statistics, Choices & Export
# ================================================================

def section_stats(cfg, n):
    """Comprehensive statistics table."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Comprehensive Statistics Table")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-stats", f"""\
print("=" * 70)
print("{cfg.title}")
print("=" * 70)
print(f"Options: {{OPTIONS}}")
print(f"cel/fed: {{diag.recommendation}}")

for seg in SEGMENTS:
    d = data[seg]
    df_settled = results[seg]["df_settled"]
    _scfg = next(sc for sc in SEGMENT_CONFIGS if sc["name"] == seg)
    fringe = " [FRINGE]" if _scfg["is_fringe"] else ""

    print(f"\\n--- {{seg}}{{fringe}} ---")
    print(f"  Total turns: {{d['n_turns']}}, Plateau: {{d['is_plateau'].sum()}}")

    for lab in ["injection", "flat-high"]:
        sub = df_settled[(df_settled["label"] == lab) & df_settled["ok_main"]]
        if len(sub) > 0:
            tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1e3)
            print(f"  {{lab:12s}}: N={{len(sub):4d}}, I={{sub['I_mean_A'].mean():.1f}} A, "
                  f"B1={{sub['B1_T'].mean():+.6f}} T, "
                  f"b2={{sub['b2_units'].mean():+.3f}}, b3={{sub['b3_units'].mean():+.3f}}, "
                  f"TF={{tf:.4f}} T/kA")"""))
    else:
        cells.append(code(f"s{n}-stats", f"""\
print("=" * 70)
print("{cfg.title}")
print("=" * 70)

for seg in SEGMENTS:
    if "{cfg.data_loader}" == "file_discovery":
        _s = summ[seg]
    else:
        _s = results[seg]["summ"]
    if _s.empty:
        print(f"\\n{{seg}}: no summary data")
        continue
    print(f"\\n--- {{seg}} ---")
    print(f"  Summary rows: {{len(_s)}}")
    if "I_nom" in _s.columns:
        print(f"  I range: {{_s['I_nom'].min():.1f}} .. {{_s['I_nom'].max():.1f}} A")
    if "B1_mean" in _s.columns:
        print(f"  B1 range: {{_s['B1_mean'].min():.6f}} .. {{_s['B1_mean'].max():.6f}} T")"""))
    return cells


def section_choices(cfg, n):
    """Analysis choices summary."""
    cells = [md(f"s{n}-hdr", f"---\n## {n}. Analysis Choices Summary\n\n"
                "Document all analysis parameters for reproducibility.")]
    cells.append(code(f"s{n}-choices", f"""\
import datetime
print("ANALYSIS CHOICES")
print("=" * 60)
print(f"Generated    : {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}}")
print(f"Title        : {cfg.title}")
print(f"Segments     : {{SEGMENTS}}")
print(f"Magnet order : {{MAGNET_ORDER}}")
print(f"R_ref        : {{R_REF}} m")
print(f"Samples/turn : {{SAMPLES_PER_TURN}}")
print(f"OPTIONS      : {{OPTIONS}}")
print(f"MIN_B1_T     : {{MIN_B1_T}}")
print(f"N_SIGMA_CLIP : {{N_SIGMA_CLIP}}")"""))
    return cells


def section_export(cfg, n):
    """CSV export."""
    out_dir = cfg.output_csv_dir
    cells = [md(f"s{n}-hdr", f"---\n## {n}. CSV Export")]

    if cfg.data_loader == "text_streaming":
        cells.append(code(f"s{n}-export", f"""\
out_dir = REPO_ROOT / "output" / "{out_dir}"
out_dir.mkdir(parents=True, exist_ok=True)

for seg in SEGMENTS:
    df_all = results[seg]["df"]
    df_settled = results[seg]["df_settled"]

    fname = f"MBB_{{seg}}_streaming_plateau.csv"
    df_all.to_csv(out_dir / fname, index=False)
    print(f"Wrote {{out_dir / fname}}  ({{len(df_all)}} rows)")

    fname_s = f"MBB_{{seg}}_streaming_settled.csv"
    df_settled.to_csv(out_dir / fname_s, index=False)
    print(f"Wrote {{out_dir / fname_s}}  ({{len(df_settled)}} rows)")

print("\\nDone.")"""))
    elif cfg.data_loader == "binary_streaming":
        label = cfg.title.split("--")[-1].strip().replace(" ", "_")
        cells.append(code(f"s{n}-export", f"""\
out_dir = REPO_ROOT / "output" / "{out_dir}"
out_dir.mkdir(parents=True, exist_ok=True)

for seg in SEGMENTS:
    _df = results[seg]["df"]
    fname = f"MC62_{{seg}}_all_turns.csv"
    _df.to_csv(out_dir / fname, index=False)
    print(f"Wrote {{out_dir / fname}}  ({{len(_df)}} rows)")

    _s = results[seg]["summ"]
    fname_s = f"MC62_{{seg}}_summary.csv"
    _s.to_csv(out_dir / fname_s, index=False)
    print(f"Wrote {{out_dir / fname_s}}  ({{len(_s)}} rows)")

print("\\nDone.")"""))
    else:  # file_discovery
        label = cfg.title.split("--")[-1].strip().replace(" ", "_")
        cells.append(code(f"s{n}-export", f"""\
out_dir = REPO_ROOT / "output" / "{out_dir}"
out_dir.mkdir(parents=True, exist_ok=True)

for seg in SEGMENTS:
    fname = f"MC62_{{seg}}_all_turns.csv"
    df[seg].to_csv(out_dir / fname, index=False)
    print(f"Wrote {{out_dir / fname}}  ({{len(df[seg])}} rows)")

    fname_s = f"MC62_{{seg}}_summary.csv"
    summ[seg].to_csv(out_dir / fname_s, index=False)
    print(f"Wrote {{out_dir / fname_s}}  ({{len(summ[seg])}} rows)")

print("\\nDone.")"""))
    return cells


# ================================================================
# Assembly functions
# ================================================================

def build_streaming_analysis(cfg):
    """Build all cells for a streaming analysis notebook."""
    section_names = _list_streaming_sections(cfg)
    n = count(1)
    cells = section_title_toc(cfg, section_names)
    cells += section_config_imports(cfg, next(n))
    cells += section_kn(cfg, next(n))
    if cfg.data_loader == "text_streaming":
        cells += section_load_text_streaming(cfg, next(n))
    else:
        cells += section_load_binary_streaming(cfg, next(n))
    cells += section_raw_signals(cfg, next(n))
    cells += section_celfed(cfg, next(n))
    cells += section_plateau_detection(cfg, next(n))
    if cfg.has_fdi:
        cells += section_fdi(cfg, next(n))
    if cfg.has_precycle:
        cells += section_precycle_id(cfg, next(n))
    cells += section_pipeline_streaming(cfg, next(n))
    if cfg.has_allturn:
        cells += section_allturn(cfg, next(n))
    if cfg.has_ffmm:
        cells += section_ffmm(cfg, next(n))
    cells += section_b1(cfg, next(n))
    cells += section_b2(cfg, next(n))
    cells += section_b3(cfg, next(n))
    cells += section_higher_harmonics(cfg, next(n))
    cells += section_spectrum(cfg, next(n))
    cells += section_tf(cfg, next(n))
    if cfg.has_inductance:
        cells += section_inductance(cfg, next(n))
    if cfg.has_eddy:
        cells += section_eddy_config(cfg, next(n))
        cells += section_eddy_raw(cfg)
        cells += section_eddy_fits(cfg, next(n))
        cells += section_eddy_tau(cfg)
        cells += section_eddy_bias(cfg, next(n))
        cells += section_eddy_nlast(cfg, next(n))
    cells += section_stats(cfg, next(n))
    cells += section_choices(cfg, next(n))
    cells += section_export(cfg, next(n))
    return cells


def build_file_discovery_analysis(cfg):
    """Build all cells for a file-discovery analysis notebook."""
    section_names = _list_file_discovery_sections(cfg)
    n = count(1)
    cells = section_title_toc(cfg, section_names)
    cells += section_config_imports(cfg, next(n))
    cells += section_kn(cfg, next(n))
    cells += section_load_file_discovery(cfg, next(n))
    cells += section_current_profile(cfg, next(n))
    cells += section_celfed(cfg, next(n))
    cells += section_pipeline_file_discovery(cfg, next(n))
    cells += section_plateau_quality(cfg, next(n))
    cells += section_b1(cfg, next(n))
    cells += section_b2(cfg, next(n))
    cells += section_b3(cfg, next(n))
    cells += section_higher_harmonics(cfg, next(n))
    cells += section_spectrum(cfg, next(n))
    cells += section_tf(cfg, next(n))
    if cfg.has_inductance:
        cells += section_inductance(cfg, next(n))
    if cfg.has_eddy:
        cells += section_eddy_config(cfg, next(n))
        cells += section_eddy_raw(cfg)
        cells += section_eddy_fits(cfg, next(n))
        cells += section_eddy_tau(cfg)
        cells += section_eddy_bias(cfg, next(n))
        cells += section_eddy_nlast(cfg, next(n))
    cells += section_stats(cfg, next(n))
    cells += section_choices(cfg, next(n))
    cells += section_export(cfg, next(n))
    return cells


def build_comparison(comp):
    """Build comparison notebook cells from pre-computed CSVs."""
    cells = []
    ds = comp.datasets
    seg_list = comp.segments

    # -- Title & TOC --------------------------------------------------
    ds_table = "\n".join(
        f"| **{d['name']}** | `{d['csv_dir']}` |" for d in ds
    )
    if comp.magnet_family == "MBB":
        toc_rows = (
            "| 1 | Configuration & Imports |\n"
            "| 2 | Load Settled CSVs |\n"
            "| 3 | B1 Comparison |\n"
            "| 4 | b2, b3 Comparison |\n"
            "| 5 | Multipole Spectrum Comparison |\n"
            "| 6 | Statistical Significance |\n"
            "| 7 | Summary |"
        )
    else:
        toc_rows = (
            "| 1 | Configuration & Imports |\n"
            "| 2 | Load Summary CSVs |\n"
            "| 3 | Overview |\n"
            "| 4 | B1 Overlay |\n"
            "| 5 | b2 Overlay |\n"
            "| 6 | b3 Overlay |\n"
            "| 7 | Transfer Function Overlay |\n"
            "| 8 | Per-Level Differences |\n"
            "| 9 | Multipole Spectrum Comparison |\n"
            "| 10 | Summary |"
        )
    cells.append(md("title", f"""# {comp.title}

| Dataset | Source |
|---------|--------|
{ds_table}

| # | Section |
|---|---------|
{toc_rows}"""))

    # -- 1. Config & Imports -------------------------------------------
    cells.append(md("c1-hdr", "---\n## 1. Configuration & Imports"))

    seg_list_str = repr(seg_list)
    out_paths = "\n".join(
        f'OUT_{i} = REPO_ROOT / "output" / "{d["csv_dir"]}"'
        for i, d in enumerate(ds)
    )
    out_asserts = "\n".join(
        f'assert OUT_{i}.exists(), f"{d["name"]} output not found: {{OUT_{i}}}"'
        for i, d in enumerate(ds)
    )
    ds_list_str = ", ".join(
        f'("{d["name"]}", OUT_{i})' for i, d in enumerate(ds)
    )
    cells.append(code("c1-config", f"""\
SEGMENTS = {seg_list_str}

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib widget
plt.rcParams.update({{"figure.figsize": (14, 5), "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100}})

REPO_ROOT = Path(".").resolve()
while REPO_ROOT != REPO_ROOT.parent:
    if (REPO_ROOT / "pyproject.toml").exists() or (REPO_ROOT / ".git").exists(): break
    REPO_ROOT = REPO_ROOT.parent

{out_paths}
{out_asserts}

DS_DIRS = [{ds_list_str}]
print("Comparison: {comp.title}")
for name, p in DS_DIRS:
    print(f"  {{name}}: {{p}}")"""))

    if comp.magnet_family == "MBB":
        _build_mbb_comparison(cells, comp)
    else:
        _build_mc62_comparison(cells, comp)

    return cells


def _build_mbb_comparison(cells, comp):
    """MBB comparison: per-turn settled CSVs, supercycle grouping."""
    ds = comp.datasets
    seg_list = comp.segments
    n_last = comp.n_last_turns

    # -- 2. Load CSVs --------------------------------------------------
    cells.append(md("c2-hdr", "---\n## 2. Load Settled CSVs"))

    cells.append(code("c2-load", f"""\
ds = {{}}
for name, out_dir in DS_DIRS:
    ds[name] = {{}}
    for seg in SEGMENTS:
        fname = f"MBB_{{seg}}_streaming_settled.csv"
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing: {{fpath}}"
        df = pd.read_csv(fpath)
        ds[name][seg] = df
        print(f"  {{name}} {{seg}}: {{len(df)}} settled turns")

# Also load eddy fit results if available
eddy_fits = {{}}
for name, out_dir in DS_DIRS:
    eddy_fits[name] = {{}}
    for seg in SEGMENTS:
        fpath = out_dir / f"b3_fits_{{seg}}.csv"
        if fpath.exists():
            eddy_fits[name][seg] = pd.read_csv(fpath)
            print(f"  {{name}} {{seg}} eddy fits: {{len(eddy_fits[name][seg])}} rows")
        else:
            eddy_fits[name][seg] = pd.DataFrame()"""))

    # -- 3. B1 Comparison -----------------------------------------------
    ds_colors = ", ".join(
        f'("{d["name"]}", "{c}")'
        for d, c in zip(ds, ["tab:blue", "tab:orange", "tab:green", "tab:red"])
    )
    cells.append(md("c3-hdr", "---\n## 3. B1 Comparison\n\nPer-supercycle B1 at injection and flat-top."))
    cells.append(code("c3-b1", f"""\
DS_COLORS = [{ds_colors}]
fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
        ax = axes[i, j]
        for ds_name, col in DS_COLORS:
            dfs = ds[ds_name][seg]
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            sc_avg = sub.groupby("sc_idx")["B1_T"].agg(["mean", "std"]).reset_index()
            ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                        fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
        ax.set_xlabel("Supercycle index"); ax.set_ylabel("B1 (T)")
        ax.set_title(f"B1 {{title_suffix}} -- {{seg}}"); ax.legend(fontsize=9)

fig.suptitle("B1 per Supercycle (settled)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # -- 4. b2, b3 Comparison -------------------------------------------
    cells.append(md("c4-hdr", "---\n## 4. b2, b3 Comparison"))
    cells.append(code("c4-harmonics", f"""\
for harm_name, harm_col, ylabel in [("b2", "b2_units", "b2 (units)"), ("b3", "b3_units", "b3 (units)")]:
    fig, axes = plt.subplots(len(SEGMENTS), 2, figsize=(14, 5 * len(SEGMENTS)))
    if len(SEGMENTS) == 1:
        axes = axes[np.newaxis, :]

    for i, seg in enumerate(SEGMENTS):
        for j, (lab, title_suffix) in enumerate([("injection", "Injection"), ("flat-high", "Flat-Top")]):
            ax = axes[i, j]
            for ds_name, col in DS_COLORS:
                dfs = ds[ds_name][seg]
                sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
                if len(sub) == 0: continue
                sc_avg = sub.groupby("sc_idx")[harm_col].agg(["mean", "std"]).reset_index()
                ax.errorbar(sc_avg["sc_idx"], sc_avg["mean"], yerr=sc_avg["std"],
                            fmt="o-", markersize=4, capsize=2, color=col, alpha=0.8, label=ds_name)
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("Supercycle index"); ax.set_ylabel(ylabel)
            ax.set_title(f"{{harm_name}} {{title_suffix}} -- {{seg}}"); ax.legend(fontsize=9)

    fig.suptitle(f"{{harm_name}} per Supercycle (settled)", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()

# Box plots
fig, axes = plt.subplots(len(SEGMENTS), 3, figsize=(16, 5 * len(SEGMENTS)))
if len(SEGMENTS) == 1:
    axes = axes[np.newaxis, :]

for i, seg in enumerate(SEGMENTS):
    for ax_idx, (col_name, ylabel, title) in enumerate([
            ("B1_T", "B1 (T)", "B1"), ("b2_units", "b2 (units)", "b2"), ("b3_units", "b3 (units)", "b3")]):
        ax = axes[i, ax_idx]
        box_data, box_labels, box_colors = [], [], []
        for ds_name, base_col in DS_COLORS:
            for lab, short in [("injection", "Inj"), ("flat-high", "FT")]:
                dfs = ds[ds_name][seg]
                sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
                if len(sub) == 0: continue
                box_data.append(sub[col_name].values)
                box_labels.append(f"{{ds_name}}\\n{{short}}\\n(N={{len(sub)}})")
                box_colors.append(base_col)
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            for patch, col in zip(bp["boxes"], box_colors): patch.set_facecolor(col); patch.set_alpha(0.5)
        ax.set_ylabel(ylabel); ax.set_title(f"{{title}} -- {{seg}}")
        ax.tick_params(axis="x", labelsize=7)

fig.suptitle("Distribution Comparison (settled turns)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # -- 5. Multipole Spectrum ------------------------------------------
    cells.append(md("c5-hdr", "---\n## 5. Multipole Spectrum Comparison\n\n"
                     "Overlay normal harmonic spectra at injection (first segment)."))
    cells.append(code("c5-spectrum", f"""\
seg = SEGMENTS[0]
lab = "injection"

bn_cols = [c for c in ds[DS_DIRS[0][0]][seg].columns if c.startswith("b") and c.endswith("_units")]
orders = sorted([int(c.replace("b", "").replace("_units", "")) for c in bn_cols])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(orders))
w = 0.8 / len(DS_DIRS)

for ax_idx, (title, yscale) in enumerate([("Linear", "linear"), ("Log", "log")]):
    ax = axes[ax_idx]
    for k, (ds_name, color) in enumerate(DS_COLORS):
        dfs = ds[ds_name][seg]
        sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
        if len(sub) == 0: continue
        means = [sub[f"b{{n}}_units"].mean() for n in orders]
        if yscale == "log":
            means = [abs(v) for v in means]
        ax.bar(x + (k - len(DS_DIRS)/2 + 0.5) * w, means, w, label=ds_name, color=color, alpha=0.8)
    if yscale == "linear":
        ax.axhline(0, color="grey", linewidth=0.5)
    else:
        ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(orders)
    ax.set_xlabel("Harmonic order n"); ax.set_ylabel("bn (units)" if yscale == "linear" else "|bn| (units)")
    ax.set_title(f"Multipole spectrum -- {{lab}} ({{title}})")
    ax.legend(fontsize=9)

fig.suptitle(f"Multipole Spectrum Comparison -- {{seg}} Injection", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # -- 6. Statistical Significance ------------------------------------
    d0name = ds[0]["name"]
    d1name = ds[1]["name"]
    cells.append(md("c6-hdr", f"---\n## 6. Statistical Significance\n\n"
                     f"Sigma = |diff| / sqrt(std1^2/N1 + std2^2/N2). > 3 sigma = real difference."))
    cells.append(code("c6-stats", f"""\
NAME_A, NAME_B = DS_DIRS[0][0], DS_DIRS[1][0]
print(f"Difference: ({{NAME_A}}) - ({{NAME_B}})  [settled turns]")
print("=" * 110)
all_results = []

for seg in SEGMENTS:
    print(f"\\n--- {{seg}} ---")
    for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
        sA = ds[NAME_A][seg]
        sA = sA[(sA["label"] == lab) & sA["ok_main"]]
        sB = ds[NAME_B][seg]
        sB = sB[(sB["label"] == lab) & sB["ok_main"]]
        if len(sA) == 0 or len(sB) == 0: continue

        results_row = {{"seg": seg, "label": lab, "desc": desc, "N_A": len(sA), "N_B": len(sB)}}
        for name, col in [("B1", "B1_T"), ("b2", "b2_units"), ("b3", "b3_units")]:
            diff = sA[col].mean() - sB[col].mean()
            err = np.sqrt((sA[col].std()**2/len(sA)) + (sB[col].std()**2/len(sB)))
            sig = abs(diff) / err if err > 0 else 0
            results_row[f"d{{name}}"] = diff
            results_row[f"sig_{{name}}"] = sig
        all_results.append(results_row)

        print(f"  {{desc:>12s}}  dB1={{results_row['dB1']:+.6f}}  db2={{results_row['db2']:+.4f}}  "
              f"db3={{results_row['db3']:+.4f}}")
        print(f"  {{'(sigma)':>12s}}  {{results_row['sig_B1']:>12.1f}}  {{results_row['sig_b2']:>14.1f}}  "
              f"{{results_row['sig_b3']:>14.1f}}")

print("\\nINTERPRETATION")
print("-" * 70)
for r in all_results:
    print(f"\\n  {{r['seg']}} -- {{r['desc']}}  (N: {{r['N_A']}} vs {{r['N_B']}} turns)")
    for name, unit in [("B1", "T"), ("b2", "units"), ("b3", "units")]:
        diff = r[f"d{{name}}"]
        sig = r[f"sig_{{name}}"]
        verdict = "REAL (>3 sigma)" if sig > 3 else ("suggestive (2-3 sigma)" if sig >= 2 else "no evidence (<2 sigma)")
        diff_str = f"{{diff*1e6:+.1f}} uT" if unit == "T" else f"{{diff:+.4f}} {{unit}}"
        print(f"    {{name:>3s}}: {{diff_str:>16s}}  ({{sig:.1f}} sigma) -> {{verdict}}")"""))

    # -- 7. Summary -----------------------------------------------------
    cells.append(md("c7-hdr", "---\n## 7. Summary"))
    out_dir = comp.output_csv_dir
    cells.append(code("c7-summary", f"""\
summary_rows = []
for ds_name, out_dir_path in DS_DIRS:
    for seg in SEGMENTS:
        dfs = ds[ds_name][seg]
        for lab, desc in [("injection", "Injection"), ("flat-high", "Flat-Top")]:
            sub = dfs[(dfs["label"] == lab) & dfs["ok_main"]]
            if len(sub) == 0: continue
            tf = sub["B1_T"].mean() / (sub["I_mean_A"].mean() / 1000.0) if "I_mean_A" in sub.columns else np.nan
            summary_rows.append({{
                "Dataset": ds_name, "Segment": seg, "Op. point": desc,
                "N turns": len(sub),
                "I mean (A)": f"{{sub['I_mean_A'].mean():.1f}}" if "I_mean_A" in sub.columns else "-",
                "B1 mean (T)": f"{{sub['B1_T'].mean():.6f}}", "B1 std (T)": f"{{sub['B1_T'].std():.6f}}",
                "b2 mean": f"{{sub['b2_units'].mean():+.4f}}", "b3 mean": f"{{sub['b3_units'].mean():+.4f}}",
                "TF (T/kA)": f"{{tf:.4f}}" if not np.isnan(tf) else "-",
            }})
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

# Export comparison summary
out_dir = REPO_ROOT / "output" / "{out_dir}"
out_dir.mkdir(parents=True, exist_ok=True)
df_summary.to_csv(out_dir / "summary_comparison_settled.csv", index=False)
print(f"\\nWrote {{out_dir / 'summary_comparison_settled.csv'}}")
print("\\nDone.")"""))


def _build_mc62_comparison(cells, comp):
    """MC62 comparison: per-run summary CSVs, hysteresis overlay."""
    ds = comp.datasets
    seg_list = comp.segments

    # -- 2. Load Summary CSVs -------------------------------------------
    cells.append(md("c2-hdr", "---\n## 2. Load Summary CSVs"))
    cells.append(code("c2-load", """\
summ = {}
for name, out_dir in DS_DIRS:
    summ[name] = {}
    for seg in SEGMENTS:
        fname = f"MC62_{seg}_summary.csv"
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing: {fpath}"
        df = pd.read_csv(fpath)
        summ[name][seg] = df
        good = df[df["quality"] == "good"]
        print(f"  {name} {seg}: {len(df)} runs ({len(good)} good)")"""))

    # -- 3. Overview table -----------------------------------------------
    cells.append(md("c3-hdr", "---\n## 3. Overview"))
    cells.append(code("c3-overview", """\
print(f"{'Dataset':<30s}  {'Segment':<12s}  {'Runs':>5s}  {'Good':>5s}  {'I range':>10s}")
print("=" * 75)
for name, _ in DS_DIRS:
    for seg in SEGMENTS:
        s = summ[name][seg]
        good = s[s["quality"] == "good"]
        I_lo = good["I_nom"].min() if len(good) else 0
        I_hi = good["I_nom"].max() if len(good) else 0
        print(f"  {name:<28s}  {seg:<12s}  {len(s):>5d}  {len(good):>5d}  "
              f"{I_lo:>+.0f} to {I_hi:>+.0f} A")"""))

    # -- 4-7. Overlay plots: B1, b2, b3, TF ----------------------------
    overlay_items = [
        (4, "B1", "B1_mean", "B1 (T)"),
        (5, "b2", "b2_units_mean", "b2 (units)"),
        (6, "b3", "b3_units_mean", "b3 (units)"),
        (7, "TF", "TF", "TF (T/kA)"),
    ]
    ds_styles = [
        ('"o"', '"#1f4e79"', '"#6fa8dc"'),
        ('"s"', '"#990000"', '"#e06666"'),
    ]
    for sec_n, harm_name, col_name, ylabel in overlay_items:
        cells.append(md(f"c{sec_n}-hdr", f"---\n## {sec_n}. {harm_name} Overlay"))

        # Build TF computation if needed
        tf_extra = ""
        if col_name == "TF":
            tf_extra = """
    # Compute TF if not already present
    if "TF" not in s.columns:
        s["TF"] = s["B1_mean"] / (s["I_nom"] / 1000.0)
"""

        cells.append(code(f"c{sec_n}-overlay", f"""\
fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(12, 4), squeeze=False)

DS_STYLES = [
    ("o", "#1f4e79", "#6fa8dc"),   # first dataset
    ("s", "#990000", "#e06666"),   # second dataset
]

for j, seg in enumerate(SEGMENTS):
    ax = axes[0, j]
    for k, (name, _) in enumerate(DS_DIRS):
        marker, c_asc, c_desc = DS_STYLES[k]
        s = summ[name][seg].copy(){tf_extra}
        good = s[(s["quality"] == "good") & (s["I_nom"].abs() > 2)]
        for br, color in [("ascending", c_asc), ("descending", c_desc)]:
            br_data = good[good["branch"] == br].sort_values("run_id")
            if len(br_data) == 0: continue
            I_vals = br_data["I_nom"].values
            breaks = [0] + [i for i in range(1, len(I_vals))
                            if abs(I_vals[i] - I_vals[i-1]) > 30] + [len(I_vals)]
            for bi in range(len(breaks)-1):
                seg_data = br_data.iloc[breaks[bi]:breaks[bi+1]]
                ax.plot(seg_data["I_nom"], seg_data["{col_name}"],
                        marker=marker, ms=4, lw=0, alpha=0.7, color=color,
                        label=f"{{name}} {{br[:3]}}" if bi == 0 else "")
    ax.set_xlabel("I (A)"); ax.set_ylabel("{ylabel}")
    ax.set_title(f"{harm_name} -- {{seg}}")
    ax.legend(fontsize=7, loc="best")

fig.suptitle("{harm_name} vs Current", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # -- 8. Per-Level Differences ----------------------------------------
    d0name = ds[0]["name"]
    d1name = ds[1]["name"]
    cells.append(md("c8-hdr", f"---\n## 8. Per-Level Differences\n\n"
                     f"Difference = ({d1name}) - ({d0name})"))
    cells.append(code("c8-diff", f"""\
NAME_A, NAME_B = DS_DIRS[0][0], DS_DIRS[1][0]

for seg in SEGMENTS:
    sA = summ[NAME_A][seg].copy()
    sB = summ[NAME_B][seg].copy()

    # Compute TF if missing
    for s in [sA, sB]:
        if "TF" not in s.columns:
            s["TF"] = s["B1_mean"] / (s["I_nom"] / 1000.0)

    merged = pd.merge(
        sA[sA["quality"] == "good"],
        sB[sB["quality"] == "good"],
        on=["I_nom", "branch"], suffixes=("_A", "_B"), how="inner"
    )
    if len(merged) == 0:
        print(f"No matching runs for {{seg}}")
        continue

    merged["dB1"] = merged["B1_mean_B"] - merged["B1_mean_A"]
    merged["db2"] = merged["b2_units_mean_B"] - merged["b2_units_mean_A"]
    merged["db3"] = merged["b3_units_mean_B"] - merged["b3_units_mean_A"]
    merged["dTF"] = merged["TF_B"] - merged["TF_A"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (col, ylabel, title) in zip(axes.flat, [
            ("dB1", "dB1 (T)", "dB1"), ("db2", "db2 (units)", "db2"),
            ("db3", "db3 (units)", "db3"), ("dTF", "dTF (T/kA)", "dTF")]):
        x_labels = [f"{{row['I_nom']:.0f}}\\n{{row['branch'][:3]}}" for _, row in merged.iterrows()]
        vals = merged[col].values
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_ylabel(ylabel); ax.set_title(f"{{title}} -- {{seg}}")

    fig.suptitle(f"Differences ({{NAME_B}}) - ({{NAME_A}}) -- {{seg}}", fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()

    # Summary statistics
    sub = merged[merged["I_nom"].abs() > 2]
    print(f"\\n{{seg}} -- matched runs with |I| > 2 A: {{len(sub)}}")
    for col in ["dB1", "db2", "db3", "dTF"]:
        vals = sub[col].values
        print(f"  {{col:>4s}}: max|d|={{np.abs(vals).max():.6f}}, "
              f"mean|d|={{np.abs(vals).mean():.6f}}, rms={{np.sqrt((vals**2).mean()):.6f}}")"""))

    # -- 9. Multipole Spectrum Comparison --------------------------------
    cells.append(md("c9-hdr", "---\n## 9. Multipole Spectrum Comparison\n\n"
                     "Overlay at peak current for the first segment."))
    cells.append(code("c9-spectrum", """\
seg = SEGMENTS[0]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
orders = list(range(2, 16))

for k, (name, _) in enumerate(DS_DIRS):
    s = summ[name][seg]
    good = s[s["quality"] == "good"]
    if len(good) == 0: continue
    peak_row = good.loc[good["I_nom"].abs().idxmax()]
    means = []
    for n in orders:
        col = f"b{n}_units_mean"
        means.append(peak_row[col] if col in good.columns else 0)

    marker, c_asc, _ = DS_STYLES[k]
    x = np.arange(len(orders))
    w = 0.35
    offset = (k - 0.5) * w

    for ax_idx, (title, yscale) in enumerate([("Linear", "linear"), ("Log", "log")]):
        ax = axes[ax_idx]
        vals = [abs(v) for v in means] if yscale == "log" else means
        ax.bar(x + offset, vals, w, label=f"{name} (I={peak_row['I_nom']:.0f} A)",
               color=c_asc, alpha=0.8)

for ax_idx, (title, yscale) in enumerate([("Linear", "linear"), ("Log", "log")]):
    ax = axes[ax_idx]
    if yscale == "linear":
        ax.axhline(0, color="grey", linewidth=0.5)
    else:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(len(orders))); ax.set_xticklabels(orders)
    ax.set_xlabel("Harmonic order n")
    ax.set_ylabel("bn (units)" if yscale == "linear" else "|bn| (units)")
    ax.set_title(f"Multipole spectrum -- peak current ({title})")
    ax.legend(fontsize=9)

fig.suptitle(f"Multipole Spectrum -- {seg}", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()"""))

    # -- 10. Summary ----------------------------------------------------
    cells.append(md("c10-hdr", "---\n## 10. Summary"))
    out_dir = comp.output_csv_dir
    cells.append(code("c10-summary", f"""\
summary_rows = []
for name, _ in DS_DIRS:
    for seg in SEGMENTS:
        s = summ[name][seg]
        good = s[s["quality"] == "good"]
        if len(good) == 0: continue
        # Peak current row
        peak = good.loc[good["I_nom"].abs().idxmax()]

        # Compute TF if missing
        if "TF" not in good.columns:
            tf_peak = peak["B1_mean"] / (peak["I_nom"] / 1000.0)
        else:
            tf_peak = peak["TF"]

        summary_rows.append({{
            "Dataset": name, "Segment": seg,
            "Good runs": len(good),
            "I range (A)": f"{{good['I_nom'].min():.0f}} to {{good['I_nom'].max():.0f}}",
            "B1 peak (T)": f"{{peak['B1_mean']:.6f}}",
            "b2 peak": f"{{peak['b2_units_mean']:+.4f}}",
            "b3 peak": f"{{peak['b3_units_mean']:+.4f}}",
            "TF peak (T/kA)": f"{{tf_peak:.4f}}",
        }})

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

# Export
out_dir = REPO_ROOT / "output" / "{out_dir}"
out_dir.mkdir(parents=True, exist_ok=True)
df_summary.to_csv(out_dir / "summary_comparison.csv", index=False)
print(f"\\nWrote {{out_dir / 'summary_comparison.csv'}}")
print("\\nDone.")"""))


# ================================================================
# Measurement configurations
# ================================================================

_MBB_KN_UPPSALA = "MBB/2026-02-25_2Hz/MBB/Kn Uppsala/Kn_values_Seg_Main_A_AC.txt"
_MBB_KN_CROSS = "MBB/2025-12-12/CRMMMMH_AV-00000001/Kn_values_Seg_Main_A_AC.txt"
_MC62_KN_INT = "MC62/2026-02-11/Kn values/Kn_R45_PCB_N1_0001_A_AC.txt"
_MC62_KN_CEN = "MC62/2026-02-11/Kn values/Kn_DQ_5_18_7_250_47x50_0001_A_AC.txt"

MEASUREMENTS = {
    # --- MBB 2 Hz (NCS + CS) ---
    "MBB_2Hz_200GeV": MeasurementConfig(
        title="SPS MBB Dipole -- 200 GeV MD1, 2 Hz",
        magnet_family="MBB",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-25_2Hz/200GeV_analysis.ipynb",
        output_csv_dir="MBB/2026-02-25_2Hz/200GeV",
        magnet_order=1, r_ref=0.02, l_coil=0.47, samples_per_turn=1024,
        segments=[
            SegmentConfig("NCS", _MBB_KN_UPPSALA),
            SegmentConfig("CS", _MBB_KN_UPPSALA, is_fringe=True),
        ],
        data_loader="text_streaming",
        session="MBB/2026-02-25_2Hz/MBB/200 GeV/20260225_183154_SPS_MBB",
        meas_subdir="20260225_183213_MBB",
        plateau_i_range_max=2.5,
        n_last_turns=18, n_last_turns_high=None,
        energy_label="200 GeV",
        has_ffmm=True, ffmm_rotate_excludes_last=True,
    ),
    "MBB_2Hz_26GeV": MeasurementConfig(
        title="SPS MBB Dipole -- 26 GeV MD1, 2 Hz",
        magnet_family="MBB",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-25_2Hz/26GeV_analysis.ipynb",
        output_csv_dir="MBB/2026-02-25_2Hz/26GeV",
        magnet_order=1, r_ref=0.02, l_coil=0.47, samples_per_turn=1024,
        segments=[
            SegmentConfig("NCS", _MBB_KN_UPPSALA),
            SegmentConfig("CS", _MBB_KN_UPPSALA, is_fringe=True),
        ],
        data_loader="text_streaming",
        session="MBB/2026-02-25_2Hz/MBB/26 GeV/20260225_181040_SPS_MBB",
        meas_subdir="20260225_181058_MBB",
        plateau_i_range_max=2.5,
        n_last_turns=18, n_last_turns_high=None,
        energy_label="26 GeV",
        has_ffmm=True, ffmm_rotate_excludes_last=True,
    ),
    # --- MBB Standard (NCS only) ---
    "MBB_std_200GeV": MeasurementConfig(
        title="SPS MBB Dipole -- 200 GeV MD1 Extended NCS",
        magnet_family="MBB",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-06_NCS_supercycle/200GeV_analysis.ipynb",
        output_csv_dir="MBB/2026-02-06_supercycle/01_200_extended",
        magnet_order=1, r_ref=0.02, l_coil=0.47, samples_per_turn=1024,
        segments=[SegmentConfig("NCS", _MBB_KN_CROSS)],
        data_loader="text_streaming",
        session="MBB/2026-02-06_supercycle/01_200_extended/20260206_144537_SPS_MBB",
        meas_subdir="20260206_144559_MBB",
        plateau_i_range_max=3.0,
        n_last_turns=18, n_last_turns_high=None,
        energy_label="200 GeV",
        has_ffmm=True, ffmm_rotate_excludes_last=True,
    ),
    "MBB_std_26GeV": MeasurementConfig(
        title="SPS MBB Dipole -- 26 GeV MD1 Extended NCS",
        magnet_family="MBB",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-06_NCS_supercycle/26GeV_analysis.ipynb",
        output_csv_dir="MBB/2026-02-06_supercycle/03_26_extended",
        magnet_order=1, r_ref=0.02, l_coil=0.47, samples_per_turn=1024,
        segments=[SegmentConfig("NCS", _MBB_KN_CROSS)],
        data_loader="text_streaming",
        session="MBB/2026-02-06_supercycle/03_26_extended/20260206_151808_SPS_MBB",
        meas_subdir="20260206_151827_MBB",
        plateau_i_range_max=3.0,
        n_last_turns=18, n_last_turns_high=None,
        energy_label="26 GeV",
        has_ffmm=True, ffmm_rotate_excludes_last=True,
    ),
    # --- MC62 File Discovery (01, 02) ---
    "MC62_01_shims": MeasurementConfig(
        title="LEAR MC62 -- 01 Staircase with Shims",
        magnet_family="MC62",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/01_with_shims/analysis.ipynb",
        output_csv_dir="MC62/01_with_shims",
        magnet_order=1, r_ref=0.033, l_coil=0.0, samples_per_turn=1024,
        segments=[
            SegmentConfig("Integral", _MC62_KN_INT),
            SegmentConfig("Central", _MC62_KN_CEN, merge_mode="abs_all"),
        ],
        data_loader="file_discovery",
        run_dir_rel="MC62/2026-02-11/01_staircase_with_shims/20260211_114759_staircase_MC62/20260211_133720_MC62",
        encoder_offset_rad=3.141592653589793,
        min_b1_T=1e-6,
        n_last_turns=170,
        has_fdi=False, has_allturn=False, has_ffmm=False,
        has_inductance=True, has_eddy=True,
    ),
    "MC62_02_no_shims": MeasurementConfig(
        title="LEAR MC62 -- 02 Staircase without Shims",
        magnet_family="MC62",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/02_without_shims/analysis.ipynb",
        output_csv_dir="MC62/02_without_shims",
        magnet_order=1, r_ref=0.033, l_coil=0.0, samples_per_turn=1024,
        segments=[
            SegmentConfig("Integral", _MC62_KN_INT),
            SegmentConfig("Central", _MC62_KN_CEN, merge_mode="abs_all"),
        ],
        data_loader="file_discovery",
        run_dir_rel="MC62/2026-02-11/02_staircase_without_shims/20260212_075344_staircase_without_shims_MC62/20260212_100000_MC62",
        encoder_offset_rad=3.141592653589793,
        min_b1_T=1e-6,
        n_last_turns=170,
        has_fdi=False, has_allturn=False, has_ffmm=False,
        has_inductance=True, has_eddy=True,
    ),
    # --- MC62 Binary Streaming (03, 04) ---
    "MC62_03_2Hz_pm": MeasurementConfig(
        title="LEAR MC62 -- 03 Staircase 2 Hz (Afternoon)",
        magnet_family="MC62",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/03_2Hz_afternoon/analysis.ipynb",
        output_csv_dir="MC62/03_2Hz_afternoon",
        magnet_order=1, r_ref=0.033, l_coil=0.0, samples_per_turn=512,
        segments=[
            SegmentConfig("Integral", _MC62_KN_INT,
                          data_path="MC62_20260216_170750_staircase_2Hz_corr_sigs_Ap_1_SegIntegral.bin"),
            SegmentConfig("Central", _MC62_KN_CEN, merge_mode="abs_all",
                          data_path="MC62_20260216_170750_staircase_2Hz_corr_sigs_Ap_1_SegCentral.bin"),
        ],
        data_loader="binary_streaming",
        session="MC62/2026-02-16_staircase_2Hz/aperture1",
        encoder_offset_rad=3.141592653589793,
        min_b1_T=1e-6, rpm=120.0,
        plateau_i_range_max=0.5, plateau_min_length=50, plateau_merge_gap=100,
        n_last_turns=340,
        has_precycle=True, has_fdi=True, has_allturn=True,
        has_ffmm=True, ffmm_r_ref=0.33,
        ffmm_options=("dri", "rot", "cel", "fed", "dit"),
        ffmm_rotate_excludes_last=False,
        has_eddy=True, has_inductance=True,
    ),
    "MC62_04_2Hz_am": MeasurementConfig(
        title="LEAR MC62 -- 04 Staircase 2 Hz (Morning)",
        magnet_family="MC62",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/04_2Hz_morning/analysis.ipynb",
        output_csv_dir="MC62/04_2Hz_morning",
        magnet_order=1, r_ref=0.033, l_coil=0.0, samples_per_turn=512,
        segments=[
            SegmentConfig("Integral", _MC62_KN_INT,
                          data_path="MC62_20260217_094521_staircase_2Hz_corr_sigs_Ap_1_SegIntegral.bin"),
            SegmentConfig("Central", _MC62_KN_CEN, merge_mode="abs_all",
                          data_path="MC62_20260217_094521_staircase_2Hz_corr_sigs_Ap_1_SegCentral.bin"),
        ],
        data_loader="binary_streaming",
        session="MC62/2026-02-17_staircase_2Hz_morning/aperture1",
        encoder_offset_rad=3.141592653589793,
        min_b1_T=1e-6, rpm=120.0,
        plateau_i_range_max=0.5, plateau_min_length=50, plateau_merge_gap=100,
        n_last_turns=340, n_skip_end=20,
        has_precycle=True, has_fdi=True, has_allturn=True,
        has_ffmm=True, ffmm_r_ref=0.33,
        ffmm_options=("dri", "rot", "cel", "fed", "dit"),
        ffmm_rotate_excludes_last=False,
        has_eddy=True, has_inductance=True,
    ),
}


# ================================================================
# Comparison configurations
# ================================================================

COMPARISONS = {
    "MBB_2Hz_compare": ComparisonConfig(
        title="B1, b2, b3 Comparison: 200 GeV vs 26 GeV (2 Hz)",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-25_2Hz/comparison.ipynb",
        magnet_family="MBB",
        segments=["NCS", "CS"],
        datasets=[
            {"name": "200 GeV", "csv_dir": "MBB/2026-02-25_2Hz/200GeV"},
            {"name": "26 GeV", "csv_dir": "MBB/2026-02-25_2Hz/26GeV"},
        ],
        output_csv_dir="MBB/2026-02-25_2Hz/compare_200_vs_26",
        n_last_turns=18,
    ),
    "MBB_std_compare": ComparisonConfig(
        title="B1, b2, b3 Comparison: 200 GeV vs 26 GeV (NCS Extended)",
        notebook_path="rotating_coil_analyzer/notebooks/SPS_MBB/2026-02-06_NCS_supercycle/comparison.ipynb",
        magnet_family="MBB",
        segments=["NCS"],
        datasets=[
            {"name": "200 GeV", "csv_dir": "MBB/2026-02-06_supercycle/01_200_extended"},
            {"name": "26 GeV", "csv_dir": "MBB/2026-02-06_supercycle/03_26_extended"},
        ],
        output_csv_dir="MBB/2026-02-06_supercycle/compare_200_vs_26",
        n_last_turns=18,
    ),
    "MC62_01_vs_02": ComparisonConfig(
        title="MC62 Shims Effect: 01 (with) vs 02 (without)",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/comparisons/shims_effect_01_vs_02/comparison.ipynb",
        magnet_family="MC62",
        segments=["Integral", "Central"],
        datasets=[
            {"name": "Test 01 (with shims)", "csv_dir": "MC62/01_with_shims"},
            {"name": "Test 02 (without shims)", "csv_dir": "MC62/02_without_shims"},
        ],
        output_csv_dir="MC62/compare_01_vs_02",
        n_last_turns=170,
    ),
    "MC62_03_vs_04": ComparisonConfig(
        title="MC62 Reproducibility: 03 (PM) vs 04 (AM)",
        notebook_path="rotating_coil_analyzer/notebooks/LEAR_MC62/comparisons/reproducibility_03_vs_04/comparison.ipynb",
        magnet_family="MC62",
        segments=["Integral", "Central"],
        datasets=[
            {"name": "Test 03 (Feb 16 PM)", "csv_dir": "MC62/03_2Hz_afternoon"},
            {"name": "Test 04 (Feb 17 AM)", "csv_dir": "MC62/04_2Hz_morning"},
        ],
        output_csv_dir="MC62/compare_03_vs_04",
        n_last_turns=340,
    ),
}


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate analysis notebooks.")
    parser.add_argument("--all", action="store_true", help="Generate all notebooks")
    parser.add_argument("--mbb", action="store_true", help="Generate MBB notebooks only")
    parser.add_argument("--mc62", action="store_true", help="Generate MC62 notebooks only")
    parser.add_argument("names", nargs="*", help="Specific measurement/comparison names")
    args = parser.parse_args()

    # Determine which to generate
    meas_names = []
    comp_names = []
    if args.all:
        meas_names = list(MEASUREMENTS.keys())
        comp_names = list(COMPARISONS.keys())
    elif args.mbb:
        meas_names = [k for k in MEASUREMENTS if k.startswith("MBB")]
        comp_names = [k for k in COMPARISONS if k.startswith("MBB")]
    elif args.mc62:
        meas_names = [k for k in MEASUREMENTS if k.startswith("MC62")]
        comp_names = [k for k in COMPARISONS if k.startswith("MC62")]
    elif args.names:
        for name in args.names:
            if name in MEASUREMENTS:
                meas_names.append(name)
            elif name in COMPARISONS:
                comp_names.append(name)
            else:
                parser.error(f"Unknown measurement/comparison: {name}")
    else:
        parser.print_help()
        print("\nAvailable measurements:", ", ".join(MEASUREMENTS.keys()))
        print("Available comparisons:", ", ".join(COMPARISONS.keys()))
        return

    print(f"Generating {len(meas_names)} analysis + {len(comp_names)} comparison notebooks...\n")

    for name in meas_names:
        cfg = MEASUREMENTS[name]
        print(f"[{name}] {cfg.title}")
        if cfg.data_loader in ("text_streaming", "binary_streaming"):
            cells = build_streaming_analysis(cfg)
        else:
            cells = build_file_discovery_analysis(cfg)
        write_notebook(Path(cfg.notebook_path), cells)

    for name in comp_names:
        comp = COMPARISONS[name]
        print(f"[{name}] {comp.title}")
        cells = build_comparison(comp)
        if cells:
            write_notebook(Path(comp.notebook_path), cells)
        else:
            print("  (comparison builder not yet implemented)")

    print(f"\nDone. Generated {len(meas_names)} analysis + {len(comp_names)} comparison notebooks.")


if __name__ == "__main__":
    main()
