"""Command-line analysis of one measurement segment: folder + kn file -> CSV.

Run ``rca-analyze --help`` (or ``python -m rotating_coil_analyzer.cli --help``).
Without a folder it runs the bundled FCC-ee quadrupole demo and compares the
result with the FFMM results file shipped with it.

The settings that decide the physics (main order m, reference radius, skew,
pipeline options) are taken from ``Parameters.txt`` when present, can be
overridden on the command line, and are printed before anything is computed.
Nothing is guessed: a missing m or Rref stops the run with a message.
"""

from __future__ import annotations

import argparse
import sys
import warnings
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

DEMO_ZIP = Path(__file__).parent / "tests" / "data" / "20231110_140210_FCC-ee_center_DC_PXMQNDI8WC-CR000001.zip"
VALID_OPTIONS = ("dit", "dri", "rot", "cel", "fed")


def _extract_demo(out_dir: Path) -> Path:
    target = out_dir / "demo_data"
    if not target.exists():
        zipfile.ZipFile(DEMO_ZIP).extractall(target)
    return next(p for p in target.iterdir() if p.is_dir())


def _find_kn(folder: Path, seg: str) -> Path:
    hits = sorted(folder.rglob(f"*Kn_values*Seg*{seg}*.txt")) or sorted(folder.rglob("*Kn_values*.txt"))
    if len(hits) != 1:
        found = ", ".join(p.name for p in hits) or "none"
        raise SystemExit(f"Could not pick a kn file automatically (found: {found}). Pass --kn PATH.")
    return hits[0]


def _parse_options(s: str) -> tuple[str, ...]:
    toks = tuple(t.strip().lower() for t in s.split(",") if t.strip())
    bad = [t for t in toks if t not in VALID_OPTIONS]
    if bad:
        raise SystemExit(f"Unknown option(s) {bad}; valid: {','.join(VALID_OPTIONS)} ('nor' is applied post-merge).")
    return toks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rca-analyze",
        description="Rotating-coil harmonics for one segment: measurement folder + kn file -> per-turn and summary CSV.",
    )
    p.add_argument("folder", nargs="?", help="Measurement folder (contains Parameters.txt). Omit to run the bundled demo.")
    p.add_argument("--kn", help="Segment kn TXT file (4 or 6 columns). Default: the single *Kn_values*Seg_<seg>*.txt found in the folder.")
    p.add_argument("--seg", help="Segment id (from the FDIs table). Default: the only one / the first one.")
    p.add_argument("--run", help="Run id. Default: the first run found.")
    p.add_argument("--order", type=int, help="Main field order m (1 dipole, 2 quadrupole, ...). Overrides Parameters.txt.")
    p.add_argument("--rref", type=float, help="Reference radius in metres. Overrides Parameters.txt refRadius.")
    p.add_argument("--skew", action="store_true", help="Main harmonic is skew (overrides magnetNormSkew).")
    p.add_argument("--normal", action="store_true", help="Main harmonic is normal (overrides magnetNormSkew).")
    p.add_argument("--options", help="Comma-separated pipeline options among dit,dri,rot,cel,fed. Default: from Parameters.txt (without nor), else dri,rot.")
    p.add_argument("--min-main-field", type=float, default=1e-4, help="Turns with |B_m| below this (T) get NaN units. Default 1e-4.")
    p.add_argument("--out", default="rca_output", help="Output directory (default ./rca_output).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
    from rotating_coil_analyzer.analysis.turns import split_into_turns
    from rotating_coil_analyzer.analysis.utility_functions import build_harmonic_rows, process_kn_pipeline
    from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
    from rotating_coil_analyzer.ingest.readers_plateau import PlateauReader
    from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader
    from rotating_coil_analyzer.models.profile import AnalysisProfile

    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demo = args.folder is None
    folder = _extract_demo(out) if demo else Path(args.folder)
    if demo:
        print(f"[demo] bundled FCC-ee quadrupole dataset extracted to {folder}")

    cat = MeasurementDiscovery().build_catalog(folder)
    for w in cat.warnings:
        print(("  ! " if w.startswith("WARNING") else "  - ") + w)

    seg_ids = [s.segment_id for s in cat.segments]
    seg = args.seg or (seg_ids[0] if seg_ids else None)
    keys = sorted(k for k in cat.segment_files if k[2] == seg and (args.run is None or k[0] == args.run))
    if not keys:
        raise SystemExit(f"No data file for segment {seg!r}{' run ' + args.run if args.run else ''}. "
                         f"Segments: {seg_ids}; runs: {list(cat.runs)}.")
    key = keys[0]
    data_file = cat.segment_files[key]

    overrides = {}
    if args.order is not None:
        overrides["magnet_order"] = args.order
    if args.rref is not None:
        overrides["r_ref_m"] = args.rref
    if args.skew or args.normal:
        overrides["skew_main"] = bool(args.skew)
    if args.options:
        overrides["options"] = _parse_options(args.options)
    elif cat.analyzer_options:
        overrides["options"] = tuple(o for o in cat.analyzer_options if o != "nor") or ("dri", "rot")
    try:
        profile = AnalysisProfile.from_catalog(cat, **overrides)
    except ValueError as e:
        raise SystemExit(f"{e}\nUse --order and/or --rref.")

    kn_path = Path(args.kn) if args.kn else _find_kn(folder, seg)
    kn = load_segment_kn_txt(str(kn_path))
    print(f"run {key[0]}  segment {seg}  file {data_file.name}")
    print(f"kn {kn_path.name} (H={kn.orders.size})  m={profile.magnet_order}  Rref={profile.r_ref_m * 1e3:g} mm  "
          f"skew={profile.skew_main}  options={','.join(profile.options)}  Ns={cat.samples_per_turn}")

    if data_file.suffix.lower() == ".bin" or "corr_sigs" in data_file.name:
        sf = StreamingReader().read(data_file, run_id=key[0], segment=seg, samples_per_turn=cat.samples_per_turn,
                                    shaft_speed_rpm=cat.shaft_speed_rpm, aperture_id=key[1])
    else:
        sf = PlateauReader().read(data_file, run_id=key[0], segment=seg, samples_per_turn=cat.samples_per_turn,
                                  aperture_id=key[1])
    for w in sf.warnings:
        if w.startswith("WARNING") or "swapped" in w:
            print("  ! " + w)
    tb = split_into_turns(sf)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res, C_merged, C_units, ok = process_kn_pipeline(
            tb.df_abs, tb.df_cmp, tb.t, tb.I, kn,
            r_ref=profile.r_ref_m, magnet_order=profile.magnet_order, options=profile.options,
            min_b1_T=args.min_main_field, skew_main=profile.skew_main,
        )
    for w in caught:
        print(f"  ! {w.message}")

    extra = None
    if tb.plateau_id is not None:
        extra = [{"plateau_id": int(tb.plateau_id[i]), "plateau_I_hint_A": float(tb.plateau_I_hint[i])}
                 for i in range(tb.n_turns)]
    df = pd.DataFrame(build_harmonic_rows(res, C_merged, C_units, ok, profile.magnet_order, extra))
    stem = f"{key[0]}_seg{seg}"
    per_turn = out / f"{stem}_per_turn.csv"
    df.to_csv(per_turn, index=False)
    group = ["plateau_id"] if "plateau_id" in df.columns else []
    summary = (df[df["ok_main"]].groupby(group).agg(["mean", "std"]) if group
               else df[df["ok_main"]].agg(["mean", "std"]).T)
    summary_path = out / f"{stem}_summary.csv"
    summary.to_csv(summary_path)

    m = profile.magnet_order
    main_col = f"A{m}_T" if profile.skew_main else f"B{m}_T"
    print(f"\n{tb.n_turns} turns, {int(ok.sum())} with |B_m| > {args.min_main_field:g} T")
    print(f"{main_col} = {df[main_col].mean():.6g} T   x = {df['x_mm'].mean():.4f} mm   y = {df['y_mm'].mean():.4f} mm")
    cols = [c for c in df.columns if c.endswith("_units")][:8]
    print("  ".join(f"{c.replace('_units', '')}={df.loc[df['ok_main'], c].mean():.4f}" for c in cols))
    print(f"\nwrote {per_turn}\nwrote {summary_path}")

    if demo:
        ref = pd.read_csv(next(folder.rglob("*_Main_results.txt")), sep="\t")
        d3 = float(np.max(np.abs(df["b3_units"].to_numpy() - ref["b3(Units)"].to_numpy())))
        print(f"[demo] max |b3 - FFMM b3| over {len(ref)} turns = {d3:.1e} units")
    return 0


if __name__ == "__main__":
    sys.exit(main())
