from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig


# Keep a single active GUI instance per kernel to avoid duplicated callbacks / stacked widgets.
_ACTIVE_GUI: Optional[w.Widget] = None


def build_catalog_gui() -> w.Widget:
    """
    Phase-1 catalog/preview GUI (Jupyter / VSCode notebooks).
    """
    global _ACTIVE_GUI

    # Close previous instance created from this module (do NOT close all widgets).
    if _ACTIVE_GUI is not None:
        try:
            _ACTIVE_GUI.close()
        except Exception:
            pass
        _ACTIVE_GUI = None

    out = w.Output(layout=w.Layout(border="1px solid #ddd", padding="8px"))

    folder = w.Text(
        description="Folder",
        placeholder=".../aperture1",
        layout=w.Layout(width="80%"),
    )
    btn_browse = w.Button(description="Browse…", layout=w.Layout(width="120px"))
    btn_load = w.Button(description="Load Catalog", button_style="primary", layout=w.Layout(width="140px"))

    dd_run = w.Dropdown(options=[], description="Run", layout=w.Layout(width="320px"))
    dd_segment = w.Dropdown(options=[], description="Segment", layout=w.Layout(width="240px"))
    dd_mode = w.Dropdown(options=[("abs", "abs"), ("cmp", "cmp")], value="abs", description="Mode", layout=w.Layout(width="220px"))

    btn_preview = w.Button(description="Preview", button_style="", layout=w.Layout(width="120px"))
    btn_diag = w.Button(description="Diagnostics", button_style="warning", layout=w.Layout(width="140px"))

    # Diagnostics parameters
    alpha_jump = w.FloatText(value=10.0, description="α jump", layout=w.Layout(width="220px"))
    tol_turn = w.FloatText(value=1.0, description="Turn tol %", layout=w.Layout(width="220px"))

    header = w.HBox([folder, btn_browse, btn_load])
    row2 = w.HBox([dd_run, dd_segment, dd_mode, btn_preview, btn_diag])
    row3 = w.HBox([alpha_jump, tol_turn])

    gui = w.VBox([header, row2, row3, out])
    _ACTIVE_GUI = gui

    # State
    state = {"catalog": None}

    def _print_catalog_debug(cat) -> None:
        print("=== CATALOG DEBUG ===")
        print("root_dir:", str(cat.root_dir))
        print("Parameters.txt:", str(cat.parameters_path))
        print("samples_per_turn:", cat.samples_per_turn)
        print("shaft_speed_rpm:", cat.shaft_speed_rpm)
        print("enabled_apertures:", cat.enabled_apertures)
        print("segments:", [(s.segment_id, s.fdi_abs, s.fdi_cmp, s.length_m) for s in cat.segments])
        print("runs_found:", len(cat.runs))
        print("segment_files_found:", len(cat.segment_files))
        if cat.warnings:
            print("\nWARNINGS:")
            for m in cat.warnings:
                print(" -", m)

    def _plot_first_turns(seg_frame, ycol: str, title_prefix: str) -> None:
        Ns = seg_frame.samples_per_turn
        n_show_turns = min(3, seg_frame.n_turns)
        n = n_show_turns * Ns
        df = seg_frame.df.iloc[:n]
        t = df["t"].to_numpy()
        y = df[ycol].to_numpy()

        plt.figure()
        plt.plot(t, y)
        plt.title(f"{title_prefix} — first {n_show_turns} turns")
        plt.xlabel("t [s]")
        plt.ylabel(f"{ycol} (selected channel)")
        plt.show()

        plt.figure()
        plt.plot(t, df["I"].to_numpy())
        plt.title(f"{title_prefix} current — first {n_show_turns} turns")
        plt.xlabel("t [s]")
        plt.ylabel("I [A]")
        plt.show()

    def _time_discontinuity_diag(seg_frame, alpha: float) -> None:
        Ns = seg_frame.samples_per_turn
        t = seg_frame.df["t"].to_numpy()
        dt = np.diff(t)
        dt_med = float(np.median(dt)) if dt.size else float("nan")
        thr = float(alpha) * dt_med

        idx = np.where(dt > thr)[0]
        print("\n=== TIME DISCONTINUITY DIAGNOSTICS (Phase-1 quick tool) ===")
        print(f"segment: {seg_frame.segment} | run: {seg_frame.run_id}")
        print(f"samples_per_turn: {Ns}")
        print(f"dt median={dt_med:.6g} s, threshold={alpha:.6g}*median={thr:.6g} s")
        print(f"n discontinuities: {int(idx.size)}")
        for k, i in enumerate(idx[:30]):
            turn = int(i // Ns) + 1
            within = int(i % Ns)
            print(f"[{k}] i={int(i)} -> i+1={int(i+1)} | dt={float(dt[i]):.6g} s | "
                  f"t[i]={float(t[i]):.6g} s, t[i+1]={float(t[i+1]):.6g} s | "
                  f"turn={turn} (0-based {turn-1}), within_turn={within}")

    def _turn_duration_qc(seg_frame, tol_percent: float) -> None:
        Ns = seg_frame.samples_per_turn
        t0 = seg_frame.df["t"].to_numpy()[::Ns]
        if t0.size < 2:
            print("\n=== TURN DURATION QC (Phase-1 quick tool) ===")
            print("Not enough turns.")
            return
        T = np.diff(t0)
        Tmed = float(np.median(T))
        tol = float(tol_percent) / 100.0
        bad = np.where(np.abs(T / Tmed - 1.0) > tol)[0]  # 0-based turn indices (between starts)
        print("\n=== TURN DURATION QC (Phase-1 quick tool) ===")
        print(f"median(T)={Tmed:.6g} s, tolerance=±{tol_percent:.6g}%")
        print(f"n bad turns: {int(bad.size)}")
        if bad.size:
            # Report in 1-based turn numbers (turn k refers to interval between start[k-1] and start[k])
            bad1 = (bad + 1).astype(int)
            print("bad turns (1-based):", bad1[:200].tolist())

    def _browse(_btn) -> None:
        # Best-effort directory browser for local environments.
        with out:
            out.clear_output()
            print("=== INPUT DEBUG ===")
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                p = filedialog.askdirectory()
                if p:
                    folder.value = p
                print("selected:", folder.value)
            except Exception as e:
                print("Browse failed (tkinter unavailable). Please paste the path manually.")
                print("error:", repr(e))

    def _load_catalog(_btn) -> None:
        with out:
            out.clear_output()
            print("=== INPUT DEBUG ===")
            root = Path(folder.value).expanduser()
            print("selected:", str(root))
            print("exists:", root.exists(), "is_dir:", root.is_dir())

            try:
                disc = MeasurementDiscovery(strict=True)
                cat = disc.build_catalog(root)
                state["catalog"] = cat
                _print_catalog_debug(cat)

                # Populate dropdowns
                dd_run.options = cat.runs
                dd_run.value = cat.runs[0] if cat.runs else None

                seg_ids = [s.segment_id for s in cat.segments]
                # Some measurements define segments in Parameters but files may not exist for all.
                dd_segment.options = seg_ids
                dd_segment.value = seg_ids[0] if seg_ids else None

            except Exception as e:
                print("ERROR:", repr(e))

    def _preview(_btn) -> None:
        with out:
            out.clear_output()
            cat = state.get("catalog", None)
            if cat is None:
                print("No catalog loaded. Click 'Load Catalog' first.")
                return

            run_id = dd_run.value
            seg = dd_segment.value
            mode = dd_mode.value

            print("=== PREVIEW DEBUG ===")
            try:
                fpath = cat.get_segment_file(run_id, seg)
            except Exception as e:
                print("ERROR:", repr(e))
                return
            print("file:", str(fpath))

            # Reader is STRICT: will raise if time is not strictly increasing or dt is inconsistent.
            reader = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=4))
            try:
                seg_frame = reader.read(
                    fpath,
                    run_id=run_id,
                    segment=seg,
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                )
            except Exception as e:
                print("ERROR:", repr(e))
                return

            for msg in seg_frame.warnings:
                print("CHECK:", msg)

            # Show first rows
            display(seg_frame.df.head(10))

            # Plot
            ycol = "df_abs" if mode == "abs" else "df_cmp"
            _plot_first_turns(seg_frame, ycol=ycol, title_prefix=f"{seg} {mode}")

    def _diagnostics(_btn) -> None:
        with out:
            out.clear_output()
            cat = state.get("catalog", None)
            if cat is None:
                print("No catalog loaded. Click 'Load Catalog' first.")
                return

            run_id = dd_run.value
            seg = dd_segment.value
            mode = dd_mode.value

            print("=== DIAGNOSTICS DEBUG ===")
            try:
                fpath = cat.get_segment_file(run_id, seg)
            except Exception as e:
                print("ERROR:", repr(e))
                return
            print("file:", str(fpath))

            reader = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=4))
            try:
                seg_frame = reader.read(
                    fpath,
                    run_id=run_id,
                    segment=seg,
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                )
            except Exception as e:
                print("ERROR:", repr(e))
                return

            for msg in seg_frame.warnings:
                print("CHECK:", msg)

            _time_discontinuity_diag(seg_frame, alpha=float(alpha_jump.value))
            _turn_duration_qc(seg_frame, tol_percent=float(tol_turn.value))

    # Wire callbacks (single time per GUI instance)
    btn_browse.on_click(_browse)
    btn_load.on_click(_load_catalog)
    btn_preview.on_click(_preview)
    btn_diag.on_click(_diagnostics)

    return gui
