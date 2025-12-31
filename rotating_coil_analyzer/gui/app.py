from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig
from rotating_coil_analyzer.ingest.readers_mba import MbaRawMeasurementReader, MbaReaderConfig


# Keep a single active GUI instance per kernel to avoid duplicated callbacks / stacked widgets.
_ACTIVE_GUI: Optional[w.Widget] = None


def _close_all_figures() -> None:
    try:
        plt.close("all")
    except Exception:
        pass


def build_catalog_gui() -> w.Widget:
    """
    Phase-1 catalog/preview GUI (Jupyter / VSCode notebooks).

    Notes on "widget multiplication":
      - This function closes the previous GUI instance created from this module.
      - If the notebook cell output itself is duplicated, use the notebook UI "Clear Outputs"
        for that cell, then re-run.
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
        placeholder=".../aperture1 (or any folder under the run; Parameters.txt may be above)",
        layout=w.Layout(width="80%"),
    )
    btn_browse = w.Button(description="Browse…", layout=w.Layout(width="120px"))
    btn_load = w.Button(description="Load Catalog", button_style="primary", layout=w.Layout(width="140px"))

    dd_run = w.Dropdown(options=[], description="Run", layout=w.Layout(width="360px"))
    dd_ap = w.Dropdown(options=[], description="Aperture", layout=w.Layout(width="200px"))
    dd_segment = w.Dropdown(options=[], description="Segment", layout=w.Layout(width="220px"))

    mode_dd = w.Dropdown(description="Mode", options=[("abs", "abs"), ("cmp", "cmp")], value="abs")
    btn_preview = w.Button(description="Preview", button_style="")
    btn_diag = w.Button(description="Diagnostics", button_style="warning")

    state = {"catalog": None}

    def _browse_for_folder() -> Optional[str]:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            picked = filedialog.askdirectory(title="Select measurement folder")
            root.destroy()
            return picked or None
        except Exception:
            return None

    def _refresh_dropdowns():
        cat = state["catalog"]
        if cat is None:
            dd_run.options = []
            dd_ap.options = []
            dd_segment.options = []
            return

        dd_run.options = [(r, r) for r in cat.runs]
        if cat.runs:
            dd_run.value = cat.runs[0]

        dd_ap.options = [(("AP" + str(ap)) if ap is not None else "AP (single)", ap) for ap in cat.logical_apertures]
        dd_ap.value = cat.logical_apertures[0]

        segs = cat.segments_for_aperture(dd_ap.value)
        dd_segment.options = [(s.segment_id, s.segment_id) for s in segs]
        if segs:
            dd_segment.value = segs[0].segment_id

    def _on_ap_change(_change):
        cat = state["catalog"]
        if cat is None:
            return
        segs = cat.segments_for_aperture(dd_ap.value)
        dd_segment.options = [(s.segment_id, s.segment_id) for s in segs]
        if segs:
            dd_segment.value = segs[0].segment_id

    dd_ap.observe(_on_ap_change, names="value")

    def _print_catalog(cat):
        print("=== CATALOG DEBUG ===")
        print("selected root_dir:", str(cat.root_dir))
        print("Parameters.txt:", str(cat.parameters_path))
        print("parameters_root:", str(cat.parameters_root))
        print("samples_per_turn:", cat.samples_per_turn)
        print("shaft_speed_rpm:", cat.shaft_speed_rpm)
        print("enabled_apertures:", cat.enabled_apertures)
        print("segments:", [(s.aperture_id, s.segment_id, s.fdi_abs, s.fdi_cmp, s.length_m) for s in cat.segments])
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

        _close_all_figures()

        plt.figure()
        plt.plot(t, y)
        plt.title(f"{title_prefix} — first {n_show_turns} turns")
        plt.xlabel("t [s]")
        plt.ylabel(ycol)
        plt.show()

        if "I" in df.columns:
            plt.figure()
            plt.plot(t, df["I"].to_numpy())
            plt.title(f"{title_prefix} current — first {n_show_turns} turns")
            plt.xlabel("t [s]")
            plt.ylabel("I")
            plt.show()

    def _plot_all_columns_vs_time(seg_frame) -> None:
        df = seg_frame.df
        t = df["t"].to_numpy()
        n = len(df)
        stride = max(1, n // 250_000)
        _close_all_figures()
        for col in df.columns:
            if col == "t":
                continue
            plt.figure()
            plt.plot(t[::stride], df[col].to_numpy()[::stride])
            plt.title(f"{seg_frame.segment}: {col} vs t (stride={stride})")
            plt.xlabel("t [s]")
            plt.ylabel(col)
            plt.show()

    def _on_browse(_):
        picked = _browse_for_folder()
        with out:
            out.clear_output(wait=True)
            if picked is None:
                print("Browse failed (headless environment). Please paste the folder path manually.")
                return
            folder.value = picked
            print(f"Selected folder: {picked}")

    def _on_load(_):
        with out:
            out.clear_output(wait=True)
            try:
                p = Path(folder.value).expanduser()
                print("=== INPUT DEBUG ===")
                print("selected:", p)
                print("exists:", p.exists(), "is_dir:", p.is_dir())

                cat = MeasurementDiscovery(strict=True).build_catalog(p)
                state["catalog"] = cat

                _print_catalog(cat)
                _refresh_dropdowns()

            except Exception as e:
                print("ERROR:", repr(e))

    def _read_selected_segment():
        cat = state["catalog"]
        if cat is None:
            raise RuntimeError("Load a catalog first.")

        run_id = dd_run.value
        ap_ui = dd_ap.value
        seg_id = dd_segment.value

        fpath = cat.get_segment_file(run_id, ap_ui, seg_id)
        ap_phys = cat.resolve_aperture(ap_ui)

        # Choose reader by filename family
        if fpath.name.lower().endswith("_raw_measurement_data.txt"):
            cfg = MbaReaderConfig()
            reader = MbaRawMeasurementReader(cfg)
            seg_frame = reader.read(
                fpath,
                run_id=run_id,
                segment=str(seg_id),
                samples_per_turn=cat.samples_per_turn,
                aperture_id=ap_phys,
            )
        else:
            cfg = Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=5)
            reader = Sm18CorrSigsReader(cfg)
            seg_frame = reader.read(
                fpath,
                run_id=run_id,
                segment=str(seg_id),
                samples_per_turn=cat.samples_per_turn,
                shaft_speed_rpm=cat.shaft_speed_rpm,
                aperture_id=ap_phys,
            )
        return cat, fpath, seg_frame

    def _on_preview(_):
        with out:
            out.clear_output(wait=True)
            try:
                cat, fpath, seg_frame = _read_selected_segment()
                print("=== PREVIEW DEBUG ===")
                print("file:", fpath)
                for msg in seg_frame.warnings:
                    print("CHECK:", msg)
                display(seg_frame.df.head(12))

                ycol = "df_abs" if mode_dd.value == "abs" else "df_cmp"
                title = f"AP{seg_frame.aperture_id} Seg{seg_frame.segment} {mode_dd.value}"
                _plot_first_turns(seg_frame, ycol=ycol, title_prefix=title)

            except Exception as e:
                print("ERROR:", repr(e))

    def _on_diag(_):
        with out:
            out.clear_output(wait=True)
            try:
                cat, fpath, seg_frame = _read_selected_segment()
                print("=== DIAGNOSTICS DEBUG ===")
                print("file:", fpath)
                for msg in seg_frame.warnings:
                    print("CHECK:", msg)

                print("\n=== PLOT ALL COLUMNS VS TIME (decimated) ===")
                _plot_all_columns_vs_time(seg_frame)

            except Exception as e:
                print("ERROR:", repr(e))

    btn_browse.on_click(_on_browse)
    btn_load.on_click(_on_load)
    btn_preview.on_click(_on_preview)
    btn_diag.on_click(_on_diag)

    top = w.HBox([folder, btn_browse, btn_load])
    mid = w.HBox([dd_run, dd_ap, dd_segment, mode_dd, btn_preview, btn_diag])
    gui = w.VBox([top, mid, out])

    _ACTIVE_GUI = gui
    return gui
