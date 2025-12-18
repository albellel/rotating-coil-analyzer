from __future__ import annotations

from pathlib import Path
import warnings as pywarnings

import ipywidgets as w
from IPython.display import display

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader


def _browse_for_folder() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select measurement folder")
        root.destroy()
        return folder or None
    except Exception:
        return None


def _clear_button_handlers(btn: w.Button) -> None:
    # Defensive: prevents stacked callbacks when code is reloaded in the same kernel.
    try:
        btn._click_handlers.callbacks = []  # type: ignore[attr-defined]
    except Exception:
        pass


def build_catalog_gui() -> w.VBox:
    discovery = MeasurementDiscovery()
    reader = Sm18CorrSigsReader()

    folder = w.Text(
        description="Folder",
        placeholder="Paste measurement folder path here (or use Browse...)",
        layout=w.Layout(width="75%"),
    )
    browse_btn = w.Button(description="Browse…", button_style="")
    load_btn = w.Button(description="Load Catalog", button_style="primary")
    close_btn = w.Button(description="Close All Widgets", button_style="danger")

    run_dd = w.Dropdown(description="Run", options=[])
    seg_dd = w.Dropdown(description="Segment", options=[])
    mode_dd = w.Dropdown(description="Mode", options=[("abs", "abs"), ("cmp", "cmp")], value="abs")

    preview_btn = w.Button(description="Preview", button_style="")
    diag_btn = w.Button(description="Diagnostics", button_style="warning")

    out = w.Output(layout=w.Layout(border="1px solid #ccc", padding="8px"))
    state = {"catalog": None}

    def _refresh_dropdowns():
        cat = state["catalog"]
        if cat is None:
            run_dd.options = []
            seg_dd.options = []
            return
        run_dd.options = [(r.run_id, r.run_id) for r in cat.runs]
        seg_dd.options = [(s.segment, s.segment) for s in cat.segments]

    def _on_close(_):
        # hard reset from within notebook
        w.Widget.close_all()

    def _on_browse(_):
        picked = _browse_for_folder()
        with out:
            out.clear_output(wait=True)
            if picked is None:
                print("Browse failed (likely headless environment). Please paste the folder path manually.")
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

                cat = discovery.build_catalog(p)
                state["catalog"] = cat

                print("\n=== CATALOG DEBUG ===")
                print("root_dir:", cat.root_dir)
                print("Parameters.txt:", cat.parameters_path)
                print("samples_per_turn:", cat.samples_per_turn)
                print("shaft_speed_rpm:", cat.shaft_speed_rpm)
                print("enabled_apertures:", cat.enabled_apertures)
                print("segments:", [(s.segment, s.fdi_abs, s.fdi_cmp, s.length_m) for s in cat.segments])
                print("runs_found:", len(cat.runs))
                print("segment_files_found:", len(cat.segment_files))

                if cat.warnings:
                    print("\nWARNINGS:")
                    for wmsg in cat.warnings:
                        print(" -", wmsg)

                _refresh_dropdowns()

            except Exception as e:
                print("ERROR:", repr(e))

    def _on_preview(_):
        with out:
            out.clear_output(wait=True)
            cat = state["catalog"]
            if cat is None:
                print("Load a catalog first.")
                return

            run_id = run_dd.value
            seg = seg_dd.value
            mode = mode_dd.value

            key = (run_id, seg)
            if key not in cat.segment_files:
                print(f"No segment file for run={run_id}, segment={seg}.")
                return

            fpath = cat.segment_files[key]
            print("=== PREVIEW DEBUG ===")
            print("file:", fpath)

            try:
                # Reader is strict; it will raise if no supported format validates.
                seg_frame = reader.read(
                    fpath,
                    run_id=run_id,
                    segment=seg,
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                )

                for msg in seg_frame.warnings:
                    print("CHECK:", msg)

                df = seg_frame.df
                if mode == "abs":
                    ch = df[["t", "df_abs", "I"]].rename(columns={"df_abs": "df"})
                else:
                    ch = df[["t", "df_cmp", "I"]].rename(columns={"df_cmp": "df"})

                print("rows:", len(ch), "n_turns:", seg_frame.n_turns)
                print("head:")
                display(ch.head(10))

            except Exception as e:
                print("ERROR:", str(e))

    def _on_diag(_):
        # Keep it minimal at Phase-1: show only what the reader already computed.
        with out:
            out.clear_output(wait=True)
            cat = state["catalog"]
            if cat is None:
                print("Load a catalog first.")
                return
            run_id = run_dd.value
            seg = seg_dd.value
            key = (run_id, seg)
            if key not in cat.segment_files:
                print(f"No segment file for run={run_id}, segment={seg}.")
                return

            fpath = cat.segment_files[key]
            print("=== DIAGNOSTICS (Phase-1) ===")
            print("file:", fpath)

            try:
                seg_frame = reader.read(
                    fpath,
                    run_id=run_id,
                    segment=seg,
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                )
                for msg in seg_frame.warnings:
                    print("CHECK:", msg)
            except Exception as e:
                print("ERROR:", str(e))

    # Prevent stacked callbacks on reload
    for b in (browse_btn, load_btn, close_btn, preview_btn, diag_btn):
        _clear_button_handlers(b)

    browse_btn.on_click(_on_browse)
    load_btn.on_click(_on_load)
    close_btn.on_click(_on_close)
    preview_btn.on_click(_on_preview)
    diag_btn.on_click(_on_diag)

    folder_row = w.HBox([folder, browse_btn, load_btn, close_btn])
    selector_row = w.HBox([run_dd, seg_dd, mode_dd, preview_btn, diag_btn])
    return w.VBox([folder_row, selector_row, out])
