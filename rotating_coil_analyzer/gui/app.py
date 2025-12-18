from __future__ import annotations

from pathlib import Path
import traceback

import ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader


_ACTIVE_GUI = None  # used to avoid multiplying widgets on repeated runs


def _reset_click_handlers(btn: w.Button) -> None:
    # Prevent "4x prints" caused by re-attaching callbacks multiple times.
    try:
        btn._click_handlers.callbacks = []
    except Exception:
        pass


def build_catalog_gui(strict: bool = True):
    global _ACTIVE_GUI

    # Close previous instance if the cell is rerun
    if _ACTIVE_GUI is not None:
        try:
            _ACTIVE_GUI.close()
        except Exception:
            pass
        _ACTIVE_GUI = None

    folder = w.Text(
        value="",
        description="Folder",
        layout=w.Layout(width="75%"),
    )

    browse_btn = w.Button(description="Browse…", layout=w.Layout(width="12%"))
    load_btn = w.Button(description="Load Catalog", button_style="primary", layout=w.Layout(width="12%"))

    run_dd = w.Dropdown(options=[], description="Run", layout=w.Layout(width="35%"))
    seg_dd = w.Dropdown(options=[], description="Segment", layout=w.Layout(width="20%"))
    mode_dd = w.Dropdown(options=[("abs", "abs"), ("cmp", "cmp")], value="abs", description="Mode", layout=w.Layout(width="20%"))

    preview_btn = w.Button(description="Preview", layout=w.Layout(width="10%"))
    diag_btn = w.Button(description="Diagnostics", button_style="warning", layout=w.Layout(width="12%"))

    out = w.Output(layout=w.Layout(border="1px solid #ddd", padding="10px"))

    discovery = MeasurementDiscovery(strict=strict)
    reader = Sm18CorrSigsReader(max_currents=4)

    state = {"catalog": None, "run": None}

    def _print_header(title: str):
        print(f"=== {title} ===")

    def _on_browse(_):
        # Minimal browse; in many Jupyter setups, OS file dialogs are blocked.
        # Keep as a no-op if tkinter is unavailable.
        with out:
            out.clear_output(wait=True)
            _print_header("BROWSE")
            print("Browse is environment-dependent. If it does not open, paste the path into the Folder field.")

    def _on_load(_):
        with out:
            out.clear_output(wait=True)
            _print_header("INPUT DEBUG")
            try:
                root = Path(folder.value).expanduser()
                print("selected:", str(root))
                print("exists:", root.exists(), "is_dir:", root.is_dir())

                cat, warnings = discovery.discover(root_dir=root, aperture=1)
                state["catalog"] = cat

                # single-run catalog in this phase
                run_ids = list(cat.runs.keys())
                run_dd.options = run_ids
                run_dd.value = run_ids[0] if run_ids else None

                if run_dd.value:
                    run = cat.runs[run_dd.value]
                    state["run"] = run

                    # Populate segments from files found; fallback to FDIs table if needed
                    segs = sorted(run.corr_sigs_files.keys(), key=lambda s: (len(s), s))
                    if not segs:
                        segs = [s.name for s in run.segments]
                    seg_dd.options = segs
                    seg_dd.value = segs[0] if segs else None

                    _print_header("CATALOG DEBUG")
                    print("root_dir:", str(run.root_dir))
                    print("Parameters.txt:", str(run.parameters_path))
                    print("samples_per_turn:", run.samples_per_turn)
                    print("shaft_speed_rpm:", run.shaft_speed_rpm)
                    print("segments_from_FDIs:", [(s.name, s.fdi_abs, s.fdi_cmp, s.length_m) for s in run.segments])
                    print("corr_sigs_files_found:", len(run.corr_sigs_files))

                if warnings:
                    print("\nWARNINGS:")
                    for m in warnings:
                        print(" -", m)

            except Exception as e:
                print("ERROR:", repr(e))
                print(traceback.format_exc())

    def _get_selected_file() -> Path:
        run = state["run"]
        if run is None:
            raise ValueError("No run selected. Click Load Catalog first.")
        seg = seg_dd.value
        if seg is None:
            raise ValueError("No segment selected.")
        if seg not in run.corr_sigs_files:
            raise ValueError(f"Segment file not found for '{seg}'. Available: {list(run.corr_sigs_files.keys())}")
        return run.corr_sigs_files[seg]

    def _on_preview(_):
        with out:
            out.clear_output(wait=True)
            _print_header("PREVIEW DEBUG")
            try:
                run = state["run"]
                if run is None:
                    raise ValueError("No run selected. Click Load Catalog first.")

                seg = str(seg_dd.value)
                fpath = _get_selected_file()
                print("file:", str(fpath))

                sf = reader.read(
                    file_path=fpath,
                    run_id=run.run_id,
                    segment=seg,
                    samples_per_turn=run.samples_per_turn,
                    shaft_speed_rpm=run.shaft_speed_rpm,
                    aperture=1,
                )

                for msg in sf.warnings:
                    print("CHECK:", msg)

                # Show head
                display(sf.df.head(10))

                # Plot first 3 turns (overlay)
                Ns = sf.samples_per_turn
                n_plot_turns = min(3, sf.n_turns)

                ycol = "df_abs" if mode_dd.value == "abs" else "df_cmp"

                if "t" in sf.df.columns:
                    x = sf.df["t"].to_numpy()
                    xlabel = "t [s] (FDI)"
                else:
                    x = np.arange(sf.n_rows)
                    xlabel = "sample index (no time available)"

                plt.figure()
                for k in range(n_plot_turns):
                    sl = slice(k * Ns, (k + 1) * Ns)
                    plt.plot(x[sl], sf.df[ycol].to_numpy()[sl])
                plt.title(f"{seg} {mode_dd.value} — first {n_plot_turns} turns")
                plt.xlabel(xlabel)
                plt.ylabel("ΔΦ (selected channel)")
                plt.show()

                # Current plot if available
                Icols = [c for c in sf.df.columns if c.startswith("I")]
                if Icols:
                    plt.figure()
                    for k in range(n_plot_turns):
                        sl = slice(k * Ns, (k + 1) * Ns)
                        plt.plot(x[sl], sf.df[Icols[0]].to_numpy()[sl])
                    plt.title(f"{seg} current — first {n_plot_turns} turns")
                    plt.xlabel(xlabel)
                    plt.ylabel(f"{Icols[0]} [A]")
                    plt.show()

            except Exception as e:
                print("ERROR:", repr(e))
                print(traceback.format_exc())

    def _on_diag(_):
        with out:
            out.clear_output(wait=True)
            _print_header("DIAGNOSTICS")
            print("Phase-1 diagnostics are intentionally minimal here; we will expand them in Phase-2.")
            print("Use Preview for format/time validation first.")

    # Attach handlers ONCE
    _reset_click_handlers(browse_btn)
    _reset_click_handlers(load_btn)
    _reset_click_handlers(preview_btn)
    _reset_click_handlers(diag_btn)

    browse_btn.on_click(_on_browse)
    load_btn.on_click(_on_load)
    preview_btn.on_click(_on_preview)
    diag_btn.on_click(_on_diag)

    header = w.HBox([folder, browse_btn, load_btn])
    controls = w.HBox([run_dd, seg_dd, mode_dd, preview_btn, diag_btn])
    gui = w.VBox([header, controls, out])

    _ACTIVE_GUI = gui
    return gui
