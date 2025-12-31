from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import ipywidgets as w
import matplotlib.pyplot as plt

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig
from rotating_coil_analyzer.ingest.readers_mba import MbaRawMeasurementReader, MbaReaderConfig

from rotating_coil_analyzer.gui.phase2 import build_phase2_panel


@dataclass
class GuiState:
    root: Optional[Path] = None
    catalog: Optional[object] = None
    # Current selection
    run_id: Optional[str] = None
    ap_ui: Optional[object] = None
    seg_id: Optional[object] = None
    file_path: Optional[Path] = None
    # Loaded data
    segf: Optional[object] = None


_ACTIVE_GUI: Optional[w.Widget] = None
_ACTIVE_STATE: Optional[GuiState] = None


def _browse_directory_dialog(initial: Optional[str] = None) -> Optional[str]:
    """
    Open a native OS directory chooser (Windows Explorer on Windows) and return the selected path.

    Uses tkinter (standard library on typical Windows Python installs).
    Returns None if tkinter is unavailable/headless or if the user cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        path = filedialog.askdirectory(
            initialdir=initial if initial else None,
            title="Select dataset root folder (contains Parameters.txt)",
            mustexist=True,
        )
        if not path:
            return None
        return str(path)
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


def build_catalog_gui() -> w.Widget:
    """
    Phase I GUI: catalog + preview/diagnostics + segment loading.
    Use build_gui() to get Phase II analysis in a second tab.
    """
    global _ACTIVE_GUI, _ACTIVE_STATE

    # Reuse existing instance if present (prevents widget multiplication when re-importing)
    if _ACTIVE_GUI is not None and _ACTIVE_STATE is not None:
        return _ACTIVE_GUI

    state = GuiState()
    _ACTIVE_STATE = state

    folder = w.Text(
        value="",
        description="Folder",
        placeholder=r"C:\path\to\dataset_root",
        layout=w.Layout(width="520px"),
    )
    btn_browse = w.Button(description="Browse…")
    btn_load_catalog = w.Button(description="Load catalog", button_style="primary")

    dd_run = w.Dropdown(options=[], description="Run", layout=w.Layout(width="260px"))
    dd_ap = w.Dropdown(options=[], description="Aperture", layout=w.Layout(width="220px"))
    dd_segment = w.Dropdown(options=[], description="Segment", layout=w.Layout(width="220px"))

    mode_dd = w.Dropdown(
        options=[("auto", "auto"), ("MBA", "mba"), ("SM18", "sm18")],
        value="auto",
        description="Mode",
        layout=w.Layout(width="220px"),
    )

    btn_load_segment = w.Button(description="Load segment")
    btn_preview = w.Button(description="Preview")
    btn_diag = w.Button(description="Diagnostics")

    append_output = w.Checkbox(
        value=False,
        description="Append output",
        indent=False,
        layout=w.Layout(width="140px"),
    )

    out = w.Output()

    def _start_action():
        """
        Apply the output policy:
        - If append_output is False: clear output + close figures
        - If append_output is True: keep previous output and figures
        """
        if not append_output.value:
            out.clear_output()
            plt.close("all")

    def _set_segments_for_current_ap():
        """
        Populate Segment dropdown based on the currently selected logical aperture.
        IMPORTANT: dd_ap.value may legitimately be None (single-aperture datasets).
        """
        cat = state.catalog
        if cat is None:
            dd_segment.options = []
            dd_segment.value = None
            return

        if not dd_ap.options:
            dd_segment.options = []
            dd_segment.value = None
            return

        ap_ui = dd_ap.value  # can be None and still valid
        segs = cat.segments_for_aperture(ap_ui)
        seg_ids = [s.segment_id for s in segs]

        dd_segment.options = seg_ids
        dd_segment.value = seg_ids[0] if seg_ids else None

    def _on_browse(_):
        _start_action()
        with out:
            initial = folder.value.strip() or None
            chosen = _browse_directory_dialog(initial=initial)
            if chosen is None:
                print("Folder browse not available (tkinter missing/headless) or cancelled.")
                print("You can still paste the folder path into the Folder field.")
                return
            folder.value = chosen
            print("Selected folder:", chosen)

    def _on_load_catalog(_):
        _start_action()
        with out:
            p = Path(folder.value).expanduser()
            if not p.exists():
                print("Folder does not exist:", p)
                return

            state.root = p
            cat = MeasurementDiscovery(strict=True).build_catalog(p)
            state.catalog = cat

            dd_run.options = list(cat.runs)
            dd_run.value = cat.runs[0] if cat.runs else None

            dd_ap.options = list(cat.logical_apertures)
            dd_ap.value = cat.logical_apertures[0] if cat.logical_apertures else None

            _set_segments_for_current_ap()

            print("Catalog loaded.")
            print("runs:", cat.runs)
            print("aps:", cat.logical_apertures)
            print("segments:", [(s.aperture_id, s.segment_id) for s in cat.segments])
            print("segment dropdown options now:", list(dd_segment.options))

    def _on_selection_change(_=None):
        # Keep silent; do not spam output when the user only changes dropdowns.
        if state.catalog is None:
            return
        if dd_run.value is None:
            return
        _set_segments_for_current_ap()

    dd_run.observe(lambda c: _on_selection_change(), names="value")
    dd_ap.observe(lambda c: _on_selection_change(), names="value")

    def _on_load_segment(_):
        _start_action()
        with out:
            cat = state.catalog
            if cat is None:
                print("Load catalog first.")
                return
            if dd_run.value is None or dd_segment.value is None:
                print("Select run and segment.")
                return

            state.run_id = dd_run.value
            state.ap_ui = dd_ap.value  # can be None and still valid
            state.seg_id = dd_segment.value

            f = cat.get_segment_file(state.run_id, state.ap_ui, state.seg_id)
            state.file_path = f
            ap_phys = cat.resolve_aperture(state.ap_ui)

            if mode_dd.value == "mba" or (mode_dd.value == "auto" and f.name.lower().endswith("_raw_measurement_data.txt")):
                segf = MbaRawMeasurementReader(MbaReaderConfig()).read(
                    f,
                    run_id=state.run_id,
                    segment=str(state.seg_id),
                    samples_per_turn=cat.samples_per_turn,
                    aperture_id=ap_phys,
                )
            else:
                segf = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True)).read(
                    f,
                    run_id=state.run_id,
                    segment=str(state.seg_id),
                    samples_per_turn=cat.samples_per_turn,
                    shaft_speed_rpm=cat.shaft_speed_rpm,
                    aperture_id=ap_phys,
                )

            state.segf = segf

            print("Loaded:", f)
            print("n_turns:", segf.n_turns, "Ns:", segf.samples_per_turn, "n_samples:", len(segf.df))
            if getattr(segf, "warnings", None):
                print("warnings:")
                for wmsg in segf.warnings:
                    print(" -", wmsg)

    def _on_preview(_):
        _start_action()
        with out:
            if state.segf is None:
                print("Load a segment first.")
                return

            df = state.segf.df
            Ns = int(state.segf.samples_per_turn)
            n = min(len(df), 10 * Ns)

            x = np.arange(n)
            plt.figure()
            plt.plot(x, df["df_cmp"].to_numpy()[:n], label="df_cmp")
            plt.plot(x, df["df_abs"].to_numpy()[:n], label="df_abs")
            plt.xlabel("sample index k (not time)")
            plt.ylabel("flux (arb.)")
            plt.title("Preview (first 10 turns)")
            plt.legend()
            plt.show()

    def _on_diag(_):
        _start_action()
        with out:
            if state.segf is None:
                print("Load a segment first.")
                return

            df = state.segf.df
            t = df["t"].to_numpy()
            dt = np.diff(t)
            finite = np.isfinite(dt)

            print("dt stats (finite only):")
            if np.any(finite):
                print("  min   :", float(np.min(dt[finite])))
                print("  median:", float(np.median(dt[finite])))
                print("  max   :", float(np.max(dt[finite])))
                print("  non-finite dt:", int(np.sum(~finite)))
            else:
                print("  all dt are non-finite (time contains NaNs/Infs).")

    btn_browse.on_click(_on_browse)
    btn_load_catalog.on_click(_on_load_catalog)
    btn_load_segment.on_click(_on_load_segment)
    btn_preview.on_click(_on_preview)
    btn_diag.on_click(_on_diag)

    top = w.HBox([folder, btn_browse, btn_load_catalog, append_output])
    mid = w.HBox([dd_run, dd_ap, dd_segment, mode_dd, btn_load_segment, btn_preview, btn_diag])
    gui = w.VBox([top, mid, out])

    _ACTIVE_GUI = gui
    return gui


def build_gui() -> w.Widget:
    """Combined GUI with Phase I and Phase II in tabs."""
    global _ACTIVE_STATE

    phase1 = build_catalog_gui()

    def _get_segf():
        return _ACTIVE_STATE.segf if _ACTIVE_STATE is not None else None

    phase2 = build_phase2_panel(_get_segf)

    tabs = w.Tab(children=[phase1, phase2])
    tabs.set_title(0, "Phase I — Catalog")
    tabs.set_title(1, "Phase II — FFT")
    return tabs
