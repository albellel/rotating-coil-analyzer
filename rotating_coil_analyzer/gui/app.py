from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
import matplotlib.pyplot as plt

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig
from rotating_coil_analyzer.ingest.readers_mba import MbaRawMeasurementReader, MbaReaderConfig

# IMPORTANT: your repo defines MeasurementCatalog (not Catalog)
from rotating_coil_analyzer.models.catalog import MeasurementCatalog
from rotating_coil_analyzer.models.frames import SegmentFrame

from rotating_coil_analyzer.gui.phase2 import build_phase2_panel
from rotating_coil_analyzer.gui.phase3_kn import build_phase3_kn_panel
from rotating_coil_analyzer.gui.phase4_plots import build_phase4_plots_panel


# Keep a single active GUI instance per kernel (prevents multiple live instances).
_ACTIVE_GUI: Optional[w.Widget] = None


def _close_all_figures() -> None:
    try:
        plt.close("all")
    except Exception:
        pass


def _df_head_to_html(df: pd.DataFrame, n: int = 12, title: str = "") -> str:
    import html as _html
    if df is None:
        return "<div style='color:#b00;'>No data.</div>"
    head = df.head(n)
    s = head.to_string(max_rows=n, max_cols=None)
    s = _html.escape(s)
    ttl = f"<b>{_html.escape(title)}</b><br/>" if title else ""
    return (
        "<div style='font-family:monospace;'>"
        f"{ttl}"
        f"<pre style='white-space:pre; overflow:auto; max-height:280px; border:1px solid #ddd; padding:8px;'>{s}</pre>"
        "</div>"
    )


def _browse_for_folder() -> Optional[str]:
    """Native folder chooser (tkinter). Returns None if unavailable/cancelled."""
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
        p = filedialog.askdirectory(title="Select measurement folder")
        return str(p) if p else None
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


@dataclass
class Phase1State:
    cat: Optional[MeasurementCatalog] = None
    segf: Optional[SegmentFrame] = None
    seg_path: Optional[Path] = None
    run_id: Optional[str] = None
    ap_ui: Optional[int] = None
    seg_id: Optional[str] = None

    fig: Optional[Any] = None
    ax: Optional[Any] = None


def _init_plot_once(state: Phase1State, plot_slot: w.Box) -> None:
    if state.fig is not None and state.ax is not None:
        return

    was_interactive = plt.isinteractive()
    try:
        plt.ioff()
        fig, ax = plt.subplots()
    finally:
        if was_interactive:
            plt.ion()

    ax.set_title("Phase I plot")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("signal")
    state.fig, state.ax = fig, ax

    if isinstance(fig.canvas, w.Widget):
        plot_slot.children = (fig.canvas,)
    else:
        plot_slot.children = (w.HTML("Non-interactive backend. In the first cell run: %matplotlib widget (ipympl)."),)


def _redraw(fig) -> None:
    if fig is None:
        return
    if hasattr(fig.canvas, "draw_idle"):
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()


def _build_phase1_panel(shared: Dict[str, Any]) -> w.Widget:
    """
    Phase I panel (catalog discovery + segment load + preview/diagnostics).
    Stores the loaded SegmentFrame into shared["segment_frame"] for Phase II.
    """
    st = Phase1State()

    folder = w.Text(
        description="Folder",
        placeholder="Select any folder inside the run (Parameters.txt may be above)",
        layout=w.Layout(width="78%"),
    )
    btn_browse = w.Button(description="Browse…", layout=w.Layout(width="120px"))
    btn_load_cat = w.Button(description="Load catalog", button_style="primary", layout=w.Layout(width="140px"))

    dd_run = w.Dropdown(options=[], description="Run", layout=w.Layout(width="360px"))
    dd_ap = w.Dropdown(options=[], description="Aperture", layout=w.Layout(width="220px"))
    dd_seg = w.Dropdown(options=[], description="Segment", layout=w.Layout(width="220px"))

    mode_dd = w.Dropdown(
        description="Channel",
        # Keep internal tokens ("abs"/"cmp") for compatibility with the rest of the pipeline.
        options=[("Absolute", "abs"), ("Compensated", "cmp")],
        value="abs",
        layout=w.Layout(width="220px"),
    )

    btn_load_seg = w.Button(description="Load segment", button_style="success", layout=w.Layout(width="140px"))
    btn_preview = w.Button(description="Preview", layout=w.Layout(width="110px"))
    btn_diag = w.Button(description="Diagnostics", button_style="warning", layout=w.Layout(width="120px"))

    append_log = w.Checkbox(value=False, description="Append output", indent=False, layout=w.Layout(width="140px"))
    status = w.HTML(value="<b>Status:</b> idle")

    log = HtmlLog(height_px=220)

    table_html = w.HTML(value="<div style='color:#666;'>No segment loaded.</div>")
    plot_slot = w.Box(layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%"))

    _init_plot_once(st, plot_slot)

    def _log(msg: str) -> None:
        s = "" if msg is None else str(msg)
        if s.startswith("ERROR:"):
            log.error(s)
        elif s.startswith("WARN") or s.startswith("WARNING") or s.startswith("CHECK:") or s.startswith("Warning"):
            log.warning(s)
        else:
            log.info(s)

    def _refresh_enabled() -> None:
        """Enable/disable widgets based on current state (catalog/segment loaded)."""
        cat_loaded = st.cat is not None
        seg_loaded = st.segf is not None

        btn_browse.disabled = False
        btn_load_cat.disabled = False

        dd_run.disabled = not cat_loaded
        dd_ap.disabled = not cat_loaded
        dd_seg.disabled = not cat_loaded
        btn_load_seg.disabled = not cat_loaded

        btn_preview.disabled = not seg_loaded
        btn_diag.disabled = not seg_loaded

    def _set_busy(busy: bool) -> None:
        """Disable all interactive controls while a long action is running."""
        if busy:
            for wdg in [
                btn_browse,
                btn_load_cat,
                dd_run,
                dd_ap,
                dd_seg,
                btn_load_seg,
                btn_preview,
                btn_diag,
            ]:
                wdg.disabled = True
        else:
            _refresh_enabled()

    def _start(msg: str) -> None:
        _set_busy(True)
        status.value = f"<b>Status:</b> {msg}"
        if not append_log.value:
            log.clear()
        _log(msg)

    def _done(msg: str = "idle") -> None:
        status.value = f"<b>Status:</b> {msg}"
        _set_busy(False)

    def _on_browse(_):
        p = _browse_for_folder()
        if p:
            folder.value = p

    def _update_segments():
        cat = st.cat
        if cat is None:
            dd_seg.options = []
            dd_seg.value = None
            return
        segs = cat.segments_for_aperture(dd_ap.value)
        opts = [(str(s.segment_id), str(s.segment_id)) for s in segs]
        dd_seg.options = opts
        dd_seg.value = opts[0][1] if opts else None

    def _on_load_catalog(_):
        try:
            _start("Loading catalog…")
            root = Path(folder.value).expanduser().resolve()
            cat = MeasurementDiscovery(strict=True).build_catalog(root)
            st.cat = cat
            shared["catalog"] = cat

            dd_run.options = [(r, r) for r in cat.runs]
            dd_run.value = cat.runs[0] if cat.runs else None

            # If there is only one aperture, do not show "None" in the GUI.
            ap_opts = []
            for a in cat.logical_apertures:
                if a is None:
                    ap_opts.append(("Single aperture measurement", None))
                else:
                    ap_opts.append((f"Aperture {a}", a))
            dd_ap.options = ap_opts
            dd_ap.value = ap_opts[0][1] if ap_opts else None

            _update_segments()

            st.segf = None
            st.seg_path = None
            shared["segment_frame"] = None
            shared["segment_path"] = None

            table_html.value = "<div style='color:#666;'>Catalog loaded. Select run/aperture/segment, then “Load segment”.</div>"

            if cat.warnings:
                _log("WARNINGS:")
                for m in cat.warnings:
                    _log(f"WARNING: {m}")

            _done("catalog ready")
        except Exception as e:
            _log(f"ERROR: {repr(e)}")
            _done("error")

    def _on_ap_change(_change):
        _update_segments()

    dd_ap.observe(_on_ap_change, names="value")

    def _read_selected_segment() -> SegmentFrame:
        cat = st.cat
        if cat is None:
            raise RuntimeError("No catalog loaded.")
        run_id = dd_run.value
        ap_ui = dd_ap.value
        seg_id = dd_seg.value
        if run_id is None or seg_id is None:
            raise RuntimeError("Select run and segment first.")
        fpath = cat.get_segment_file(run_id, ap_ui, seg_id)
        ap_phys = cat.resolve_aperture(ap_ui)

        if fpath.name.lower().endswith("_raw_measurement_data.txt"):
            reader = MbaRawMeasurementReader(MbaReaderConfig())
            segf = reader.read(
                fpath,
                run_id=run_id,
                segment=str(seg_id),
                samples_per_turn=cat.samples_per_turn,
                aperture_id=ap_phys,
                magnet_order=cat.magnet_order,
            )
        else:
            reader = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=5))
            segf = reader.read(
                fpath,
                run_id=run_id,
                segment=str(seg_id),
                samples_per_turn=cat.samples_per_turn,
                shaft_speed_rpm=cat.shaft_speed_rpm,
                aperture_id=ap_phys,
                magnet_order=cat.magnet_order,
            )

        st.seg_path = fpath
        st.run_id = run_id
        st.ap_ui = ap_ui
        st.seg_id = str(seg_id)
        st.segf = segf

        shared["segment_frame"] = segf
        shared["segment_path"] = fpath
        return segf

    def _plot_first_turns(segf: SegmentFrame, ycol: str, title: str):
        Ns = int(segf.samples_per_turn)
        n_show_turns = min(3, int(segf.n_turns))
        n = n_show_turns * Ns

        t = segf.df["t"].to_numpy()[:n]
        y = segf.df[ycol].to_numpy()[:n]

        ax = st.ax
        ax.clear()
        ax.plot(t, y, "-", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ycol)
        _redraw(st.fig)

    def _on_load_segment(_):
        try:
            _start("Loading segment (this can take a few seconds)…")
            segf = _read_selected_segment()
            _log(f"Loaded: {st.seg_path}")
            _log(f"n_turns={segf.n_turns}  Ns={segf.samples_per_turn}  n_samples={len(segf.df)}")
            for m in segf.warnings:
                _log(f"CHECK: {m}")
            table_html.value = _df_head_to_html(segf.df, n=12, title="SegmentFrame.df head(12)")
            _done("segment loaded (Phase II ready)")
        except Exception as e:
            _log(f"ERROR: {repr(e)}")
            _done("error")

    def _on_preview(_):
        try:
            _start("Preview…")
            if st.segf is None:
                _log("WARNING: No segment loaded yet. Click “Load segment” first.")
                _done("no segment")
                return
            segf = st.segf
            ycol = "df_abs" if mode_dd.value == "abs" else "df_cmp"
            mode_label = "absolute" if mode_dd.value == "abs" else "compensated"
            title = f"Preview: run={st.run_id}  ap={segf.aperture_id}  seg={st.seg_id}  channel={mode_label}"
            table_html.value = _df_head_to_html(segf.df, n=12, title="SegmentFrame.df head(12)")
            _plot_first_turns(segf, ycol=ycol, title=title)
            _done("preview ready")
        except Exception as e:
            _log(f"ERROR: {repr(e)}")
            _done("error")

    def _on_diag(_):
        try:
            _start("Diagnostics…")
            if st.segf is None:
                _log("WARNING: No segment loaded yet. Click “Load segment” first.")
                _done("no segment")
                return
            segf = st.segf
            df = segf.df

            t = df["t"].to_numpy()
            dt = np.diff(t)
            dt_f = dt[np.isfinite(dt)]

            _log("=== DIAGNOSTICS ===")
            _log(f"file: {st.seg_path}")
            _log(f"rows: {len(df)}   Ns: {segf.samples_per_turn}   n_turns: {segf.n_turns}")
            _log(f"t finite: {int(np.sum(np.isfinite(t)))} / {len(t)}")
            _log(f"dt finite: {int(np.sum(np.isfinite(dt)))} / {len(dt)}")
            if dt_f.size:
                _log(f"dt min/median/max: {dt_f.min():.6g} / {np.median(dt_f):.6g} / {dt_f.max():.6g}")
                _log(f"n dt < 0: {int(np.sum(dt_f < 0))}")
            else:
                _log("dt stats: no finite dt values")

            for m in segf.warnings:
                _log(f"CHECK: {m}")

            for col in ["df_abs", "df_cmp", "I"]:
                if col in df.columns:
                    _log(f"{col} finite: {int(np.sum(np.isfinite(df[col].to_numpy())))} / {len(df)}")

            _done("diagnostics ready")
        except Exception as e:
            _log(f"ERROR: {repr(e)}")
            _done("error")

    def _clear_button_handlers(btn: w.Button) -> None:
        try:
            btn._click_handlers.callbacks.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    for b in [btn_browse, btn_load_cat, btn_load_seg, btn_preview, btn_diag]:
        _clear_button_handlers(b)

    btn_browse.on_click(_on_browse)
    btn_load_cat.on_click(_on_load_catalog)
    btn_load_seg.on_click(_on_load_segment)
    btn_preview.on_click(_on_preview)
    btn_diag.on_click(_on_diag)

    top = w.HBox([folder, btn_browse, btn_load_cat])
    mid = w.HBox([dd_run, dd_ap, dd_seg, mode_dd, btn_load_seg, btn_preview, btn_diag, append_log])

    plot_box = w.VBox([w.HTML("<b>Plot</b>"), plot_slot], layout=w.Layout(width="100%"))
    table_box = w.VBox([w.HTML("<b>Table preview</b>"), table_html], layout=w.Layout(width="100%"))
    diag_box = w.VBox(
        [
            status,
            w.HTML("<b>Log</b>"),
            log.widget,
        ],
        layout=w.Layout(width="34%", min_width="360px"),
    )

    main_box = w.VBox([plot_box, table_box], layout=w.Layout(width="66%", min_width="620px"))

    _refresh_enabled()

    # Two-column layout: results on the left, diagnostics on the right.
    # This avoids vertical scrolling to reach the log/status.
    return w.VBox([top, mid, w.HBox([main_box, diag_box], layout=w.Layout(width="100%"))])


def build_gui(*, clear_cell_output: bool = True) -> w.Widget:
    """
    Combined Phase I + Phase II + Phase III + Plots GUI (four tabs).

    VS Code notebook rule:
      If you re-run the launch cell without clearing the cell output, you may end up with
      multiple renderings. One click then updates all renderings -> repeated outputs.
    """
    global _ACTIVE_GUI

    # Clear the launch-cell output so the GUI doesn't stack visually.
    if clear_cell_output:
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except Exception:
            pass

    # Close previous GUI instance from this module.
    if _ACTIVE_GUI is not None:
        try:
            _ACTIVE_GUI.close()
        except Exception:
            pass
        _ACTIVE_GUI = None

    _close_all_figures()

    shared: Dict[str, Any] = {
        "catalog": None,
        "segment_frame": None,
        "segment_path": None,
    }

    phase1 = _build_phase1_panel(shared)
    phase2 = build_phase2_panel(lambda: shared.get("segment_frame"))
    phase3 = build_phase3_kn_panel(lambda: shared.get("segment_frame"), lambda: shared.get("segment_path"))
    phase4 = build_phase4_plots_panel(lambda: shared.get("segment_frame"), lambda: shared.get("segment_path"))

    tabs = w.Tab(children=[phase1, phase2, phase3, phase4])
    tabs.set_title(0, "Phase I — Catalog")
    tabs.set_title(1, "Phase II — FFT")
    tabs.set_title(2, "Phase III — Kn")
    tabs.set_title(3, "Plots")

    _ACTIVE_GUI = tabs
    return tabs


def build_catalog_gui() -> w.Widget:
    """Return only the Phase I panel (catalog/preview/diagnostics)."""
    shared: Dict[str, Any] = {"catalog": None, "segment_frame": None, "segment_path": None}
    return _build_phase1_panel(shared)


if __name__ == "__main__":
    # Running this file in a terminal will not show the ipywidgets GUI.
    # This message prevents confusion when executing `python .../gui/app.py`.
    print("This module provides an ipywidgets GUI for Jupyter/VS Code notebooks.")
    print("Use it in a notebook cell:")
    print("  %matplotlib widget")
    print("  from rotating_coil_analyzer.gui.app import build_gui")
    print("  build_gui()")
