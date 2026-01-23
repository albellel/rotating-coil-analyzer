"""Phase IV: read-only exploration plots.

Goals
-----
* Provide quick, safe visualization of the currently loaded segment.
* Read-only: must not mutate analysis outputs.
* Downsampling is decimation only (no interpolation): keep every K-th sample.
* Respect the project's "no synthetic time" rule: always plot against measured time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import ipywidgets as w
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.gui.log_view import HtmlLog


@dataclass
class Phase4State:
    fig: Optional[plt.Figure] = None
    last_source: Optional[str] = None


def _decimate(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return x[::k]


def _basic_time_stats(t: np.ndarray) -> dict:
    if t.size < 2:
        return {"n": int(t.size)}
    dt = np.diff(t)
    finite = np.isfinite(dt)
    dtf = dt[finite] if np.any(finite) else dt
    nonmono = int(np.sum(dtf <= 0)) if dtf.size else 0
    return {
        "n": int(t.size),
        "t_min": float(np.nanmin(t)),
        "t_max": float(np.nanmax(t)),
        "dt_min": float(np.nanmin(dtf)) if dtf.size else float("nan"),
        "dt_med": float(np.nanmedian(dtf)) if dtf.size else float("nan"),
        "dt_max": float(np.nanmax(dtf)) if dtf.size else float("nan"),
        "nonmono": nonmono,
        "nonmono_pct": 100.0 * nonmono / max(1, int(dtf.size)),
    }


def build_phase4_plots_panel(
    get_segmentframe_callable: Callable[[], Optional[SegmentFrame]],
    get_segmentpath_callable: Optional[Callable[[], Optional[str]]] = None,
) -> w.Widget:
    """Build Phase IV panel.

    Parameters
    ----------
    get_segmentframe_callable:
        Callable returning the currently loaded SegmentFrame (or None).
    """

    state = Phase4State()

    status = w.HTML("<b>Status:</b> idle")
    log = HtmlLog()

    out_plot = w.Output(layout=w.Layout(border="1px solid #ddd", padding="4px"))
    out_stats = w.HTML("<div style='color:#666;'>Load a segment in Phase I, then come here to plot.</div>")

    downsample_k = w.BoundedIntText(
        value=1,
        min=1,
        max=1_000_000,
        description="Downsample K:",
        layout=w.Layout(width="240px"),
    )

    cb_turn_markers = w.Checkbox(value=False, description="Show turn markers")

    btn_plot_current = w.Button(description="Plot I(t)", button_style="", layout=w.Layout(width="140px"))
    btn_plot_abs = w.Button(description="Plot absolute signal", button_style="", layout=w.Layout(width="180px"))
    btn_plot_cmp = w.Button(description="Plot compensated signal", button_style="", layout=w.Layout(width="210px"))

    dd_format = w.Dropdown(
        options=[("SVG", "svg"), ("PDF", "pdf"), ("PNG", "png")],
        value="svg",
        description="Format:",
        layout=w.Layout(width="200px"),
    )
    btn_save = w.Button(description="Save figure...", button_style="", layout=w.Layout(width="160px"))

    def _get_seg() -> Optional[SegmentFrame]:
        return get_segmentframe_callable()

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _refresh_stats(seg: SegmentFrame) -> None:
        t = np.asarray(seg.df["t"].to_numpy())
        stats = _basic_time_stats(t)
        # sampling stats
        html = ["<b>Segment statistics</b>"]
        html.append(f"<div><b>Samples:</b> {stats.get('n','?')}</div>")
        if "t_min" in stats:
            html.append(
                "<div>"
                f"<b>t range:</b> {stats['t_min']:.6g} … {stats['t_max']:.6g} s"
                "</div>"
            )
            html.append(
                "<div>"
                f"<b>Δt:</b> min {stats['dt_min']:.6g} / med {stats['dt_med']:.6g} / max {stats['dt_max']:.6g} s"
                "</div>"
            )
            html.append(
                "<div>"
                f"<b>Non‑monotonic Δt:</b> {stats['nonmono']} samples ({stats['nonmono_pct']:.3g}%)"
                "</div>"
            )
        html.append(f"<div><b>Samples/turn:</b> {int(seg.samples_per_turn)}</div>")
        html.append(f"<div><b>Turns:</b> {int(seg.n_turns)}</div>")
        html.append(f"<div><b>Source:</b> {seg.source_path}</div>")
        out_stats.value = "\n".join(html)

    def _draw_fig() -> None:
        if state.fig is None:
            return
        try:
            state.fig.tight_layout()
        except Exception:
            pass
        with out_plot:
            clear_output(wait=True)
            display(state.fig)

    def _plot_series(y: np.ndarray, label: str) -> None:
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return

        _set_status("plotting…")
        try:
            k = int(downsample_k.value)
            t = np.asarray(seg.df["t"].to_numpy())
            y = np.asarray(y)

            if t.shape[0] != y.shape[0]:
                raise ValueError(f"Length mismatch: t has {t.shape[0]} samples but y has {y.shape[0]} samples")

            t_d = _decimate(t, k)
            y_d = _decimate(y, k)

            state.fig = plt.figure(figsize=(8.5, 4.2))
            ax = state.fig.add_subplot(1, 1, 1)
            ax.plot(t_d, y_d)
            ax.set_xlabel("t (s)")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True)

            if bool(cb_turn_markers.value):
                # Turn starts at indices 0, N, 2N, ... in the original sampling.
                n = int(seg.samples_per_turn)
                if n > 0:
                    idx = np.arange(0, int(seg.n_turns) * n, n, dtype=int)
                    idx = idx[idx < t.size]
                    # Decimate marker indices consistently: keep markers whose sample index survived the decimation.
                    if k > 1:
                        idx = idx[(idx % k) == 0] // k
                    ax.vlines(t_d[idx], ymin=np.nanmin(y_d), ymax=np.nanmax(y_d), linestyles="dotted")

            _refresh_stats(seg)
            _draw_fig()
            state.last_source = str(seg.source_path)
            log.write(f"Plotted {label} (downsample K={k}).")
        except Exception as exc:
            log.write(f"ERROR: {exc}")
        finally:
            _set_status("idle")

    def _on_plot_current(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        y = seg.df["I"].to_numpy()
        _plot_series(y, "I(t) [A]")

    def _on_plot_abs(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        _plot_series(seg.df["df_abs"].to_numpy(), "Absolute signal")

    def _on_plot_cmp(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        _plot_series(seg.df["df_cmp"].to_numpy(), "Compensated signal")

    def _saveas_dialog(ext: str) -> Optional[str]:
        """Open a native Save-As dialog (works in desktop Jupyter; no-op in pure web environments)."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return None

        root = tk.Tk()
        root.withdraw()
        initialdir = None
        try:
            if get_segmentpath_callable is not None:
                seg_path = get_segmentpath_callable()
                if seg_path:
                    initialdir = os.path.dirname(os.path.abspath(seg_path))
        except Exception:
            initialdir = None
        path = filedialog.asksaveasfilename(
            defaultextension=f".{ext}",
            filetypes=[(ext.upper(), f"*.{ext}"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        root.destroy()
        return path or None

    def _on_save(_btn):
        if state.fig is None:
            log.write("No figure to save yet. Plot something first.")
            return
        ext = str(dd_format.value)
        path = _saveas_dialog(ext)
        if not path:
            log.write("Save cancelled.")
            return
        try:
            state.fig.savefig(path)
            log.write(f"Saved: {path}")
        except Exception as exc:
            log.write(f"ERROR saving figure: {exc}")

    btn_plot_current.on_click(_on_plot_current)
    btn_plot_abs.on_click(_on_plot_abs)
    btn_plot_cmp.on_click(_on_plot_cmp)
    btn_save.on_click(_on_save)

    header = w.HTML(
        "<h3>Phase IV — Exploration plots</h3>"
        "<div style='color:#666;'>Read-only plots using measured time (no resampling). Downsampling is decimation only.</div>"
    )

    controls = w.HBox([downsample_k, cb_turn_markers, dd_format, btn_save])
    btns = w.HBox([btn_plot_current, btn_plot_abs, btn_plot_cmp])

    left = w.VBox(
        [
            header,
            controls,
            btns,
            out_stats,
            out_plot,
        ],
        layout=w.Layout(width="100%"),
    )

    right = w.VBox([
        status,
        w.HTML("<b>Log</b>"),
        log.widget,
    ], layout=w.Layout(width="420px"))

    panel = w.HBox([
        w.VBox([left], layout=w.Layout(flex="1 1 auto")),
        w.VBox([right], layout=w.Layout(flex="0 0 420px")),
    ])

    return panel
