"""Phase IV: read-only exploration plots.

Goals
-----
* Provide quick, safe visualization of the currently loaded segment.
* Read-only: must not mutate analysis outputs.
* Downsampling is decimation only (no interpolation): keep every K-th sample.
* Respect the project's "no synthetic time" rule: always plot against measured time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import ipywidgets as w
from IPython.display import clear_output, display

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.gui.log_view import HtmlLog


@dataclass
class Phase4State:
    fig: Optional[object] = None  # Matplotlib Figure
    canvas: Optional[object] = None  # ipympl canvas if used
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


def _have_ipympl() -> bool:
    try:
        # Presence check only; rendering uses FigureCanvasNbAgg
        import ipympl  # noqa: F401

        return True
    except Exception:
        return False


def _make_figure(interactive: bool) -> Tuple[object, object, Optional[object]]:
    """Create a figure.
    If interactive=True and ipympl is available, return (fig, ax, canvas) where canvas is zoomable.
    Otherwise return (fig, ax, None) and caller should display(fig).
    """
    if interactive and _have_ipympl():
        from matplotlib.figure import Figure
        from ipympl.backend_nbagg import FigureCanvasNbAgg  # type: ignore

        fig = Figure(figsize=(8.5, 4.2))
        canvas = FigureCanvasNbAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax, canvas

    # Static fallback
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    return fig, ax, None


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

    cb_interactive = w.Checkbox(
        value=True,
        description="Interactive (zoom/pan)",
        layout=w.Layout(width="220px"),
    )

    cb_append = w.Checkbox(
        value=False,
        description="Append plots",
        layout=w.Layout(width="160px"),
    )

    dd_secondary = w.Dropdown(
        options=[
            ("None", "none"),
            ("Current I(t) [A]", "current"),
        ],
        value="none",
        description="Secondary y:",
        layout=w.Layout(width="280px"),
    )

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
        html = ["<b>Segment statistics</b>"]
        html.append(f"<div><b>Samples:</b> {stats.get('n','?')}</div>")
        if "t_min" in stats:
            html.append(f"<div><b>t range:</b> {stats['t_min']:.6g} … {stats['t_max']:.6g} s</div>")
            html.append(
                "<div>"
                f"<b>Δt:</b> min {stats['dt_min']:.6g} / med {stats['dt_med']:.6g} / max {stats['dt_max']:.6g} s"
                "</div>"
            )
            html.append(
                "<div>"
                f"<b>Non-monotonic Δt:</b> {stats['nonmono']} samples ({stats['nonmono_pct']:.3g}%)"
                "</div>"
            )
        html.append(f"<div><b>Samples/turn:</b> {int(seg.samples_per_turn)}</div>")
        html.append(f"<div><b>Turns:</b> {int(seg.n_turns)}</div>")
        html.append(f"<div><b>Source:</b> {seg.source_path}</div>")
        out_stats.value = "\n".join(html)

    def _render(fig: object, canvas: Optional[object]) -> None:
        with out_plot:
            if not bool(cb_append.value):
                clear_output(wait=True)

            # If we have a zoomable canvas, display that; otherwise display the fig (static).
            if canvas is not None:
                display(canvas)
            else:
                display(fig)

    def _maybe_add_turn_markers(seg: SegmentFrame, ax, t_d: np.ndarray, y_d: np.ndarray, k: int) -> None:
        if not bool(cb_turn_markers.value):
            return

        n = int(seg.samples_per_turn)
        if n <= 0:
            return

        t = np.asarray(seg.df["t"].to_numpy())
        idx = np.arange(0, int(seg.n_turns) * n, n, dtype=int)
        idx = idx[idx < t.size]

        if k > 1:
            idx = idx[(idx % k) == 0] // k

        if idx.size == 0:
            return

        ymin = float(np.nanmin(y_d))
        ymax = float(np.nanmax(y_d))
        ax.vlines(t_d[idx], ymin=ymin, ymax=ymax, linestyles="dotted")

    def _plot_series(y: np.ndarray, label: str, allow_secondary_current: bool) -> None:
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

            interactive = bool(cb_interactive.value)
            fig, ax, canvas = _make_figure(interactive=interactive)

            ax.plot(t_d, y_d, label=label)
            ax.set_xlabel("t (s)")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs time")
            ax.grid(True)

            # Optional secondary axis: current in red
            ax2 = None
            if allow_secondary_current and str(dd_secondary.value) == "current":
                I = np.asarray(seg.df["I"].to_numpy())
                I_d = _decimate(I, k)
                ax2 = ax.twinx()
                ax2.plot(t_d, I_d, color="red", linestyle="--", label="I(t) [A]")
                ax2.set_ylabel("I(t) [A]")

            self_handles, self_labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                self_handles += h2
                self_labels += l2
            if self_handles:
                ax.legend(self_handles, self_labels, loc="best")

            _maybe_add_turn_markers(seg, ax, t_d, y_d, k)

            try:
                fig.tight_layout()
            except Exception:
                pass

            state.fig = fig
            state.canvas = canvas
            state.last_source = str(seg.source_path)

            _refresh_stats(seg)
            _render(fig, canvas)

            # ipympl availability hint (only if requested interactive but not available)
            if interactive and not _have_ipympl():
                log.write("NOTE: Interactive mode requires ipympl. Install with: py -3.13 -m pip install ipympl")

            sec_txt = "None"
            if allow_secondary_current and str(dd_secondary.value) == "current":
                sec_txt = "Current I(t) [A]"

            log.write(f"Plotted {label} (downsample K={k}). Secondary y: {sec_txt}")
        except Exception as exc:
            log.write(f"ERROR: {exc}")
        finally:
            _set_status("idle")

    def _on_plot_current(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        _plot_series(seg.df["I"].to_numpy(), "I(t) [A]", allow_secondary_current=False)

    def _on_plot_abs(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        _plot_series(seg.df["df_abs"].to_numpy(), "Absolute signal", allow_secondary_current=True)

    def _on_plot_cmp(_btn):
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        _plot_series(seg.df["df_cmp"].to_numpy(), "Compensated signal", allow_secondary_current=True)

    def _saveas_dialog(ext: str) -> Optional[str]:
        """Open a native Save-As dialog (works in desktop Jupyter; may no-op in pure web environments)."""
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
            # ipympl canvas does not change saving: save the underlying figure.
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
        "<div style='color:#666;'>Read-only plots using measured time (no resampling). "
        "Downsampling is decimation only. Interactive zoom/pan requires ipympl.</div>"
    )

    controls_row1 = w.HBox([downsample_k, cb_turn_markers, cb_interactive, cb_append, dd_secondary])
    controls_row2 = w.HBox([dd_format, btn_save])

    btns = w.HBox([btn_plot_current, btn_plot_abs, btn_plot_cmp])

    left = w.VBox(
        [
            header,
            controls_row1,
            btns,
            controls_row2,
            out_stats,
            out_plot,
        ],
        layout=w.Layout(width="100%"),
    )

    right = w.VBox(
        [
            status,
            w.HTML("<b>Log</b>"),
            log.widget,
        ],
        layout=w.Layout(width="420px"),
    )

    panel = w.HBox(
        [
            w.VBox([left], layout=w.Layout(flex="1 1 auto")),
            w.VBox([right], layout=w.Layout(flex="0 0 420px")),
        ]
    )

    return panel
