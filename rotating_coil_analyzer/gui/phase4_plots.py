"""
Phase IV — Exploration plots (read-only).

Design goals
- Must not mutate analysis results.
- Must respect "no synthetic time": use measured time vector as-is.
- Plot-only downsampling uses decimation (keep every K-th sample, no interpolation).
- Optional interactive zoom/pan when an ipympl/widget backend is active.

Notes on interactivity
- We do NOT attempt to force-enable a widget backend from inside the GUI because matplotlib backends
  are typically decided by the host environment (Jupyter/VSCode/Qt).
- If you want interactive zoom/pan, in a notebook run: %matplotlib widget
  and ensure ipympl is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import io

import numpy as np
import pandas as pd

import ipywidgets as widgets
from IPython.display import display, Image, clear_output

import matplotlib
import matplotlib.pyplot as plt


def build_phase4_plots_panel(
    get_segment_frame: Callable[[], object | None],
    log_append: Callable[[widgets.Textarea, str], None],
) -> widgets.Widget:
    """Build Phase IV panel.

    Parameters
    ----------
    get_segment_frame:
        Callable returning the current SegmentFrame (or None). SegmentFrame is expected
        to carry a pandas DataFrame in `.df`.
    log_append:
        Logging callback used elsewhere in the GUI.
    """

    @dataclass
    class _State:
        segment_df: Optional[pd.DataFrame] = None
        last_fig: Optional[plt.Figure] = None
        last_title: str = ""

    st = _State()

    # ----------------------------
    # Helpers
    # ----------------------------
    def _log(msg: str) -> None:
        log_append(log, msg)

    def _extract_df(seg: object | None) -> Optional[pd.DataFrame]:
        if seg is None:
            return None
        # SegmentFrame normally has `.df`
        if hasattr(seg, "df"):
            df = getattr(seg, "df")
            if isinstance(df, pd.DataFrame):
                return df
        # Sometimes the DF might be passed directly
        if isinstance(seg, pd.DataFrame):
            return seg
        return None

    def _safe_int(x: str, default: int) -> int:
        try:
            return int(str(x).strip())
        except Exception:
            return default

    def _safe_float(x: str) -> Optional[float]:
        s = str(x).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _get_time_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["t", "time", "timestamp"]:
            if c in df.columns:
                return c
        for c in df.columns:
            if str(c).lower() == "t":
                return c
        return None

    def _get_current_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["I", "i", "current", "I[A]"]:
            if c in df.columns:
                return c
        return None

    def _get_abs_col(df: pd.DataFrame) -> Optional[str]:
        # Project uses df_abs/df_cmp (your Phase I table preview), but older code used abs/cmp.
        for c in ["df_abs", "abs", "absolute", "Absolute"]:
            if c in df.columns:
                return c
        return None

    def _get_cmp_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["df_cmp", "cmp", "compensated", "Compensated"]:
            if c in df.columns:
                return c
        return None

    def _apply_window_and_decimate(
        t: np.ndarray,
        y: np.ndarray,
        K: int,
        tmin: Optional[float],
        tmax: Optional[float],
    ):
        mask = np.ones_like(t, dtype=bool)
        if tmin is not None:
            mask &= (t >= tmin)
        if tmax is not None:
            mask &= (t <= tmax)

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return t[:0], y[:0]

        sel = idx if K <= 1 else idx[::K]
        return t[sel], y[sel]

    def _is_widget_backend_active() -> bool:
        b = str(matplotlib.get_backend()).lower()
        return ("ipympl" in b) or ("widget" in b) or ("nbagg" in b)

    def _render_static_png(fig: plt.Figure) -> Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return Image(data=buf.getvalue())

    def _display_figure(fig: plt.Figure) -> None:
        # If interactive requested and a widget backend is active, display the actual figure (zoom/pan).
        if interactive.value and _is_widget_backend_active():
            display(fig)
        else:
            # Always reliable in VSCode/Jupyter: render to PNG bytes and display as an Image.
            display(_render_static_png(fig))

    def _refresh_state_and_columns() -> None:
        seg = get_segment_frame()
        st.segment_df = _extract_df(seg)

        if st.segment_df is None:
            custom_channel.options = []
            secondary.options = ["None", "Current I(t) [A]"]
            _log("No segment loaded yet. Load a segment in Phase I/II, then come here.")
            return

        cols = list(st.segment_df.columns)

        preferred = []
        for c in ["t", "I", "df_abs", "df_cmp", "abs", "cmp"]:
            if c in cols:
                preferred.append(c)
        rest = [c for c in cols if c not in preferred]

        full = preferred + rest
        custom_channel.options = full
        # Secondary overlay can be current or any column
        secondary.options = ["None", "Current I(t) [A]"] + full

        _log(f"Columns refreshed ({len(cols)} columns).")

    def _clear_plots(_=None) -> None:
        with plot_out:
            clear_output(wait=True)
        st.last_fig = None
        st.last_title = ""
        _log("Plots cleared.")

    def _plot_time_series(
        *,
        title: str,
        y_col: str,
        y_label: str,
        secondary_choice: str,
    ) -> None:
        if st.segment_df is None:
            _refresh_state_and_columns()
            return
        df = st.segment_df

        tcol = _get_time_col(df)
        if tcol is None:
            _log("ERROR: could not find a time column (expected 't').")
            return
        if y_col not in df.columns:
            _log(f"ERROR: column '{y_col}' not found in segment.")
            return

        t = np.asarray(df[tcol].to_numpy())
        y = np.asarray(df[y_col].to_numpy())

        K = max(1, _safe_int(downsample.value, 1))
        tmin = _safe_float(tmin_w.value)
        tmax = _safe_float(tmax_w.value)

        t_plot, y_plot = _apply_window_and_decimate(t, y, K, tmin, tmax)
        if t_plot.size == 0:
            _log("WARNING: selected plot window is empty (no samples).")
            return

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t_plot, y_plot, label=y_label)
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(y_label)
        ax.grid(True)

        # Secondary y-axis overlay
        if secondary_choice != "None":
            if secondary_choice == "Current I(t) [A]":
                ccol = _get_current_col(df)
            else:
                ccol = secondary_choice

            if ccol is None or ccol not in df.columns:
                _log(f"WARNING: secondary channel '{secondary_choice}' not found; skipping overlay.")
            else:
                yc = np.asarray(df[ccol].to_numpy())
                t2, yc2 = _apply_window_and_decimate(t, yc, K, tmin, tmax)
                ax2 = ax.twinx()
                # Make the current clearly distinguishable: solid red line.
                ax2.plot(t2, yc2, color="red", label=str(secondary_choice))
                ax2.set_ylabel(str(secondary_choice))

                # Combined legend
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, loc="best")
        else:
            ax.legend(loc="best")

        with plot_out:
            if not append_plots.value:
                clear_output(wait=True)
            _display_figure(fig)

        st.last_fig = fig
        st.last_title = title

        _log(
            f"Plotted '{y_col}' (K={K}). Secondary y: {secondary_choice}. "
            f"Interactive={'on' if (interactive.value and _is_widget_backend_active()) else 'off'}."
        )

    def _plot_current(_=None) -> None:
        if st.segment_df is None:
            _refresh_state_and_columns()
            return
        df = st.segment_df
        ccol = _get_current_col(df)
        if ccol is None:
            _log("ERROR: could not find current column (expected 'I').")
            return

        _plot_time_series(
            title="Current I(t) [A] vs time",
            y_col=ccol,
            y_label="I(t) [A]",
            secondary_choice="None",
        )

    def _plot_abs(_=None) -> None:
        if st.segment_df is None:
            _refresh_state_and_columns()
            return
        df = st.segment_df
        col = _get_abs_col(df)
        if col is None:
            _log("ERROR: no absolute-signal column found (expected 'df_abs' or 'abs').")
            return

        _plot_time_series(
            title="Absolute signal vs time",
            y_col=col,
            y_label=str(col),
            secondary_choice=secondary.value,
        )

    def _plot_cmp(_=None) -> None:
        if st.segment_df is None:
            _refresh_state_and_columns()
            return
        df = st.segment_df
        col = _get_cmp_col(df)
        if col is None:
            _log("ERROR: no compensated-signal column found (expected 'df_cmp' or 'cmp').")
            return

        _plot_time_series(
            title="Compensated signal vs time",
            y_col=col,
            y_label=str(col),
            secondary_choice=secondary.value,
        )

    def _plot_custom(_=None) -> None:
        if st.segment_df is None:
            _refresh_state_and_columns()
            return
        df = st.segment_df

        col = custom_channel.value
        if col is None or col == "":
            _log("ERROR: choose a custom channel first.")
            return
        if col not in df.columns:
            _log(f"ERROR: column '{col}' not found.")
            return

        _plot_time_series(
            title=f"{col} vs time",
            y_col=col,
            y_label=str(col),
            secondary_choice=secondary.value,
        )

    def _save_last(_=None) -> None:
        if st.last_fig is None:
            _log("No figure available to save yet.")
            return

        fmt = str(save_format.value).lower()
        if fmt not in {"svg", "png", "pdf"}:
            _log(f"Unsupported format: {fmt}")
            return

        from tkinter import Tk
        from tkinter.filedialog import asksaveasfilename

        Tk().withdraw()
        suggested = (st.last_title or "plot").replace(" ", "_").replace("/", "_")
        path = asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")],
            initialfile=f"{suggested}.{fmt}",
            title="Save plot",
        )
        if not path:
            return

        try:
            st.last_fig.savefig(path, format=fmt, bbox_inches="tight")
            _log(f"Saved: {path}")
        except Exception as e:
            _log(f"ERROR saving plot: {e}")

    # ----------------------------
    # Widgets
    # ----------------------------
    title = widgets.HTML("<h3>Phase IV — Exploration plots</h3>")
    intro = widgets.HTML(
        "<p style='margin-top:0'>Read-only plots using measured time (no resampling). "
        "Downsampling is decimation only. For zoom/pan, enable <b>Interactive</b> "
        "and run <code>%matplotlib widget</code> in your notebook environment.</p>"
    )

    downsample = widgets.Text(value="1", description="Downsample K", layout=widgets.Layout(width="160px"))
    show_turn_markers = widgets.Checkbox(value=False, description="Show turn markers")
    interactive = widgets.Checkbox(value=False, description="Interactive (zoom/pan)")
    append_plots = widgets.Checkbox(value=False, description="Append plots")

    tmin_w = widgets.Text(value="", description="tmin [s]", layout=widgets.Layout(width="220px"))
    tmax_w = widgets.Text(value="", description="tmax [s]", layout=widgets.Layout(width="220px"))

    refresh_cols = widgets.Button(description="Refresh columns")
    refresh_cols.on_click(lambda _=None: _refresh_state_and_columns())

    clear_btn = widgets.Button(description="Clear plots", button_style="warning")
    clear_btn.on_click(_clear_plots)

    secondary = widgets.Dropdown(
        options=["None", "Current I(t) [A]"],
        value="Current I(t) [A]",
        description="Secondary y",
        layout=widgets.Layout(width="280px"),
    )

    btn_I = widgets.Button(description="Plot I(t)")
    btn_abs = widgets.Button(description="Plot absolute signal")
    btn_cmp = widgets.Button(description="Plot compensated signal")
    btn_custom = widgets.Button(description="Plot custom channel")

    btn_I.on_click(_plot_current)
    btn_abs.on_click(_plot_abs)
    btn_cmp.on_click(_plot_cmp)
    btn_custom.on_click(_plot_custom)

    custom_channel = widgets.Dropdown(options=[], description="Custom channel", layout=widgets.Layout(width="520px"))

    save_format = widgets.Dropdown(
        options=["SVG", "PNG", "PDF"], value="SVG", description="Format", layout=widgets.Layout(width="160px")
    )
    save_btn = widgets.Button(description="Save figure…")
    save_btn.on_click(_save_last)

    plot_out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", height="520px", overflow="auto"))
    log = widgets.Textarea(value="", layout=widgets.Layout(width="100%", height="220px"), disabled=True)

    row1 = widgets.HBox([downsample, show_turn_markers, interactive, append_plots, secondary])
    row2 = widgets.HBox([tmin_w, tmax_w, refresh_cols, clear_btn])
    row3 = widgets.HBox([btn_I, btn_abs, btn_cmp, btn_custom])
    row4 = widgets.HBox([custom_channel])
    row5 = widgets.HBox([save_format, save_btn])

    left = widgets.VBox([title, intro, row1, row2, row3, row4, row5, plot_out], layout=widgets.Layout(width="70%"))
    right = widgets.VBox(
        [widgets.HTML("<b>Status:</b> idle"), widgets.HTML("<b>Log</b>"), log],
        layout=widgets.Layout(width="30%"),
    )

    _refresh_state_and_columns()
    return widgets.HBox([left, right])
