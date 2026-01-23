"""Phase IV — Exploration plots.

Read-only plots using measured time (no resampling). Downsampling is decimation
only (keep every Kth sample). Optional interactive zoom/pan when ipympl is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import ipywidgets as w
from IPython.display import display


@dataclass
class Phase4State:
    # NOTE: Phase I/II store a SegmentFrame object in shared state.
    # That object exposes the pandas DataFrame as `.df` (not directly as a DataFrame).
    # Phase IV keeps both the original object (for metadata) and the extracted df.
    segment_obj: Optional[Any] = None
    segment_df: Optional[pd.DataFrame] = None
    segment_meta: Optional[dict] = None


def _extract_df(segment_obj: Any) -> Optional[pd.DataFrame]:
    """Return a pandas DataFrame from a SegmentFrame-like object.

    Accepts:
    - pandas DataFrame (returned as-is)
    - SegmentFrame wrapper (must expose `.df`)
    - None (returns None)
    """
    if segment_obj is None:
        return None
    if isinstance(segment_obj, pd.DataFrame):
        return segment_obj
    # SegmentFrame wrapper used by this project
    df = getattr(segment_obj, "df", None)
    if isinstance(df, pd.DataFrame):
        return df
    return None


_ACTIVE_PHASE4_PANEL = None


def _log_clear(log: w.Textarea) -> None:
    log.value = ""


def _log_append(log: w.Textarea, msg: str) -> None:
    log.value = (log.value + ("\n" if log.value else "") + msg)


def _parse_int(text: str, default: int) -> int:
    try:
        v = int(str(text).strip())
        return v
    except Exception:
        return default


def _parse_float_or_none(text: str) -> Optional[float]:
    s = str(text).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _decimate(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return x[::k]


def _prepare_window_and_decimate(
    t: np.ndarray,
    y: np.ndarray,
    k: int,
    tmin: Optional[float],
    tmax: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    if tmin is None and tmax is None:
        tt = _decimate(t, k)
        yy = _decimate(y, k)
        return tt, yy

    m = np.ones_like(t, dtype=bool)
    if tmin is not None:
        m &= (t >= tmin)
    if tmax is not None:
        m &= (t <= tmax)

    tt = t[m]
    yy = y[m]
    tt = _decimate(tt, k)
    yy = _decimate(yy, k)
    return tt, yy


def _try_enable_ipympl(interactive: bool, log: w.Textarea) -> bool:
    """Try to enable ipympl widget backend.

    Returns True if ipympl backend was enabled.

    This must run *before* importing matplotlib.pyplot.
    """
    if not interactive:
        return False

    try:
        import matplotlib  # noqa: WPS433

        # Ensure ipympl is importable.
        import ipympl  # noqa: F401,WPS433

        matplotlib.use("module://ipympl.backend_nbagg")
        return True
    except Exception as e:
        _log_append(
            log,
            "NOTE: interactive mode requires a working ipympl backend. "
            f"Falling back to static plots. Error: {type(e).__name__}: {e}",
        )
        return False


def build_phase4_plots_panel(
    get_segmentframe_callable: Callable[[], Optional[pd.DataFrame]],
    get_segmentmeta_callable: Callable[[], Optional[dict]],
    get_turns_callable: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
) -> w.Widget:
    """Build Phase IV tab.

    Parameters
    ----------
    get_segmentframe_callable:
        Returns the currently loaded segment DataFrame (or None).
    get_segmentmeta_callable:
        Returns metadata dict about the loaded segment (or None).
    get_turns_callable:
        Optional callable returning turns DataFrame with per-turn timing info.
    """

    global _ACTIVE_PHASE4_PANEL

    if _ACTIVE_PHASE4_PANEL is not None:
        return _ACTIVE_PHASE4_PANEL

    st = Phase4State()

    # --- controls ---
    downsample = w.IntText(value=1, description="Downsample K", layout=w.Layout(width="160px"))
    show_turns = w.Checkbox(value=False, description="Show turn markers")
    interactive = w.Checkbox(value=False, description="Interactive (zoom/pan)")
    append_out = w.Checkbox(value=False, description="Append plots")

    tmin_txt = w.Text(value="", placeholder="(blank = start)", description="tmin [s]", layout=w.Layout(width="220px"))
    tmax_txt = w.Text(value="", placeholder="(blank = end)", description="tmax [s]", layout=w.Layout(width="220px"))

    custom_channel = w.Dropdown(options=[], description="Custom channel", layout=w.Layout(width="520px"))
    secondary = w.Dropdown(
        options=[("None", "none"), ("Current I(t) [A]", "current")],
        value="current",
        description="Secondary y",
        layout=w.Layout(width="260px"),
    )

    btn_refresh = w.Button(description="Refresh columns", button_style="")
    btn_clear = w.Button(description="Clear plots", button_style="warning")

    btn_plot_current = w.Button(description="Plot I(t)")
    btn_plot_abs = w.Button(description="Plot absolute signal")
    btn_plot_cmp = w.Button(description="Plot compensated signal")
    btn_plot_custom = w.Button(description="Plot custom channel")

    fmt = w.Dropdown(options=["SVG", "PNG", "PDF"], value="SVG", description="Format", layout=w.Layout(width="160px"))
    btn_save = w.Button(description="Save figure…")

    status = w.HTML("<b>Status:</b> idle")
    log = w.Textarea(value="", layout=w.Layout(width="100%", height="220px"))

    plot_out = w.Output(layout=w.Layout(width="100%"))

    # we keep last figure handle for save
    last_fig = {"fig": None}

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _refresh_state_and_columns() -> None:
        st.segment_frame = get_segmentframe_callable()
        st.segment_meta = get_segmentmeta_callable()
        st.segment_df = _extract_df(st.segment_frame)

        if st.segment_df is None:
            custom_channel.options = []
            _log_append(log, "No segment loaded yet. Load a segment in Phase I/II, then come here.")
            return

        df = st.segment_df
        cols = list(df.columns)
        # Prefer time-like columns first (helps users spot it), but keep full list.
        def _score(c: str) -> tuple[int, str]:
            lc = c.lower()
            if lc in {"t", "time", "timestamp", "utc", "t_s"}:
                return (0, c)
            return (1, c)

        cols_sorted = sorted(cols, key=_score)
        custom_channel.options = cols_sorted
        if "abs" in df.columns:
            custom_channel.value = "abs"
        elif "cmp" in df.columns:
            custom_channel.value = "cmp"
        else:
            custom_channel.value = cols_sorted[0] if cols_sorted else None

    def _get_time_and_current(df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        # Time column is expected to be `t` in our pipeline, but keep fallbacks.
        if "t" in df.columns:
            t = df["t"].to_numpy(dtype=float)
        elif "time" in df.columns:
            t = df["time"].to_numpy(dtype=float)
        else:
            raise ValueError("No time column found. Expected 't' (or fallback 'time').")

        I = None
        for cand in ["I", "i", "current", "I_A", "I_meas"]:
            if cand in df.columns:
                I = df[cand].to_numpy(dtype=float)
                break
        return t, I

    def _plot(
        title: str,
        y: np.ndarray,
        y_label: str,
        include_current: bool,
        k: int,
        tmin: Optional[float],
        tmax: Optional[float],
    ) -> None:
        if st.segment_df is None:
            _log_append(log, "No segment loaded.")
            return

        df = st.segment_df
        t, I = _get_time_and_current(df)

        tt, yy = _prepare_window_and_decimate(t, y, k, tmin, tmax)

        if include_current:
            if I is None:
                _log_append(log, "Secondary current requested, but no current column was found in the segment.")
                include_current = False
            else:
                tI, II = _prepare_window_and_decimate(t, I, k, tmin, tmax)
                # Ensure the same t vector when plotting on twin axis (use common indices approach).
                # If windows/decimation differ due to mask, align via the shortest.
                n = min(len(tt), len(tI))
                tt = tt[:n]
                yy = yy[:n]
                II = II[:n]

        # Backend selection must happen before importing pyplot.
        _log_clear(log)
        _set_status("plotting")

        _try_enable_ipympl(bool(interactive.value), log)
        import matplotlib.pyplot as plt  # noqa: WPS433

        if not append_out.value:
            with plot_out:
                plot_out.clear_output(wait=True)

        # Close previous figure to reduce memory/leaks.
        try:
            if last_fig["fig"] is not None:
                plt.close(last_fig["fig"])
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tt, yy, label=y_label)
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(y_label)
        ax.grid(True)

        if include_current and I is not None:
            ax2 = ax.twinx()
            ax2.plot(tt, II, label="I(t) [A]", color="red", alpha=0.85)
            ax2.set_ylabel("I(t) [A]")

            # Combined legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(loc="best")

        # Optional turn markers: requires turns table with per-turn start times
        if show_turns.value and get_turns_callable is not None:
            try:
                turns_df = get_turns_callable()
                if turns_df is not None and "t0" in turns_df.columns:
                    t0s = turns_df["t0"].to_numpy(dtype=float)
                    # Plot only those in window
                    if tmin is not None:
                        t0s = t0s[t0s >= tmin]
                    if tmax is not None:
                        t0s = t0s[t0s <= tmax]
                    for t0 in t0s:
                        ax.axvline(t0, color="k", alpha=0.05, linewidth=0.8)
            except Exception as e:
                _log_append(log, f"Turn marker overlay failed: {type(e).__name__}: {e}")

        last_fig["fig"] = fig

        with plot_out:
            if bool(interactive.value):
                # ipympl provides a proper widget canvas with zoom/pan.
                display(fig.canvas)
            else:
                display(fig)
                # Prevent implicit "auto-display" of the figure outside our Output.
                plt.close(fig)

        _set_status("idle")

    def _plot_current(_=None) -> None:
        _refresh_state_and_columns()
        if st.segment_df is None:
            return
        df = st.segment_df
        t, I = _get_time_and_current(df)
        if I is None:
            _log_append(log, "No current column found in segment.")
            return
        k = max(1, int(downsample.value or 1))
        tmin = _parse_float_or_none(tmin_txt.value)
        tmax = _parse_float_or_none(tmax_txt.value)
        _plot("I(t) [A] vs time", I, "I(t) [A]", include_current=False, k=k, tmin=tmin, tmax=tmax)

    def _plot_abs(_=None) -> None:
        _refresh_state_and_columns()
        if st.segment_df is None:
            return
        if "abs" not in st.segment_df.columns:
            _log_append(log, "No 'abs' column found in segment.")
            return
        y = st.segment_df["abs"].to_numpy(dtype=float)
        k = max(1, int(downsample.value or 1))
        tmin = _parse_float_or_none(tmin_txt.value)
        tmax = _parse_float_or_none(tmax_txt.value)
        incI = (secondary.value == "current")
        _plot("Absolute signal vs time", y, "Absolute signal", include_current=incI, k=k, tmin=tmin, tmax=tmax)

    def _plot_cmp(_=None) -> None:
        _refresh_state_and_columns()
        if st.segment_df is None:
            return
        if "cmp" not in st.segment_df.columns:
            _log_append(log, "No 'cmp' column found in segment.")
            return
        y = st.segment_df["cmp"].to_numpy(dtype=float)
        k = max(1, int(downsample.value or 1))
        tmin = _parse_float_or_none(tmin_txt.value)
        tmax = _parse_float_or_none(tmax_txt.value)
        incI = (secondary.value == "current")
        _plot("Compensated signal vs time", y, "Compensated signal", include_current=incI, k=k, tmin=tmin, tmax=tmax)

    def _plot_custom(_=None) -> None:
        _refresh_state_and_columns()
        if st.segment_df is None:
            return
        ch = custom_channel.value
        if ch is None or ch not in st.segment_df.columns:
            _log_append(log, "Choose a valid custom channel.")
            return
        y = st.segment_df[ch].to_numpy(dtype=float)
        k = max(1, int(downsample.value or 1))
        tmin = _parse_float_or_none(tmin_txt.value)
        tmax = _parse_float_or_none(tmax_txt.value)
        incI = (secondary.value == "current")
        _plot(f"{ch} vs time", y, ch, include_current=incI, k=k, tmin=tmin, tmax=tmax)

    def _clear_plots(_=None) -> None:
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433

            if last_fig["fig"] is not None:
                plt.close(last_fig["fig"])
        except Exception:
            pass
        last_fig["fig"] = None
        with plot_out:
            plot_out.clear_output(wait=True)
        _log_append(log, "Plots cleared.")

    def _save_fig(_=None) -> None:
        if last_fig["fig"] is None:
            _log_append(log, "No plot to save yet.")
            return
        # Minimal saver: uses a file chooser dialog via browser download is not available;
        # in JupyterLab this typically saves to the working directory via a text prompt.
        # We keep this aligned with previous export mechanism when integrated.
        ext = fmt.value.lower()
        fname = f"phase4_plot.{ext}"
        try:
            last_fig["fig"].savefig(fname, bbox_inches="tight")
            _log_append(log, f"Saved {fname}")
        except Exception as e:
            _log_append(log, f"Save failed: {type(e).__name__}: {e}")

    # wire callbacks
    btn_refresh.on_click(lambda _: _refresh_state_and_columns())
    btn_clear.on_click(_clear_plots)
    btn_plot_current.on_click(_plot_current)
    btn_plot_abs.on_click(_plot_abs)
    btn_plot_cmp.on_click(_plot_cmp)
    btn_plot_custom.on_click(_plot_custom)
    btn_save.on_click(_save_fig)

    # initial populate
    _refresh_state_and_columns()

    # --- layout ---
    controls_row1 = w.HBox([
        downsample,
        show_turns,
        interactive,
        append_out,
        secondary,
    ])
    controls_row2 = w.HBox([
        tmin_txt,
        tmax_txt,
        btn_refresh,
        btn_clear,
    ])
    controls_row3 = w.HBox([
        btn_plot_current,
        btn_plot_abs,
        btn_plot_cmp,
        btn_plot_custom,
    ])
    controls_row4 = w.HBox([
        custom_channel,
    ])
    controls_row5 = w.HBox([
        fmt,
        btn_save,
    ])

    left = w.VBox([
        w.HTML("<h3>Phase IV — Exploration plots</h3>"),
        w.HTML(
            "Read-only plots using measured time (no resampling). "
            "Downsampling is decimation only. "
            "For zoom/pan, enable ‘Interactive’ and ensure ipympl works."
        ),
        controls_row1,
        controls_row2,
        controls_row3,
        controls_row4,
        controls_row5,
        plot_out,
    ], layout=w.Layout(width="100%"))

    right = w.VBox([
        status,
        w.HTML("<b>Log</b>"),
        log,
    ], layout=w.Layout(width="420px"))

    panel = w.HBox([
        left,
        right,
    ], layout=w.Layout(width="100%"))

    _ACTIVE_PHASE4_PANEL = panel
    return panel
