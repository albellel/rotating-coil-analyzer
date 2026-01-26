"""
Phase IV (Plots) — Exploration plots.

Design goals:
- Read-only with respect to analysis pipeline.
- No synthetic time: always plot against measured time vector `t` as stored.
- Downsampling is decimation only: keep every Kth sample (no interpolation).
- Optional interactive zoom/pan using Matplotlib interactive backends when available.

Notes on zoom/pan:
- True zoom/pan requires a working interactive backend in Jupyter.
- We try ipympl first, then nbagg.
- If neither works, plots will still render, but they will be static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import os

import numpy as np
import ipywidgets as w

from .log_view import HtmlLog


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _decimate(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return x
    return x[::k]


def _parse_float_or_none(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _resolve_column(df_cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate present in df_cols, else None."""
    colset = set(df_cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def _is_current_col(col: str) -> bool:
    return col in ("I", "I_meas", "I_cmd", "current", "Current")


def _is_time_col(col: str) -> bool:
    return col in ("t", "time", "Time", "timestamp", "Timestamp")


def _try_enable_interactive_backend() -> Tuple[bool, str]:
    """
    Try to enable an interactive Matplotlib backend suitable for Jupyter.

    Returns:
        (ok, message)
    """
    import matplotlib as mpl

    # Prefer ipympl widget backend if available
    try:
        import ipympl  # noqa: F401
        try:
            mpl.use("module://ipympl.backend_nbagg", force=True)
            return True, "Interactive backend: ipympl widget (module://ipympl.backend_nbagg)."
        except Exception:
            pass
    except Exception:
        pass

    # Fall back to nbagg (Matplotlib's notebook backend)
    try:
        mpl.use("nbagg", force=True)
        return True, "Interactive backend: Matplotlib nbagg."
    except Exception as exc:
        return False, f"Interactive backend unavailable (fallback to static): {exc}"


def _get_pyplot():
    """Import pyplot lazily (backend must already be configured)."""
    import matplotlib.pyplot as plt  # late import by design
    return plt


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------

@dataclass
class _PlotsState:
    fig: object | None = None


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def build_phase4_plots_panel(
    get_segmentFrame_callable: Callable[[], object | None],
    get_segmentpath_callable: Optional[Callable[[], Optional[str]]] = None,
) -> w.Widget:
    """
    Build Phase IV panel.

    Args:
        get_segmentFrame_callable: returns the current SegmentFrame (or None).
        get_segmentpath_callable: returns the current segment path (optional; used for Save-As default dir).

    Expected SegmentFrame interface (duck-typed):
        - seg.df: pandas.DataFrame with at least `t` and (often) `I`, `df_abs`, `df_cmp`.
        - seg.samples_per_turn: int
        - seg.n_turns: int
        - seg.source_path: pathlib.Path | str
    """
    state = _PlotsState()
    log = HtmlLog(title="Log", height_px=220)

    status = w.HTML("<b>Status:</b> idle")

    def _set_status(s: str) -> None:
        status.value = f"<b>Status:</b> {s}"

    def _get_seg():
        try:
            return get_segmentFrame_callable()
        except Exception:
            return None

    def _get_df(seg):
        return getattr(seg, "df", None)

    # -----------------------
    # Widgets (controls)
    # -----------------------

    downsample_k = w.BoundedIntText(
        value=1,
        min=1,
        max=1_000_000,
        step=1,
        description="Downsample K",
        style={"description_width": "initial"},
        layout=w.Layout(width="210px"),
    )

    cb_turn_markers = w.Checkbox(value=False, description="Show turn markers")
    cb_interactive = w.Checkbox(value=False, description="Interactive (zoom/pan)")
    cb_append = w.Checkbox(value=False, description="Append plots")

    tmin_box = w.Text(
        value="",
        placeholder="(blank = start)",
        description="tmin [s]",
        style={"description_width": "initial"},
        layout=w.Layout(width="220px"),
    )
    tmax_box = w.Text(
        value="",
        placeholder="(blank = end)",
        description="tmax [s]",
        style={"description_width": "initial"},
        layout=w.Layout(width="220px"),
    )

    btn_refresh_cols = w.Button(description="Refresh columns")
    btn_clear = w.Button(description="Clear plots", button_style="warning")

    # Primary/Secondary y selection (two Y-axes)
    dd_primary = w.Dropdown(
        options=[("—", "")],
        value="",
        description="Primary y",
        layout=w.Layout(width="360px"),
    )
    dd_secondary = w.Dropdown(
        options=[("None", "")],
        value="",
        description="Secondary y",
        layout=w.Layout(width="360px"),
    )

    # Convenience buttons
    btn_plot_current = w.Button(description="Plot I(t)")
    btn_plot_abs = w.Button(description="Plot absolute signal")
    btn_plot_cmp = w.Button(description="Plot compensated signal")
    btn_plot_selected = w.Button(description="Plot selected y's")

    # Save
    dd_format = w.Dropdown(
        options=["svg", "pdf", "png"],
        value="svg",
        description="Format",
        layout=w.Layout(width="160px"),
    )
    btn_save = w.Button(description="Save figure…")

    out_plot = w.Output(
        layout=w.Layout(border="1px solid #ddd", padding="6px", height="560px", overflow="auto")
    )

    # -----------------------
    # Column refresh
    # -----------------------

    def _refresh_columns() -> None:
        seg = _get_seg()
        if seg is None:
            dd_primary.options = [("— (no segment loaded)", "")]
            dd_primary.value = ""
            dd_secondary.options = [("None", "")]
            dd_secondary.value = ""
            return

        df = _get_df(seg)
        if df is None:
            dd_primary.options = [("— (segment has no DataFrame)", "")]
            dd_primary.value = ""
            dd_secondary.options = [("None", "")]
            dd_secondary.value = ""
            return

        cols = list(df.columns)

        # Preferred ordering: time, current, abs/cmp, then rest.
        def _score(c: str) -> Tuple[int, str]:
            if _is_time_col(c):
                return (0, c)
            if _is_current_col(c):
                return (1, c)
            if c in ("df_abs", "df_cmp"):
                return (2, c)
            return (3, c)

        cols_sorted = sorted(cols, key=_score)
        plottable = [c for c in cols_sorted if not _is_time_col(c)]

        if not plottable:
            dd_primary.options = [("— (no plottable columns)", "")]
            dd_primary.value = ""
            dd_secondary.options = [("None", "")]
            dd_secondary.value = ""
            return

        dd_primary.options = [(c, c) for c in plottable]
        dd_secondary.options = [("None", "")] + [(c, c) for c in plottable]

        # Default primary: df_abs if present, else df_cmp, else first.
        pref_primary = _resolve_column(plottable, ["df_abs", "df_cmp", "I"]) or plottable[0]
        dd_primary.value = pref_primary

        # Default secondary: current if available and not already primary.
        if "I" in plottable and dd_primary.value != "I":
            dd_secondary.value = "I"
        else:
            dd_secondary.value = ""

    # -----------------------
    # Plot logic
    # -----------------------

    def _slice_by_time(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tmin = _parse_float_or_none(tmin_box.value)
        tmax = _parse_float_or_none(tmax_box.value)
        if tmin is None and tmax is None:
            return t, y
        m = np.ones_like(t, dtype=bool)
        if tmin is not None:
            m &= (t >= tmin)
        if tmax is not None:
            m &= (t <= tmax)
        return t[m], y[m]

    def _plot(primary_col: str, secondary_col: Optional[str], title: Optional[str] = None) -> None:
        seg = _get_seg()
        if seg is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        df = _get_df(seg)
        if df is None:
            log.write("ERROR: segment has no DataFrame (.df).")
            return

        t_col = _resolve_column(df.columns, ["t", "time", "Time"])
        if t_col is None:
            log.write("ERROR: no time column found (expected 't').")
            return

        if not primary_col:
            log.write("Select a primary y column.")
            return

        if primary_col not in df.columns:
            log.write(f"ERROR: primary column '{primary_col}' not found in segment.")
            return

        if secondary_col:
            if secondary_col not in df.columns:
                log.write(f"ERROR: secondary column '{secondary_col}' not found in segment.")
                return
            if secondary_col == primary_col:
                secondary_col = None

        _set_status("plotting…")
        try:
            k = int(downsample_k.value)
            interactive = bool(cb_interactive.value)

            if interactive:
                ok, msg = _try_enable_interactive_backend()
                log.write(msg)

            plt = _get_pyplot()

            t = np.asarray(df[t_col].to_numpy())
            y1 = np.asarray(df[primary_col].to_numpy())

            if t.shape[0] != y1.shape[0]:
                raise ValueError(
                    f"Length mismatch: {t_col} has {t.shape[0]} samples but {primary_col} has {y1.shape[0]} samples"
                )

            # Window first, then decimate (keeps exact measured times)
            t_w, y1_w = _slice_by_time(t, y1)
            t_d = _decimate(t_w, k)
            y1_d = _decimate(y1_w, k)

            # Critical: clear unless Append is checked
            if not cb_append.value:
                out_plot.clear_output(wait=True)

            with out_plot:
                fig = plt.figure(figsize=(10.0, 4.8))
                ax1 = fig.add_subplot(1, 1, 1)

                # Primary color: red if current, else default cycle
                c1 = "red" if _is_current_col(primary_col) else None
                ax1.plot(t_d, y1_d, label=primary_col, color=c1)
                ax1.set_xlabel(f"{t_col} (s)" if t_col == "t" else t_col)
                ax1.set_ylabel(primary_col)
                ax1.grid(True)

                if title is None:
                    title = f"{primary_col} vs time"
                ax1.set_title(title)

                ax2 = None
                if secondary_col:
                    y2 = np.asarray(df[secondary_col].to_numpy())
                    t_w2, y2_w = _slice_by_time(t, y2)
                    # Mask is computed from the same t, so this should match
                    if t_w2.shape[0] != t_w.shape[0]:
                        raise ValueError("Internal error: time-window slicing mismatch between primary and secondary.")
                    y2_d = _decimate(y2_w, k)

                    ax2 = ax1.twinx()
                    # Secondary line: always red (requested for current visibility); dashed for clarity
                    ax2.plot(t_d, y2_d, label=secondary_col, color="red", linestyle="--")
                    ax2.set_ylabel(secondary_col)

                # Turn markers (optional)
                if bool(cb_turn_markers.value):
                    n = int(getattr(seg, "samples_per_turn", 0) or 0)
                    n_turns = int(getattr(seg, "n_turns", 0) or 0)
                    if n > 0 and n_turns > 0:
                        idx = np.arange(0, n_turns * n, n, dtype=int)
                        idx = idx[idx < t.size]

                        tmin = _parse_float_or_none(tmin_box.value)
                        tmax = _parse_float_or_none(tmax_box.value)
                        if tmin is not None:
                            idx = idx[t[idx] >= tmin]
                        if tmax is not None:
                            idx = idx[t[idx] <= tmax]

                        if k > 1:
                            idx = idx[(idx % k) == 0] // k

                        idx = idx[(idx >= 0) & (idx < t_d.size)]
                        if idx.size > 0:
                            ymin = np.nanmin(y1_d)
                            ymax = np.nanmax(y1_d)
                            ax1.vlines(t_d[idx], ymin=ymin, ymax=ymax, linestyles="dotted", linewidth=0.8)

                # Legend
                h1, l1 = ax1.get_legend_handles_labels()
                if ax2 is not None:
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1 + h2, l1 + l2, loc="best")
                else:
                    ax1.legend(loc="best")

                plt.show()

                state.fig = fig

            log.write(
                f"Plotted primary='{primary_col}'"
                + (f", secondary='{secondary_col}'" if secondary_col else "")
                + f" (K={k}, interactive={interactive}, append={bool(cb_append.value)})."
            )
        except Exception as exc:
            log.write(f"ERROR: {exc}")
        finally:
            _set_status("idle")

    # -----------------------
    # Button callbacks
    # -----------------------

    def _on_refresh(_btn):
        _refresh_columns()
        log.write("Columns refreshed.")

    def _on_clear(_btn):
        out_plot.clear_output(wait=True)
        state.fig = None
        log.write("Plots cleared.")

    def _on_plot_current(_btn):
        seg = _get_seg()
        if seg is None or _get_df(seg) is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        df = _get_df(seg)
        col = _resolve_column(df.columns, ["I", "current", "Current"])
        if col is None:
            log.write("ERROR: no current column found (expected 'I').")
            return
        _plot(col, dd_secondary.value or None, title="I(t) [A] vs time")

    def _on_plot_abs(_btn):
        seg = _get_seg()
        if seg is None or _get_df(seg) is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        df = _get_df(seg)
        col = _resolve_column(df.columns, ["df_abs", "abs", "Abs", "absolute", "Absolute"])
        if col is None:
            log.write("ERROR: no absolute-signal column found (expected 'df_abs').")
            return
        _plot(col, dd_secondary.value or None, title="Absolute signal vs time")

    def _on_plot_cmp(_btn):
        seg = _get_seg()
        if seg is None or _get_df(seg) is None:
            log.write("No segment loaded. Load a segment in Phase I first.")
            return
        df = _get_df(seg)
        col = _resolve_column(df.columns, ["df_cmp", "cmp", "Cmp", "compensated", "Compensated"])
        if col is None:
            log.write("ERROR: no compensated-signal column found (expected 'df_cmp').")
            return
        _plot(col, dd_secondary.value or None, title="Compensated signal vs time")

    def _on_plot_selected(_btn):
        _plot(dd_primary.value, dd_secondary.value or None)

    def _saveas_dialog(ext: str) -> Optional[str]:
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

    # Wire callbacks
    btn_refresh_cols.on_click(_on_refresh)
    btn_clear.on_click(_on_clear)
    btn_plot_current.on_click(_on_plot_current)
    btn_plot_abs.on_click(_on_plot_abs)
    btn_plot_cmp.on_click(_on_plot_cmp)
    btn_plot_selected.on_click(_on_plot_selected)
    btn_save.on_click(_on_save)

    # Initial population
    _refresh_columns()

    # -----------------------
    # Layout
    # -----------------------

    header = w.HTML(
        "<h3>Phase IV — Exploration plots</h3>"
        "<div style='color:#666;'>Read-only plots using measured time (no resampling). "
        "Downsampling is decimation only. Enable <b>Interactive</b> for zoom/pan (requires a working Matplotlib Jupyter backend).</div>"
    )

    row1 = w.HBox([downsample_k, cb_turn_markers, cb_interactive, cb_append, dd_secondary])
    row2 = w.HBox([tmin_box, tmax_box, btn_refresh_cols, btn_clear])
    row3 = w.HBox([btn_plot_current, btn_plot_abs, btn_plot_cmp, btn_plot_selected])
    row4 = w.HBox([dd_primary, dd_format, btn_save])

    left = w.VBox([header, row1, row2, row3, row4, out_plot], layout=w.Layout(width="70%"))
    right = w.VBox([status, log.panel], layout=w.Layout(width="30%"))

    return w.HBox([left, right], layout=w.Layout(width="100%"))
