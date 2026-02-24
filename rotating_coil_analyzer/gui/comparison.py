"""Cross-measurement Comparison GUI tab.

Load two exported CSV files and compare operating points side-by-side.
All comparison functions live in ``analysis.utility_functions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_level_stats,
    diff_sigma,
)


@dataclass
class ComparisonState:
    """Local mutable state for the Comparison tab."""

    df1: Optional[pd.DataFrame] = None
    df2: Optional[pd.DataFrame] = None
    fig: Optional[Any] = None
    busy: bool = False


def _browse_for_file() -> Optional[str]:
    """Native file chooser (tkinter). Returns None if unavailable/cancelled."""
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
        p = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        return str(p) if p else None
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


def build_comparison_panel() -> w.Widget:
    """Build the Cross-measurement Comparison panel (standalone, no shared state)."""

    st = ComparisonState()
    log = HtmlLog(title="Comparison log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- File pickers ----
    path1 = w.Text(
        value="",
        description="CSV 1:",
        placeholder="Path to first export CSV",
        layout=w.Layout(width="60%"),
        style={"description_width": "60px"},
    )
    btn_browse1 = w.Button(description="Browse…", layout=w.Layout(width="100px"))

    path2 = w.Text(
        value="",
        description="CSV 2:",
        placeholder="Path to second export CSV",
        layout=w.Layout(width="60%"),
        style={"description_width": "60px"},
    )
    btn_browse2 = w.Button(description="Browse…", layout=w.Layout(width="100px"))

    label_col_dd = w.Dropdown(
        options=[("label", "label"), ("I_nom", "I_nom")],
        value="label",
        description="Match by:",
        style={"description_width": "80px"},
        layout=w.Layout(width="220px"),
    )

    btn_compare = w.Button(description="Load & compare", button_style="primary")
    btn_plot_diff = w.Button(description="Plot differences", button_style="")
    btn_plot_diff.disabled = True

    result_html = w.HTML("<i>Load two CSVs to compare.</i>")
    out_plot = w.Output(
        layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%", height="400px")
    )

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _on_browse1(_btn):
        p = _browse_for_file()
        if p:
            path1.value = p

    def _on_browse2(_btn):
        p = _browse_for_file()
        if p:
            path2.value = p

    def _detect_label_col(df: pd.DataFrame) -> Optional[str]:
        """Detect the label/level column in an exported CSV."""
        for candidate in ["label", "I_nom", "I_nom_A", "level"]:
            if candidate in df.columns:
                return candidate
        return None

    def _ensure_required_cols(df: pd.DataFrame, name: str) -> bool:
        """Check that essential columns exist."""
        required = {"ok_main", "I_mean_A", "B1_T"}
        missing = required - set(df.columns)
        if missing:
            log.write(f"WARNING: {name} missing columns: {missing}. "
                       "Attempting partial comparison.")
        return True

    def _on_compare(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("loading & comparing...")

            p1 = path1.value.strip()
            p2 = path2.value.strip()
            if not p1 or not p2:
                log.write("Please provide both CSV paths.")
                return

            df1 = pd.read_csv(p1)
            df2 = pd.read_csv(p2)
            st.df1 = df1
            st.df2 = df2

            log.write(f"Loaded CSV 1: {len(df1)} rows, {len(df1.columns)} cols")
            log.write(f"Loaded CSV 2: {len(df2)} rows, {len(df2.columns)} cols")

            # Detect label column
            label_col = str(label_col_dd.value)
            lcol1 = _detect_label_col(df1) or label_col
            lcol2 = _detect_label_col(df2) or label_col

            if lcol1 not in df1.columns:
                log.write(f"ERROR: Column '{lcol1}' not found in CSV 1.")
                return
            if lcol2 not in df2.columns:
                log.write(f"ERROR: Column '{lcol2}' not found in CSV 2.")
                return

            _ensure_required_cols(df1, "CSV 1")
            _ensure_required_cols(df2, "CSV 2")

            # Ensure ok_main exists
            if "ok_main" not in df1.columns:
                df1["ok_main"] = True
            if "ok_main" not in df2.columns:
                df2["ok_main"] = True

            # Match operating points
            levels1 = set(df1[lcol1].unique())
            levels2 = set(df2[lcol2].unique())
            common = sorted(levels1 & levels2, key=str)

            if not common:
                result_html.value = (
                    "<b style='color:#b00;'>No common operating points found.</b><br>"
                    f"CSV 1 levels: {sorted(levels1, key=str)}<br>"
                    f"CSV 2 levels: {sorted(levels2, key=str)}"
                )
                return

            # Compute stats per level
            rows: list[dict] = []
            for level in common:
                s1 = compute_level_stats(df1, level, label_col=lcol1)
                s2 = compute_level_stats(df2, level, label_col=lcol2)
                if not s1 or not s2:
                    continue

                row: dict = {"level": level, "N1": s1["N"], "N2": s2["N"]}

                for key in ["B1", "b3", "TF"]:
                    try:
                        d, err, sig = diff_sigma(s1, s2, key)
                        row[f"d_{key}"] = f"{d:.4g}"
                        row[f"err_{key}"] = f"{err:.4g}"
                        row[f"sig_{key}"] = f"{sig:.1f}"
                    except (KeyError, ZeroDivisionError):
                        row[f"d_{key}"] = "—"
                        row[f"err_{key}"] = "—"
                        row[f"sig_{key}"] = "—"

                rows.append(row)

            if rows:
                df_cmp = pd.DataFrame(rows)
                result_html.value = (
                    f"<b>Comparison ({len(rows)} common levels):</b><br>"
                    + df_cmp.to_html(index=False, classes="table table-sm")
                )
                btn_plot_diff.disabled = False
            else:
                result_html.value = "<b>No comparable data found.</b>"

            log.write(f"Comparison complete: {len(rows)} levels matched.")
            _set_status("comparison done")

        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    def _on_plot_diff(_btn) -> None:
        if st.busy or st.df1 is None or st.df2 is None:
            return
        st.busy = True
        try:
            import matplotlib.pyplot as plt

            _set_status("plotting differences...")

            label_col = str(label_col_dd.value)
            lcol1 = _detect_label_col(st.df1) or label_col
            lcol2 = _detect_label_col(st.df2) or label_col

            levels1 = set(st.df1[lcol1].unique())
            levels2 = set(st.df2[lcol2].unique())
            common = sorted(levels1 & levels2, key=str)

            if not common:
                log.write("No common levels to plot.")
                return

            keys = ["B1", "b3", "TF"]
            deltas = {k: [] for k in keys}
            labels_list = []

            for level in common:
                s1 = compute_level_stats(st.df1, level, label_col=lcol1)
                s2 = compute_level_stats(st.df2, level, label_col=lcol2)
                if not s1 or not s2:
                    continue
                labels_list.append(str(level))
                for key in keys:
                    try:
                        d, _, sig = diff_sigma(s1, s2, key)
                        deltas[key].append(sig)
                    except (KeyError, ZeroDivisionError):
                        deltas[key].append(0.0)

            out_plot.clear_output(wait=True)
            try:
                plt.close("all")
            except Exception:
                pass

            with out_plot:
                fig, ax = plt.subplots(figsize=(8, 4))
                st.fig = fig

                x = np.arange(len(labels_list))
                width = 0.25
                colors = ["tab:blue", "tab:orange", "tab:green"]

                for i, key in enumerate(keys):
                    ax.bar(x + i * width, deltas[key], width,
                           label=f"|delta {key}| / sigma", color=colors[i], alpha=0.8)

                ax.set_xlabel("Operating point")
                ax.set_ylabel("Significance [sigma]")
                ax.set_title("Difference significance by operating point")
                ax.set_xticks(x + width)
                ax.set_xticklabels(labels_list, rotation=45, ha="right")
                ax.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="2-sigma")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    # ---- Wire handlers ----
    btn_browse1.on_click(_on_browse1)
    btn_browse2.on_click(_on_browse2)
    btn_compare.on_click(_on_compare)
    btn_plot_diff.on_click(_on_plot_diff)

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Cross-measurement Comparison</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Load two exported CSVs and compare operating points. "
        "Computes delta, propagated error, and sigma significance for B1, b3, TF."
        "</div>"
    )

    file_box = w.VBox([
        w.HBox([path1, btn_browse1]),
        w.HBox([path2, btn_browse2]),
        w.HBox([label_col_dd, btn_compare, btn_plot_diff]),
    ])

    main_panel = w.VBox(
        [header, file_box, w.HTML("<hr>"), result_html, out_plot],
        layout=w.Layout(width="70%", min_width="640px"),
    )

    diag_panel = w.VBox(
        [status, log.panel],
        layout=w.Layout(width="30%", min_width="300px"),
    )

    panel = w.HBox([main_panel, diag_panel], layout=w.Layout(width="100%"))
    return panel
