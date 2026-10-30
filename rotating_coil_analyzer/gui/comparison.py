"""Cross-measurement Comparison GUI tab.

Load two exported CSV files and compare operating points side-by-side.
Accepts the Harmonic Merge ``*_MERGED.csv`` export (``mrg_`` prefixed,
``B{n}_T`` for n <= m, ``b{n}_units`` above) for any main order m, and the
legacy summary layout (``label``, ``ok_main``, ``B1_T``, ``b3_units``).
The significance test is ``analysis.utility_functions.diff_sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import re

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.analysis.utility_functions import diff_sigma


@dataclass
class ComparisonState:
    """Local mutable state for the Comparison tab."""

    df1: Optional[pd.DataFrame] = None
    df2: Optional[pd.DataFrame] = None
    keys: list = field(default_factory=list)
    fig: Optional[Any] = None
    busy: bool = False


def _browse_for_file() -> Optional[str]:
    """Native file chooser (tkinter). Returns None if unavailable/cancelled."""
    from rotating_coil_analyzer.gui._dialogs import open_file_dialog
    return open_file_dialog(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )


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

    round_step = w.FloatText(
        value=1.0,
        description="Round I [A]:",
        tooltip="Used only when a CSV has no label / I_nom column: operating points = I_mean_A rounded to this step.",
        style={"description_width": "80px"},
        layout=w.Layout(width="170px"),
    )
    harm_text = w.Text(
        value="b3",
        description="Harmonics:",
        placeholder="e.g. b3, a2, b6",
        style={"description_width": "80px"},
        layout=w.Layout(width="260px"),
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

    def _resolve_label_col(df: pd.DataFrame, requested: str) -> Optional[str]:
        """Resolve the column to match on.

        Honours the user's "Match by" selection when that column is present
        in the CSV; only falls back to auto-detection when the requested
        column is missing. Returns None when no label-like column exists
        (the caller then groups by the rounded mean current).
        """
        if requested in df.columns:
            return requested
        return _detect_label_col(df)

    def _prepare(df_raw: pd.DataFrame, name: str) -> Optional[Tuple[pd.DataFrame, dict]]:
        """Bring an exported CSV to a common layout.

        Accepts the Harmonic Merge export (``mrg_`` prefixed, ``B{n}_T`` for
        n <= m and ``b{n}_units`` / ``a{n}_units`` above) and the legacy
        summary layout (``B1_T``, ``b3_units``, ``label``, ``ok_main``).
        Returns (df, info) or None after logging what is missing.
        """
        df = df_raw.copy()
        df.columns = [c[len("mrg_"):] if str(c).startswith("mrg_") else c for c in df.columns]

        missing: list[str] = []

        # Main field: the highest-order B{n}_T column is the main harmonic B{m}
        # (mixed format writes B_n in T for every n <= m).
        b_orders = sorted(
            int(mm.group(1)) for c in df.columns
            if (mm := re.fullmatch(r"B(\d+)_T", str(c)))
        )
        if not b_orders:
            missing.append("main-field column B{m}_T (or mrg_B{m}_T)")
            m = None
            main_col = None
        else:
            m = b_orders[-1]
            main_col = f"B{m}_T"
            a_col = f"A{m}_T"
            # A skew main lives on the skew component: use whichever is dominant.
            if a_col in df.columns:
                b_abs = float(np.nanmedian(np.abs(pd.to_numeric(df[main_col], errors="coerce"))))
                a_abs = float(np.nanmedian(np.abs(pd.to_numeric(df[a_col], errors="coerce"))))
                if np.isfinite(a_abs) and np.isfinite(b_abs) and a_abs > 10.0 * b_abs:
                    main_col = a_col
                    log.write(f"{name}: main field taken from {a_col} (skew main).")

        cur_col = "I_mean_A" if "I_mean_A" in df.columns else ("I_nom" if "I_nom" in df.columns else None)
        if cur_col is None:
            missing.append("current column I_mean_A (or I_nom)")

        if missing:
            log.write(
                f"ERROR: {name} cannot be compared, missing: {', '.join(missing)}. "
                f"Columns found: {', '.join(map(str, df_raw.columns[:25]))}"
                + (" …" if len(df_raw.columns) > 25 else "")
            )
            return None

        out = pd.DataFrame(index=df.index)
        out["I_mean_A"] = pd.to_numeric(df[cur_col], errors="coerce")
        out["main"] = pd.to_numeric(df[main_col], errors="coerce")

        harm_cols = [c for c in df.columns if re.fullmatch(r"[ab]\d+_units", str(c))]
        for c in harm_cols:
            out[c] = pd.to_numeric(df[c], errors="coerce")

        if "ok_main" in df.columns:
            out["ok_main"] = df["ok_main"].astype(bool)
        else:
            out["ok_main"] = np.isfinite(out["main"].to_numpy(dtype=float))
            log.write(f"{name}: no ok_main column; using turns with a finite main field.")

        # Operating-point label
        lcol = _resolve_label_col(df, str(label_col_dd.value))
        if lcol is not None:
            out["level"] = df[lcol]
            label_desc = lcol
        else:
            step = float(round_step.value) if float(round_step.value) > 0 else 1.0
            out["level"] = (np.round(out["I_mean_A"] / step) * step).round(6)
            label_desc = f"I_mean_A rounded to {step:g} A"
            log.write(f"{name}: no label / I_nom column; operating points = {label_desc}.")

        info = {"m": m, "main_col": main_col, "label": label_desc, "harm_cols": harm_cols}
        return out, info

    def _level_stats(df: pd.DataFrame, level, keys: list[tuple[str, str]]) -> dict:
        """Mean / std / N for each (key, column) at one operating point."""
        sub = df[(df["level"] == level) & df["ok_main"]]
        if len(sub) == 0:
            return {}
        stats: dict = {"N": len(sub), "I_mean": float(sub["I_mean_A"].mean())}
        cols = {
            "main": sub["main"],
            "TF": sub["main"] / (sub["I_mean_A"] / 1000.0),
        }
        for key, col in keys:
            series = cols.get(col, sub[col] if col in sub.columns else None)
            if series is None:
                continue
            stats[f"{key}_mean"] = float(series.mean())
            stats[f"{key}_std"] = float(series.std())
        return stats

    def _comparison_keys(info1: dict, info2: dict) -> list[tuple[str, str]]:
        """Keys to compare: main B{m}, TF, and the requested harmonics present in both files."""
        m = info1["m"]
        keys: list[tuple[str, str]] = [(f"B{m}", "main")]
        common = set(info1["harm_cols"]) & set(info2["harm_cols"])
        req = [t.strip() for t in str(harm_text.value).split(",") if t.strip()]
        for r in req:
            col = r if r.endswith("_units") else f"{r}_units"
            if col in common:
                keys.append((col[: -len("_units")], col))
            else:
                log.write(f"Harmonic '{r}' not available in both files; skipped.")
        keys.append(("TF", "TF"))
        return keys

    def _load_both() -> Optional[tuple]:
        p1 = path1.value.strip()
        p2 = path2.value.strip()
        if not p1 or not p2:
            log.write("Please provide both CSV paths.")
            return None
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        st.df1 = df1
        st.df2 = df2
        log.write(f"Loaded CSV 1: {len(df1)} rows, {len(df1.columns)} cols")
        log.write(f"Loaded CSV 2: {len(df2)} rows, {len(df2.columns)} cols")
        r1 = _prepare(df1, "CSV 1")
        r2 = _prepare(df2, "CSV 2")
        if r1 is None or r2 is None:
            result_html.value = "<b style='color:#b00;'>Cannot compare: see the log for the missing columns.</b>"
            return None
        (d1, i1), (d2, i2) = r1, r2
        if i1["m"] != i2["m"]:
            log.write(
                f"WARNING: main order differs (CSV 1: B{i1['m']}, CSV 2: B{i2['m']}); "
                "main field and TF are compared as given."
            )
        return d1, i1, d2, i2

    def _on_compare(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("loading & comparing...")

            loaded = _load_both()
            if loaded is None:
                return
            d1, i1, d2, i2 = loaded

            # Match operating points
            levels1 = set(d1["level"].dropna().unique())
            levels2 = set(d2["level"].dropna().unique())
            common = sorted(levels1 & levels2, key=str)

            if not common:
                result_html.value = (
                    "<b style='color:#b00;'>No common operating points found.</b><br>"
                    f"CSV 1 levels ({i1['label']}): {sorted(levels1, key=str)}<br>"
                    f"CSV 2 levels ({i2['label']}): {sorted(levels2, key=str)}"
                )
                return

            keys = _comparison_keys(i1, i2)
            st.keys = keys

            # Compute stats per level
            rows: list[dict] = []
            for level in common:
                s1 = _level_stats(d1, level, keys)
                s2 = _level_stats(d2, level, keys)
                if not s1 or not s2:
                    continue

                row: dict = {"level": level, "N1": s1["N"], "N2": s2["N"]}

                for key, _col in keys:
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
                    f"<b>Comparison ({len(rows)} common levels; main field B{i1['m']} from "
                    f"{i1['main_col']}):</b><br>"
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

            loaded = _load_both()
            if loaded is None:
                return
            d1, i1, d2, i2 = loaded

            levels1 = set(d1["level"].dropna().unique())
            levels2 = set(d2["level"].dropna().unique())
            common = sorted(levels1 & levels2, key=str)

            if not common:
                log.write("No common levels to plot.")
                return

            keys = _comparison_keys(i1, i2)
            deltas = {k: [] for k, _ in keys}
            labels_list = []

            for level in common:
                s1 = _level_stats(d1, level, keys)
                s2 = _level_stats(d2, level, keys)
                if not s1 or not s2:
                    continue
                labels_list.append(str(level))
                for key, _col in keys:
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
                n_keys = max(1, len(keys))
                width = 0.8 / n_keys

                for i, (key, _col) in enumerate(keys):
                    ax.bar(x + i * width, deltas[key], width,
                           label=f"|delta {key}| / sigma", alpha=0.8)

                ax.set_xlabel("Operating point")
                ax.set_ylabel("Significance [sigma]")
                ax.set_title("Difference significance by operating point")
                ax.set_xticks(x + width * (n_keys - 1) / 2.0)
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
        "Computes delta, propagated error, and sigma significance for the main field B{m}, "
        "the listed harmonics and TF. Accepts the Harmonic Merge *_MERGED.csv export."
        "</div>"
    )

    file_box = w.VBox([
        w.HBox([path1, btn_browse1]),
        w.HBox([path2, btn_browse2]),
        w.HBox([label_col_dd, round_step, harm_text]),
        w.HBox([btn_compare, btn_plot_diff]),
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
