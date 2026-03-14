"""Physics Plots GUI tab — hysteresis, transfer function, Ld, eddy-current settling.

Provides interactive plots for magnet characterisation from plateau-based
measurements.  All backend functions live in ``analysis.utility_functions``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.analysis.utility_functions import (
    plateau_summary,
    plot_hysteresis,
    build_run_averages,
)
from tools_for_data_analysis.eddy import (
    fit_eddy_per_run,
    EddyFitResult,
)


@dataclass
class PhysicsPlotsState:
    """Local mutable state for the Physics Plots tab."""

    summary_df: Optional[pd.DataFrame] = None
    eddy_results: Optional[List[EddyFitResult]] = None
    fig: Optional[Any] = None
    busy: bool = False


def build_physics_plots_panel(
    get_merge_result: Callable[[], Optional[Any]],
    get_plateau_info: Callable[[], Optional[Dict[str, Any]]],
    get_segmentframe: Callable[[], Optional[Any]],
    set_n_last_recommended: Callable[[int], None] | None = None,
) -> w.Widget:
    """Build the Physics Plots panel.

    Parameters
    ----------
    get_merge_result
        Returns the current MergeResult from shared state.
    get_plateau_info
        Returns plateau detection info dict from shared state.
    get_segmentframe
        Returns the SegmentFrame from shared state.
    set_n_last_recommended
        Called with recommended N_LAST from eddy analysis.
    """
    st = PhysicsPlotsState()
    log = HtmlLog(title="Physics Plots log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- Widgets ----
    n_last = w.IntText(
        value=10,
        description="N last turns:",
        style={"description_width": "100px"},
        layout=w.Layout(width="200px"),
    )
    btn_summary = w.Button(description="Compute summary", button_style="primary")
    summary_html = w.HTML("<i>Compute summary to see per-run statistics.</i>")

    btn_hysteresis = w.Button(description="Hysteresis B1 vs I", button_style="")
    btn_tf = w.Button(description="Transfer function vs I", button_style="")
    btn_ld = w.Button(description="Differential inductance", button_style="")
    harmonic_dd = w.Dropdown(
        options=[("b3_units", "b3_units"), ("b5_units", "b5_units"),
                 ("a2_units", "a2_units"), ("b2_units", "b2_units")],
        value="b3_units",
        description="Harmonic:",
        style={"description_width": "80px"},
        layout=w.Layout(width="250px"),
    )
    btn_harmonic = w.Button(description="Harmonic vs I", button_style="")

    for b in [btn_hysteresis, btn_tf, btn_ld, btn_harmonic]:
        b.disabled = True

    # Eddy-current section
    btn_eddy = w.Button(description="Fit eddy tau", button_style="warning")
    btn_eddy_hist = w.Button(description="Plot tau histogram", button_style="")
    btn_eddy_vs_I = w.Button(description="Plot tau vs I", button_style="")
    eddy_html = w.HTML("")
    for b in [btn_eddy_hist, btn_eddy_vs_I]:
        b.disabled = True

    out_plot = w.Output(
        layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%", height="420px")
    )

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _build_analysis_df(mr, plateau_info) -> Optional[pd.DataFrame]:
        """Build a DataFrame suitable for plateau_summary from merge result + plateau info."""
        if mr is None:
            log.write("No merge result available. Apply merge in Harmonic Merge tab first.")
            return None

        n_turns = mr.C_merged.shape[0]
        m = int(mr.magnet_order)
        orders = mr.orders

        # Build per-turn data
        data: dict = {
            "turn": np.arange(n_turns),
            "B1_T": np.real(mr.C_merged[:, m - 1]) if m >= 1 else np.real(mr.C_merged[:, 0]),
        }

        # Add harmonic columns from C_merged
        # For C_units we need to recompute — use C_merged for Tesla columns
        # and attempt units from the merge result
        if hasattr(mr, 'C_abs') and mr.C_abs is not None:
            data["I_mean_A"] = np.zeros(n_turns)  # will be filled from plateau_info or result
        else:
            data["I_mean_A"] = np.zeros(n_turns)

        # Try to get I_mean and ok_main from the kn_provenance
        if hasattr(mr, 'kn_provenance') and mr.kn_provenance is not None:
            data["ok_main"] = np.ones(n_turns, dtype=bool)
        else:
            data["ok_main"] = np.ones(n_turns, dtype=bool)

        # Harmonic unit columns (b/a for orders > m)
        for j, n_ord in enumerate([int(x) for x in orders]):
            if n_ord > m:
                data[f"b{n_ord}_units"] = np.real(mr.C_merged[:, j])
                data[f"a{n_ord}_units"] = np.imag(mr.C_merged[:, j])

        # Fill plateau info
        if plateau_info is not None:
            I_mean = plateau_info.get("I_mean")
            if I_mean is not None and len(I_mean) == n_turns:
                data["I_mean_A"] = I_mean

            groups = plateau_info.get("groups", [])
            labels = plateau_info.get("labels")
            plateau_id = plateau_info.get("plateau_id")

            # Assign run_id, turn_in_run, I_nom, branch
            run_id = np.full(n_turns, -1, dtype=int)
            turn_in_run = np.zeros(n_turns, dtype=int)
            I_nom = np.zeros(n_turns)
            branch = np.full(n_turns, "unknown", dtype=object)

            prev_I = 0.0
            for gid, (gs, ge) in enumerate(groups):
                run_id[gs:ge + 1] = gid
                for t in range(gs, ge + 1):
                    turn_in_run[t] = t - gs
                I_grp = float(np.mean(data["I_mean_A"][gs:ge + 1]))
                I_nom[gs:ge + 1] = I_grp
                branch[gs:ge + 1] = "ascending" if I_grp > prev_I else "descending"
                prev_I = I_grp

            data["run_id"] = run_id
            data["turn_in_run"] = turn_in_run
            data["I_nom"] = I_nom
            data["branch"] = branch
        else:
            # No plateau info — single run
            data["run_id"] = np.zeros(n_turns, dtype=int)
            data["turn_in_run"] = np.arange(n_turns)
            data["I_nom"] = data["I_mean_A"]
            data["branch"] = np.full(n_turns, "unknown", dtype=object)

        return pd.DataFrame(data)

    def _on_summary(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("computing summary...")

            mr = get_merge_result()
            plateau_info = get_plateau_info()
            df = _build_analysis_df(mr, plateau_info)
            if df is None:
                return

            # Filter to plateau turns only
            if plateau_info is not None:
                is_plateau = plateau_info.get("is_plateau")
                if is_plateau is not None and len(is_plateau) == len(df):
                    df = df[is_plateau].reset_index(drop=True)

            if df.empty or df["run_id"].max() < 0:
                summary_html.value = "<b>No plateau runs found.</b> Run plateau detection first."
                return

            summ = plateau_summary(df, n_last=int(n_last.value))
            st.summary_df = summ

            summary_html.value = (
                f"<b>Summary ({len(summ)} runs):</b><br>"
                + summ.to_html(index=False, classes="table table-sm", float_format="%.6g")
            )

            for b in [btn_hysteresis, btn_tf, btn_ld, btn_harmonic]:
                b.disabled = False

            log.write(f"Summary computed: {len(summ)} runs.")
            _set_status("summary ready")

        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    def _plot_with(plot_fn) -> None:
        """Helper: clear output and call plot_fn(fig, ax)."""
        import matplotlib.pyplot as plt

        out_plot.clear_output(wait=True)
        try:
            plt.close("all")
        except Exception:
            pass

        with out_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            st.fig = fig
            plot_fn(fig, ax)
            plt.tight_layout()
            plt.show()

    def _on_hysteresis(_btn) -> None:
        if st.busy or st.summary_df is None:
            return
        st.busy = True
        try:
            _set_status("plotting hysteresis...")
            summ = st.summary_df

            def plot_fn(fig, ax):
                plot_hysteresis(ax, summ, "I_nom", "B1_mean", "B1_std")
                ax.set_xlabel("I_nom [A]")
                ax.set_ylabel("B1 [T]")
                ax.set_title("Hysteresis: B1 vs I")
                ax.legend()
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    def _on_tf(_btn) -> None:
        if st.busy or st.summary_df is None:
            return
        st.busy = True
        try:
            _set_status("plotting TF...")
            summ = st.summary_df

            def plot_fn(fig, ax):
                good = (summ["quality"] == "good") & summ["TF"].notna()
                if not good.any():
                    ax.text(0.5, 0.5, "No good runs", ha="center", va="center",
                            transform=ax.transAxes)
                    return
                seg = summ[good].sort_values("I_nom")
                for br, col in [("ascending", "tab:blue"), ("descending", "tab:red")]:
                    m = seg["branch"] == br
                    if m.any():
                        ax.plot(seg.loc[m, "I_nom"], seg.loc[m, "TF"], "o-",
                                color=col, ms=4, label=br)
                ax.set_xlabel("I_nom [A]")
                ax.set_ylabel("TF [T/kA]")
                ax.set_title("Transfer Function vs I")
                ax.legend()
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    def _on_ld(_btn) -> None:
        if st.busy or st.summary_df is None:
            return
        st.busy = True
        try:
            _set_status("plotting Ld...")
            summ = st.summary_df

            def plot_fn(fig, ax):
                good = (summ["quality"] == "good") & summ["B1_mean"].notna()
                if good.sum() < 2:
                    ax.text(0.5, 0.5, "Need >= 2 good runs for Ld", ha="center",
                            va="center", transform=ax.transAxes)
                    return
                seg = summ[good].sort_values("I_nom")
                I_vals = seg["I_nom"].values
                B1_vals = seg["B1_mean"].values
                # Ld = dB1/dI (central differences, A -> kA)
                dI = np.diff(I_vals) / 1000.0  # kA
                dB = np.diff(B1_vals)
                Ld = dB / dI
                I_mid = (I_vals[:-1] + I_vals[1:]) / 2.0
                ax.plot(I_mid, Ld, "o-", ms=4)
                ax.set_xlabel("I [A]")
                ax.set_ylabel("Ld = dB1/dI [T/kA]")
                ax.set_title("Differential Inductance vs I")
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    def _on_harmonic(_btn) -> None:
        if st.busy or st.summary_df is None:
            return
        st.busy = True
        try:
            _set_status("plotting harmonic...")
            summ = st.summary_df
            col = str(harmonic_dd.value)
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"

            def plot_fn(fig, ax):
                good = (summ["quality"] == "good")
                if mean_col not in summ.columns:
                    ax.text(0.5, 0.5, f"Column {mean_col} not found", ha="center",
                            va="center", transform=ax.transAxes)
                    return
                good = good & summ[mean_col].notna()
                if not good.any():
                    ax.text(0.5, 0.5, "No good runs", ha="center", va="center",
                            transform=ax.transAxes)
                    return
                seg = summ[good].sort_values("I_nom")
                yerr = seg[std_col] if std_col in seg.columns else None
                for br, c in [("ascending", "tab:blue"), ("descending", "tab:red")]:
                    m = seg["branch"] == br
                    if m.any():
                        kw = dict(yerr=seg.loc[m, std_col]) if yerr is not None else {}
                        ax.errorbar(seg.loc[m, "I_nom"], seg.loc[m, mean_col],
                                    fmt="o-", color=c, ms=4, capsize=2,
                                    label=br, **kw)
                ax.set_xlabel("I_nom [A]")
                ax.set_ylabel(col)
                ax.set_title(f"{col} vs I")
                ax.legend()
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    def _on_eddy(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("fitting eddy tau...")

            mr = get_merge_result()
            plateau_info = get_plateau_info()
            df = _build_analysis_df(mr, plateau_info)
            if df is None:
                return

            if plateau_info is None or not plateau_info.get("groups"):
                log.write("No plateau info. Run plateau detection first.")
                return

            groups = plateau_info["groups"]
            m = int(mr.magnet_order)
            B1_all = np.real(mr.C_merged[:, m - 1]) if m >= 1 else np.real(mr.C_merged[:, 0])
            I_mean_all = plateau_info.get("I_mean", np.zeros(len(B1_all)))

            results: list[EddyFitResult] = []
            prev_I = 0.0
            for gid, (gs, ge) in enumerate(groups):
                turns = np.arange(ge - gs + 1, dtype=float)
                B1_run = B1_all[gs:ge + 1]
                I_run = I_mean_all[gs:ge + 1]
                I_nom = float(np.mean(I_run))
                branch = "ascending" if I_nom > prev_I else "descending"
                prev_I = I_nom

                if len(turns) < 10:
                    continue

                res = fit_eddy_per_run(
                    turns=turns,
                    B1=B1_run,
                    run_id=gid,
                    I_nom=I_nom,
                    branch=branch,
                    I_mean=I_run,
                )
                results.append(res)

            st.eddy_results = results

            # Display results table
            if results:
                rows = []
                for r in results:
                    rows.append({
                        "run_id": r.run_id,
                        "I_nom": f"{r.I_nom:.0f}",
                        "branch": r.branch,
                        "tau": f"{r.tau:.2f}" if np.isfinite(r.tau) else "—",
                        "tau_err": f"{r.tau_err:.2f}" if np.isfinite(r.tau_err) else "—",
                        "R2": f"{r.r2:.4f}" if np.isfinite(r.r2) else "—",
                        "quality": r.quality,
                    })
                df_res = pd.DataFrame(rows)
                eddy_html.value = (
                    f"<b>Eddy fits ({len(results)} runs):</b><br>"
                    + df_res.to_html(index=False, classes="table table-sm")
                )

                # Compute recommended N_LAST
                good_taus = [r.tau for r in results
                             if r.quality == "GOOD" and np.isfinite(r.tau)]
                if good_taus:
                    tau_max = max(good_taus)
                    n_last_rec = math.ceil(5 * tau_max)
                    log.write(f"Max tau (GOOD fits) = {tau_max:.1f} turns -> "
                              f"recommended N_LAST = {n_last_rec}")
                    if set_n_last_recommended is not None:
                        set_n_last_recommended(n_last_rec)
                else:
                    log.write("No GOOD eddy fits — cannot recommend N_LAST.")
            else:
                eddy_html.value = "<b>No eddy fits could be performed.</b>"

            for b in [btn_eddy_hist, btn_eddy_vs_I]:
                b.disabled = not bool(results)

            log.write(f"Eddy fitting complete: {len(results)} runs.")
            _set_status("eddy fitting done")

        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    def _on_eddy_hist(_btn) -> None:
        if st.busy or not st.eddy_results:
            return
        st.busy = True
        try:
            import matplotlib.pyplot as plt

            taus = [r.tau for r in st.eddy_results
                    if r.quality in ("GOOD", "MARGINAL") and np.isfinite(r.tau)]
            if not taus:
                log.write("No valid tau values to plot.")
                return

            def plot_fn(fig, ax):
                ax.hist(taus, bins=max(5, len(taus) // 2), edgecolor="black", alpha=0.7)
                ax.set_xlabel("tau [turns]")
                ax.set_ylabel("Count")
                ax.set_title("Eddy tau histogram")
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    def _on_eddy_vs_I(_btn) -> None:
        if st.busy or not st.eddy_results:
            return
        st.busy = True
        try:
            import matplotlib.pyplot as plt

            good = [r for r in st.eddy_results
                    if r.quality in ("GOOD", "MARGINAL") and np.isfinite(r.tau)]
            if not good:
                log.write("No valid tau values to plot.")
                return

            def plot_fn(fig, ax):
                for br, col in [("ascending", "tab:blue"), ("descending", "tab:red")]:
                    sub = [r for r in good if r.branch == br]
                    if sub:
                        I_vals = [r.I_nom for r in sub]
                        tau_vals = [r.tau for r in sub]
                        tau_errs = [r.tau_err for r in sub]
                        ax.errorbar(I_vals, tau_vals, yerr=tau_errs,
                                    fmt="o", color=col, ms=5, capsize=3, label=br)
                ax.set_xlabel("I_nom [A]")
                ax.set_ylabel("tau [turns]")
                ax.set_title("Eddy tau vs I")
                ax.legend()
                ax.grid(True, alpha=0.3)

            _plot_with(plot_fn)
            _set_status("plot ready")
        except Exception as e:
            log.write(f"ERROR: {e}")
        finally:
            st.busy = False

    # ---- Wire handlers ----
    btn_summary.on_click(_on_summary)
    btn_hysteresis.on_click(_on_hysteresis)
    btn_tf.on_click(_on_tf)
    btn_ld.on_click(_on_ld)
    btn_harmonic.on_click(_on_harmonic)
    btn_eddy.on_click(_on_eddy)
    btn_eddy_hist.on_click(_on_eddy_hist)
    btn_eddy_vs_I.on_click(_on_eddy_vs_I)

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Physics Plots</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Hysteresis loops, transfer function, differential inductance, "
        "eddy-current settling analysis. Requires merge result + plateau detection."
        "</div>"
    )

    summary_box = w.VBox([
        w.HTML("<b>Summary</b>"),
        w.HBox([n_last, btn_summary]),
        summary_html,
    ])

    plot_buttons = w.VBox([
        w.HTML("<b>Physics plots</b>"),
        w.HBox([btn_hysteresis, btn_tf, btn_ld]),
        w.HBox([harmonic_dd, btn_harmonic]),
    ])

    eddy_box = w.VBox([
        w.HTML("<b>Eddy-current settling</b>"),
        w.HBox([btn_eddy, btn_eddy_hist, btn_eddy_vs_I]),
        eddy_html,
    ])

    main_panel = w.VBox(
        [header, summary_box, w.HTML("<hr>"), plot_buttons, w.HTML("<hr>"),
         eddy_box, w.HTML("<hr>"), out_plot],
        layout=w.Layout(width="70%", min_width="640px"),
    )

    diag_panel = w.VBox(
        [status, log.panel],
        layout=w.Layout(width="30%", min_width="300px"),
    )

    panel = w.HBox([main_panel, diag_panel], layout=w.Layout(width="100%"))
    return panel
