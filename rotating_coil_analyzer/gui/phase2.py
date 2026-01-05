from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Dict, Iterable

import numpy as np
import pandas as pd
import ipywidgets as w
import matplotlib.pyplot as plt

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.analysis.fourier import dft_per_turn


@dataclass
class Phase2State:
    segf: Optional[SegmentFrame] = None
    tb: Any = None
    valid_turn: Optional[np.ndarray] = None
    ok_t: Optional[np.ndarray] = None
    ok_abs: Optional[np.ndarray] = None
    ok_cmp: Optional[np.ndarray] = None

    H_abs: Any = None
    H_cmp: Any = None

    # Tables
    df_out: Optional[pd.DataFrame] = None
    df_out_mean: Optional[pd.DataFrame] = None
    df_out_std: Optional[pd.DataFrame] = None

    df_ns: Optional[pd.DataFrame] = None
    mean_ns: Optional[pd.DataFrame] = None
    std_ns: Optional[pd.DataFrame] = None

    # One persistent plot
    fig: Optional[Any] = None
    ax: Optional[Any] = None


def _saveas_dialog(initialfile: str, defaultextension: str, filetypes: list[tuple[str, str]]) -> Optional[str]:
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

        path = filedialog.asksaveasfilename(
            title="Save file",
            initialfile=initialfile,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        return str(path) if path else None
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


def _ensure_full_turns(segf: SegmentFrame) -> tuple[SegmentFrame, int]:
    Ns = int(segf.samples_per_turn)
    n = len(segf.df)
    rem = n % Ns
    if rem == 0:
        return segf, 0

    n_keep = n - rem
    df2 = segf.df.iloc[:n_keep, :].reset_index(drop=True)
    segf2 = SegmentFrame(
        source_path=segf.source_path,
        run_id=segf.run_id,
        segment=segf.segment,
        samples_per_turn=Ns,
        n_turns=n_keep // Ns,
        df=df2,
        warnings=tuple(segf.warnings) + (f"analysis trim: removed tail remainder={rem} samples",),
        aperture_id=segf.aperture_id,
    )
    return segf2, rem


def _phase_reference_from(H_ref, n_ref: int = 1) -> np.ndarray:
    orders = np.asarray(H_ref.orders)
    j = int(np.where(orders == n_ref)[0][0])
    return np.angle(H_ref.coeff[:, j])


def _rotate_harmonics(H, phi_ref: np.ndarray) -> np.ndarray:
    orders = np.asarray(H.orders)
    C = np.array(H.coeff, copy=True)
    for j, n in enumerate(orders):
        C[:, j] = C[:, j] * np.exp(-1j * float(n) * phi_ref)
    return C


def _normal_skew_from_rotated(C_rot: np.ndarray, orders: Sequence[int], prefix: str) -> pd.DataFrame:
    out: Dict[str, Any] = {}
    orders = [int(x) for x in orders]
    for j, n in enumerate(orders):
        if n == 0:
            out[f"{prefix}N0"] = np.real(C_rot[:, j])
            out[f"{prefix}S0"] = 0.0 * np.real(C_rot[:, j])
        else:
            out[f"{prefix}N{n}"] = 2.0 * np.real(C_rot[:, j])
            out[f"{prefix}S{n}"] = -2.0 * np.imag(C_rot[:, j])
    return pd.DataFrame(out)


def build_phase2_panel(get_segmentframe_callable, *, default_n_max: int = 20) -> w.Widget:
    state = Phase2State()

    # ---------------------------
    # Controls (with explicit widths so labels don't truncate)
    # ---------------------------
    nmax = w.BoundedIntText(
        value=default_n_max, min=1, max=200, step=1, description="N_max",
        layout=w.Layout(width="190px"),
    )
    require_finite_t = w.Checkbox(value=True, description="Drop turns with non-finite t", indent=False)

    append_log = w.Checkbox(value=False, description="Append output", indent=False, layout=w.Layout(width="140px"))
    status = w.HTML(value="<b>Status:</b> idle")

    btn_compute = w.Button(
        description="Compute FFT (per turn)",
        button_style="primary",
        layout=w.Layout(width="210px"),
    )

    dd_channel = w.Dropdown(
        options=[("compensated (df_cmp)", "cmp"), ("absolute (df_abs)", "abs")],
        value="cmp",
        description="Channel",
        layout=w.Layout(width="320px"),
    )
    dd_harm = w.BoundedIntText(value=1, min=1, max=200, step=1, description="Harm n", layout=w.Layout(width="190px"))
    btn_plot_amp = w.Button(description="Plot |C_n| vs current", layout=w.Layout(width="180px"))

    dd_plateau = w.Dropdown(options=[], description="plateau_id", layout=w.Layout(width="220px"))
    btn_plot_ns = w.Button(description="Plot Normal/Skew bars", layout=w.Layout(width="190px"))

    save_plot_fmt = w.Dropdown(
        options=[("SVG (vector)", "svg"), ("PDF (vector)", "pdf")],
        value="svg",
        description="Plot",
        layout=w.Layout(width="240px"),
    )
    btn_save_plot = w.Button(description="Save plot…", layout=w.Layout(width="120px"))

    table_choice = w.Dropdown(
        options=[
            ("Per-turn table (df_out)", "df_out"),
            ("Per-plateau mean (df_out_mean)", "df_out_mean"),
            ("Per-plateau std (df_out_std)", "df_out_std"),
            ("Per-turn Normal/Skew (df_ns)", "df_ns"),
            ("Per-plateau mean Normal/Skew (mean_ns)", "mean_ns"),
            ("Per-plateau std Normal/Skew (std_ns)", "std_ns"),
        ],
        value="df_out",
        description="Table",
        layout=w.Layout(width="420px"),
    )
    btn_save_table = w.Button(description="Save table (CSV)…", layout=w.Layout(width="150px"))
    btn_show_head = w.Button(description="Show table head", layout=w.Layout(width="140px"))

    # Outputs
    out_log = w.Output()
    out_table = w.Output()  # dedicated table pane (prevents duplicated prints)

    # Plot slot (single persistent canvas)
    plot_slot = w.Box(layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%"))

    # ---------------------------
    # Helpers
    # ---------------------------
    def _set_status(msg: str):
        status.value = f"<b>Status:</b> {msg}"

    def _set_enabled(btns: Iterable[w.Button], enabled: bool):
        for b in btns:
            b.disabled = not enabled

    def _start_action(msg: str):
        if not append_log.value:
            out_log.clear_output(wait=True)
        _set_status(msg)
        _set_enabled(
            [btn_compute, btn_plot_amp, btn_plot_ns, btn_save_plot, btn_save_table, btn_show_head],
            False,
        )
        with out_log:
            print(msg)

    def _end_action(ok: bool = True, msg: str = "idle"):
        _set_enabled(
            [btn_compute, btn_plot_amp, btn_plot_ns, btn_save_plot, btn_save_table, btn_show_head],
            True,
        )
        _set_status(msg if ok else f"ERROR: {msg}")

    def _refresh_plateau_dropdown(df_out: pd.DataFrame):
        if "plateau_id" in df_out.columns:
            pids = sorted(pd.unique(df_out["plateau_id"]))
            dd_plateau.options = [(str(p), float(p)) for p in pids]
            dd_plateau.value = float(pids[0]) if pids else None
        else:
            dd_plateau.options = []
            dd_plateau.value = None

    def _init_plot_once():
        if state.fig is not None and state.ax is not None:
            return

        was_interactive = plt.isinteractive()
        try:
            plt.ioff()
            fig, ax = plt.subplots()
        finally:
            if was_interactive:
                plt.ion()

        ax.set_title("Phase II plot")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        state.fig, state.ax = fig, ax

        if isinstance(fig.canvas, w.Widget):
            plot_slot.children = (fig.canvas,)
        else:
            plot_slot.children = (w.HTML("Non-interactive backend. In the first cell run: %matplotlib widget (ipympl)."),)

    def _redraw():
        if state.fig is None:
            return
        if hasattr(state.fig.canvas, "draw_idle"):
            state.fig.canvas.draw_idle()
        else:
            state.fig.canvas.draw()

    def _get_table(key: str) -> Optional[pd.DataFrame]:
        return getattr(state, key, None)

    # ---------------------------
    # Actions
    # ---------------------------
    def _compute(_):
        try:
            _start_action("Computing turn-QC + per-turn FFT…")
            _init_plot_once()

            segf = get_segmentframe_callable()
            if segf is None:
                with out_log:
                    print("No SegmentFrame loaded yet. In Phase I, click 'Load segment' first.")
                _end_action(ok=False, msg="no segment loaded")
                return

            segf2, rem = _ensure_full_turns(segf)
            if rem != 0:
                with out_log:
                    print(f"INCOMPLETE TURN DETECTED: trimmed tail remainder={rem} samples before analysis.")
            else:
                with out_log:
                    print(f"Full-turn length OK: total samples={len(segf2.df)} is n_turns*Ns ({segf2.n_turns}*{segf2.samples_per_turn}).")

            state.segf = segf2

            tb = split_into_turns(segf2)
            state.tb = tb

            ok_abs = np.all(np.isfinite(tb.df_abs), axis=1)
            ok_cmp = np.all(np.isfinite(tb.df_cmp), axis=1)
            ok_t = np.all(np.isfinite(tb.t), axis=1) if require_finite_t.value else np.ones(tb.df_abs.shape[0], dtype=bool)
            valid_turn = ok_t & ok_abs & ok_cmp

            state.ok_t, state.ok_abs, state.ok_cmp = ok_t, ok_abs, ok_cmp
            state.valid_turn = valid_turn

            with out_log:
                print("TURN QC REPORT")
                print("  total turns:", int(tb.df_abs.shape[0]))
                print("  keep turns :", int(np.sum(valid_turn)))
                print("  drop turns :", int(np.sum(~valid_turn)))
                print("   - drop (non-finite t)      :", int(np.sum(~ok_t)))
                print("   - drop (finite t, bad abs) :", int(np.sum(ok_t & ~ok_abs)))
                print("   - drop (finite t+abs, bad cmp):", int(np.sum(ok_t & ok_abs & ~ok_cmp)))

            abs_turns = tb.df_abs[valid_turn, :]
            cmp_turns = tb.df_cmp[valid_turn, :]
            I_turns = tb.I[valid_turn, :]

            with out_log:
                print(f"Running FFT up to N_max={int(nmax.value)}…")
            H_abs = dft_per_turn(abs_turns, n_max=int(nmax.value))
            H_cmp = dft_per_turn(cmp_turns, n_max=int(nmax.value))
            state.H_abs, state.H_cmp = H_abs, H_cmp

            turn_idx = np.arange(H_cmp.coeff.shape[0], dtype=int)

            def coeff_table(H, prefix: str):
                C = H.coeff
                orders = [int(x) for x in H.orders]
                cols = {}
                for jj, nn in enumerate(orders):
                    cols[f"{prefix}A{nn}"] = np.abs(C[:, jj])
                    cols[f"{prefix}phi{nn}"] = np.angle(C[:, jj])
                return pd.DataFrame(cols, index=turn_idx)

            df_h = pd.concat([coeff_table(H_abs, "abs_"), coeff_table(H_cmp, "cmp_")], axis=1)

            meta = {"I_mean": np.mean(I_turns, axis=1)}
            if getattr(tb, "plateau_id", None) is not None:
                meta["plateau_id"] = tb.plateau_id[valid_turn]
            if getattr(tb, "plateau_step", None) is not None:
                meta["plateau_step"] = tb.plateau_step[valid_turn]
            if getattr(tb, "plateau_I_hint", None) is not None:
                meta["plateau_I_hint"] = tb.plateau_I_hint[valid_turn]

            df_out = pd.concat([pd.DataFrame(meta, index=turn_idx), df_h], axis=1)
            state.df_out = df_out
            _refresh_plateau_dropdown(df_out)

            if "plateau_id" in df_out.columns:
                g = df_out.groupby("plateau_id", sort=True)
                state.df_out_mean = g.mean(numeric_only=True)
                state.df_out_std = g.std(numeric_only=True)
            else:
                state.df_out_mean = df_out.mean(numeric_only=True).to_frame().T
                state.df_out_std = df_out.std(numeric_only=True).to_frame().T

            phi_ref = _phase_reference_from(H_cmp, n_ref=1)
            C_cmp_rot = _rotate_harmonics(H_cmp, phi_ref)
            df_ns_cmp = _normal_skew_from_rotated(C_cmp_rot, H_cmp.orders, prefix="cmp_")

            df_ns = pd.concat(
                [
                    df_out[[c for c in ["plateau_id", "I_mean", "plateau_I_hint", "plateau_step"] if c in df_out.columns]].reset_index(drop=True),
                    df_ns_cmp.reset_index(drop=True),
                ],
                axis=1,
            )
            state.df_ns = df_ns

            if "plateau_id" in df_ns.columns:
                gn = df_ns.groupby("plateau_id", sort=True)
                state.mean_ns = gn.mean(numeric_only=True)
                state.std_ns = gn.std(numeric_only=True)
            else:
                state.mean_ns = df_ns.mean(numeric_only=True).to_frame().T
                state.std_ns = df_ns.std(numeric_only=True).to_frame().T

            with out_log:
                print("Computed tables ready.")

            # Clear table pane on compute (keeps UI tidy)
            out_table.clear_output(wait=True)

            _end_action(ok=True, msg="FFT ready")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _plot_amp(_):
        try:
            _start_action("Updating amplitude plot…")
            _init_plot_once()

            if state.df_out is None:
                with out_log:
                    print("Nothing computed yet. Click 'Compute FFT (per turn)' first.")
                _end_action(ok=False, msg="no FFT yet")
                return

            df = state.df_out
            ch = dd_channel.value
            nn = int(dd_harm.value)
            col = f"{ch}_A{nn}"
            if col not in df.columns:
                with out_log:
                    print(f"Missing column {col}. Increase N_max or choose a smaller n.")
                _end_action(ok=False, msg="missing harmonic")
                return

            ax = state.ax
            ax.clear()

            x_turn = df["I_mean"].to_numpy()
            y_turn = df[col].to_numpy()
            ax.plot(x_turn, y_turn, ".", label="per-turn")

            if "plateau_id" in df.columns:
                g = df.groupby("plateau_id", sort=True)
                mean = g.mean(numeric_only=True)
                std = g.std(numeric_only=True)
                x_plateau = mean["I_mean"].to_numpy()
                y_mean = mean[col].to_numpy()
                y_std = std[col].to_numpy()
                ax.errorbar(x_plateau, y_mean, yerr=y_std, fmt="o", label="per-plateau mean ± std")

            ax.set_xlabel("current (A)")
            ax.set_ylabel(f"{ch} |C_{nn}|")
            ax.set_title(f"Harmonic amplitude vs current ({ch}, n={nn})")
            ax.legend()

            _redraw()
            _end_action(ok=True, msg="plot ready")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _plot_ns(_):
        try:
            _start_action("Updating Normal/Skew bars…")
            _init_plot_once()

            if state.mean_ns is None:
                with out_log:
                    print("Nothing computed yet. Click 'Compute FFT (per turn)' first.")
                _end_action(ok=False, msg="no FFT yet")
                return

            pid = dd_plateau.value
            if pid is None or pid not in state.mean_ns.index:
                with out_log:
                    print("Select a valid plateau_id first.")
                _end_action(ok=False, msg="invalid plateau_id")
                return

            ax = state.ax
            ax.clear()

            row = state.mean_ns.loc[pid]
            nmax_eff = int(nmax.value)
            n = np.arange(1, nmax_eff + 1)
            N_vals = np.array([row.get(f"cmp_N{k}", np.nan) for k in n])
            S_vals = np.array([row.get(f"cmp_S{k}", np.nan) for k in n])

            wbar = 0.4
            ax.bar(n - wbar / 2, N_vals, width=wbar, label="Normal (cos)")
            ax.bar(n + wbar / 2, S_vals, width=wbar, label="Skew (sin)")
            ax.set_xlabel("harmonic order n")
            ax.set_ylabel("cmp component (phase-referenced to n=1)")
            ax.set_title(f"Normal/Skew bars (cmp), plateau_id={pid}")
            ax.legend()

            _redraw()
            _end_action(ok=True, msg="bars ready")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _save_plot(_):
        try:
            _start_action("Saving plot…")
            _init_plot_once()

            if state.fig is None:
                with out_log:
                    print("No plot exists.")
                _end_action(ok=False, msg="no plot")
                return

            fmt = str(save_plot_fmt.value).lower().strip()
            if fmt not in ("svg", "pdf"):
                with out_log:
                    print("Unsupported format:", fmt)
                _end_action(ok=False, msg="unsupported format")
                return

            if fmt == "svg":
                path = _saveas_dialog("phase2_plot.svg", ".svg", [("SVG (vector)", "*.svg"), ("All files", "*.*")])
            else:
                path = _saveas_dialog("phase2_plot.pdf", ".pdf", [("PDF (vector)", "*.pdf"), ("All files", "*.*")])

            if not path:
                with out_log:
                    print("Save cancelled.")
                _end_action(ok=True, msg="idle")
                return

            state.fig.savefig(path)
            with out_log:
                print("Saved:", path)
            _end_action(ok=True, msg="plot saved")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _save_table(_):
        try:
            _start_action("Saving table (CSV)…")
            key = str(table_choice.value)
            df = _get_table(key)
            if df is None:
                with out_log:
                    print(f"Table '{key}' is not available yet. Compute FFT first.")
                _end_action(ok=False, msg="no table")
                return

            path = _saveas_dialog(f"{key}.csv", ".csv", [("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                with out_log:
                    print("Save cancelled.")
                _end_action(ok=True, msg="idle")
                return

            df.to_csv(path)
            with out_log:
                print("Saved:", path)
            _end_action(ok=True, msg="table saved")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _show_head(_):
        """
        IMPORTANT: write into a dedicated Output pane (out_table), not the shared log,
        and always clear first. This prevents the “printed 4 times” behavior in VS Code.
        """
        try:
            _start_action("Showing table head…")
            key = str(table_choice.value)
            df = _get_table(key)
            if df is None:
                with out_log:
                    print(f"Table '{key}' is not available yet. Compute FFT first.")
                _end_action(ok=False, msg="no table")
                return

            out_table.clear_output(wait=True)
            with out_table:
                print(f"Head of {key}:")
                print(df.head(15))

            _end_action(ok=True, msg="shown")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    # Wire callbacks
    btn_compute.on_click(_compute)
    btn_plot_amp.on_click(_plot_amp)
    btn_plot_ns.on_click(_plot_ns)
    btn_save_plot.on_click(_save_plot)
    btn_save_table.on_click(_save_table)
    btn_show_head.on_click(_show_head)

    # Init plot once so “Plot” is always one canvas
    _init_plot_once()

    # Layout
    row1 = w.HBox([nmax, require_finite_t, btn_compute, append_log])
    row2 = w.HBox([dd_channel, dd_harm, btn_plot_amp, dd_plateau, btn_plot_ns])
    row3 = w.HBox([save_plot_fmt, btn_save_plot, table_choice, btn_save_table, btn_show_head])

    plot_box = w.VBox([w.HTML("<b>Plot</b>"), plot_slot])
    table_box = w.VBox([w.HTML("<b>Table</b>"), out_table])
    log_box = w.VBox([w.HTML("<b>Log</b>"), out_log])

    return w.VBox([row1, row2, row3, status, plot_box, table_box, log_box])
