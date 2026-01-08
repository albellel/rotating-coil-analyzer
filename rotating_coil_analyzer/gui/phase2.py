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
    # QC approval workflow (Option B): preview QC actions, then apply before FFT
    qc_plan: Optional[dict] = None
    qc_plan_cfg: Optional[tuple] = None
    qc_source: Optional[str] = None
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
        magnet_order=getattr(segf, "magnet_order", None),
    )
    return segf2, rem


def _phase_zero_from_main_harmonic(H_ref, main_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-turn phase zero Φ_out consistent with the legacy C++ rotation.

    Legacy convention (MatlabAnalyzerRotCoil.cpp):
      - Let m = magnetOrder (main field order).
      - Compute SignalPhase = arg(C_abs[m-1]) (after internal scaling; scaling does not affect phase).
      - Apply a π-wrap into [-π/2, +π/2] to stabilize the choice.
      - Define Φ_out = SignalPhase / m.
      - Rotate each harmonic n by exp(-i n Φ_out).

    Here we replicate the same idea using the reference coefficient in H_ref at order m.
    Returns:
      phi_out: shape (n_turns,), the per-turn Φ_out
      bad: boolean mask (True where reference harmonic is tiny or non-finite and Φ_out was forced to 0)
    """
    m = int(main_order)
    if m <= 0:
        raise ValueError(f"main_order must be > 0, got {m}")

    orders = np.asarray(H_ref.orders, dtype=int)
    idx = np.where(orders == m)[0]
    if idx.size == 0:
        raise ValueError(f"Reference harmonic m={m} not available. Increase N_max.")
    j = int(idx[0])

    c_m = np.asarray(H_ref.coeff)[:, j]
    mag = np.abs(c_m)
    phi = np.angle(c_m)

    # Replicate the C++ phase wrap: keep main phase in [-π/2, +π/2] by shifting by π if needed.
    phi_wrapped = np.array(phi, copy=True)
    phi_wrapped[phi_wrapped > (np.pi / 2.0)] -= np.pi
    phi_wrapped[phi_wrapped < (-np.pi / 2.0)] += np.pi

    bad = (~np.isfinite(phi_wrapped)) | (~np.isfinite(mag)) | (mag < 1e-20)

    phi_out = phi_wrapped / float(m)
    if np.any(bad):
        phi_out = np.array(phi_out, copy=True)
        phi_out[bad] = 0.0
    return phi_out, bad


def _rotate_harmonics(H, phi_ref: np.ndarray) -> np.ndarray:
    orders = np.asarray(H.orders)
    C = np.array(H.coeff, copy=True)
    for j, n in enumerate(orders):
        C[:, j] = C[:, j] * np.exp(-1j * float(n) * phi_ref)
    return C


def _ba_from_rotated(C_rot: np.ndarray, orders: Sequence[int], prefix: str) -> pd.DataFrame:
    """Compute legacy-compatible (B/A) coefficients from rotated complex harmonics.

    With the internal convention C_n = FFT/Ns, the legacy magnet-style coefficient is M_n = 2*C_n for n>=1.
    After rotation, we report:
      B_n = Re(M_n) = 2*Re(C_n)
      A_n = Im(M_n) = 2*Im(C_n)

    Column naming uses B/A to match the legacy C++ outputs (bn/an).
    """
    out: Dict[str, Any] = {}
    orders = [int(x) for x in orders]
    for j, n in enumerate(orders):
        if n == 0:
            out[f"{prefix}B0"] = np.real(C_rot[:, j])
            out[f"{prefix}A0"] = 0.0 * np.real(C_rot[:, j])
        else:
            out[f"{prefix}B{n}"] = 2.0 * np.real(C_rot[:, j])
            out[f"{prefix}A{n}"] = 2.0 * np.imag(C_rot[:, j])
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

    main_order = w.BoundedIntText(
        value=1, min=1, max=50, step=1, description="Main m",
        layout=w.Layout(width="190px"),
    )

    integrate_to_flux = w.Checkbox(
        value=True, description="Integrate df→flux (C++ style)", indent=False, layout=w.Layout(width="230px")
    )
    drift_correction = w.Checkbox(value=False, description="Drift corr (dri)", indent=False, layout=w.Layout(width="150px"))

    require_finite_t = w.Checkbox(value=True, description="Drop turns with non-finite t", indent=False)

    append_log = w.Checkbox(value=False, description="Append output", indent=False, layout=w.Layout(width="140px"))
    status = w.HTML(value="<b>Status:</b> idle")

    btn_preview_qc = w.Button(
        description="Preview QC actions",
        button_style="warning",
        layout=w.Layout(width="190px"),
    )

    btn_apply_qc = w.Button(
        description="Apply QC + Compute FFT",
        button_style="primary",
        layout=w.Layout(width="220px"),
    )
    btn_apply_qc.disabled = True

    dd_channel = w.Dropdown(
        options=[("compensated (df_cmp)", "cmp"), ("absolute (df_abs)", "abs")],
        value="cmp",
        description="Channel",
        layout=w.Layout(width="320px"),
    )
    dd_harm = w.BoundedIntText(value=1, min=1, max=200, step=1, description="Harm n", layout=w.Layout(width="190px"))
    btn_plot_amp = w.Button(description="Plot |C_n| vs current", layout=w.Layout(width="180px"))

    dd_plateau = w.Dropdown(options=[], description="plateau_id", layout=w.Layout(width="220px"))
    btn_plot_ns = w.Button(description="Plot Normal/Skew (B/A) bars", layout=w.Layout(width="210px"))

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
            [btn_preview_qc, btn_apply_qc, btn_plot_amp, btn_plot_ns, btn_save_plot, btn_save_table, btn_show_head],
            False,
        )
        with out_log:
            print(msg)

    def _end_action(ok: bool = True, msg: str = "idle"):
        _set_enabled(
            [btn_preview_qc, btn_apply_qc, btn_plot_amp, btn_plot_ns, btn_save_plot, btn_save_table, btn_show_head],
            True,
        )
        _set_status(msg if ok else f"ERROR: {msg}")
        _refresh_qc_buttons()

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
    def _qc_cfg() -> tuple:
        # QC plan depends only on whether time is required to be finite/monotonic.
        return (bool(require_finite_t.value),)

    def _current_source() -> Optional[str]:
        segf_cur = get_segmentframe_callable()
        if segf_cur is None:
            return None
        return str(segf_cur.source_path)

    def _format_turn_list(idxs0: np.ndarray, *, max_show: int = 12) -> str:
        if idxs0.size == 0:
            return "(none)"
        idxs1 = (idxs0.astype(int) + 1).tolist()  # user-facing 1-based turn numbers
        if len(idxs1) > max_show:
            head = ", ".join(str(x) for x in idxs1[:max_show])
            return f"{head}, … (+{len(idxs1) - max_show} more)"
        return ", ".join(str(x) for x in idxs1)

    def _refresh_qc_buttons() -> None:
        src = _current_source()
        ok = (
            state.qc_plan is not None
            and state.qc_source == src
            and state.qc_plan_cfg == _qc_cfg()
        )
        btn_apply_qc.disabled = not ok

    def _preview_qc(_):
        try:
            _start_action("Previewing QC actions…")
            _init_plot_once()

            segf = get_segmentframe_callable()
            if segf is None:
                with out_log:
                    print("No SegmentFrame loaded yet. In Phase I, click 'Load segment' first.")
                state.qc_plan = None
                state.qc_plan_cfg = None
                state.qc_source = None
                _end_action(ok=False, msg="no segment loaded")
                return

            # Invalidate any prior FFT tables if the source changed.
            src = str(segf.source_path)
            if state.qc_source is not None and state.qc_source != src:
                state.df_out = None
                state.df_out_mean = None
                state.df_out_std = None
                state.df_ns = None
                state.mean_ns = None
                state.std_ns = None

            segf2, rem = _ensure_full_turns(segf)
            tb = split_into_turns(segf2)

            # Per-turn validity masks
            ok_abs = np.all(np.isfinite(tb.df_abs), axis=1)
            ok_cmp = np.all(np.isfinite(tb.df_cmp), axis=1)
            ok_I = np.all(np.isfinite(tb.I), axis=1)
            finite_t = np.all(np.isfinite(tb.t), axis=1)
            mono_t = finite_t & np.all(np.diff(tb.t, axis=1) > 0.0, axis=1)

            if require_finite_t.value:
                ok_t = mono_t
            else:
                ok_t = np.ones(tb.n_turns, dtype=bool)

            valid_turn = ok_abs & ok_cmp & ok_I & ok_t

            # Reason breakdown (always reported, even if require_finite_t=False)
            bad_abs = np.where(~ok_abs)[0]
            bad_cmp = np.where(~ok_cmp)[0]
            bad_I = np.where(~ok_I)[0]
            bad_t_nonfinite = np.where(~finite_t)[0]
            bad_t_nonmono = np.where(finite_t & ~np.all(np.diff(tb.t, axis=1) > 0.0, axis=1))[0]
            dropped = np.where(~valid_turn)[0]

            # Store QC plan (not yet applied) for the follow-up 'Apply QC + Compute FFT' step.
            state.qc_plan = {
                "segf_trimmed": segf2,
                "rem_samples": int(rem),
                "tb": tb,
                "valid_turn": valid_turn,
                "ok_abs": ok_abs,
                "ok_cmp": ok_cmp,
                "ok_I": ok_I,
                "finite_t": finite_t,
                "mono_t": mono_t,
            }
            state.qc_plan_cfg = _qc_cfg()
            state.qc_source = src

            # Display a user-facing QC preview summary.
            n_total = int(tb.n_turns)
            n_keep = int(np.sum(valid_turn))
            n_drop = int(n_total - n_keep)

            with out_log:
                print("QC PREVIEW (no changes applied yet)")
                print(f" - Source: {src}")
                if rem != 0:
                    print(f" - Tail remainder: {int(rem)} samples would be trimmed to reach full turns.")
                else:
                    print(" - Tail remainder: none (already full turns).")

                print(f" - Turns: total={n_total}, keep={n_keep}, drop={n_drop}")
                print(" - Drop rule set:")
                print(f"    * abs finite: required (bad turns: {bad_abs.size})")
                print(f"    * cmp finite: required (bad turns: {bad_cmp.size})")
                print(f"    * I   finite: required (bad turns: {bad_I.size})")
                if require_finite_t.value:
                    print(f"    * t finite + strictly increasing within turn: required (bad non-finite: {bad_t_nonfinite.size}, bad non-monotonic: {bad_t_nonmono.size})")
                else:
                    print(f"    * t finite/monotonic: NOT required for dropping (detected non-finite: {bad_t_nonfinite.size}, non-monotonic: {bad_t_nonmono.size})")

                print(" - Turn indices (1-based) by reason (first items):")
                print(f"    * abs non-finite: { _format_turn_list(bad_abs) }")
                print(f"    * cmp non-finite: { _format_turn_list(bad_cmp) }")
                print(f"    * I non-finite:   { _format_turn_list(bad_I) }")
                print(f"    * t non-finite:   { _format_turn_list(bad_t_nonfinite) }")
                print(f"    * t non-monotonic:{ _format_turn_list(bad_t_nonmono) }")
                print(f"    * total dropped:  { _format_turn_list(dropped) }")

                if getattr(tb, "plateau_id", None) is not None:
                    uniq = np.unique(tb.plateau_id)
                    print(" - Plateau summary (keep/total):")
                    for pid in uniq[:20]:
                        mask = (tb.plateau_id == pid)
                        kk = int(np.sum(valid_turn[mask]))
                        tt = int(np.sum(mask))
                        print(f"    * plateau_id={int(pid)}: {kk}/{tt}")
                    if uniq.size > 20:
                        print(f"    ... ({int(uniq.size - 20)} more plateaus)")

                print("ACTION REQUIRED: If you agree with these QC actions, click 'Apply QC + Compute FFT'.")

            _end_action(ok=True, msg="QC preview ready")
            _refresh_qc_buttons()

        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            state.qc_plan = None
            state.qc_plan_cfg = None
            state.qc_source = None
            _end_action(ok=False, msg=str(e))
            _refresh_qc_buttons()

    def _apply_qc_and_compute(_):
        try:
            _start_action("Applying QC + computing per-turn FFT…")
            _init_plot_once()

            if state.qc_plan is None or state.qc_source != _current_source() or state.qc_plan_cfg != _qc_cfg():
                with out_log:
                    print("No valid QC plan is available (or it is stale). Click 'Preview QC actions' first.")
                _end_action(ok=False, msg="QC not approved")
                _refresh_qc_buttons()
                return

            plan = state.qc_plan
            segf2 = plan["segf_trimmed"]
            rem = int(plan["rem_samples"])
            tb = plan["tb"]
            valid_turn = plan["valid_turn"]

            # Apply the approved plan into the Phase II state (Phase I data stays untouched).
            state.segf = segf2
            state.tb = tb
            state.valid_turn = valid_turn
            state.ok_abs = plan.get("ok_abs")
            state.ok_cmp = plan.get("ok_cmp")
            state.ok_t = plan.get("mono_t") if require_finite_t.value else np.ones(tb.n_turns, dtype=bool)

            if rem != 0:
                with out_log:
                    print(f"APPLY: trimming tail remainder={rem} samples to reach full turns.")

            n_total = int(tb.n_turns)
            n_keep = int(np.sum(valid_turn))
            n_drop = int(n_total - n_keep)
            with out_log:
                print(f"APPLY: turns keep={n_keep} / total={n_total} (drop={n_drop})")

            # Default main field order (magnetOrder) from Parameters.txt if available.
            if getattr(segf2, "magnet_order", None) is not None and int(segf2.magnet_order) > 0:
                main_order.value = int(segf2.magnet_order)
                with out_log:
                    print(f"Phase reference: using magnetOrder m={int(segf2.magnet_order)} from Parameters.txt")
            else:
                with out_log:
                    print(f"Phase reference: magnetOrder not available; using GUI value m={int(main_order.value)}")

            # Turn matrices for FFT (approved subset).
            abs_turns = tb.df_abs[valid_turn, :]
            cmp_turns = tb.df_cmp[valid_turn, :]
            I_turns = tb.I[valid_turn, :]

            if integrate_to_flux.value:
                if drift_correction.value:
                    # C++ option "dri": flux = cumsum(df - mean(df)) - mean(cumsum(df))
                    abs0 = abs_turns - np.mean(abs_turns, axis=1, keepdims=True)
                    cmp0 = cmp_turns - np.mean(cmp_turns, axis=1, keepdims=True)
                    flux_abs = np.cumsum(abs0, axis=1) - np.mean(np.cumsum(abs_turns, axis=1), axis=1, keepdims=True)
                    flux_cmp = np.cumsum(cmp0, axis=1) - np.mean(np.cumsum(cmp_turns, axis=1), axis=1, keepdims=True)
                    with out_log:
                        print("FFT input: flux = cumsum(df) with drift correction (dri)")
                else:
                    flux_abs = np.cumsum(abs_turns, axis=1)
                    flux_cmp = np.cumsum(cmp_turns, axis=1)
                    with out_log:
                        print("FFT input: flux = cumsum(df) (C++ style)")
                sig_abs = flux_abs
                sig_cmp = flux_cmp
            else:
                sig_abs = abs_turns
                sig_cmp = cmp_turns
                with out_log:
                    print("FFT input: raw df (no integration)")

            with out_log:
                print(f"Running FFT up to N_max={int(nmax.value)}…")
            H_abs = dft_per_turn(sig_abs, n_max=int(nmax.value))
            H_cmp = dft_per_turn(sig_cmp, n_max=int(nmax.value))
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

            m = int(main_order.value)
            phi_out, bad_phi = _phase_zero_from_main_harmonic(H_abs, m)
            if int(np.sum(bad_phi)):
                with out_log:
                    print(
                        f"WARNING: reference harmonic m={m} is tiny/non-finite for {int(np.sum(bad_phi))} turns; "
                        "using Φ_out=0 for those turns."
                    )

            C_abs_rot = _rotate_harmonics(H_abs, phi_out)
            C_cmp_rot = _rotate_harmonics(H_cmp, phi_out)

            df_ba_abs = _ba_from_rotated(C_abs_rot, H_abs.orders, prefix="abs_")
            df_ba_cmp = _ba_from_rotated(C_cmp_rot, H_cmp.orders, prefix="cmp_")

            df_ns = pd.concat(
                [
                    df_out[[c for c in ["plateau_id", "I_mean", "plateau_I_hint", "plateau_step"] if c in df_out.columns]].reset_index(drop=True),
                    df_ba_abs.reset_index(drop=True),
                    df_ba_cmp.reset_index(drop=True),
                ],
                axis=1,
            )
            state.df_ns = df_ns

            if "plateau_id" in df_ns.columns:
                g2 = df_ns.groupby("plateau_id", sort=True)
                state.mean_ns = g2.mean(numeric_only=True)
                state.std_ns = g2.std(numeric_only=True)
            else:
                state.mean_ns = df_ns.mean(numeric_only=True).to_frame().T
                state.std_ns = df_ns.std(numeric_only=True).to_frame().T

            with out_log:
                print("DONE: FFT computed.")
                print("Tip: you can change N_max or plotting options and recompute without redoing QC if the QC plan remains valid.")

            _end_action(ok=True, msg="FFT computed")
            _refresh_qc_buttons()

        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))
            _refresh_qc_buttons()
    def _plot_amp(_):
        try:
            _start_action("Updating amplitude plot…")
            _init_plot_once()

            if state.df_out is None:
                with out_log:
                    print("Nothing computed yet. Click 'Apply QC + Compute FFT' first (after 'Preview QC actions').")
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
                    print("Nothing computed yet. Click 'Apply QC + Compute FFT' first (after 'Preview QC actions').")
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
            B_vals = np.array([row.get(f"cmp_B{k}", np.nan) for k in n])
            A_vals = np.array([row.get(f"cmp_A{k}", np.nan) for k in n])

            wbar = 0.4
            ax.bar(n - wbar / 2, B_vals, width=wbar, label="Normal B (cos)")
            ax.bar(n + wbar / 2, A_vals, width=wbar, label="Skew A (sin)")
            ax.set_xlabel("harmonic order n")
            ax.set_ylabel(f"cmp component (phase-referenced to main order m={int(main_order.value)})")
            ax.set_title(f"Normal/Skew (B/A) bars (cmp), plateau_id={pid}")
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
    btn_preview_qc.on_click(_preview_qc)
    btn_apply_qc.on_click(_apply_qc_and_compute)
    btn_plot_amp.on_click(_plot_amp)
    btn_plot_ns.on_click(_plot_ns)
    btn_save_plot.on_click(_save_plot)
    btn_save_table.on_click(_save_table)
    btn_show_head.on_click(_show_head)

    # Init plot once so “Plot” is always one canvas
    _init_plot_once()

    # Layout
    row1 = w.HBox([nmax, main_order, integrate_to_flux, drift_correction, require_finite_t, btn_preview_qc, btn_apply_qc, append_log])
    row2 = w.HBox([dd_channel, dd_harm, btn_plot_amp, dd_plateau, btn_plot_ns])
    row3 = w.HBox([save_plot_fmt, btn_save_plot, table_choice, btn_save_table, btn_show_head])

    plot_box = w.VBox([w.HTML("<b>Plot</b>"), plot_slot])
    table_box = w.VBox([w.HTML("<b>Table</b>"), out_table])
    log_box = w.VBox([w.HTML("<b>Log</b>"), out_log])

    return w.VBox([row1, row2, row3, status, plot_box, table_box, log_box])
