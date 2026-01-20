from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Dict

import time
import html

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
import matplotlib.pyplot as plt

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.analysis.fourier import dft_per_turn
from rotating_coil_analyzer.analysis.preprocess import (
    apply_di_dt_to_channels,
    integrate_to_flux as integrate_turns_to_flux,
    provenance_columns,
    format_preproc_tag,
    append_tag_to_path,
)


# Keep a single active Phase II panel per kernel (defensive: prevents stacked live instances).
_ACTIVE_PHASE2_PANEL: Optional[w.Widget] = None


@dataclass
class Phase2State:
    segf: Optional[SegmentFrame] = None
    tb: Any = None
    valid_turn: Optional[np.ndarray] = None

    # Two-step workflow: preview cuts, then apply before computing harmonics
    dq_plan: Optional[dict] = None
    dq_plan_cfg: Optional[tuple] = None
    dq_source: Optional[str] = None

    # FFT results
    H_abs: Any = None
    H_cmp: Any = None

    # Tables
    df_out: Optional[pd.DataFrame] = None           # per turn (amplitude/phase)
    df_out_mean: Optional[pd.DataFrame] = None      # per plateau mean
    df_out_std: Optional[pd.DataFrame] = None       # per plateau std

    df_ba: Optional[pd.DataFrame] = None            # normal/skew per turn (phase referenced)
    mean_ba: Optional[pd.DataFrame] = None          # normal/skew per plateau mean
    std_ba: Optional[pd.DataFrame] = None           # normal/skew per plateau std

    # Plateau mapping for UI (plateau_id -> mean current)
    plateau_current_map: Optional[Dict[int, float]] = None

    # Persistent plot
    fig: Optional[Any] = None
    ax: Optional[Any] = None

    # Prevent accidental re-entrancy
    busy: bool = False

    # Debounce repeated clicks (defensive)
    last_action_key: Optional[str] = None
    last_action_t: float = 0.0


def _saveas_dialog(
    *,
    initialfile: str,
    defaultextension: str,
    filetypes: list[tuple[str, str]],
    title: str = "Save file",
) -> Optional[str]:
    """Open a native Save-As dialog (best effort). Returns None if tkinter is unavailable."""
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
            title=title,
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
    """Trim tail samples so that len(df) == n_turns * samples_per_turn.

    Removes partial tail data (cannot form a full turn). Does not modify time values.
    """
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


def _wrap_arg_to_pm_pi_over_2(phi: np.ndarray) -> np.ndarray:
    """Wrap angles into [-pi/2, +pi/2] by adding/subtracting pi (legacy convention)."""
    out = np.asarray(phi, dtype=float).copy()
    out[out > (np.pi / 2.0)] -= np.pi
    out[out < (-np.pi / 2.0)] += np.pi
    return out


def _phase_zero_from_main_harmonic(H_ref, main_order: int, *, eps: float = 1e-20) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-turn phase zero Φ_out from the main field order m (legacy compatible).

    Using stored coefficients C_n (FFT/Ns):

      Φ_out = wrap(arg(C_m)) / m

    Then each harmonic n is rotated by exp(-i*n*Φ_out).
    """
    m = int(main_order)
    if m <= 0:
        raise ValueError(f"Main field order must be > 0, got {m}")

    orders = np.asarray(H_ref.orders, dtype=int)
    idx = np.where(orders == m)[0]
    if idx.size == 0:
        raise ValueError(f"Main field order m={m} is not available. Increase the maximum harmonic order.")
    j = int(idx[0])

    c_m = np.asarray(H_ref.coeff[:, j], dtype=complex)
    mag = np.abs(c_m)
    arg = np.angle(c_m)

    bad = (~np.isfinite(mag)) | (mag < eps) | (~np.isfinite(arg))

    arg_wrapped = _wrap_arg_to_pm_pi_over_2(arg)
    phi_out = arg_wrapped / float(m)
    if np.any(bad):
        phi_out = np.array(phi_out, copy=True)
        phi_out[bad] = 0.0
    return phi_out, bad


def _rotate_harmonics(H, phi_out: np.ndarray) -> np.ndarray:
    """Rotate harmonics by exp(-i*n*Φ_out) per turn."""
    C = np.array(H.coeff, dtype=complex, copy=True)
    orders = np.asarray(H.orders, dtype=int)
    for j, n in enumerate(orders):
        C[:, j] = C[:, j] * np.exp(-1j * float(n) * phi_out)
    return C


def _ba_from_rotated(C_rot: np.ndarray, orders: Sequence[int], prefix: str) -> pd.DataFrame:
    """Compute legacy-compatible normal/skew (B/A) from rotated complex harmonics.

    Internal convention: C_n = FFT/Ns.
    Legacy magnet convention: M_n = 2*C_n for n>=1.

      normal_Bn = Re(M_n),  skew_An = Im(M_n)

    Important:
    ---------
    The DC component (n=0) is treated as diagnostic only and is excluded from
    the main harmonic outputs.
    """
    cols: Dict[str, np.ndarray] = {}
    for j, n in enumerate([int(x) for x in orders]):
        if n == 0:
            continue
        M = 2.0 * C_rot[:, j]
        cols[f"{prefix}normal_B{n}"] = np.real(M)
        cols[f"{prefix}skew_A{n}"] = np.imag(M)
    return pd.DataFrame(cols)


def _pick_current_column_name(df: pd.DataFrame) -> str:
    """Backward-compatible current column lookup."""
    if "mean_current_A" in df.columns:
        return "mean_current_A"
    if "I_mean" in df.columns:
        return "I_mean"
    raise KeyError("No mean-current column found (expected 'mean_current_A' or legacy 'I_mean').")


def _wrap_title(text: str, width: int = 44) -> str:
    """Wrap a plot title by inserting newlines (matplotlib does not auto-wrap titles reliably)."""
    s = " ".join(str(text).split())
    if len(s) <= width:
        return s
    words = s.split(" ")
    lines: list[str] = []
    cur: list[str] = []
    ncur = 0
    for w0 in words:
        add = len(w0) + (1 if cur else 0)
        if ncur + add > width and cur:
            lines.append(" ".join(cur))
            cur = [w0]
            ncur = len(w0)
        else:
            cur.append(w0)
            ncur += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _clear_button_handlers(btn: w.Button) -> None:
    """Defensive: remove any accumulated click handlers (avoids duplicated logs if widgets stack)."""
    try:
        btn._click_handlers.callbacks.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def build_phase2_panel(get_segmentframe_callable, *, default_n_max: int = 20) -> w.Widget:
    global _ACTIVE_PHASE2_PANEL

    # Close any previous Phase II panel (defensive against stacked live instances).
    if _ACTIVE_PHASE2_PANEL is not None:
        try:
            _ACTIVE_PHASE2_PANEL.close()
        except Exception:
            pass
        _ACTIVE_PHASE2_PANEL = None

    state = Phase2State()

    STYLE_WIDE = {"description_width": "190px"}
    STYLE_MED = {"description_width": "150px"}

    # ---------------------------
    # Step 1 — preview/apply
    # ---------------------------
    max_harm = w.BoundedIntText(
        value=default_n_max, min=1, max=200, step=1,
        description="Maximum harmonic order",
        layout=w.Layout(width="420px"),
        style=STYLE_WIDE,
    )
    main_order = w.BoundedIntText(
        value=1, min=1, max=50, step=1,
        description="Main field order",
        layout=w.Layout(width="320px"),
        style=STYLE_MED,
    )

    integrate_to_flux = w.Checkbox(
        value=True,
        description="Integrate differential signal to flux (legacy convention)",
        indent=False,
        layout=w.Layout(width="650px"),
    )
    drift_correction = w.Checkbox(
        value=False,
        description="Apply drift correction (recommended only if needed)",
        indent=False,
        layout=w.Layout(width="650px"),
    )


    di_dt_correction = w.Checkbox(
        value=False,
        description="Apply di/dt correction (legacy current-ramp correction)",
        indent=False,
        layout=w.Layout(width="650px"),
    )

    drift_mode = w.Dropdown(
        options=[
            ("Legacy (C++) — uniform Δt", "legacy"),
            ("Bottura/Pentella — Δt-weighted", "weighted"),
        ],
        value="legacy",
        description="Drift mode",
        layout=w.Layout(width="420px"),
        style=STYLE_MED,
        disabled=True,
    )
    require_valid_time = w.Checkbox(
        value=True,
        description="Require valid time (finite and strictly increasing within each turn)",
        indent=False,
        layout=w.Layout(width="750px"),
    )

    append_log = w.Checkbox(value=False, description="Append output", indent=False, layout=w.Layout(width="160px"))

    btn_preview_dq = w.Button(
        description="Preview data-quality cuts",
        button_style="warning",
        layout=w.Layout(width="300px", height="46px"),
    )
    btn_apply_and_compute = w.Button(
        description="Apply cuts and compute harmonics (FFT)",
        button_style="primary",
        layout=w.Layout(width="420px", height="46px"),
    )
    btn_apply_and_compute.disabled = True

    help_text = w.HTML(
        value=(
            "<div style='color:#444; line-height:1.35;'>"
            "<b>Signal preparation for the spectrum</b><br>"
            "<ul style='margin-top:6px;'>"
            "<li><b>Integrate differential signal to flux</b>: compute the spectrum on "
            "<i>flux</i> (cumulative sum of the measured differential signal over a turn). "
            "This matches the legacy analyzer and is the default.</li>"
            "<li><b>Drift correction</b>: removes per-turn offset and recenters the integrated flux "
            "to reduce spurious low-order leakage. Use only if you observe obvious drift or a large baseline.</li>"
            "<li><b>di/dt correction</b>: applies the legacy current-ramp correction on the incremental signal "
            "before integration/FFT when the current is ramping (rule: dI/dt &gt; 0.1 A/s and mean(I) &gt; 10 A).</li>"
            "</ul>"
            "</div>"
        )
    )

    def _refresh_drift_mode_enabled(*_):
        # Drift mode is meaningful only when integrating to flux AND drift correction is enabled.
        drift_mode.disabled = not (bool(integrate_to_flux.value) and bool(drift_correction.value))

    integrate_to_flux.observe(_refresh_drift_mode_enabled, names="value")
    drift_correction.observe(_refresh_drift_mode_enabled, names="value")
    _refresh_drift_mode_enabled()

    # ---------------------------
    # View 1 — amplitude versus current
    # ---------------------------
    dd_channel = w.Dropdown(
        options=[("Compensated channel", "cmp"), ("Absolute channel", "abs")],
        value="cmp",
        description="Signal channel",
        layout=w.Layout(width="420px"),
        style=STYLE_MED,
    )
    harm_order = w.BoundedIntText(
        value=1, min=1, max=200, step=1,
        description="Harmonic order",
        layout=w.Layout(width="300px"),
        style=STYLE_MED,
    )
    btn_plot_amp = w.Button(
        description="Plot amplitude versus current",
        layout=w.Layout(width="360px", height="46px"),
    )

    # ---------------------------
    # View 2 — normal/skew versus harmonic order
    # ---------------------------
    dd_plateau = w.Dropdown(
        options=[],
        value=None,
        description="Plateau selection",
        layout=w.Layout(width="720px"),
        style={"description_width": "190px"},
        disabled=True,
    )
    plateau_info = w.HTML(value="<div style='color:#666;'>No plateau information loaded.</div>")

    hide_main_in_ns = w.Checkbox(
        value=True,
        description="Hide main field order in this plot (recommended to view smaller components)",
        indent=False,
        layout=w.Layout(width="750px"),
    )

    btn_plot_ns = w.Button(
        description="Plot normal and skew versus harmonic order",
        layout=w.Layout(width="420px", height="46px"),
        disabled=True,
    )

    # ---------------------------
    # Export
    # ---------------------------
    save_plot_fmt = w.Dropdown(
        options=[("SVG", "svg"), ("PDF", "pdf")],
        value="svg",
        description="Plot format",
        layout=w.Layout(width="260px"),
        style=STYLE_MED,
    )
    btn_save_plot = w.Button(description="Save plot…", layout=w.Layout(width="200px", height="46px"))

    table_choice = w.Dropdown(
        options=[
            ("Per turn (amplitude and phase)", "df_out"),
            ("Per plateau (mean)", "df_out_mean"),
            ("Per plateau (standard deviation)", "df_out_std"),
            ("Normal/skew per turn (phase referenced)", "df_ba"),
            ("Normal/skew per plateau (mean)", "mean_ba"),
            ("Normal/skew per plateau (standard deviation)", "std_ba"),
        ],
        value="df_out",
        description="Data table",
        layout=w.Layout(width="520px"),
        style=STYLE_MED,
    )
    btn_save_table = w.Button(description="Save table…", layout=w.Layout(width="200px", height="46px"))
    btn_show_head = w.Button(description="Show first rows", layout=w.Layout(width="200px", height="46px"))

    status = w.HTML(value="<b>Status:</b> idle")

    log = HtmlLog(height_px=220)
    out_log = log.output_proxy()
    table_html = w.HTML(value="<div style='color:#666;'>Table is empty.</div>")

    plot_slot = w.Box(layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%"))

    # ---------------------------
    # Helpers
    # ---------------------------
    def _set_status(msg: str):
        status.value = f"<b>Status:</b> {msg}"

    def _init_plot_once():
        if state.fig is not None and state.ax is not None:
            return

        was_interactive = plt.isinteractive()
        try:
            plt.ioff()
            fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
        finally:
            if was_interactive:
                plt.ion()

        state.fig, state.ax = fig, ax
        plot_slot.children = (fig.canvas,)

    def _clear_ax():
        if state.ax is not None:
            state.ax.clear()

    def _draw():
        if state.fig is None:
            return
        # Reserve extra top margin for wrapped titles (defensive).
        try:
            state.fig.subplots_adjust(top=0.86)
        except Exception:
            pass
        if hasattr(state.fig.canvas, "draw_idle"):
            state.fig.canvas.draw_idle()
        else:
            state.fig.canvas.draw()

    def _get_table(key: str) -> Optional[pd.DataFrame]:
        return getattr(state, key, None)

    def _current_source() -> Optional[str]:
        segf_cur = get_segmentframe_callable()
        return str(segf_cur.source_path) if segf_cur is not None else None

    def _dq_cfg() -> tuple:
        return (bool(require_valid_time.value),)

    def _export_preproc_tag() -> str:
        """Return a file-name-safe tag describing the current preprocessing configuration."""

        integrate_on = bool(integrate_to_flux.value)
        drift_on = integrate_on and bool(drift_correction.value)
        mode = str(drift_mode.value) if drift_on and drift_mode.value is not None else None

        # DC (n=0) is excluded from the main harmonic outputs by design.
        return format_preproc_tag(
            di_dt_enabled=bool(di_dt_correction.value),
            integrate_to_flux_enabled=integrate_on,
            drift_enabled=drift_on,
            drift_mode=mode,
            main_order=int(main_order.value),
            include_dc=False,
        )

    def _refresh_apply_button() -> None:
        src = _current_source()
        ok = (
            state.dq_plan is not None
            and state.dq_source == src
            and state.dq_plan_cfg == _dq_cfg()
        )
        btn_apply_and_compute.disabled = not ok

    def _set_plateau_controls_enabled(enabled: bool, message: str) -> None:
        dd_plateau.disabled = not enabled
        btn_plot_ns.disabled = not enabled
        plateau_info.value = message

    def _refresh_plateau_dropdown_from_df(df_out: pd.DataFrame) -> None:
        if df_out is None or "plateau_id" not in df_out.columns:
            state.plateau_current_map = None
            dd_plateau.options = []
            dd_plateau.value = None
            _set_plateau_controls_enabled(False, "<div style='color:#666;'>This measurement has no plateau information.</div>")
            return

        current_col = _pick_current_column_name(df_out)
        g = df_out.groupby("plateau_id", sort=True)[current_col].mean()

        plateau_current_map: Dict[int, float] = {int(pid): float(I) for pid, I in g.items()}
        state.plateau_current_map = plateau_current_map

        opts: list[tuple[str, int]] = []
        for pid in sorted(plateau_current_map.keys()):
            I = plateau_current_map[pid]
            opts.append((f"Plateau {pid} — mean current {I:.6g} A", pid))

        dd_plateau.options = opts
        dd_plateau.value = opts[0][1] if opts else None

        if dd_plateau.value is None:
            _set_plateau_controls_enabled(False, "<div style='color:#666;'>No plateau steps found.</div>")
        else:
            _set_plateau_controls_enabled(True, "<div style='color:#444;'>Select a plateau to plot normal and skew.</div>")

    def _update_plateau_info(*_):
        if dd_plateau.value is None or state.plateau_current_map is None:
            plateau_info.value = "<div style='color:#666;'>No plateau selected.</div>"
            return
        pid = int(dd_plateau.value)
        I = state.plateau_current_map.get(pid, float("nan"))
        if np.isfinite(I):
            plateau_info.value = (
                f"<div style='color:#444;'>Selected: <b>Plateau {pid}</b> — "
                f"mean current <b>{I:.6g} A</b></div>"
            )
        else:
            plateau_info.value = f"<div style='color:#444;'>Selected: <b>Plateau {pid}</b></div>"

    dd_plateau.observe(_update_plateau_info, names="value")

    def _start_action(key: str, msg: str) -> bool:
        # Debounce identical action triggers (defensive against multi-event clicks).
        now = time.monotonic()
        if state.last_action_key == key and (now - state.last_action_t) < 0.25:
            return False
        state.last_action_key = key
        state.last_action_t = now

        # Return False if already busy (prevents duplicate runs and duplicate logs).
        if state.busy:
            return False
        state.busy = True

        if not append_log.value:
            out_log.clear_output(wait=True)

        _set_status(msg)

        # Disable UI during the action
        btn_preview_dq.disabled = True
        btn_apply_and_compute.disabled = True
        btn_plot_amp.disabled = True
        btn_plot_ns.disabled = True
        btn_save_plot.disabled = True
        btn_save_table.disabled = True
        btn_show_head.disabled = True

        with out_log:
            print(msg)
        return True

    def _end_action(ok: bool = True, msg: str = "idle"):
        btn_preview_dq.disabled = False
        btn_plot_amp.disabled = False
        btn_save_plot.disabled = False
        btn_save_table.disabled = False
        btn_show_head.disabled = False

        # Plateau plot button stays disabled unless plateau info exists
        btn_plot_ns.disabled = dd_plateau.disabled

        _set_status(msg if ok else f"ERROR: {msg}")
        state.busy = False
        _refresh_apply_button()

    # ---------------------------
    # Actions
    # ---------------------------
    def _preview_data_quality(_):
        if not _start_action("preview", "Previewing data-quality cuts…"):
            return
        try:
            _init_plot_once()

            segf = get_segmentframe_callable()
            if segf is None:
                with out_log:
                    print("No segment is loaded yet. In Phase I, click 'Load segment' first.")
                state.dq_plan = None
                state.dq_plan_cfg = None
                state.dq_source = None
                _end_action(ok=False, msg="no segment loaded")
                return

            src = str(segf.source_path)

            segf2, rem = _ensure_full_turns(segf)
            tb = split_into_turns(segf2)

            ok_abs = np.all(np.isfinite(tb.df_abs), axis=1)
            ok_cmp = np.all(np.isfinite(tb.df_cmp), axis=1)
            ok_I = np.all(np.isfinite(tb.I), axis=1)

            finite_t = np.all(np.isfinite(tb.t), axis=1)
            strict_inc = finite_t & np.all(np.diff(tb.t, axis=1) > 0.0, axis=1)

            ok_t = strict_inc if require_valid_time.value else np.ones(tb.n_turns, dtype=bool)
            valid_turn = ok_abs & ok_cmp & ok_I & ok_t

            state.dq_plan = {
                "segf_trimmed": segf2,
                "rem_samples": int(rem),
                "tb": tb,
                "valid_turn": valid_turn,
            }
            state.dq_plan_cfg = _dq_cfg()
            state.dq_source = src

            n_total = int(tb.n_turns)
            n_keep = int(np.sum(valid_turn))
            n_drop = int(n_total - n_keep)

            with out_log:
                print("DATA-QUALITY PREVIEW (no cuts applied yet)")
                print(f" - Source: {src}")
                print(f" - Tail samples to trim for full turns: {int(rem)}")
                print(f" - Turns: total={n_total}, keep={n_keep}, drop={n_drop}")
                print("Next step: if you agree, click 'Apply cuts and compute harmonics (FFT)'.")
            _end_action(ok=True, msg="preview ready")

        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            state.dq_plan = None
            state.dq_plan_cfg = None
            state.dq_source = None
            _end_action(ok=False, msg=str(e))

    def _apply_and_compute(_):
        if not _start_action("apply", "Applying cuts and computing harmonics (FFT)…"):
            return
        try:
            if state.dq_plan is None or state.dq_source != _current_source() or state.dq_plan_cfg != _dq_cfg():
                with out_log:
                    print("No valid data-quality preview is available (or it is stale). Click 'Preview data-quality cuts' first.")
                _end_action(ok=False, msg="preview required")
                return

            plan = state.dq_plan
            segf2 = plan["segf_trimmed"]
            rem = int(plan["rem_samples"])
            tb = plan["tb"]
            valid_turn = plan["valid_turn"]

            state.segf = segf2
            state.tb = tb
            state.valid_turn = valid_turn

            if rem != 0:
                with out_log:
                    print(f"Applied tail trimming: removed {rem} sample(s) to reach full turns.")

            n_total = int(tb.n_turns)
            n_keep = int(np.sum(valid_turn))
            n_drop = int(n_total - n_keep)
            with out_log:
                print(f"Applied turn dropping: kept {n_keep}/{n_total} turn(s), dropped {n_drop}.")

            if getattr(segf2, "magnet_order", None) is not None and int(segf2.magnet_order) > 0:
                main_order.value = int(segf2.magnet_order)
                with out_log:
                    print(f"Phase reference: using main field order {int(segf2.magnet_order)} from Parameters.")
            else:
                with out_log:
                    print(f"Phase reference: using GUI main field order {int(main_order.value)}.")

            abs_turns = tb.df_abs[valid_turn, :]
            cmp_turns = tb.df_cmp[valid_turn, :]
            I_turns = tb.I[valid_turn, :]
            t_turns = getattr(tb, "t", None)
            t_turns = t_turns[valid_turn, :] if t_turns is not None else None

            # Optional: legacy di/dt ("dit") correction, applied to the incremental signal BEFORE integration/FFT.
            di_dt_res = None
            if di_dt_correction.value:
                if t_turns is None:
                    with out_log:
                        print("Warning: di/dt correction requested, but no time array is available; skipping di/dt correction.")
                else:
                    abs_turns, cmp_turns, di_dt_res = apply_di_dt_to_channels(abs_turns, cmp_turns, t_turns, I_turns)
                    with out_log:
                        n_app = int(np.sum(di_dt_res.applied))
                        n_tot = int(di_dt_res.applied.size)
                        print(
                            f"di/dt correction enabled: applied to {n_app}/{n_tot} turn(s) "
                            "(rule: dI/dt>0.1 A/s and mean(I)>10 A)."
                        )

            drift_abs = None
            drift_cmp = None

            if integrate_to_flux.value:
                if drift_correction.value:
                    mode = str(drift_mode.value)
                    if mode == "weighted" and t_turns is None:
                        raise ValueError("Δt-weighted drift correction requires a time array per turn.")

                    flux_abs, drift_abs = integrate_turns_to_flux(abs_turns, drift=True, drift_mode=mode, t_turns=t_turns)
                    flux_cmp, drift_cmp = integrate_turns_to_flux(cmp_turns, drift=True, drift_mode=mode, t_turns=t_turns)

                    with out_log:
                        label = "Legacy (C++)" if mode == "legacy" else "Bottura/Pentella (Δt-weighted)"
                        print(f"Signal used for spectrum: integrated flux with drift correction ({label}).")
                else:
                    flux_abs, drift_abs = integrate_turns_to_flux(abs_turns, drift=False)
                    flux_cmp, drift_cmp = integrate_turns_to_flux(cmp_turns, drift=False)
                    with out_log:
                        print("Signal used for spectrum: integrated flux (legacy convention).")

                sig_abs = flux_abs
                sig_cmp = flux_cmp
            else:
                sig_abs = abs_turns
                sig_cmp = cmp_turns
                with out_log:
                    print("Signal used for spectrum: raw differential signal (no integration).")

            with out_log:
                print(f"Computing harmonics up to order {int(max_harm.value)}…")

            H_abs = dft_per_turn(sig_abs, n_max=int(max_harm.value))
            H_cmp = dft_per_turn(sig_cmp, n_max=int(max_harm.value))
            state.H_abs, state.H_cmp = H_abs, H_cmp

            turn_idx = np.arange(H_cmp.coeff.shape[0], dtype=int)

            def per_turn_amp_phase(H, prefix: str) -> pd.DataFrame:
                C = H.coeff
                orders = [int(x) for x in H.orders]
                cols: Dict[str, Any] = {}
                for jj, nn in enumerate(orders):
                    if int(nn) == 0:
                        continue
                    cols[f"{prefix}amplitude_C{nn}"] = np.abs(C[:, jj])
                    cols[f"{prefix}phase_C{nn}"] = np.angle(C[:, jj])
                return pd.DataFrame(cols, index=turn_idx)

            prov = provenance_columns(
                n_turns=int(I_turns.shape[0]),
                di_dt_enabled=bool(di_dt_correction.value),
                di_dt_res=di_dt_res,
                integrate_to_flux_enabled=bool(integrate_to_flux.value),
                drift_enabled=bool(integrate_to_flux.value and drift_correction.value),
                drift_mode=(str(drift_mode.value) if (integrate_to_flux.value and drift_correction.value) else None),
                drift_abs=drift_abs,
                drift_cmp=drift_cmp,
            )

            meta: Dict[str, Any] = {"mean_current_A": np.mean(I_turns, axis=1)}
            meta.update(prov)
            if getattr(tb, "plateau_id", None) is not None:
                meta["plateau_id"] = np.asarray(tb.plateau_id[valid_turn]).astype(int)

            df_out = pd.concat(
                [
                    pd.DataFrame(meta, index=turn_idx),
                    per_turn_amp_phase(H_abs, "absolute_"),
                    per_turn_amp_phase(H_cmp, "compensated_"),
                ],
                axis=1,
            )
            state.df_out = df_out

            if "plateau_id" in df_out.columns:
                g = df_out.groupby("plateau_id", sort=True)
                state.df_out_mean = g.mean(numeric_only=True)
                state.df_out_std = g.std(numeric_only=True)
            else:
                state.df_out_mean = df_out.mean(numeric_only=True).to_frame().T
                state.df_out_std = df_out.std(numeric_only=True).to_frame().T


            # Add constant preprocessing descriptors (string/bool) to the summary tables.
            # Note: per-turn provenance numeric columns (e.g. dI/dt, applied masks) are already
            # included in the group-by mean/std computations via numeric_only=True.
            summary_const = {
                "preproc_di_dt_enabled": bool(di_dt_correction.value),
                "preproc_integrate_to_flux": bool(integrate_to_flux.value),
                "preproc_drift_enabled": bool(integrate_to_flux.value and drift_correction.value),
                "preproc_drift_mode": str(drift_mode.value) if (integrate_to_flux.value and drift_correction.value) else "",
            }
            for k, v in summary_const.items():
                if state.df_out_mean is not None:
                    state.df_out_mean[k] = v
                if state.df_out_std is not None:
                    state.df_out_std[k] = v

            # NOTE (do not forget): the legacy C++ analyzer computes the rotation reference after applying k_n
            # (k_n is complex and can shift the phase). Once k_n calibration is implemented, compute phi_out
            # from the calibrated main harmonic (post-k_n), not from the raw FFT coefficient.
            m = int(main_order.value)
            phi_out, bad_phi = _phase_zero_from_main_harmonic(H_abs, m)
            if int(np.sum(bad_phi)):
                with out_log:
                    print(
                        f"Warning: main harmonic (order {m}) is too small or non-finite in {int(np.sum(bad_phi))} turn(s). "
                        "For those turns, the phase reference is set to 0."
                    )

            C_abs_rot = _rotate_harmonics(H_abs, phi_out)
            C_cmp_rot = _rotate_harmonics(H_cmp, phi_out)

            df_ba_abs = _ba_from_rotated(C_abs_rot, H_abs.orders, prefix="absolute_")
            df_ba_cmp = _ba_from_rotated(C_cmp_rot, H_cmp.orders, prefix="compensated_")

            prov_cols = [
                c
                for c in df_out.columns
                if c.startswith("preproc_") or c.startswith("absolute_preproc_") or c.startswith("compensated_preproc_")
            ]

            front_cols: list[str] = []
            if "plateau_id" in df_out.columns:
                front_cols.append("plateau_id")
            front_cols.append("mean_current_A")
            front_cols.extend(prov_cols)

            df_ba = pd.concat(
                [df_out[front_cols].reset_index(drop=True), df_ba_abs, df_ba_cmp],
                axis=1,
            )

            state.df_ba = df_ba

            if "plateau_id" in df_ba.columns:
                g2 = df_ba.groupby("plateau_id", sort=True)
                state.mean_ba = g2.mean(numeric_only=True)
                state.std_ba = g2.std(numeric_only=True)
            else:
                state.mean_ba = df_ba.mean(numeric_only=True).to_frame().T
                state.std_ba = df_ba.std(numeric_only=True).to_frame().T


            # Add constant preprocessing descriptors to the normal/skew summary tables.
            summary_const_ba = {
                "preproc_di_dt_enabled": bool(di_dt_correction.value),
                "preproc_integrate_to_flux": bool(integrate_to_flux.value),
                "preproc_drift_enabled": bool(integrate_to_flux.value and drift_correction.value),
                "preproc_drift_mode": str(drift_mode.value) if (integrate_to_flux.value and drift_correction.value) else "",
            }
            for k, v in summary_const_ba.items():
                if state.mean_ba is not None:
                    state.mean_ba[k] = v
                if state.std_ba is not None:
                    state.std_ba[k] = v

            _refresh_plateau_dropdown_from_df(df_out)
            _update_plateau_info()

            with out_log:
                print("Done.")
            _end_action(ok=True, msg="harmonics computed")

        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _plot_amplitude_vs_current(_):
        if not _start_action("plot_amp", "Updating amplitude-versus-current plot…"):
            return
        try:
            _init_plot_once()

            if state.df_out is None:
                with out_log:
                    print("No results available yet. Run: Preview → Apply cuts and compute harmonics (FFT).")
                _end_action(ok=False, msg="no results")
                return

            df = state.df_out
            n = int(harm_order.value)

            if dd_channel.value == "cmp":
                amp_col = f"compensated_amplitude_C{n}"
                title_chan = "compensated"
            else:
                amp_col = f"absolute_amplitude_C{n}"
                title_chan = "absolute"

            if amp_col not in df.columns:
                with out_log:
                    print("This harmonic order is not available. Increase the maximum harmonic order and recompute.")
                _end_action(ok=False, msg="missing harmonic")
                return

            current_col = _pick_current_column_name(df)

            _clear_ax()

            if "plateau_id" in df.columns:
                g = df.groupby("plateau_id", sort=True)
                x = g[current_col].mean().values
                y = g[amp_col].mean().values
                yerr = g[amp_col].std().values
                state.ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
                ttl = f"Amplitude versus current — order {n} ({title_chan} channel) — plateau mean ± standard deviation"
            else:
                x = df[current_col].values
                y = df[amp_col].values
                state.ax.plot(x, y, marker="o", linestyle="None")
                ttl = f"Amplitude versus current — order {n} ({title_chan} channel)"

            t = state.ax.set_title(_wrap_title(ttl, width=44), fontsize=11, pad=10)
            try:
                t.set_wrap(True)
            except Exception:
                pass

            state.ax.set_xlabel("Mean current [A]")
            state.ax.set_ylabel(f"Amplitude |C_{n}| (internal units)")
            state.ax.grid(True)
            _draw()

            _end_action(ok=True, msg="plot updated")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _plot_normal_skew_vs_harmonic(_):
        if not _start_action("plot_ns", "Updating normal/skew versus harmonic order…"):
            return
        try:
            _init_plot_once()

            if state.mean_ba is None or state.std_ba is None:
                with out_log:
                    print("No normal/skew results available yet. Run: Preview → Apply cuts and compute harmonics (FFT).")
                _end_action(ok=False, msg="no results")
                return

            if dd_plateau.value is None:
                with out_log:
                    print("Select a plateau first.")
                _end_action(ok=False, msg="no plateau selected")
                return

            pid = int(dd_plateau.value)

            dfm = state.mean_ba
            dfs = state.std_ba
            if pid not in dfm.index:
                with out_log:
                    print("The selected plateau is not present in the computed results.")
                _end_action(ok=False, msg="plateau missing")
                return

            prefix = "compensated_" if dd_channel.value == "cmp" else "absolute_"
            title_chan = "compensated" if dd_channel.value == "cmp" else "absolute"

            Nmax = int(max_harm.value)
            orders = np.arange(1, Nmax + 1, dtype=int)

            if hide_main_in_ns.value:
                m = int(main_order.value)
                orders = orders[orders != m]

            B = np.full(orders.shape, np.nan, dtype=float)
            A = np.full(orders.shape, np.nan, dtype=float)
            sB = np.full(orders.shape, np.nan, dtype=float)
            sA = np.full(orders.shape, np.nan, dtype=float)

            for i, n in enumerate(orders):
                colB = f"{prefix}normal_B{n}"
                colA = f"{prefix}skew_A{n}"
                if colB in dfm.columns:
                    B[i] = float(dfm.loc[pid, colB])
                if colA in dfm.columns:
                    A[i] = float(dfm.loc[pid, colA])
                if colB in dfs.columns:
                    sB[i] = float(dfs.loc[pid, colB])
                if colA in dfs.columns:
                    sA[i] = float(dfs.loc[pid, colA])

            keep = np.isfinite(B) | np.isfinite(A)
            orders = orders[keep]
            B = B[keep]
            A = A[keep]
            sB = sB[keep]
            sA = sA[keep]

            if orders.size == 0:
                with out_log:
                    print("No normal/skew harmonic columns were found for the selected channel. Increase maximum harmonic order and recompute.")
                _end_action(ok=False, msg="no harmonic columns")
                return

            _clear_ax()

            x = np.arange(orders.size, dtype=float)
            width = 0.42

            state.ax.bar(x - width / 2, B, width=width, label="Normal (B_n)")
            state.ax.bar(x + width / 2, A, width=width, label="Skew (A_n)")

            if np.any(np.isfinite(sB)):
                state.ax.errorbar(x - width / 2, B, yerr=sB, fmt="none", capsize=3)
            if np.any(np.isfinite(sA)):
                state.ax.errorbar(x + width / 2, A, yerr=sA, fmt="none", capsize=3)

            state.ax.set_xticks(x)
            state.ax.set_xticklabels([str(int(n)) for n in orders])
            state.ax.set_xlabel("Harmonic order n")
            state.ax.set_ylabel("Normal / Skew (legacy convention)")

            I = state.plateau_current_map.get(pid, float("nan")) if state.plateau_current_map else float("nan")
            if np.isfinite(I):
                ttl = f"Normal and skew versus harmonic order — {title_chan} channel — Plateau {pid} at {I:.6g} A"
            else:
                ttl = f"Normal and skew versus harmonic order — {title_chan} channel — Plateau {pid}"

            if hide_main_in_ns.value:
                ttl += f" (main field order {int(main_order.value)} hidden)"

            t = state.ax.set_title(_wrap_title(ttl, width=44), fontsize=11, pad=10)
            try:
                t.set_wrap(True)
            except Exception:
                pass

            state.ax.grid(True, axis="y")
            state.ax.legend()
            _draw()

            _end_action(ok=True, msg="plot updated")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _save_plot(_):
        if not _start_action("save_plot", "Saving plot…"):
            return
        try:
            if state.fig is None:
                with out_log:
                    print("No plot is available.")
                _end_action(ok=False, msg="no plot")
                return

            fmt = str(save_plot_fmt.value)
            tag = _export_preproc_tag()
            path = _saveas_dialog(
                title="Save plot",
                initialfile=f"phase2_plot_{tag}.{fmt}",
                defaultextension=f".{fmt}",
                filetypes=[(fmt.upper(), f"*.{fmt}")],
            )
            if not path:
                _end_action(ok=True, msg="save cancelled")
                return

            path = append_tag_to_path(path, tag)

            state.fig.savefig(path, bbox_inches="tight")
            with out_log:
                print(f"Saved plot to: {path}")
            _end_action(ok=True, msg="plot saved")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _save_table(_):
        if not _start_action("save_table", "Saving table…"):
            return
        try:
            key = str(table_choice.value)
            df = _get_table(key)
            if df is None:
                with out_log:
                    print("The selected table is not available yet.")
                _end_action(ok=False, msg="no table")
                return

            tag = _export_preproc_tag()
            path = _saveas_dialog(
                title="Save table (comma-separated values)",
                initialfile=f"{key}_{tag}.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not path:
                _end_action(ok=True, msg="save cancelled")
                return

            path = append_tag_to_path(path, tag)

            df.to_csv(path, index=True)
            with out_log:
                print(f"Saved table to: {path}")
            _end_action(ok=True, msg="table saved")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    def _show_head(_):
        if not _start_action("show_head", "Showing table…"):
            return
        try:
            key = str(table_choice.value)
            df = _get_table(key)
            if df is None:
                with out_log:
                    print("The selected table is not available yet.")
                _end_action(ok=False, msg="no table")
                return

            txt = df.head(15).to_string()
            esc = html.escape(txt)
            table_html.value = (
                "<div style='border:1px solid #ddd; padding:8px; height:260px; overflow-y:auto; background:#fff;'>"
                f"<div style='margin-bottom:6px; color:#444;'><b>First rows of:</b> {html.escape(key)}</div>"
                f"<pre style='margin:0; white-space:pre; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;'>{esc}</pre>"
                "</div>"
            )
            _end_action(ok=True, msg="table shown")
        except Exception as e:
            with out_log:
                print("Exception:", repr(e))
            _end_action(ok=False, msg=str(e))

    # Defensive: clear click handlers before wiring callbacks (prevents handler accumulation).
    for b in [btn_preview_dq, btn_apply_and_compute, btn_plot_amp, btn_plot_ns, btn_save_plot, btn_save_table, btn_show_head]:
        _clear_button_handlers(b)

    btn_preview_dq.on_click(_preview_data_quality)
    btn_apply_and_compute.on_click(_apply_and_compute)
    btn_plot_amp.on_click(_plot_amplitude_vs_current)
    btn_plot_ns.on_click(_plot_normal_skew_vs_harmonic)
    btn_save_plot.on_click(_save_plot)
    btn_save_table.on_click(_save_table)
    btn_show_head.on_click(_show_head)

    _init_plot_once()

    # ---------------------------
    # Layout: two columns
    # ---------------------------
    compute_box = w.VBox(
        [
            w.HTML("<b>Step 1 — Preview and apply data-quality cuts</b>"),
            w.HBox([max_harm, main_order]),
            integrate_to_flux,
            drift_correction,
            drift_mode,
            di_dt_correction,
            require_valid_time,
            w.HBox([btn_preview_dq, btn_apply_and_compute, append_log]),
            help_text,
        ],
        layout=w.Layout(border="1px solid #ddd", padding="10px", width="100%"),
    )

    # View 1: button on its own row (prevents clipping)
    view1_box = w.VBox(
        [
            w.HTML("<b>View 1 — Amplitude versus current</b><div style='color:#666;'>This view ignores plateau selection.</div>"),
            w.HBox([dd_channel, harm_order]),
            btn_plot_amp,
        ],
        layout=w.Layout(border="1px solid #ddd", padding="10px", width="100%"),
    )

    view2_box = w.VBox(
        [
            w.HTML("<b>View 2 — Normal and skew components by plateau</b><div style='color:#666;'>This view plots harmonics versus harmonic order for the selected plateau.</div>"),
            dd_plateau,
            plateau_info,
            hide_main_in_ns,
            btn_plot_ns,
        ],
        layout=w.Layout(border="1px solid #ddd", padding="10px", width="100%"),
    )

    export_box = w.VBox(
        [
            w.HTML("<b>Export</b>"),
            w.HBox([save_plot_fmt, btn_save_plot]),
            w.HBox([table_choice, btn_save_table, btn_show_head]),
        ],
        layout=w.Layout(border="1px solid #ddd", padding="10px", width="100%"),
    )

    # Two-column layout:
    #   left = controls + results (plot/table)
    #   right = diagnostics (status/log)
    # This prevents having to scroll down to see the log.
    left_panel = w.VBox(
        [
            compute_box,
            view1_box,
            view2_box,
            export_box,
            w.HTML("<b>Plot</b>"),
            plot_slot,
            w.HTML("<b>Table</b>"),
            table_html,
        ],
        layout=w.Layout(width="68%", min_width="640px"),
    )

    right_panel = w.VBox(
        [
            status,
            w.HTML("<b>Log</b>"),
            log.widget,
        ],
        layout=w.Layout(width="32%", min_width="360px"),
    )

    def _refresh_apply_button_outer():
        _refresh_apply_button()

    _refresh_apply_button_outer()

    panel = w.HBox([left_panel, right_panel], layout=w.Layout(width="100%"))
    _ACTIVE_PHASE2_PANEL = panel
    return panel
