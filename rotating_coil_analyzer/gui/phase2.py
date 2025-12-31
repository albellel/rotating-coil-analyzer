from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Dict

import numpy as np
import pandas as pd
import ipywidgets as w
from IPython.display import display
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
    df_out: Optional[pd.DataFrame] = None

    df_ns: Optional[pd.DataFrame] = None
    mean_ns: Optional[pd.DataFrame] = None
    std_ns: Optional[pd.DataFrame] = None


def _ensure_full_turns(segf: SegmentFrame) -> tuple[SegmentFrame, int]:
    """Trim any tail remainder so length is an integer number of turns."""
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
    """
    Fourier-series convention after phase referencing:
      Normal: N_n =  2 Re(C_n)
      Skew  : S_n = -2 Im(C_n)
    """
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
    """
    Phase II GUI panel.

    get_segmentframe_callable: () -> SegmentFrame | None
      Must return the SegmentFrame currently loaded in Phase I.
    """
    state = Phase2State()

    # Controls
    nmax = w.BoundedIntText(value=default_n_max, min=1, max=200, step=1, description="N_max", layout=w.Layout(width="180px"))
    require_finite_t = w.Checkbox(value=True, description="Drop turns with non-finite t", indent=False)

    append_output = w.Checkbox(value=False, description="Append output", indent=False, layout=w.Layout(width="140px"))

    btn_compute = w.Button(description="Compute FFT (per turn)", button_style="primary")

    dd_channel = w.Dropdown(
        options=[("compensated (df_cmp)", "cmp"), ("absolute (df_abs)", "abs")],
        value="cmp",
        description="Channel",
        layout=w.Layout(width="260px"),
    )
    dd_harm = w.BoundedIntText(value=1, min=1, max=200, step=1, description="Harm n", layout=w.Layout(width="180px"))
    btn_plot_amp = w.Button(description="Plot |C_n| vs current")

    dd_plateau = w.Dropdown(options=[], description="plateau_id", layout=w.Layout(width="220px"))
    btn_plot_ns = w.Button(description="Plot Normal/Skew bars")

    btn_show_tables = w.Button(description="Show tables")

    out = w.Output()

    def _start_action():
        if not append_output.value:
            out.clear_output()
            plt.close("all")

    def _refresh_plateau_dropdown(df_out: pd.DataFrame):
        if "plateau_id" in df_out.columns:
            pids = sorted(pd.unique(df_out["plateau_id"]))
            dd_plateau.options = [(str(p), float(p)) for p in pids]
            dd_plateau.value = float(pids[0]) if pids else None
        else:
            dd_plateau.options = []
            dd_plateau.value = None

    def _compute(_):
        _start_action()
        with out:
            segf = get_segmentframe_callable()
            if segf is None:
                print("No SegmentFrame loaded yet. In Phase I, click 'Load segment' first.")
                return

            # 0) Full-turn remainder check
            segf2, rem = _ensure_full_turns(segf)
            if rem != 0:
                print(f"INCOMPLETE TURN DETECTED: trimmed tail remainder={rem} samples before analysis.")
            else:
                print(f"Full-turn length OK: total samples={len(segf2.df)} is n_turns*Ns ({segf2.n_turns}*{segf2.samples_per_turn}).")

            state.segf = segf2

            # 1) Turn split
            tb = split_into_turns(segf2)
            state.tb = tb

            # 2) QC mask
            ok_abs = np.all(np.isfinite(tb.df_abs), axis=1)
            ok_cmp = np.all(np.isfinite(tb.df_cmp), axis=1)
            ok_t = np.all(np.isfinite(tb.t), axis=1) if require_finite_t.value else np.ones(tb.df_abs.shape[0], dtype=bool)

            valid_turn = ok_t & ok_abs & ok_cmp

            state.ok_t, state.ok_abs, state.ok_cmp = ok_t, ok_abs, ok_cmp
            state.valid_turn = valid_turn

            print("TURN QC REPORT")
            print("  total turns:", int(tb.df_abs.shape[0]))
            print("  keep turns :", int(np.sum(valid_turn)))
            print("  drop turns :", int(np.sum(~valid_turn)))
            print("   - drop (non-finite t)      :", int(np.sum(~ok_t)))
            print("   - drop (finite t, bad abs) :", int(np.sum(ok_t & ~ok_abs)))
            print("   - drop (finite t+abs, bad cmp):", int(np.sum(ok_t & ok_abs & ~ok_cmp)))

            bad = np.where(~valid_turn)[0]
            if bad.size:
                # within-plateau index if available
                within = None
                if getattr(tb, "plateau_id", None) is not None:
                    pid_turn = tb.plateau_id
                    within = np.zeros_like(pid_turn, dtype=int)
                    for pid in np.unique(pid_turn):
                        m = np.where(pid_turn == pid)[0]
                        within[m] = np.arange(m.size, dtype=int)

                rows = []
                for j in bad:
                    rows.append({
                        "turn_idx": int(j),
                        "plateau_id": float(tb.plateau_id[j]) if getattr(tb, "plateau_id", None) is not None else None,
                        "plateau_step": float(tb.plateau_step[j]) if getattr(tb, "plateau_step", None) is not None else None,
                        "plateau_I_hint": float(tb.plateau_I_hint[j]) if getattr(tb, "plateau_I_hint", None) is not None else None,
                        "turn_in_plateau": int(within[j]) if within is not None else None,
                        "bad_t": bool(~ok_t[j]),
                        "bad_abs": bool(~ok_abs[j]),
                        "bad_cmp": bool(~ok_cmp[j]),
                    })
                print("\nDROPPED TURN DETAILS")
                display(pd.DataFrame(rows))

            # 3) Apply mask
            abs_turns = tb.df_abs[valid_turn, :]
            cmp_turns = tb.df_cmp[valid_turn, :]
            I_turns = tb.I[valid_turn, :]

            # 4) FFT
            H_abs = dft_per_turn(abs_turns, n_max=int(nmax.value))
            H_cmp = dft_per_turn(cmp_turns, n_max=int(nmax.value))
            state.H_abs, state.H_cmp = H_abs, H_cmp

            # 5) Per-turn table
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

            # 6) Normal/Skew (phase-reference to cmp n=1)
            phi_ref = _phase_reference_from(H_cmp, n_ref=1)
            C_cmp_rot = _rotate_harmonics(H_cmp, phi_ref)
            df_ns_cmp = _normal_skew_from_rotated(C_cmp_rot, H_cmp.orders, prefix="cmp_")

            df_ns = pd.concat(
                [
                    df_out[[c for c in ["plateau_id", "I_mean", "plateau_I_hint", "plateau_step"] if c in df_out.columns]].reset_index(drop=True),
                    df_ns_cmp,
                ],
                axis=1,
            )
            state.df_ns = df_ns

            if "plateau_id" in df_ns.columns:
                g = df_ns.groupby("plateau_id", sort=True)
                state.mean_ns = g.mean(numeric_only=True)
                state.std_ns = g.std(numeric_only=True)
            else:
                state.mean_ns = df_ns.mean(numeric_only=True).to_frame().T
                state.std_ns = df_ns.std(numeric_only=True).to_frame().T

            print("\nComputed:")
            print("  df_out (per-turn):", df_out.shape)
            if "plateau_id" in df_out.columns:
                print("  plateaus:", len(pd.unique(df_out["plateau_id"])))

    def _plot_amp(_):
        _start_action()
        with out:
            if state.df_out is None:
                print("Nothing computed yet. Click 'Compute FFT (per turn)' first.")
                return

            df = state.df_out
            ch = dd_channel.value
            nn = int(dd_harm.value)

            col = f"{ch}_A{nn}"
            if col not in df.columns:
                print(f"Missing column {col}. Increase N_max or choose a smaller n.")
                return

            x_turn = df["I_mean"].to_numpy()
            y_turn = df[col].to_numpy()

            if "plateau_id" in df.columns:
                g = df.groupby("plateau_id", sort=True)
                mean = g.mean(numeric_only=True)
                std = g.std(numeric_only=True)
                x_plateau = mean["I_mean"].to_numpy()
                y_mean = mean[col].to_numpy()
                y_std = std[col].to_numpy()
            else:
                x_plateau = np.array([np.mean(x_turn)])
                y_mean = np.array([np.mean(y_turn)])
                y_std = np.array([np.std(y_turn)])

            plt.figure()
            plt.plot(x_turn, y_turn, ".", label="per-turn")
            plt.errorbar(x_plateau, y_mean, yerr=y_std, fmt="o", label="per-plateau mean ± std")
            plt.xlabel("current (A)")
            plt.ylabel(f"{ch} |C_{nn}|")
            plt.title(f"Harmonic amplitude vs current ({ch}, n={nn})")
            plt.legend()
            plt.show()

    def _plot_ns(_):
        _start_action()
        with out:
            if state.mean_ns is None:
                print("Nothing computed yet. Click 'Compute FFT (per turn)' first.")
                return
            pid = dd_plateau.value
            if pid is None:
                print("No plateau_id available (not MBA data or not computed yet).")
                return

            mean_ns = state.mean_ns
            if pid not in mean_ns.index:
                print(f"plateau_id={pid} not found.")
                return

            row = mean_ns.loc[pid]
            nmax_eff = int(nmax.value)
            n = np.arange(1, nmax_eff + 1)

            N_vals = np.array([row.get(f"cmp_N{k}", np.nan) for k in n])
            S_vals = np.array([row.get(f"cmp_S{k}", np.nan) for k in n])

            plt.figure()
            wbar = 0.4
            plt.bar(n - wbar / 2, N_vals, width=wbar, label="Normal (cos)")
            plt.bar(n + wbar / 2, S_vals, width=wbar, label="Skew (sin)")
            plt.xlabel("harmonic order n")
            plt.ylabel("cmp component (phase-referenced to n=1)")
            plt.title(f"Normal/Skew bars (cmp), plateau_id={pid}")
            plt.legend()
            plt.show()

    def _show_tables(_):
        _start_action()
        with out:
            if state.df_out is None:
                print("Nothing computed yet. Click 'Compute FFT (per turn)' first.")
                return
            print("df_out (per-turn):")
            display(state.df_out.head(30))

            if state.mean_ns is not None and "plateau_id" in state.df_out.columns:
                print("\nNormal/Skew per-plateau mean (first 30):")
                display(state.mean_ns.head(30))

    btn_compute.on_click(_compute)
    btn_plot_amp.on_click(_plot_amp)
    btn_plot_ns.on_click(_plot_ns)
    btn_show_tables.on_click(_show_tables)

    row1 = w.HBox([nmax, require_finite_t, btn_compute, append_output])
    row2 = w.HBox([dd_channel, dd_harm, btn_plot_amp, dd_plateau, btn_plot_ns, btn_show_tables])
    return w.VBox([row1, row2, out])
