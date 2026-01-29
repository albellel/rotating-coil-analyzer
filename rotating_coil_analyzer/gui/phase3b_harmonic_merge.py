from __future__ import annotations

"""Phase 3B GUI: Harmonic Merge (apply kn + per-n channel selection).

This tab is responsible for:
1. Applying the kn calibration (from Phase 3A) to compute Abs/Cmp harmonics per turn.
2. Displaying diagnostics and allowing per-harmonic-order source selection.
3. Providing preset merge modes and custom per-n selection.
4. Exporting merged results with full provenance.

Inputs:
- SegmentFrame from Phase I
- KnBundle from Phase 3A

Output:
- MergeResult with full traceability (kn provenance, compensation scheme, per-n map)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import ipywidgets as w
import matplotlib.pyplot as plt

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns

from rotating_coil_analyzer.analysis.kn_pipeline import (
    LegacyKnPerTurn,
    compute_legacy_kn_per_turn,
    merge_coefficients,
)
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice, MergeDiagnostics
from rotating_coil_analyzer.analysis.kn_bundle import (
    KnBundle,
    MergeResult,
    CHANNEL_ABS,
    CHANNEL_CMP,
    CHANNEL_EXT,
    CHANNEL_NAMES,
)


_ACTIVE_PHASE3B_PANEL: Optional[w.Widget] = None


@dataclass
class Phase3BState:
    """Local state for the Harmonic Merge tab."""
    segf: Optional[SegmentFrame] = None
    seg_path: Optional[str] = None
    kn_bundle: Optional[KnBundle] = None

    # Computed harmonics (pre-merge)
    result: Optional[LegacyKnPerTurn] = None

    # Merge diagnostics and recommendation
    diag: Optional[MergeDiagnostics] = None
    choice_recommended: Optional[np.ndarray] = None

    # User's per-n selection (mutable during interaction)
    per_n_selection: Optional[np.ndarray] = None

    # Applied merge result
    merge_result: Optional[MergeResult] = None

    # Plot state
    fig: Optional[Any] = None
    ax: Optional[Any] = None

    busy: bool = False


def _saveas_dialog(
    *,
    initialfile: str,
    defaultextension: str,
    filetypes: list[tuple[str, str]],
    title: str = "Save file",
) -> Optional[str]:
    """Open a native Save-As dialog (best effort)."""
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


def _clear_button_handlers(btn: w.Button) -> None:
    """Defensive: remove any accumulated click handlers."""
    try:
        btn._click_handlers.callbacks.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _ba_table_from_C(C: np.ndarray, orders: np.ndarray, *, prefix: str = "") -> pd.DataFrame:
    """Convert complex coefficients to legacy B/A tables per turn.

    Convention: M_n = 2*C_n (n>=1), B_n=Re(M_n), A_n=Im(M_n).
    """
    out: Dict[str, np.ndarray] = {}
    for j, n in enumerate([int(x) for x in orders]):
        M = 2.0 * C[:, j]
        out[f"{prefix}normal_B{n}"] = np.real(M)
        out[f"{prefix}skew_A{n}"] = np.imag(M)
    return pd.DataFrame(out)


def _ensure_full_turns(segf: SegmentFrame) -> Tuple[SegmentFrame, int]:
    """Trim tail remainder samples to keep full turns."""
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
        aperture_id=getattr(segf, "aperture_id", None),
        magnet_order=getattr(segf, "magnet_order", None),
    )
    return segf2, rem


def build_phase3b_harmonic_merge_panel(
    get_segmentframe_callable: Callable[[], Optional[SegmentFrame]],
    get_segmentpath_callable: Callable[[], Optional[str]] | None = None,
    get_kn_bundle_callable: Callable[[], Optional[KnBundle]] | None = None,
    set_merge_result_callable: Callable[[Optional[MergeResult]], None] | None = None,
) -> w.Widget:
    """Build the Phase 3B (Harmonic Merge) panel.

    Parameters
    ----------
    get_segmentframe_callable
        Returns the current SegmentFrame from shared state.
    get_segmentpath_callable
        Returns the current segment file path.
    get_kn_bundle_callable
        Returns the KnBundle from Phase 3A.
    set_merge_result_callable
        Called when a MergeResult is created.
    """
    global _ACTIVE_PHASE3B_PANEL

    if _ACTIVE_PHASE3B_PANEL is not None:
        try:
            _ACTIVE_PHASE3B_PANEL.close()
        except Exception:
            pass
        _ACTIVE_PHASE3B_PANEL = None

    st = Phase3BState()

    log = HtmlLog(title="Harmonic Merge log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- Context display ----
    context_html = w.HTML(
        "<div style='color:#666; padding:4px; background:#f8f8f8; border:1px solid #ddd;'>"
        "Waiting for inputs from Phase I (segment) and Phase 3A (kn)."
        "</div>"
    )

    # ---- Numeric parameters ----
    rref_mm = w.FloatText(
        value=17.0,
        description="Rref [mm]:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )
    abs_calib = w.FloatText(
        value=1.0,
        description="Abs calibration:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )
    magnet_order = w.IntText(
        value=2,
        description="Main order m:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )
    skew_main = w.Checkbox(value=False, description="Skew main harmonic")

    # ---- Processing options (legacy ordering) ----
    opt_dit = w.Checkbox(value=False, description="di/dt correction")
    opt_dri = w.Checkbox(value=True, description="drift correction")
    opt_rot = w.Checkbox(value=True, description="rotation (post-Kn)")
    opt_cel = w.Checkbox(value=False, description="CEL")
    opt_fed = w.Checkbox(value=False, description="feeddown")
    opt_nor = w.Checkbox(value=False, description="normalization")

    drift_mode = w.Dropdown(
        options=[("legacy", "legacy"), ("weighted", "weighted")],
        value="legacy",
        description="Drift mode:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )

    # ---- Compensation scheme (metadata) ----
    comp_scheme = w.Text(
        value="",
        description="Compensation scheme:",
        placeholder="e.g. A-C, ABCD, or leave blank",
        layout=w.Layout(width="400px"),
        style={"description_width": "150px"},
    )
    comp_scheme_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "This is documentation metadata only. The physical compensation is already applied in the measurement."
        "</i>"
    )

    # ---- Actions ----
    btn_compute = w.Button(description="Apply Kn (Compute Harmonics)", button_style="info")
    btn_recommend = w.Button(description="Recommend Merge", button_style="")
    btn_recommend.disabled = True

    # ---- Per-n selection table ----
    per_n_table_html = w.HTML("<i>Compute harmonics first to see per-n selection table.</i>")

    # Per-n dropdowns (will be populated dynamically)
    per_n_dropdowns: List[w.Dropdown] = []
    per_n_container = w.VBox([])

    # ---- Merge mode preset ----
    merge_preset = w.Dropdown(
        options=[
            ("Main from Abs, others from Cmp (Recommended)", "abs_main_cmp_others"),
            ("Use recommendation", "recommended"),
            ("All from Abs", "abs_all"),
            ("All from Cmp", "cmp_all"),
            ("Abs up to m, Cmp above", "abs_upto_m_cmp_above"),
            ("Custom (use dropdowns below)", "custom"),
        ],
        value="abs_main_cmp_others",
        description="Merge preset:",
        style={"description_width": "110px"},
        layout=w.Layout(width="450px"),
    )

    # ---- Approval and apply ----
    approve_merge = w.Checkbox(value=False, description="I approve applying this merge")
    btn_apply_merge = w.Button(description="Apply Merge", button_style="warning")
    btn_apply_merge.disabled = True

    btn_export = w.Button(description="Export CSV (all tables)", button_style="success")
    btn_export.disabled = True

    # ---- Diagnostics display ----
    diag_table_html = w.HTML("<i>Diagnostics will appear here after recommendation.</i>")

    # ---- Plot output ----
    out_plot = w.Output()

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _refresh_context() -> bool:
        """Refresh context from Phase I and Phase 3A. Returns True if ready."""
        segf = get_segmentframe_callable()
        kn_bundle = get_kn_bundle_callable() if get_kn_bundle_callable else None

        if segf is None:
            context_html.value = (
                "<div style='color:#b00; padding:4px; background:#fee; border:1px solid #c88;'>"
                "No segment loaded. Load a measurement in Phase I first."
                "</div>"
            )
            st.segf = None
            st.kn_bundle = None
            return False

        if kn_bundle is None:
            context_html.value = (
                "<div style='color:#b00; padding:4px; background:#fee; border:1px solid #c88;'>"
                "No kn loaded. Load/compute kn in Phase 3A (Coil Calibration) first."
                "</div>"
            )
            st.segf = segf
            st.kn_bundle = None
            return False

        segf2, rem = _ensure_full_turns(segf)
        st.segf = segf2
        if get_segmentpath_callable:
            st.seg_path = get_segmentpath_callable()
        st.kn_bundle = kn_bundle

        # Auto-fill magnet order if available
        m0 = getattr(segf2, "magnet_order", None)
        if m0 is not None:
            try:
                magnet_order.value = int(m0)
            except Exception:
                pass

        # Auto-fill compensation scheme from kn bundle if head_csv
        if kn_bundle.head_cmp_connection and not comp_scheme.value.strip():
            comp_scheme.value = kn_bundle.head_cmp_connection

        ap_str = f"Aperture {segf2.aperture_id}" if segf2.aperture_id is not None else "Single aperture"
        m_str = f"m={segf2.magnet_order}" if segf2.magnet_order is not None else "m=?"
        kn_src = kn_bundle.source_type
        kn_H = len(kn_bundle.kn.orders)

        context_html.value = (
            f"<div style='padding:4px; background:#e8f4e8; border:1px solid #8c8;'>"
            f"<b>Segment:</b> Run={segf2.run_id}, {ap_str}, Seg={segf2.segment}, "
            f"{m_str}, n_turns={segf2.n_turns}<br>"
            f"<b>Kn:</b> source={kn_src}, H={kn_H}, "
            f"ext={'yes' if kn_bundle.kn.kn_ext is not None else 'no'}"
            f"</div>"
        )

        if rem > 0:
            log.write(f"Trimmed {rem} samples from tail to ensure full turns.")

        return True

    def _build_per_n_table() -> None:
        """Build the per-n selection dropdowns based on computed harmonics."""
        nonlocal per_n_dropdowns

        if st.result is None:
            per_n_table_html.value = "<i>No harmonics computed yet.</i>"
            per_n_container.children = []
            per_n_dropdowns = []
            return

        H = st.result.H
        orders = st.result.orders
        has_ext = st.kn_bundle and st.kn_bundle.kn.kn_ext is not None

        # Initialize selection array
        if st.per_n_selection is None or len(st.per_n_selection) != H:
            # Default: main from abs, others from cmp
            m = int(magnet_order.value)
            st.per_n_selection = np.ones(H, dtype=int)  # all cmp
            if 1 <= m <= H:
                st.per_n_selection[m - 1] = 0  # main from abs

        # Build dropdown widgets
        per_n_dropdowns = []
        rows = []

        options = [("Abs", CHANNEL_ABS), ("Cmp", CHANNEL_CMP)]
        if has_ext:
            options.append(("Ext", CHANNEL_EXT))

        for j, n in enumerate([int(x) for x in orders]):
            dd = w.Dropdown(
                options=options,
                value=int(st.per_n_selection[j]),
                description=f"n={n}:",
                style={"description_width": "50px"},
                layout=w.Layout(width="150px"),
            )

            # Capture j in closure
            def make_observer(idx):
                def on_change(change):
                    if st.per_n_selection is not None:
                        st.per_n_selection[idx] = int(change["new"])
                return on_change

            dd.observe(make_observer(j), names="value")
            per_n_dropdowns.append(dd)
            rows.append(dd)

        # Arrange in grid (4 per row)
        grid_rows = []
        for i in range(0, len(rows), 4):
            grid_rows.append(w.HBox(rows[i:i+4]))

        per_n_container.children = grid_rows
        per_n_table_html.value = "<b>Per-harmonic source selection:</b>"

    def _apply_preset_to_dropdowns() -> None:
        """Apply merge preset to the per-n selection dropdowns."""
        if st.result is None or st.per_n_selection is None:
            return

        H = st.result.H
        m = int(magnet_order.value)
        preset = merge_preset.value

        if preset == "abs_all":
            st.per_n_selection[:] = CHANNEL_ABS
        elif preset == "cmp_all":
            st.per_n_selection[:] = CHANNEL_CMP
        elif preset == "abs_main_cmp_others":
            st.per_n_selection[:] = CHANNEL_CMP
            if 1 <= m <= H:
                st.per_n_selection[m - 1] = CHANNEL_ABS
        elif preset == "abs_upto_m_cmp_above":
            st.per_n_selection[:] = CHANNEL_CMP
            if 1 <= m <= H:
                st.per_n_selection[:m] = CHANNEL_ABS
        elif preset == "recommended":
            if st.choice_recommended is not None:
                st.per_n_selection[:] = st.choice_recommended
            else:
                log.write("<b style='color:#b00'>No recommendation computed. Click 'Recommend Merge' first.</b>")
                return
        # "custom" does nothing - user manually adjusts

        # Update dropdowns to reflect
        for j, dd in enumerate(per_n_dropdowns):
            if j < len(st.per_n_selection):
                dd.value = int(st.per_n_selection[j])

    def _on_preset_change(_change) -> None:
        _apply_preset_to_dropdowns()

    merge_preset.observe(_on_preset_change, names="value")

    def _plot_main_field(res: LegacyKnPerTurn) -> None:
        with out_plot:
            out_plot.clear_output(wait=True)
            try:
                plt.close("all")
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            st.fig, st.ax = fig, ax

            t_idx = np.arange(res.main_field.size)
            y = np.real(res.main_field)
            ax.plot(t_idx, y, marker=".", linestyle="-", label="main field (Re)")

            if np.any(res.phi_bad):
                bad = np.where(res.phi_bad)[0]
                ax.plot(t_idx[bad], y[bad], marker="x", linestyle="None", color="red", label="phi_bad")

            ax.set_xlabel("turn index")
            ax.set_ylabel("Re(main_field)")
            ax.set_title("Main field after kn pipeline")
            ax.legend()
            plt.tight_layout()
            plt.show()

    def _on_compute(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("computing harmonics...")

            if not _refresh_context():
                return

            segf = st.segf
            kn = st.kn_bundle.kn

            tb = split_into_turns(segf)

            # Collect options
            opts = []
            if opt_dit.value:
                opts.append("dit")
            if opt_dri.value:
                opts.append("dri")
            if opt_rot.value:
                opts.append("rot")
            if opt_cel.value:
                opts.append("cel")
            if opt_fed.value:
                opts.append("fed")
            if opt_nor.value:
                opts.append("nor")

            Rref_m = float(rref_mm.value) * 1e-3

            res = compute_legacy_kn_per_turn(
                df_abs_turns=tb.df_abs,
                df_cmp_turns=tb.df_cmp,
                t_turns=tb.t,
                I_turns=tb.I,
                kn=kn,
                Rref_m=Rref_m,
                magnet_order=int(magnet_order.value),
                absCalib=float(abs_calib.value),
                options=tuple(opts),
                drift_mode=str(drift_mode.value),
                skew_main=bool(skew_main.value),
            )
            st.result = res

            # Reset merge state
            st.diag = None
            st.choice_recommended = None
            st.per_n_selection = None
            st.merge_result = None
            approve_merge.value = False

            # Build per-n table
            _build_per_n_table()

            # Enable next steps
            btn_recommend.disabled = False
            btn_apply_merge.disabled = False
            btn_export.disabled = True

            log.write(
                f"Computed harmonics: n_turns={res.C_abs.shape[0]}, H={res.H}. "
                "Now select merge options and apply."
            )
            _plot_main_field(res)

        except Exception as e:
            log.write(f"<b style='color:#b00'>Compute failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_recommend(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("computing merge recommendation...")

            if st.result is None:
                log.write("<b style='color:#b00'>No harmonics computed. Click 'Apply Kn' first.</b>")
                return

            res = st.result
            choice, diag = recommend_merge_choice(
                C_abs=res.C_abs,
                C_cmp=res.C_cmp,
                magnet_order=int(magnet_order.value),
                orders=res.orders,
            )
            st.choice_recommended = choice
            st.diag = diag

            # Display diagnostics table
            df = pd.DataFrame({
                "n": diag.orders,
                "noise_abs": [f"{x:.3e}" for x in diag.noise_abs],
                "noise_cmp": [f"{x:.3e}" for x in diag.noise_cmp],
                "mismatch": [f"{x:.3e}" for x in diag.mismatch],
                "recommended": [CHANNEL_NAMES.get(int(c), "?") for c in diag.selected],
                "flags": diag.flags,
            })
            diag_table_html.value = (
                "<b>Merge Diagnostics:</b><br>" +
                df.to_html(index=False, classes="table table-sm")
            )

            log.write("Computed merge recommendation. Select 'Use recommendation' preset to apply it.")

        except Exception as e:
            log.write(f"<b style='color:#b00'>Recommendation failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_apply_merge(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("applying merge...")

            if st.result is None:
                log.write("<b style='color:#b00'>No harmonics computed.</b>")
                return

            if not approve_merge.value:
                log.write("<b style='color:#b00'>Merge not applied. Tick the approval checkbox first.</b>")
                return

            if st.per_n_selection is None:
                log.write("<b style='color:#b00'>No per-n selection available.</b>")
                return

            res = st.result
            H = res.H

            # Use per_n_selection for custom merge
            # Note: merge_coefficients only supports abs(0) and cmp(1)
            # For ext channel, we need custom logic
            has_ext_selection = np.any(st.per_n_selection == CHANNEL_EXT)

            if has_ext_selection:
                # Custom merge with ext channel
                C_merged = np.empty_like(res.C_abs)
                for j in range(H):
                    ch = int(st.per_n_selection[j])
                    if ch == CHANNEL_ABS:
                        C_merged[:, j] = res.C_abs[:, j]
                    elif ch == CHANNEL_CMP:
                        C_merged[:, j] = res.C_cmp[:, j]
                    elif ch == CHANNEL_EXT:
                        # Ext channel not computed in LegacyKnPerTurn
                        # This would require extending the pipeline
                        log.write(f"<b style='color:#b00'>Ext channel selection for n={j+1} not yet supported.</b>")
                        C_merged[:, j] = res.C_abs[:, j]  # fallback
                choice = np.array(st.per_n_selection, copy=True)
            else:
                # Standard merge (abs/cmp only)
                C_merged, choice = merge_coefficients(
                    C_abs=res.C_abs,
                    C_cmp=res.C_cmp,
                    magnet_order=int(magnet_order.value),
                    mode="custom",
                    per_order_choice=st.per_n_selection.tolist(),
                )

            # Create MergeResult with full provenance
            merge_result = MergeResult(
                C_merged=C_merged,
                orders=res.orders,
                per_n_source_map=choice,
                compensation_scheme=comp_scheme.value.strip() or "unspecified",
                magnet_order=int(magnet_order.value),
                kn_provenance=st.kn_bundle,
                merge_mode=merge_preset.value,
                timestamp=KnBundle.now_iso(),
                diagnostics=st.diag,
                C_abs=res.C_abs,
                C_cmp=res.C_cmp,
            )
            st.merge_result = merge_result

            if set_merge_result_callable:
                set_merge_result_callable(merge_result)

            btn_export.disabled = False
            log.write(
                f"Merge applied: {merge_result.n_turns} turns, H={merge_result.H}. "
                f"Compensation scheme: '{merge_result.compensation_scheme}'. "
                "Ready to export."
            )

        except Exception as e:
            log.write(f"<b style='color:#b00'>Apply merge failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_export(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("exporting...")

            if st.result is None or st.merge_result is None:
                log.write("<b style='color:#b00'>Nothing to export. Apply merge first.</b>")
                return

            res = st.result
            mr = st.merge_result

            # Default file prefix
            prefix = "phase3b_merge"
            if st.seg_path:
                try:
                    base = os.path.basename(str(st.seg_path))
                    prefix = base.replace(".bin", "").replace(".txt", "")
                except Exception:
                    pass

            base_path = _saveas_dialog(
                initialfile=f"{prefix}_export.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Save export (base filename; suffixes will be appended)",
            )
            if not base_path:
                log.write("Export cancelled.")
                return

            root, ext = os.path.splitext(base_path)

            # Common scalar columns
            scalars = pd.DataFrame({
                "turn": np.arange(res.I_mean_A.size, dtype=int),
                "I_mean_A": res.I_mean_A,
                "dI_dt_A_per_s": res.dI_dt_A_per_s,
                "duration_s": res.duration_s,
                "time_median_s": res.time_median_s,
                "phi_out_rad": res.phi_out_rad,
                "phi_bad": res.phi_bad.astype(int),
                "x_m": res.x_m,
                "y_m": res.y_m,
            })

            # Pre-merge tables
            df_abs = pd.concat([scalars, _ba_table_from_C(res.C_abs, res.orders, prefix="abs_")], axis=1)
            df_cmp = pd.concat([scalars, _ba_table_from_C(res.C_cmp, res.orders, prefix="cmp_")], axis=1)

            abs_path = f"{root}_ABS.csv"
            cmp_path = f"{root}_CMP.csv"
            df_abs.to_csv(abs_path, index=False)
            df_cmp.to_csv(cmp_path, index=False)
            log.write(f"Saved: {abs_path}")
            log.write(f"Saved: {cmp_path}")

            # Merged table
            df_m = pd.concat([scalars, _ba_table_from_C(mr.C_merged, mr.orders, prefix="mrg_")], axis=1)
            mrg_path = f"{root}_MERGED.csv"
            df_m.to_csv(mrg_path, index=False)
            log.write(f"Saved: {mrg_path}")

            # Per-n source map
            df_map = mr.source_map_dataframe()
            map_path = f"{root}_SOURCE_MAP.csv"
            df_map.to_csv(map_path, index=False)
            log.write(f"Saved: {map_path}")

            # Full metadata / provenance
            meta = mr.to_metadata_dict()
            df_meta = pd.DataFrame([meta])
            meta_path = f"{root}_PROVENANCE.csv"
            df_meta.to_csv(meta_path, index=False)
            log.write(f"Saved: {meta_path}")

            # Diagnostics if available
            if st.diag is not None:
                d = st.diag
                df_d = pd.DataFrame({
                    "n": d.orders,
                    "noise_abs": d.noise_abs,
                    "noise_cmp": d.noise_cmp,
                    "mismatch": d.mismatch,
                    "selected": d.selected,
                    "flags": d.flags,
                })
                diag_path = f"{root}_DIAGNOSTICS.csv"
                df_d.to_csv(diag_path, index=False)
                log.write(f"Saved: {diag_path}")

            log.write("<b>Export complete.</b>")

        except Exception as e:
            log.write(f"<b style='color:#b00'>Export failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    # ---- Wire handlers ----
    _clear_button_handlers(btn_compute)
    btn_compute.on_click(_on_compute)

    _clear_button_handlers(btn_recommend)
    btn_recommend.on_click(_on_recommend)

    _clear_button_handlers(btn_apply_merge)
    btn_apply_merge.on_click(_on_apply_merge)

    _clear_button_handlers(btn_export)
    btn_export.on_click(_on_export)

    # Initial context refresh
    _refresh_context()

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Phase 3B: Harmonic Merge (Channel Selection)</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Apply kn calibration to compute harmonics, then select Abs/Cmp source per harmonic order. "
        "All outputs include full provenance (kn source, compensation scheme, per-n map)."
        "</div>"
    )

    params_box = w.VBox([
        w.HTML("<b>Parameters</b>"),
        w.HBox([rref_mm, abs_calib, magnet_order]),
        w.HBox([skew_main]),
    ])

    opts_box = w.VBox([
        w.HTML("<b>Processing options (legacy ordering)</b>"),
        w.HBox([opt_dit, opt_dri, opt_rot, opt_cel, opt_fed, opt_nor, drift_mode]),
    ])

    comp_box = w.VBox([
        w.HTML("<b>Compensation scheme (metadata)</b>"),
        w.HBox([comp_scheme]),
        comp_scheme_help,
    ])

    merge_box = w.VBox([
        w.HTML("<b>Merge configuration</b>"),
        w.HBox([merge_preset]),
        per_n_table_html,
        per_n_container,
        w.HBox([approve_merge, btn_apply_merge, btn_export]),
    ])

    main_panel = w.VBox(
        [
            context_html,
            w.HTML("<hr>"),
            params_box,
            opts_box,
            w.HBox([btn_compute, btn_recommend]),
            w.HTML("<hr>"),
            comp_box,
            w.HTML("<hr>"),
            merge_box,
            w.HTML("<hr>"),
            diag_table_html,
            out_plot,
        ],
        layout=w.Layout(width="68%", min_width="640px"),
    )

    diag_panel = w.VBox(
        [
            status,
            log.panel,
        ],
        layout=w.Layout(width="32%", min_width="320px"),
    )

    panel = w.VBox([header, w.HBox([main_panel, diag_panel])])

    _ACTIVE_PHASE3B_PANEL = panel
    return panel
