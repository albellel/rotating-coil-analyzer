from __future__ import annotations

"""Phase III GUI: $k_n$ loading, application, and user-approved Abs/Cmp merge.

Design goals
------------
- Keep Phase II GUI unchanged (already dense).
- Implement a dedicated Phase III tab with:
  (i) segment $k_n$ TXT loader (4 or 6 columns),
  (ii) application of $k_n$ with rotation reference computed post-$k_n$,
  (iii) CEL + feeddown + normalization ordering consistent with legacy,
  (iv) explicit, auditable merge that is *never* applied without user approval.

This tab intentionally focuses on *workflow clarity*:
- You compute Abs and Cmp calibrated harmonics first.
- You export/save pre-merge tables (Abs + Cmp) before any merge.
- You compute merge diagnostics and a recommended choice.
- You apply a merge only if you tick an explicit approval checkbox.

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import ipywidgets as w

import matplotlib.pyplot as plt

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns

from rotating_coil_analyzer.analysis.kn_pipeline import (
    SegmentKn,
    LegacyKnPerTurn,
    load_segment_kn_txt,
    compute_legacy_kn_per_turn,
    merge_coefficients,
)
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice, MergeDiagnostics
from rotating_coil_analyzer.analysis.kn_head import (
    compute_head_kn_from_csv,
    compute_segment_kn_from_head,
    write_segment_kn_txt,
)


# Keep a single active Phase III panel per kernel (defensive).
_ACTIVE_PHASE3_PANEL: Optional[w.Widget] = None


@dataclass
class Phase3State:
    segf: Optional[SegmentFrame] = None
    seg_path: Optional[str] = None

    kn: Optional[SegmentKn] = None
    kn_path: Optional[str] = None

    result: Optional[LegacyKnPerTurn] = None

    # merge recommendation
    diag: Optional[MergeDiagnostics] = None
    choice_recommended: Optional[np.ndarray] = None

    # merged coefficients (post-user-approval)
    C_merged: Optional[np.ndarray] = None
    choice_applied: Optional[np.ndarray] = None

    fig: Optional[Any] = None
    ax: Optional[Any] = None

    busy: bool = False


def _ensure_full_turns(segf: SegmentFrame) -> tuple[SegmentFrame, int]:
    """Trim tail remainder samples to keep full turns (allowed by project policy)."""
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


def _openfile_dialog(
    *,
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> Optional[str]:
    """Open a native Open dialog (best effort). Returns None if tkinter is unavailable."""
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

        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes or [("All files", "*.*")],
        )
        return str(path) if path else None
    finally:
        try:
            if root is not None:
                root.destroy()
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


def _clear_button_handlers(btn: w.Button) -> None:
    """Defensive: remove any accumulated click handlers."""
    try:
        btn._click_handlers.callbacks.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def build_phase3_kn_panel(
    get_segmentframe_callable: Callable[[], Optional[SegmentFrame]],
    get_segmentpath_callable: Callable[[], Optional[str]] | None = None,
) -> w.Widget:
    """Build the Phase III (k_n) panel."""
    global _ACTIVE_PHASE3_PANEL

    # Close any previous Phase III panel (defensive against stacked live instances).
    if _ACTIVE_PHASE3_PANEL is not None:
        try:
            _ACTIVE_PHASE3_PANEL.close()
        except Exception:
            pass
        _ACTIVE_PHASE3_PANEL = None

    st = Phase3State()

    log = HtmlLog(title="Phase III log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- k_n source selection ----
    src_radio = w.ToggleButtons(
        options=[("Segment k_n TXT", "segment_txt"), ("Head geometry CSV", "head_csv")],
        value="segment_txt",
        description="k_n source:",
        style={"description_width": "110px"},
    )

    kn_path = w.Text(
        value="",
        description="k_n TXT:",
        placeholder="path to Kn_values_*.txt",
        layout=w.Layout(width="780px"),
        style={"description_width": "110px"},
    )
    kn_browse = w.Button(description="Browse", button_style="")

    head_csv_path = w.Text(
        value="",
        description="Head CSV:",
        placeholder="path to measurement head CSV (geometry)",
        layout=w.Layout(width="780px"),
        style={"description_width": "110px"},
        disabled=True,
    )
    head_browse = w.Button(description="Browse", disabled=True)

    # ---- head-CSV computation controls (enabled only when src=head_csv) ----
    # Defaults: warm=True and use_design_radius=True (typical case when calibrated radius is empty)
    head_warm = w.Checkbox(value=True, description="Warm geometry", disabled=True)
    head_use_design_radius = w.Checkbox(value=True, description="Use design radius", disabled=True)
    head_strict_header = w.Checkbox(value=True, description="Strict header", disabled=True)
    head_n_multipoles = w.IntText(
        value=15,
        description="N multipoles:",
        style={"description_width": "110px"},
        layout=w.Layout(width="240px"),
        disabled=True,
    )
    head_abs_conn = w.Text(
        value="",
        description="Abs conn:",
        placeholder="e.g. 1.1-1.3 (A.C terms)",
        layout=w.Layout(width="520px"),
        style={"description_width": "110px"},
        disabled=True,
    )
    head_cmp_conn = w.Text(
        value="",
        description="Cmp conn:",
        placeholder="e.g. 1.2 (A.C terms)",
        layout=w.Layout(width="520px"),
        style={"description_width": "110px"},
        disabled=True,
    )
    head_ext_conn = w.Text(
        value="",
        description="Ext conn:",
        placeholder="optional",
        layout=w.Layout(width="520px"),
        style={"description_width": "110px"},
        disabled=True,
    )
    btn_export_kn_txt = w.Button(description="Export segment k_n TXT", button_style="")
    btn_export_kn_txt.disabled = True

    # ---- numeric parameters ----
    rref_mm = w.FloatText(
        value=17.0,
        description="Rref [mm]:",
        style={"description_width": "110px"},
        layout=w.Layout(width="240px"),
    )

    abs_calib = w.FloatText(
        value=1.0,
        description="absCalib:",
        style={"description_width": "110px"},
        layout=w.Layout(width="240px"),
    )

    magnet_order = w.IntText(
        value=2,
        description="Main m:",
        style={"description_width": "110px"},
        layout=w.Layout(width="240px"),
    )

    skew_main = w.Checkbox(value=False, description="skew main (legacy skw)")

    # ---- processing options (legacy ordering) ----
    opt_dit = w.Checkbox(value=False, description="dit")
    opt_dri = w.Checkbox(value=True, description="dri")
    opt_rot = w.Checkbox(value=True, description="rot")
    opt_cel = w.Checkbox(value=False, description="cel")
    opt_fed = w.Checkbox(value=False, description="fed")
    opt_nor = w.Checkbox(value=False, description="nor")

    drift_mode = w.Dropdown(
        options=[("legacy", "legacy"), ("weighted", "weighted")],
        value="legacy",
        description="drift mode:",
        style={"description_width": "110px"},
        layout=w.Layout(width="240px"),
    )

    # ---- actions ----
    btn_load_kn = w.Button(description="Load k_n", button_style="info")
    btn_compute = w.Button(description="Compute (Abs/Cmp)", button_style="success")
    btn_recommend = w.Button(description="Recommend merge", button_style="")

    merge_mode = w.Dropdown(
        options=[
            ("recommended", "recommended"),
            ("abs main, cmp others (legacy default)", "abs_main_cmp_others"),
            ("abs all", "abs_all"),
            ("cmp all", "cmp_all"),
            ("abs up to m, cmp above", "abs_upto_m_cmp_above"),
        ],
        value="recommended",
        description="merge:",
        style={"description_width": "110px"},
        layout=w.Layout(width="520px"),
    )

    approve_merge = w.Checkbox(value=False, description="I approve applying the merge")
    btn_apply_merge = w.Button(description="Apply merge", button_style="warning")

    btn_export = w.Button(description="Export CSV (pre-merge + merge)", button_style="")

    # ---- displays ----
    diag_table = w.HTML("<i>Merge diagnostics will appear here.</i>")

    # matplotlib output area (simple, non-interactive by default)
    out_plot = w.Output()

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _refresh_segmentframe() -> bool:
        segf = get_segmentframe_callable()
        if segf is None:
            log.write("<b style='color:#b00'>No SegmentFrame loaded.</b> Load a measurement in Phase I first.")
            return False

        segf2, rem = _ensure_full_turns(segf)
        if rem:
            log.write(f"Trimmed tail remainder: removed {rem} samples to keep full turns.")
        st.segf = segf2

        if get_segmentpath_callable is not None:
            st.seg_path = get_segmentpath_callable()

        # auto-fill magnet order if present
        m0 = getattr(segf2, "magnet_order", None)
        if m0 is not None:
            try:
                magnet_order.value = int(m0)
            except Exception:
                pass

        return True

    def _on_kn_browse(_btn) -> None:
        p = _openfile_dialog(title="Select segment k_n TXT", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if p:
            kn_path.value = p

    def _on_head_browse(_btn) -> None:
        p = _openfile_dialog(title="Select head CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            head_csv_path.value = p

    def _update_source_ui() -> None:
        is_txt = (src_radio.value == "segment_txt")

        # Segment TXT widgets
        kn_path.disabled = not is_txt
        kn_browse.disabled = not is_txt

        # Head CSV widgets
        head_csv_path.disabled = is_txt
        head_browse.disabled = is_txt
        head_warm.disabled = is_txt
        head_use_design_radius.disabled = is_txt
        head_strict_header.disabled = is_txt
        head_n_multipoles.disabled = is_txt
        head_abs_conn.disabled = is_txt
        head_cmp_conn.disabled = is_txt
        head_ext_conn.disabled = is_txt

        # Actions
        btn_load_kn.disabled = False

        # Export TXT is only meaningful for head-csv computed segment k_n
        btn_export_kn_txt.disabled = (is_txt or st.kn is None)

    def _on_src_change(_chg) -> None:
        _update_source_ui()

    def _on_load_kn(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("loading k_n")
            if src_radio.value == "segment_txt":
                p = kn_path.value.strip()
                if not p:
                    log.write("<b style='color:#b00'>Please provide a k_n TXT path.</b>")
                    return

                kn = load_segment_kn_txt(p)
                st.kn = kn
                st.kn_path = p
                btn_export_kn_txt.disabled = True
                log.write(f"Loaded segment k_n: {p} (H={len(kn.orders)} harmonics)")
            else:
                p = head_csv_path.value.strip()
                if not p:
                    log.write("<b style='color:#b00'>Please provide a head CSV path.</b>")
                    return
                if not head_abs_conn.value.strip() or not head_cmp_conn.value.strip():
                    log.write("<b style='color:#b00'>Please fill Abs conn and Cmp conn.</b>")
                    return

                head = compute_head_kn_from_csv(
                    p,
                    warm_geometry=bool(head_warm.value),
                    n_multipoles=int(head_n_multipoles.value),
                    use_design_radius=bool(head_use_design_radius.value),
                    strict_header=bool(head_strict_header.value),
                )
                seg_kn = compute_segment_kn_from_head(
                    head,
                    abs_connection=head_abs_conn.value.strip(),
                    cmp_connection=head_cmp_conn.value.strip(),
                    ext_connection=head_ext_conn.value.strip() if head_ext_conn.value.strip() else None,
                    source_label=f"head_csv:{p}",
                )
                st.kn = seg_kn
                st.kn_path = f"head_csv:{p}"
                btn_export_kn_txt.disabled = False
                log.write(
                    f"Computed segment k_n from head CSV: {p} (H={len(seg_kn.orders)}). "
                    f"Abs='{head_abs_conn.value.strip()}', Cmp='{head_cmp_conn.value.strip()}'"
                )
        except Exception as e:
            log.write(f"<b style='color:#b00'>k_n load failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_export_kn_txt(_btn) -> None:
        if st.kn is None:
            log.write("<b style='color:#b00'>No k_n available to export.</b>")
            return
        if src_radio.value != "head_csv":
            log.write("<b style='color:#b00'>Export segment k_n TXT is only for head-CSV computed k_n.</b>")
            return

        initial = "Kn_values_Seg_from_head.txt"
        p = _saveas_dialog(
            title="Save segment k_n TXT",
            initialfile=initial,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not p:
            log.write("Save-as cancelled (or tkinter not available).")
            return
        try:
            write_segment_kn_txt(st.kn, p)
            log.write(f"Saved segment k_n TXT: {p}")
        except Exception as e:
            log.write(f"<b style='color:#b00'>Failed to export segment k_n TXT:</b> {e}")
            raise

    def _compute_stage() -> Optional[LegacyKnPerTurn]:
        if not _refresh_segmentframe():
            return None
        if st.kn is None:
            log.write("<b style='color:#b00'>No k_n loaded.</b> Load k_n first.")
            return None

        tb = split_into_turns(st.segf)

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
            kn=st.kn,
            Rref_m=Rref_m,
            magnet_order=int(magnet_order.value),
            absCalib=float(abs_calib.value),
            options=tuple(opts),
            drift_mode=str(drift_mode.value),
            skew_main=bool(skew_main.value),
        )
        return res

    def _plot_main_field(res: LegacyKnPerTurn) -> None:
        with out_plot:
            out_plot.clear_output(wait=True)
            try:
                plt.close("all")
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            st.fig, st.ax = fig, ax

            # Show real(main_field) vs turn index, plus a marker for phi_bad
            t_idx = np.arange(res.main_field.size)
            y = np.real(res.main_field)
            ax.plot(t_idx, y, marker=".", linestyle="-")

            if np.any(res.phi_bad):
                bad = np.where(res.phi_bad)[0]
                ax.plot(t_idx[bad], y[bad], marker="x", linestyle="None")

            ax.set_xlabel("turn index")
            ax.set_ylabel("Re(main_field)")
            ax.set_title("Main field after k_n pipeline (Re)")
            plt.show()

    def _on_compute(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("computing")
            res = _compute_stage()
            if res is None:
                return
            st.result = res

            # clear any previous merge artifacts
            st.diag = None
            st.choice_recommended = None
            st.C_merged = None
            st.choice_applied = None
            approve_merge.value = False
            diag_table.value = "<i>Merge diagnostics will appear here.</i>"

            log.write(
                "Computed per-turn k_n pipeline for Abs/Cmp. "
                "Pre-merge data (Abs & Cmp) is preserved in memory; export will save both." 
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
            _set_status("recommending merge")
            if st.result is None:
                log.write("<b style='color:#b00'>No computed result.</b> Click Compute first.")
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

            df = pd.DataFrame(
                {
                    "n": diag.orders,
                    "noise_abs": diag.noise_abs,
                    "noise_cmp": diag.noise_cmp,
                    "mismatch": diag.mismatch,
                    "selected(0=abs,1=cmp)": diag.selected,
                    "flags": diag.flags,
                }
            )
            diag_table.value = df.to_html(index=False)
            log.write("Computed merge diagnostics and a recommended per-order choice. Review and then approve/apply explicitly.")
        except Exception as e:
            log.write(f"<b style='color:#b00'>Recommend failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_apply_merge(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("applying merge")
            if st.result is None:
                log.write("<b style='color:#b00'>No computed result.</b> Click Compute first.")
                return
            if not approve_merge.value:
                log.write("<b style='color:#b00'>Merge not applied.</b> Tick the approval checkbox first.")
                return

            res = st.result
            mode = str(merge_mode.value)
            if mode == "recommended":
                if st.choice_recommended is None:
                    log.write("<b style='color:#b00'>No recommendation computed.</b> Click Recommend merge first.")
                    return
                C_merged, choice = merge_coefficients(
                    C_abs=res.C_abs,
                    C_cmp=res.C_cmp,
                    magnet_order=int(magnet_order.value),
                    mode="custom",
                    per_order_choice=st.choice_recommended,
                )
            else:
                C_merged, choice = merge_coefficients(
                    C_abs=res.C_abs,
                    C_cmp=res.C_cmp,
                    magnet_order=int(magnet_order.value),
                    mode=mode,
                )

            st.C_merged = C_merged
            st.choice_applied = choice
            log.write("Merge applied in-memory. Export will save Abs, Cmp, and Merged tables + the merge choice.")
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
            _set_status("exporting")
            if st.result is None:
                log.write("<b style='color:#b00'>Nothing to export.</b> Compute first.")
                return

            res = st.result

            # default file prefix
            prefix = "phase3_kn"
            if st.seg_path:
                try:
                    import os
                    base = os.path.basename(st.seg_path)
                    prefix = base.replace(".bin", "").replace(".txt", "")
                except Exception:
                    pass

            # Choose a base filename via Save-As. We will append suffixes.
            base_path = _saveas_dialog(
                initialfile=f"{prefix}_kn_export.csv",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title="Save export (base filename; suffixes will be appended)",
            )
            if not base_path:
                log.write("Export canceled.")
                return

            import os
            root, ext = os.path.splitext(base_path)

            # Common scalar columns
            scalars = pd.DataFrame(
                {
                    "turn": np.arange(res.I_mean_A.size, dtype=int),
                    "I_mean_A": res.I_mean_A,
                    "dI_dt_A_per_s": res.dI_dt_A_per_s,
                    "duration_s": res.duration_s,
                    "time_median_s": res.time_median_s,
                    "phi_out_rad": res.phi_out_rad,
                    "phi_bad": res.phi_bad.astype(int),
                    "x_m": res.x_m,
                    "y_m": res.y_m,
                }
            )

            df_abs = pd.concat([scalars, _ba_table_from_C(res.C_abs, res.orders, prefix="abs_")], axis=1)
            df_cmp = pd.concat([scalars, _ba_table_from_C(res.C_cmp, res.orders, prefix="cmp_")], axis=1)

            abs_path = f"{root}_ABS.csv"
            cmp_path = f"{root}_CMP.csv"
            df_abs.to_csv(abs_path, index=False)
            df_cmp.to_csv(cmp_path, index=False)

            log.write(f"Saved pre-merge per-turn tables: {abs_path} and {cmp_path}")

            if st.C_merged is not None and st.choice_applied is not None:
                df_m = pd.concat([scalars, _ba_table_from_C(st.C_merged, res.orders, prefix="mrg_")], axis=1)
                mrg_path = f"{root}_MERGED.csv"
                df_m.to_csv(mrg_path, index=False)

                ch = pd.DataFrame({"n": res.orders, "choice(0=abs,1=cmp)": st.choice_applied})
                ch_path = f"{root}_MERGE_CHOICE.csv"
                ch.to_csv(ch_path, index=False)

                log.write(f"Saved merged table and merge choice: {mrg_path} and {ch_path}")
            else:
                log.write("No merge was applied; only ABS and CMP tables were exported.")

            if st.diag is not None:
                d = st.diag
                df_d = pd.DataFrame(
                    {
                        "n": d.orders,
                        "noise_abs": d.noise_abs,
                        "noise_cmp": d.noise_cmp,
                        "mismatch": d.mismatch,
                        "selected(0=abs,1=cmp)": d.selected,
                        "flags": d.flags,
                    }
                )
                d_path = f"{root}_MERGE_DIAGNOSTICS.csv"
                df_d.to_csv(d_path, index=False)
                log.write(f"Saved merge diagnostics: {d_path}")

        except Exception as e:
            log.write(f"<b style='color:#b00'>Export failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    # ---- wire handlers ----
    _clear_button_handlers(kn_browse)
    kn_browse.on_click(_on_kn_browse)

    _clear_button_handlers(head_browse)
    head_browse.on_click(_on_head_browse)

    src_radio.observe(_on_src_change, names="value")

    _clear_button_handlers(btn_load_kn)
    btn_load_kn.on_click(_on_load_kn)

    _clear_button_handlers(btn_export_kn_txt)
    btn_export_kn_txt.on_click(_on_export_kn_txt)

    _clear_button_handlers(btn_compute)
    btn_compute.on_click(_on_compute)

    _clear_button_handlers(btn_recommend)
    btn_recommend.on_click(_on_recommend)

    _clear_button_handlers(btn_apply_merge)
    btn_apply_merge.on_click(_on_apply_merge)

    _clear_button_handlers(btn_export)
    btn_export.on_click(_on_export)

    _update_source_ui()

    # ---- layout ----
    row_kn = w.HBox([kn_path, kn_browse])
    row_head = w.HBox([head_csv_path, head_browse])
    row_head_opts = w.HBox([head_warm, head_use_design_radius, head_strict_header, head_n_multipoles, btn_export_kn_txt])
    row_head_conn1 = w.HBox([head_abs_conn])
    row_head_conn2 = w.HBox([head_cmp_conn])
    row_head_conn3 = w.HBox([head_ext_conn])

    row_params1 = w.HBox([rref_mm, abs_calib, magnet_order])
    row_params2 = w.HBox([skew_main])

    row_opts = w.HBox([opt_dit, opt_dri, opt_rot, opt_cel, opt_fed, opt_nor, drift_mode])
    row_actions = w.HBox([btn_load_kn, btn_compute, btn_recommend])

    merge_box = w.VBox([
        w.HBox([merge_mode, approve_merge, btn_apply_merge]),
        w.HBox([btn_export]),
        diag_table,
    ])

    panel = w.VBox(
        [
            w.HTML("<h3>Phase III — k_n loading, application, and merge</h3>"),
            src_radio,
            row_kn,
            row_head,
            row_head_opts,
            row_head_conn1,
            row_head_conn2,
            row_head_conn3,
            w.HTML("<hr>"),
            w.HTML("<b>Parameters</b>"),
            row_params1,
            row_params2,
            w.HTML("<b>Options (legacy ordering)</b>"),
            row_opts,
            row_actions,
            w.HTML("<hr>"),
            merge_box,
            w.HTML("<hr>"),
            out_plot,
            status,
            log.panel,
        ]
    )

    _ACTIVE_PHASE3_PANEL = panel
    return panel
