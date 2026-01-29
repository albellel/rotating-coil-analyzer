from __future__ import annotations

"""Phase 3A GUI: Coil Calibration (kn loading and computation).

This tab is responsible for:
1. Loading segment kn from a TXT file, OR
2. Computing segment kn from a measurement-head geometry CSV.

Output: A KnBundle object stored in shared state, which carries full provenance
metadata (source type, file paths, connection specs, timestamps).

Non-goals (handled by Phase 3B):
- Applying kn to harmonics
- Merging Abs/Cmp channels
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.models.frames import SegmentFrame

from rotating_coil_analyzer.analysis.kn_pipeline import (
    SegmentKn,
    load_segment_kn_txt,
)
from rotating_coil_analyzer.analysis.kn_head import (
    compute_head_kn_from_csv,
    compute_segment_kn_from_head,
    write_segment_kn_txt,
)
from rotating_coil_analyzer.analysis.kn_bundle import KnBundle


_ACTIVE_PHASE3A_PANEL: Optional[w.Widget] = None


@dataclass
class Phase3AState:
    """Local state for the Coil Calibration tab."""
    segf: Optional[SegmentFrame] = None
    seg_path: Optional[str] = None
    kn: Optional[SegmentKn] = None
    kn_bundle: Optional[KnBundle] = None
    busy: bool = False


def _openfile_dialog(
    *,
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> Optional[str]:
    """Open a native Open dialog (best effort). Returns None if tkinter unavailable."""
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


def _saveas_dialog(
    *,
    initialfile: str,
    defaultextension: str,
    filetypes: list[tuple[str, str]],
    title: str = "Save file",
) -> Optional[str]:
    """Open a native Save-As dialog (best effort). Returns None if tkinter unavailable."""
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


def build_phase3a_coil_calibration_panel(
    get_segmentframe_callable: Callable[[], Optional[SegmentFrame]],
    get_segmentpath_callable: Callable[[], Optional[str]] | None = None,
    set_kn_bundle_callable: Callable[[Optional[KnBundle]], None] | None = None,
) -> w.Widget:
    """Build the Phase 3A (Coil Calibration) panel.

    Parameters
    ----------
    get_segmentframe_callable
        Returns the current SegmentFrame from shared state (for context).
    get_segmentpath_callable
        Returns the current segment file path (for context).
    set_kn_bundle_callable
        Called when a KnBundle is created, to store it in shared state.
    """
    global _ACTIVE_PHASE3A_PANEL

    if _ACTIVE_PHASE3A_PANEL is not None:
        try:
            _ACTIVE_PHASE3A_PANEL.close()
        except Exception:
            pass
        _ACTIVE_PHASE3A_PANEL = None

    st = Phase3AState()

    log = HtmlLog(title="Coil Calibration log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- Context display ----
    context_html = w.HTML(
        "<div style='color:#666; padding:4px; background:#f8f8f8; border:1px solid #ddd;'>"
        "No segment loaded. Load a measurement in Phase I first."
        "</div>"
    )

    # ---- Kn source selection ----
    src_radio = w.ToggleButtons(
        options=[("Segment Kn TXT", "segment_txt"), ("Head geometry CSV", "head_csv")],
        value="segment_txt",
        description="Kn source:",
        style={"description_width": "110px"},
    )

    # Segment TXT widgets
    kn_path = w.Text(
        value="",
        description="Kn TXT:",
        placeholder="path to Kn_values_*.txt (segment Kn)",
        layout=w.Layout(width="700px"),
        style={"description_width": "110px"},
    )
    kn_browse = w.Button(description="Browse", button_style="")

    # Head CSV widgets
    head_csv_path = w.Text(
        value="",
        description="Head CSV:",
        placeholder="path to measurement head CSV (geometry)",
        layout=w.Layout(width="700px"),
        style={"description_width": "110px"},
        disabled=True,
    )
    head_browse = w.Button(description="Browse", disabled=True)

    # Head-CSV computation controls
    head_warm = w.Checkbox(value=True, description="Warm geometry", disabled=True)
    head_use_design_radius = w.Checkbox(value=True, description="Use design radius", disabled=True)
    head_strict_header = w.Checkbox(value=True, description="Strict header", disabled=True)
    head_n_multipoles = w.IntText(
        value=15,
        description="N multipoles:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
        disabled=True,
    )

    # Connection specifications
    head_abs_conn = w.Text(
        value="",
        description="Absolute conn:",
        placeholder="e.g. 1.2 (single coil)",
        layout=w.Layout(width="400px"),
        style={"description_width": "120px"},
        disabled=True,
    )
    head_cmp_conn = w.Text(
        value="",
        description="Compensated conn:",
        placeholder="e.g. 1.1-1.3 (A-C scheme)",
        layout=w.Layout(width="400px"),
        style={"description_width": "120px"},
        disabled=True,
    )
    head_ext_conn = w.Text(
        value="",
        description="External conn:",
        placeholder="optional",
        layout=w.Layout(width="400px"),
        style={"description_width": "120px"},
        disabled=True,
    )

    # ---- Actions ----
    btn_load_kn = w.Button(description="Load / Compute Kn", button_style="info")
    btn_export_kn = w.Button(description="Export Kn TXT", button_style="")
    btn_export_kn.disabled = True

    # ---- Result display ----
    result_html = w.HTML("<i>No kn loaded yet.</i>")

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _refresh_context() -> None:
        """Update context display from Phase I."""
        segf = get_segmentframe_callable()
        if segf is None:
            context_html.value = (
                "<div style='color:#666; padding:4px; background:#f8f8f8; border:1px solid #ddd;'>"
                "No segment loaded. Load a measurement in Phase I first."
                "</div>"
            )
            st.segf = None
            st.seg_path = None
        else:
            st.segf = segf
            if get_segmentpath_callable:
                st.seg_path = get_segmentpath_callable()
            ap_str = f"Aperture {segf.aperture_id}" if segf.aperture_id is not None else "Single aperture"
            m_str = f"m={segf.magnet_order}" if segf.magnet_order is not None else "m=?"
            context_html.value = (
                f"<div style='padding:4px; background:#e8f4e8; border:1px solid #8c8;'>"
                f"<b>Context:</b> Run={segf.run_id}, {ap_str}, Segment={segf.segment}, "
                f"{m_str}, n_turns={segf.n_turns}, Ns={segf.samples_per_turn}"
                f"</div>"
            )

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

    def _on_src_change(_chg) -> None:
        _update_source_ui()

    def _on_kn_browse(_btn) -> None:
        p = _openfile_dialog(
            title="Select segment kn TXT",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if p:
            kn_path.value = p

    def _on_head_browse(_btn) -> None:
        p = _openfile_dialog(
            title="Select head CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if p:
            head_csv_path.value = p

    def _on_load_kn(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _refresh_context()
            _set_status("loading kn...")

            if src_radio.value == "segment_txt":
                # Load from TXT file
                p = kn_path.value.strip()
                if not p:
                    log.write("<b style='color:#b00'>Please provide a kn TXT path.</b>")
                    return

                kn = load_segment_kn_txt(p)
                st.kn = kn

                # Create KnBundle with provenance
                bundle = KnBundle(
                    kn=kn,
                    source_type="segment_txt",
                    source_path=p,
                    timestamp=KnBundle.now_iso(),
                    segment_id=st.segf.segment if st.segf else None,
                    aperture_id=st.segf.aperture_id if st.segf else None,
                )
                st.kn_bundle = bundle

                btn_export_kn.disabled = True  # No need to export if loaded from file
                log.write(f"Loaded segment kn from TXT: {p}")
                log.write(f"  H={len(kn.orders)} harmonics, ext={'yes' if kn.kn_ext is not None else 'no'}")

            else:
                # Compute from head CSV
                p = head_csv_path.value.strip()
                if not p:
                    log.write("<b style='color:#b00'>Please provide a head CSV path.</b>")
                    return
                if not head_abs_conn.value.strip() or not head_cmp_conn.value.strip():
                    log.write("<b style='color:#b00'>Please fill Absolute conn and Compensated conn.</b>")
                    return

                abs_conn = head_abs_conn.value.strip()
                cmp_conn = head_cmp_conn.value.strip()
                ext_conn = head_ext_conn.value.strip() if head_ext_conn.value.strip() else None

                head = compute_head_kn_from_csv(
                    p,
                    warm_geometry=bool(head_warm.value),
                    n_multipoles=int(head_n_multipoles.value),
                    use_design_radius=bool(head_use_design_radius.value),
                    strict_header=bool(head_strict_header.value),
                )
                kn = compute_segment_kn_from_head(
                    head,
                    abs_connection=abs_conn,
                    cmp_connection=cmp_conn,
                    ext_connection=ext_conn,
                    source_label=f"head_csv:{p}",
                )
                st.kn = kn

                # Create KnBundle with full head-CSV provenance
                bundle = KnBundle(
                    kn=kn,
                    source_type="head_csv",
                    source_path=p,
                    timestamp=KnBundle.now_iso(),
                    segment_id=st.segf.segment if st.segf else None,
                    aperture_id=st.segf.aperture_id if st.segf else None,
                    head_abs_connection=abs_conn,
                    head_cmp_connection=cmp_conn,
                    head_ext_connection=ext_conn,
                    head_warm_geometry=bool(head_warm.value),
                    head_n_multipoles=int(head_n_multipoles.value),
                )
                st.kn_bundle = bundle

                btn_export_kn.disabled = False  # Can export computed kn
                log.write(f"Computed segment kn from head CSV: {p}")
                log.write(f"  Abs='{abs_conn}', Cmp='{cmp_conn}'" + (f", Ext='{ext_conn}'" if ext_conn else ""))
                log.write(f"  H={len(kn.orders)} harmonics, warm={head_warm.value}")

            # Store in shared state
            if set_kn_bundle_callable:
                set_kn_bundle_callable(st.kn_bundle)

            # Update result display
            kn = st.kn
            lines = [
                f"<b>Kn loaded successfully</b>",
                f"<br>Source: {st.kn_bundle.source_type}",
                f"<br>Path: <code>{st.kn_bundle.source_path}</code>",
                f"<br>Harmonics: n=1..{len(kn.orders)}",
                f"<br>External channel: {'yes' if kn.kn_ext is not None else 'no'}",
                f"<br>Timestamp: {st.kn_bundle.timestamp}",
            ]
            if st.kn_bundle.source_type == "head_csv":
                lines.append(f"<br>Abs connection: {st.kn_bundle.head_abs_connection}")
                lines.append(f"<br>Cmp connection: {st.kn_bundle.head_cmp_connection}")
                if st.kn_bundle.head_ext_connection:
                    lines.append(f"<br>Ext connection: {st.kn_bundle.head_ext_connection}")
            result_html.value = "".join(lines)

        except Exception as e:
            log.write(f"<b style='color:#b00'>Kn load failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_export_kn(_btn) -> None:
        if st.kn is None:
            log.write("<b style='color:#b00'>No kn available to export.</b>")
            return

        initial = "Kn_values_Seg_computed.txt"
        p = _saveas_dialog(
            title="Save segment kn TXT",
            initialfile=initial,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not p:
            log.write("Export cancelled.")
            return

        try:
            write_segment_kn_txt(st.kn, p)
            log.write(f"Exported segment kn to: {p}")
        except Exception as e:
            log.write(f"<b style='color:#b00'>Export failed:</b> {e}")
            raise

    # ---- Wire handlers ----
    _clear_button_handlers(kn_browse)
    kn_browse.on_click(_on_kn_browse)

    _clear_button_handlers(head_browse)
    head_browse.on_click(_on_head_browse)

    src_radio.observe(_on_src_change, names="value")

    _clear_button_handlers(btn_load_kn)
    btn_load_kn.on_click(_on_load_kn)

    _clear_button_handlers(btn_export_kn)
    btn_export_kn.on_click(_on_export_kn)

    _update_source_ui()
    _refresh_context()

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Phase 3A: Coil Calibration (Sensitivity Coefficients)</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Load kn from a TXT file or compute from measurement-head geometry CSV. "
        "The resulting KnBundle is passed to the Harmonic Merge tab (Phase 3B)."
        "</div>"
    )

    row_src = w.VBox([
        w.HTML("<b>Kn Source Selection</b>"),
        src_radio,
    ])

    row_txt = w.VBox([
        w.HTML("<b>Option 1: Load from segment kn TXT file</b>"),
        w.HBox([kn_path, kn_browse]),
    ])

    row_head = w.VBox([
        w.HTML("<b>Option 2: Compute from measurement-head geometry CSV</b>"),
        w.HBox([head_csv_path, head_browse]),
        w.HBox([head_warm, head_use_design_radius, head_strict_header, head_n_multipoles]),
        w.HTML("<i>Connection specifications (coil wiring):</i>"),
        head_abs_conn,
        head_cmp_conn,
        head_ext_conn,
    ])

    row_actions = w.HBox([btn_load_kn, btn_export_kn])

    main_panel = w.VBox(
        [
            context_html,
            w.HTML("<hr>"),
            row_src,
            w.HTML("<hr>"),
            row_txt,
            w.HTML("<hr>"),
            row_head,
            w.HTML("<hr>"),
            row_actions,
            w.HTML("<hr>"),
            w.HTML("<b>Result</b>"),
            result_html,
        ],
        layout=w.Layout(width="65%", min_width="600px"),
    )

    diag_panel = w.VBox(
        [
            status,
            log.panel,
        ],
        layout=w.Layout(width="35%", min_width="320px"),
    )

    panel = w.VBox([header, w.HBox([main_panel, diag_panel])])

    _ACTIVE_PHASE3A_PANEL = panel
    return panel
