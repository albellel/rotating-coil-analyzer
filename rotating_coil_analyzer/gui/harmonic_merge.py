from __future__ import annotations

"""Harmonic Merge GUI tab (apply kn + per-n channel selection).

This tab is responsible for:
1. Applying the kn calibration (from Coil Calibration tab) to compute Abs/Cmp harmonics per turn.
2. Displaying diagnostics and allowing per-harmonic-order source selection.
3. Providing preset merge modes and custom per-n selection.
4. Exporting merged results with full provenance.

Inputs:
- SegmentFrame from Catalog tab
- KnBundle from Coil Calibration tab

Output:
- MergeResult with full traceability (kn provenance, compensation scheme, per-n map)
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
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
    bucked_cmp_orders,
    compute_legacy_kn_per_turn,
    merge_coefficients,
    safe_normalize_to_units,
)
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice, MergeDiagnostics
from rotating_coil_analyzer.analysis.utility_functions import (
    ba_table_from_C,
    mixed_format_table,
    diagnose_cel_fed,
    CelFedDiagnostic,
    mad_sigma_clip,
    plateau_summary,
)
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
class HarmonicMergeState:
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

    # Post-merge normalization results
    C_units: Optional[np.ndarray] = None
    ok_main: Optional[np.ndarray] = None

    # Options used in last compute (to detect whether 'nor' was checked)
    opts: Optional[Tuple[str, ...]] = None

    # Outlier removal mask (True = keep)
    outlier_keep_mask: Optional[np.ndarray] = None

    # Parameters actually used at the last "Apply Kn" (m, Rref, skew, absCalib).
    # The merge, the units and the presets use THESE, never the live widgets.
    used_params: Optional[dict] = None

    # Identity of the segment / catalog the widgets were last pre-filled for.
    seg_key: Optional[tuple] = None
    cat_key: Optional[str] = None
    # Last (segment, kn) pair for which a kn/segment mismatch was reported.
    kn_warned_key: Optional[tuple] = None

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
    from rotating_coil_analyzer.gui._dialogs import saveas_dialog
    return saveas_dialog(
        initialfile=initialfile,
        defaultextension=defaultextension,
        filetypes=filetypes,
        title=title,
    )


def _clear_button_handlers(btn: w.Button) -> None:
    """Defensive: remove any accumulated click handlers."""
    try:
        btn._click_handlers.callbacks.clear()  # type: ignore[attr-defined]
    except Exception:
        pass


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


def segment_key(segf: Optional[SegmentFrame]) -> Optional[tuple]:
    """Identity of a loaded segment: (source file, run, aperture, segment)."""
    if segf is None:
        return None
    return (
        str(getattr(segf, "source_path", "")),
        str(getattr(segf, "run_id", "")),
        getattr(segf, "aperture_id", None),
        str(getattr(segf, "segment", "")),
    )


def kn_segment_mismatch(bundle: Optional[KnBundle], segf: Optional[SegmentFrame]) -> Optional[str]:
    """Describe a mismatch between the segment a kn bundle was loaded for and *segf*.

    Returns None when they are consistent, or when the bundle does not record
    a segment / aperture (e.g. a kn file loaded before any segment).
    """
    if bundle is None or segf is None:
        return None
    problems: List[str] = []
    kn_seg = getattr(bundle, "segment_id", None)
    if kn_seg is not None and str(kn_seg) != str(segf.segment):
        problems.append(f"kn segment '{kn_seg}' vs loaded segment '{segf.segment}'")
    kn_ap = getattr(bundle, "aperture_id", None)
    seg_ap = getattr(segf, "aperture_id", None)
    if kn_ap is not None and seg_ap is not None and int(kn_ap) != int(seg_ap):
        problems.append(f"kn aperture {kn_ap} vs loaded aperture {seg_ap}")
    if not problems:
        return None
    return "; ".join(problems)


def build_phase3b_harmonic_merge_panel(
    get_segmentframe_callable: Callable[[], Optional[SegmentFrame]],
    get_segmentpath_callable: Callable[[], Optional[str]] | None = None,
    get_kn_bundle_callable: Callable[[], Optional[KnBundle]] | None = None,
    set_merge_result_callable: Callable[[Optional[MergeResult]], None] | None = None,
    get_profile_callable: Callable | None = None,
    get_n_last_recommended: Callable[[], Optional[int]] | None = None,
    get_catalog_callable: Callable[[], Optional[Any]] | None = None,
) -> w.Widget:
    """Build the Harmonic Merge panel.

    Parameters
    ----------
    get_segmentframe_callable
        Returns the current SegmentFrame from shared state.
    get_segmentpath_callable
        Returns the current segment file path.
    get_kn_bundle_callable
        Returns the KnBundle from Coil Calibration tab.
    set_merge_result_callable
        Called when a MergeResult is created.
    get_profile_callable
        Optional.  Returns an :class:`~rotating_coil_analyzer.models.profile.AnalysisProfile`
        to pre-populate widget values (Rref, magnet order, options, etc.).
    get_n_last_recommended
        Accepted for backward compatibility; unused (the N-last-turns choice
        lives in the Physics Plots tab, which computes the plateau summary).
    get_catalog_callable
        Optional.  Returns the current MeasurementCatalog; its Parameters.txt
        settings (refRadius, magnet order/type, norm/skew, analyzer options)
        pre-fill the widgets whenever a new segment is loaded.
    """
    global _ACTIVE_PHASE3B_PANEL

    if _ACTIVE_PHASE3B_PANEL is not None:
        try:
            _ACTIVE_PHASE3B_PANEL.close()
        except Exception:
            pass
        _ACTIVE_PHASE3B_PANEL = None

    st = HarmonicMergeState()

    log = HtmlLog(title="Harmonic Merge log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- Context display ----
    context_html = w.HTML(
        "<div style='color:#666; padding:4px; background:#f8f8f8; border:1px solid #ddd;'>"
        "Waiting for inputs: segment from Catalog tab and kn from Coil Calibration tab."
        "</div>"
    )

    # ---- Numeric parameters ----
    # Rref and m start at 0 = "unknown": they are pre-filled from Parameters.txt
    # when a segment is loaded, otherwise the user must type them ("Apply Kn"
    # refuses values <= 0). There is deliberately no silent magnet default.
    rref_mm = w.FloatText(
        value=0.0,
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
        value=0,
        description="Main order m:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )
    skew_main = w.Checkbox(value=False, description="Skew main harmonic")
    params_source_html = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Rref and m are taken from Parameters.txt when a segment is loaded; "
        "0 means unknown and must be entered before 'Apply Kn'."
        "</i>"
    )

    # ---- Processing options (legacy ordering) ----
    opt_dit = w.Checkbox(value=False, description="di/dt correction")
    opt_dri = w.Checkbox(value=True, description="drift correction")
    opt_rot = w.Checkbox(value=True, description="rotation (post-Kn)")
    opt_cel = w.Checkbox(value=False, description="CEL")
    opt_fed = w.Checkbox(value=False, description="feeddown")
    opt_nor = w.Checkbox(value=False, description="normalization (auto, do not check)",
                         disabled=True)

    opt_dit_signed = w.Checkbox(value=False, description="dit signed (FFMM parity)")
    opt_dit_signed.disabled = True  # only enabled when dit is checked

    max_zR_input = w.FloatText(
        value=0.0,
        description="max |zR|:",
        style={"description_width": "80px"},
        layout=w.Layout(width="180px"),
    )
    max_zR_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "0 = no clamp. Auto-set to 0.01 when feeddown checked."
        "</i>"
    )

    opt_legacy_rotate = w.Checkbox(
        value=False,
        description="Legacy rotate excludes last (SM18)",
    )

    encoder_offset = w.FloatText(
        value=0.0,
        description="Encoder offset [rad]:",
        style={"description_width": "140px"},
        layout=w.Layout(width="240px"),
    )
    encoder_offset_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Known encoder trigger offset from nominal zero (radians). "
        "Pre-rotates data before rotation step. 0 = no pre-rotation."
        "</i>"
    )

    flip_polarity = w.Checkbox(value=False, description="Flip signal polarity")
    flip_polarity_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Negate all harmonics before rotation/normalization. "
        "Use when B1 is negative at positive current (inverted coil/cable polarity)."
        "</i>"
    )

    def _on_dit_change(change):
        opt_dit_signed.disabled = not opt_dit.value

    def _on_fed_change_zr(change):
        if opt_fed.value and max_zR_input.value == 0.0:
            max_zR_input.value = 0.01
        elif not opt_fed.value:
            max_zR_input.value = 0.0

    opt_dit.observe(_on_dit_change, names="value")
    opt_fed.observe(_on_fed_change_zr, names="value")

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

    # ---- CEL/FED safety diagnostic ----
    btn_diagnose_cel = w.Button(
        description="Diagnose CEL/FED",
        button_style="warning",
        layout=w.Layout(width="160px"),
    )
    cel_diag_html = w.HTML("<i style='color:#666;'>Click to diagnose cel/fed safety.</i>")

    # ---- MAD outlier removal ----
    mad_n_sigma = w.FloatText(
        value=5.0,
        description="MAD n_sigma:",
        style={"description_width": "90px"},
        layout=w.Layout(width="180px"),
    )
    mad_clip_col = w.Dropdown(
        options=[("B1_T", "B1_T"), ("b3_units", "b3_units")],
        value="B1_T",
        description="Clip col:",
        style={"description_width": "70px"},
        layout=w.Layout(width="200px"),
    )
    btn_mad_clip = w.Button(
        description="Remove outliers",
        button_style="",
        layout=w.Layout(width="140px"),
    )
    mad_result_html = w.HTML("")

    # (The former "N last turns" widget was removed: no export used it. The
    #  N-last choice is made in the Physics Plots tab, where the plateau
    #  summary is computed.)

    # ---- Diagnostics display ----
    diag_table_html = w.HTML("<i>Diagnostics will appear here after recommendation.</i>")

    # ---- Plot output ----
    out_plot = w.Output()

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _warn(msg: str) -> None:
        log.warning(f"WARNING: {msg}")

    def _reset_compute_state() -> None:
        """Forget everything derived from the last 'Apply Kn' (per-compute state)."""
        st.result = None
        st.opts = None
        st.used_params = None
        st.diag = None
        st.choice_recommended = None
        st.per_n_selection = None
        st.merge_result = None
        st.C_units = None
        st.ok_main = None
        st.outlier_keep_mask = None
        mad_result_html.value = ""
        approve_merge.value = False
        btn_recommend.disabled = True
        btn_apply_merge.disabled = True
        btn_export.disabled = True
        per_n_container.children = []
        per_n_table_html.value = "<i>Compute harmonics first to see per-n selection table.</i>"
        diag_table_html.value = "<i>Diagnostics will appear here after recommendation.</i>"

    def _prefill_from_parameters(segf2: SegmentFrame) -> None:
        """Pre-fill Rref / m / skew / options from Parameters.txt for a NEW segment.

        Called only when the loaded segment changes, so values typed by the
        user for the current segment are never overwritten by a refresh.
        """
        cat = None
        if get_catalog_callable is not None:
            try:
                cat = get_catalog_callable()
            except Exception as e:
                _warn(f"could not read the catalog for Parameters pre-fill: {e!r}")
                cat = None
        cat_key = str(getattr(cat, "parameters_path", "")) if cat is not None else None
        same_catalog = cat_key is not None and cat_key == st.cat_key
        st.cat_key = cat_key

        taken: List[str] = []
        unknown: List[str] = []

        # --- main field order m ---
        m0 = getattr(segf2, "magnet_order", None)
        if (m0 is None or int(m0) <= 0) and cat is not None:
            m0 = getattr(cat, "magnet_order", None)
        if m0 is not None and int(m0) > 0:
            magnet_order.value = int(m0)
            mtype = getattr(cat, "magnet_type", None) if cat is not None else None
            taken.append(f"m={int(m0)}" + (f" ({mtype})" if mtype else ""))
        elif same_catalog and int(magnet_order.value) > 0:
            unknown.append(f"main order m (keeping the value you entered, m={int(magnet_order.value)})")
        else:
            magnet_order.value = 0
            unknown.append("main order m")

        # --- reference radius ---
        r0 = getattr(cat, "r_ref_m", None) if cat is not None else None
        if r0 is not None and np.isfinite(float(r0)) and float(r0) > 0:
            rref_mm.value = float(r0) * 1000.0
            taken.append(f"Rref={float(r0) * 1000.0:g} mm")
        elif same_catalog and float(rref_mm.value) > 0:
            unknown.append(f"Rref (keeping the value you entered, {float(rref_mm.value):g} mm)")
        else:
            rref_mm.value = 0.0
            unknown.append("Rref")

        # --- normal / skew main ---
        sk = getattr(cat, "skew_main", None) if cat is not None else None
        if sk is not None:
            skew_main.value = bool(sk)
            taken.append("skew main" if bool(sk) else "normal main")

        # --- processing options (never 'nor': normalisation is done post-merge) ---
        aopts = getattr(cat, "analyzer_options", None) if cat is not None else None
        if aopts is not None:
            toks = {str(t).strip().lower() for t in aopts}
            for tok, cb in (("dit", opt_dit), ("dri", opt_dri), ("rot", opt_rot),
                            ("cel", opt_cel), ("fed", opt_fed)):
                cb.value = tok in toks
            on = [t for t in ("dit", "dri", "rot", "cel", "fed") if t in toks]
            taken.append("options=" + (",".join(on) if on else "none"))
            if "nor" in toks:
                log.write("Parameters.txt has nor=true; ignored here (normalisation to units "
                          "is always done after the merge).")

        # --- explicit AnalysisProfile overrides Parameters (legacy behaviour) ---
        if get_profile_callable is not None:
            try:
                _profile = get_profile_callable()
                if _profile is not None:
                    rref_mm.value = _profile.r_ref_m * 1000.0
                    magnet_order.value = _profile.magnet_order
                    abs_calib.value = _profile.abs_calib
                    drift_mode.value = _profile.drift_mode
                    skew_main.value = _profile.skew_main
                    taken.append("AnalysisProfile (overrides Parameters)")
            except Exception as e:
                _warn(f"AnalysisProfile pre-fill failed: {e!r}")

        if taken:
            log.write("Pre-filled from Parameters.txt: " + "; ".join(taken) + ".")
        if unknown:
            _warn("not found in Parameters.txt: " + ", ".join(unknown)
                  + ". Enter the value(s) before 'Apply Kn'.")
        src = "; ".join(taken) if taken else "nothing"
        miss = (" | <b style='color:#b00;'>unknown: " + ", ".join(unknown) + "</b>") if unknown else ""
        params_source_html.value = (
            "<span style='font-size:11px; color:#555;'>From Parameters.txt: "
            f"{src}{miss}</span>"
        )

    def _refresh_context() -> bool:
        """Refresh context from Catalog tab and Coil Calibration tab. Returns True if ready."""
        segf = get_segmentframe_callable()
        kn_bundle = get_kn_bundle_callable() if get_kn_bundle_callable else None

        if segf is None:
            context_html.value = (
                "<div style='color:#b00; padding:4px; background:#fee; border:1px solid #c88;'>"
                "<b>No segment loaded.</b><br>"
                "Go to the <b>Catalog</b> tab, browse to a measurement folder, "
                "select a run/aperture/segment, then click <b>'Load segment'</b>."
                "</div>"
            )
            st.segf = None
            st.kn_bundle = None
            return False

        # A different segment was loaded: drop every per-compute result and
        # pre-fill the parameters for it (only now, never on a plain refresh).
        key = segment_key(segf)
        if key != st.seg_key:
            if st.seg_key is not None:
                log.write("New segment loaded: previous harmonics / merge cleared.")
                if set_merge_result_callable:
                    set_merge_result_callable(None)
            st.seg_key = key
            _reset_compute_state()
            _prefill_from_parameters(segf)

        if kn_bundle is None:
            context_html.value = (
                "<div style='color:#960; padding:4px; background:#ffc; border:1px solid #c90;'>"
                "<b>Segment loaded</b>, but <b>no kn calibration</b> available.<br>"
                "Go to the <b>Coil Calibration</b> tab to load or compute kn coefficients first."
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

        # Auto-fill compensation scheme from kn bundle if head_csv
        if kn_bundle.head_cmp_connection and not comp_scheme.value.strip():
            comp_scheme.value = kn_bundle.head_cmp_connection

        ap_str = f"Aperture {segf2.aperture_id}" if segf2.aperture_id is not None else "Single aperture"
        m_str = f"m={segf2.magnet_order}" if segf2.magnet_order is not None else "m=?"
        kn_src = kn_bundle.source_type
        kn_H = len(kn_bundle.kn.orders)
        kn_for = kn_bundle.segment_id if kn_bundle.segment_id is not None else "unspecified"
        mismatch = kn_segment_mismatch(kn_bundle, segf2)

        if mismatch:
            box = "padding:4px; background:#fee; border:1px solid #c88;"
            mis_html = (
                f"<br><b style='color:#b00;'>kn / segment mismatch: {mismatch}.</b> "
                "Reload kn for this segment in the Coil Calibration tab."
            )
            warn_key = (key, id(kn_bundle))
            if st.kn_warned_key != warn_key:
                st.kn_warned_key = warn_key
                _warn(f"kn / segment mismatch ({mismatch}). 'Apply Kn' is refused until kn "
                      "for this segment is loaded in the Coil Calibration tab.")
        else:
            box = "padding:4px; background:#e8f4e8; border:1px solid #8c8;"
            mis_html = ""

        context_html.value = (
            f"<div style='{box}'>"
            f"<b>Segment:</b> Run={segf2.run_id}, {ap_str}, Seg={segf2.segment}, "
            f"{m_str}, n_turns={segf2.n_turns}<br>"
            f"<b>Kn:</b> source={kn_src}, H={kn_H}, loaded for segment={kn_for}, "
            f"ext={'yes' if kn_bundle.kn.kn_ext is not None else 'no'}"
            f"{mis_html}"
            f"</div>"
        )

        if rem > 0:
            log.write(f"Trimmed {rem} samples from tail to ensure full turns.")

        return True

    def _validate_params() -> Optional[str]:
        """Return an error message when Rref / m are unknown or invalid."""
        problems: List[str] = []
        try:
            r = float(rref_mm.value)
        except Exception:
            r = float("nan")
        if not np.isfinite(r) or r <= 0.0:
            problems.append("reference radius Rref [mm] is unknown (enter a value > 0)")
        if int(magnet_order.value) <= 0:
            problems.append("main field order m is unknown (enter m >= 1: 1 dipole, 2 quadrupole, ...)")
        if problems:
            return "; ".join(problems)
        return None

    def _current_params() -> dict:
        return {
            "m": int(magnet_order.value),
            "rref_mm": float(rref_mm.value),
            "skew_main": bool(skew_main.value),
            "abs_calib": float(abs_calib.value),
        }

    def _m_used() -> int:
        """Main order used at the last 'Apply Kn' (falls back to the widget)."""
        if st.used_params is not None:
            return int(st.used_params["m"])
        return int(magnet_order.value)

    def _main_component(C_col: np.ndarray) -> np.ndarray:
        """Main-field component: Im for a skew main, Re otherwise (as used at Apply Kn)."""
        skew = bool(st.used_params["skew_main"]) if st.used_params is not None else bool(skew_main.value)
        return np.imag(C_col) if skew else np.real(C_col)

    def _mad_harmonic_order(m: int, H: int) -> Optional[int]:
        """Order offered as the 'harmonic' MAD clip column: b3, or b{m+1} when b3 is not a harmonic."""
        if m < 3 and H >= 3:
            return 3
        if m + 1 <= H:
            return m + 1
        return None

    def _update_mad_options() -> None:
        """Relabel the MAD clip column choices for the order actually used."""
        if st.result is None:
            return
        m = _m_used()
        n_h = _mad_harmonic_order(m, int(st.result.H))
        opts = [(f"B{m}_T (main)", "B1_T")]
        if n_h is not None:
            opts.append((f"b{n_h}_units", "b3_units"))
        prev = mad_clip_col.value
        mad_clip_col.options = opts
        vals = [v for _, v in opts]
        mad_clip_col.value = prev if prev in vals else vals[0]

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
            m = _m_used()
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
        m = _m_used()
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

            # Refuse silently-wrong inputs: unknown Rref / m, or kn for another segment.
            bad = _validate_params()
            if bad:
                log.write(f"<b style='color:#b00'>Apply Kn refused:</b> {bad}.")
                return
            mismatch = kn_segment_mismatch(st.kn_bundle, st.segf)
            if mismatch:
                log.write(
                    f"<b style='color:#b00'>Apply Kn refused: kn / segment mismatch ({mismatch}).</b> "
                    "Load or compute kn for this segment in the Coil Calibration tab."
                )
                return

            # Every per-compute result (incl. the outlier mask) belongs to the previous run.
            _reset_compute_state()

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
            zR_val = float(max_zR_input.value)
            max_zR_param = zR_val if zR_val > 0 else None

            # Auto-run CEL/FED safety diagnostic when cel or fed is checked
            if "cel" in opts or "fed" in opts:
                try:
                    _set_status("checking cel/fed safety...")
                    max_zR_diag = zR_val if zR_val > 0 else 0.01
                    diag_result = diagnose_cel_fed(
                        flux_abs_turns=tb.df_abs,
                        flux_cmp_turns=tb.df_cmp,
                        t_turns=tb.t,
                        I_turns=tb.I,
                        kn=kn,
                        r_ref=Rref_m,
                        magnet_order=int(magnet_order.value),
                        max_zR=max_zR_diag,
                        dit_signed=bool(opt_dit_signed.value),
                        drift_mode=str(drift_mode.value),
                    )
                    rec = diag_result.recommendation
                    # Update diagnostic display
                    if rec == "SAFE":
                        color, bg = "#080", "#e8f4e8"
                    elif rec == "UNSAFE":
                        color, bg = "#b00", "#fee"
                    else:
                        color, bg = "#960", "#ffc"
                    cel_diag_html.value = (
                        f"<div style='padding:6px; background:{bg}; border:1px solid {color}; "
                        f"border-radius:4px; margin:4px 0;'>"
                        f"<b style='color:{color};'>{rec}</b>: {diag_result.reason}"
                        f"</div>"
                    )
                    if rec == "UNSAFE":
                        opt_cel.value = False
                        opt_fed.value = False
                        opts = [o for o in opts if o not in ("cel", "fed")]
                        log.write(
                            f"WARNING: cel/fed UNSAFE — auto-disabled. "
                            f"{diag_result.n_suspect}/{diag_result.n_total} suspect turns. "
                            f"Computing without cel/fed."
                        )
                    else:
                        log.write(f"cel/fed diagnostic: {rec} — {diag_result.reason}")
                    _set_status("computing harmonics...")
                except Exception as diag_err:
                    log.write(f"cel/fed diagnostic skipped: {diag_err}")

            # Snapshot of the parameters used: the merge, units and presets use these.
            params = _current_params()
            res = compute_legacy_kn_per_turn(
                df_abs_turns=tb.df_abs,
                df_cmp_turns=tb.df_cmp,
                t_turns=tb.t,
                I_turns=tb.I,
                kn=kn,
                Rref_m=Rref_m,
                magnet_order=params["m"],
                absCalib=params["abs_calib"],
                options=tuple(opts),
                drift_mode=str(drift_mode.value),
                skew_main=params["skew_main"],
                dit_signed=bool(opt_dit_signed.value),
                legacy_rotate_excludes_last=bool(opt_legacy_rotate.value),
                max_zR=max_zR_param,
                encoder_offset_rad=float(encoder_offset.value),
                flip_signal_polarity=bool(flip_polarity.value),
            )
            st.result = res
            st.opts = tuple(opts)
            st.used_params = params

            # Pipeline diagnostics (cel not applied, |zR|>=1, |C_m|>20 T, wrap boundary, ...)
            for msg in getattr(res, "warnings", ()) or ():
                _warn(f"pipeline: {msg}")

            # Build per-n table
            _build_per_n_table()
            _update_mad_options()

            # Enable next steps
            btn_recommend.disabled = False
            btn_apply_merge.disabled = False
            btn_export.disabled = True

            # Log zR clamping info
            if max_zR_param is not None and hasattr(res, 'zR_clamped'):
                n_clamped = int(np.sum(res.zR_clamped))
                if n_clamped > 0:
                    log.write(
                        f"WARNING: {n_clamped}/{res.C_abs.shape[0]} turns had |zR| clamped "
                        f"(max_zR={max_zR_param:.4f})"
                    )

            # Diagnostic summary for b2 sign investigation
            phi_med = float(np.median(res.phi_out_rad))
            phi_std = float(np.std(res.phi_out_rad))
            m = _m_used()
            B1_med = float(np.median(_main_component(res.C_abs[:, m - 1]))) if m >= 1 else 0.0
            b2_abs_med = float(np.median(np.real(res.C_abs[:, 1]))) if res.H >= 2 else 0.0
            b2_cmp_med = float(np.median(np.real(res.C_cmp[:, 1]))) if res.H >= 2 else 0.0
            log.write(
                f"Computed harmonics: n_turns={res.C_abs.shape[0]}, H={res.H}. "
                "Now select merge options and apply."
            )
            log.write(
                f"Diagnostics: phi_out median={phi_med:.4f} rad (std={phi_std:.4f}), "
                f"B_main median={B1_med:.6e} T, "
                f"b2_abs median={b2_abs_med:.6e}, b2_cmp median={b2_cmp_med:.6e}"
            )
            _plot_main_field(res)

        except Exception as e:
            log.write(f"<b style='color:#b00'>Compute failed:</b> {e}")
            raise
        finally:
            _set_status("idle")
            st.busy = False

    def _on_diagnose_cel(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("diagnosing cel/fed...")

            if not _refresh_context():
                return

            bad = _validate_params()
            if bad:
                log.write(f"<b style='color:#b00'>Diagnose cel/fed refused:</b> {bad}.")
                return

            segf = st.segf
            kn = st.kn_bundle.kn

            tb = split_into_turns(segf)
            Rref_m = float(rref_mm.value) * 1e-3
            zR_val = float(max_zR_input.value)
            max_zR_diag = zR_val if zR_val > 0 else 0.01

            diag_result = diagnose_cel_fed(
                flux_abs_turns=tb.df_abs,
                flux_cmp_turns=tb.df_cmp,
                t_turns=tb.t,
                I_turns=tb.I,
                kn=kn,
                r_ref=Rref_m,
                magnet_order=int(magnet_order.value),
                max_zR=max_zR_diag,
                dit_signed=bool(opt_dit_signed.value),
                drift_mode=str(drift_mode.value),
            )

            rec = diag_result.recommendation
            if rec == "SAFE":
                color = "#080"
                bg = "#e8f4e8"
            elif rec == "UNSAFE":
                color = "#b00"
                bg = "#fee"
            else:
                color = "#960"
                bg = "#ffc"

            cel_diag_html.value = (
                f"<div style='padding:6px; background:{bg}; border:1px solid {color}; "
                f"border-radius:4px; margin:4px 0;'>"
                f"<b style='color:{color};'>{rec}</b>: {diag_result.reason}"
                f"</div>"
            )

            if rec == "UNSAFE":
                opt_cel.value = False
                opt_fed.value = False
                log.write(f"WARNING: cel/fed UNSAFE — auto-unchecked cel and fed. "
                          f"{diag_result.n_suspect}/{diag_result.n_total} suspect turns.")
            else:
                log.write(f"cel/fed diagnostic: {rec} — {diag_result.reason}")

        except Exception as e:
            log.write(f"<b style='color:#b00'>Diagnose cel/fed failed:</b> {e}")
        finally:
            _set_status("idle")
            st.busy = False

    def _on_mad_clip(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("removing outliers...")

            if st.result is None or st.merge_result is None:
                log.write("<b style='color:#b00'>Apply merge first before removing outliers.</b>")
                return

            res = st.result
            mr = st.merge_result
            m = _m_used()
            clip_col = str(mad_clip_col.value)

            # Build a temporary DataFrame with the clip column. The dropdown
            # values stay "B1_T" / "b3_units" (stable tokens); the real column
            # is the main B{m} (Im for a skew main) and b3, or b{m+1} when b3
            # is not a harmonic for this magnet.
            n_turns = mr.C_merged.shape[0]
            if clip_col == "B1_T":
                values = _main_component(mr.C_merged[:, m - 1])
                clip_label = f"B{m}_T"
            elif clip_col == "b3_units":
                n_h = _mad_harmonic_order(m, int(mr.C_merged.shape[1]))
                if n_h is None or st.C_units is None:
                    log.write(
                        "<b style='color:#b00'>MAD clip refused:</b> no harmonic above the main "
                        f"order m={m} is available."
                    )
                    return
                values = np.real(st.C_units[:, n_h - 1])
                clip_label = f"b{n_h}_units"
            else:
                log.write(f"<b style='color:#b00'>MAD clip refused:</b> unknown column '{clip_col}'.")
                return

            # Use a simple label (all same group if no plateau info)
            labels = np.full(n_turns, "all", dtype=object)
            tmp_df = pd.DataFrame({clip_label: values, "label": labels})

            df_clean, removed = mad_sigma_clip(
                tmp_df,
                col=clip_label,
                n_sigma=float(mad_n_sigma.value),
                label_col="label",
            )
            clip_col = clip_label

            keep_mask = np.ones(n_turns, dtype=bool)
            removed_indices = tmp_df.index.difference(df_clean.index)
            keep_mask[removed_indices] = False
            st.outlier_keep_mask = keep_mask

            total_removed = sum(removed.values())
            if total_removed > 0:
                mad_result_html.value = (
                    f"<span style='color:#b00;'>Removed {total_removed} outlier(s) "
                    f"from {clip_col} (n_sigma={float(mad_n_sigma.value):.1f}). "
                    f"Will be filtered on export.</span>"
                )
                log.write(f"MAD clip: removed {total_removed} turns on {clip_col}.")
            else:
                mad_result_html.value = (
                    "<span style='color:#080;'>No outliers found.</span>"
                )
                log.write("MAD clip: no outliers found.")

        except Exception as e:
            log.write(f"<b style='color:#b00'>MAD clip failed:</b> {e}")
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
                magnet_order=_m_used(),
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

            # The merge must use the parameters of the last "Apply Kn".
            used = st.used_params or {}
            cur = _current_params()
            changed = [
                f"{k}: {used.get(k)!r} at Apply Kn, now {cur[k]!r}"
                for k in ("m", "rref_mm", "skew_main", "abs_calib")
                if k in used and used.get(k) != cur[k]
            ]
            if changed:
                log.write(
                    "<b style='color:#b00'>Merge refused: parameters changed since 'Apply Kn'</b> ("
                    + "; ".join(changed)
                    + "). Click 'Apply Kn' again to recompute with the new values."
                )
                return

            res = st.result
            H = res.H

            # The Ext channel is not processed by the pipeline (LegacyKnPerTurn
            # has no C_ext): refuse rather than substitute Abs and record Ext.
            ext_orders = [int(res.orders[j]) for j in range(H) if int(st.per_n_selection[j]) == CHANNEL_EXT]
            if ext_orders:
                log.write(
                    "<b style='color:#b00'>Merge refused: Ext channel not processed by the pipeline yet</b> "
                    f"(selected for n={ext_orders}). Choose Abs or Cmp for these orders."
                )
                return

            # Standard merge (abs/cmp only)
            C_merged, choice = merge_coefficients(
                C_abs=res.C_abs,
                C_cmp=res.C_cmp,
                magnet_order=_m_used(),
                mode="custom",
                per_order_choice=st.per_n_selection.tolist(),
            )

            # Post-merge normalization to units (Bottura 3.7 mixed format).
            # When 'nor' was checked, normalization already happened inside
            # compute_legacy_kn_per_turn — C_merged is already in units.
            # When 'nor' was NOT checked, C_merged is in Tesla and we need
            # safe_normalize_to_units for the n>m columns.
            nor_was_checked = st.opts is not None and "nor" in st.opts
            if nor_was_checked:
                # All harmonics already in units; C_units = C_merged
                st.C_units = C_merged.copy()
                st.ok_main = np.ones(C_merged.shape[0], dtype=bool)
            else:
                C_units, ok_main = safe_normalize_to_units(
                    C_merged,
                    magnet_order=_m_used(),
                    absCalib=float(used.get("abs_calib", abs_calib.value)),
                    skew_main=bool(used.get("skew_main", skew_main.value)),
                )
                st.C_units = C_units
                st.ok_main = ok_main

            # Create MergeResult with full provenance
            merge_result = MergeResult(
                C_merged=C_merged,
                orders=res.orders,
                per_n_source_map=choice,
                compensation_scheme=comp_scheme.value.strip() or "unspecified",
                magnet_order=_m_used(),
                kn_provenance=st.kn_bundle,
                merge_mode=merge_preset.value,
                timestamp=KnBundle.now_iso(),
                diagnostics=st.diag,
                C_abs=res.C_abs,
                C_cmp=res.C_cmp,
                C_units=st.C_units,
                ok_main=st.ok_main,
                I_mean_A=res.I_mean_A,
            )
            st.merge_result = merge_result

            if set_merge_result_callable:
                set_merge_result_callable(merge_result)

            btn_export.disabled = False

            # Re-state the pipeline diagnostics and flag orders taken from a bucked Cmp.
            for msg in getattr(res, "warnings", ()) or ():
                _warn(f"pipeline: {msg}")
            try:
                bucked = bucked_cmp_orders(st.kn_bundle.kn, choice)
            except Exception as e:
                bucked = []
                _warn(f"bucked-Cmp check failed: {e!r}")
            if bucked:
                _warn(
                    f"orders n={list(bucked)} are merged from Cmp although the Cmp channel bucks them "
                    "(|kn_cmp| ~ 0): these values are noise / ~0. Take them from Abs."
                )

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
            prefix = "harmonic_merge"
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
            df_abs = pd.concat([scalars, ba_table_from_C(res.C_abs, res.orders, prefix="abs_")], axis=1)
            df_cmp = pd.concat([scalars, ba_table_from_C(res.C_cmp, res.orders, prefix="cmp_")], axis=1)

            abs_path = f"{root}_ABS.csv"
            cmp_path = f"{root}_CMP.csv"
            df_abs.to_csv(abs_path, index=False)
            df_cmp.to_csv(cmp_path, index=False)
            log.write(f"Saved: {abs_path}")
            log.write(f"Saved: {cmp_path}")

            # Merged table (Bottura 3.7 mixed format)
            nor_was_checked = st.opts is not None and "nor" in st.opts
            df_m = pd.concat([
                scalars,
                mixed_format_table(
                    mr.C_merged,
                    st.C_units,
                    mr.orders,
                    _m_used(),
                    nor_was_checked=nor_was_checked,
                ),
            ], axis=1)
            # Apply outlier mask if present
            if st.outlier_keep_mask is not None:
                mask = st.outlier_keep_mask
                n_removed = int(np.sum(~mask))
                df_abs = df_abs.loc[mask].reset_index(drop=True)
                df_cmp = df_cmp.loc[mask].reset_index(drop=True)
                df_m = df_m.loc[mask].reset_index(drop=True)
                if n_removed > 0:
                    log.write(f"Applied outlier mask: {n_removed} rows removed from exports.")

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
            # Parameters used at "Apply Kn" (added columns; existing ones unchanged).
            if st.used_params is not None:
                meta.setdefault("r_ref_mm", st.used_params["rref_mm"])
                meta.setdefault("skew_main", st.used_params["skew_main"])
                meta.setdefault("abs_calib", st.used_params["abs_calib"])
            if st.opts is not None:
                meta.setdefault("pipeline_options", ",".join(st.opts))
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

    _clear_button_handlers(btn_diagnose_cel)
    btn_diagnose_cel.on_click(_on_diagnose_cel)

    _clear_button_handlers(btn_mad_clip)
    btn_mad_clip.on_click(_on_mad_clip)

    # Initial context refresh
    _refresh_context()

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Harmonic Merge</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Apply kn calibration to compute harmonics, then select Abs/Cmp source per harmonic order. "
        "All outputs include full provenance (kn source, compensation scheme, per-n map)."
        "</div>"
    )

    params_box = w.VBox([
        w.HTML("<b>Parameters</b>"),
        w.HBox([rref_mm, abs_calib, magnet_order]),
        w.HBox([skew_main]),
        params_source_html,
    ])

    opts_box = w.VBox([
        w.HTML("<b>Processing options (legacy ordering)</b>"),
        w.HBox([opt_dit, opt_dit_signed, opt_dri, opt_rot, opt_cel, opt_fed, opt_nor, drift_mode]),
        w.HBox([max_zR_input, max_zR_help, opt_legacy_rotate]),
        w.HBox([encoder_offset, encoder_offset_help]),
        w.HBox([flip_polarity, flip_polarity_help]),
    ])

    comp_box = w.VBox([
        w.HTML("<b>Compensation scheme (metadata)</b>"),
        w.HBox([comp_scheme]),
        comp_scheme_help,
    ])

    cel_diag_box = w.VBox([
        w.HTML("<b>CEL/FED safety diagnostic</b>"),
        w.HBox([btn_diagnose_cel, cel_diag_html]),
    ])

    mad_box = w.VBox([
        w.HTML("<b>Outlier removal (MAD sigma clip)</b>"),
        w.HBox([mad_n_sigma, mad_clip_col, btn_mad_clip]),
        mad_result_html,
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
            cel_diag_box,
            w.HBox([btn_compute, btn_recommend]),
            w.HTML("<hr>"),
            comp_box,
            w.HTML("<hr>"),
            merge_box,
            w.HTML("<hr>"),
            mad_box,
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

    # Expose refresh hook so app.py can auto-refresh context on tab switch
    setattr(panel, "_harmonic_merge_refresh_context", _refresh_context)
    # Parameters (m, rref_mm, skew_main, abs_calib) used by the current merge,
    # read by the Physics Plots tab (MergeResult carries no skew flag).
    setattr(panel, "_harmonic_merge_used_params", lambda: (dict(st.used_params) if st.used_params else None))

    _ACTIVE_PHASE3B_PANEL = panel
    return panel
