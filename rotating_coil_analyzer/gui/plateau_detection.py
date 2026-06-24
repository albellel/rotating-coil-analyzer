"""Plateau Detection GUI tab for streaming supercycle measurements.

Provides automatic detection of current plateaus in streaming data using
block-averaged range analysis with three-rule detection.  Matches the
notebook workflow: detect → merge (I_nom-aware) → classify turns as
precycle/ramp/settling/used → share "used" indices downstream.

All backend functions already exist in ``analysis.utility_functions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import ipywidgets as w

from rotating_coil_analyzer.gui.log_view import HtmlLog
from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.analysis.utility_functions import (
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
)


@dataclass
class PlateauDetectionState:
    """Local mutable state for the Plateau Detection tab."""

    segf: Optional[SegmentFrame] = None
    I_range: Optional[np.ndarray] = None
    I_blocks: Optional[np.ndarray] = None
    I_mean: Optional[np.ndarray] = None
    groups: Optional[List] = None
    # Per-turn classification: "precycle", "ramp", "settling", "used"
    turn_category: Optional[np.ndarray] = None
    fig: Optional[Any] = None
    busy: bool = False


def build_plateau_detection_panel(
    get_segmentframe: Callable[[], Optional[SegmentFrame]],
    get_segmentpath: Callable[[], Optional[str]] | None = None,
    set_plateau_info: Callable[[Optional[Dict[str, Any]]], None] | None = None,
) -> w.Widget:
    """Build the Plateau Detection panel.

    Parameters
    ----------
    get_segmentframe
        Returns the current SegmentFrame from shared state.
    get_segmentpath
        Returns the current segment file path.
    set_plateau_info
        Called with a dict of plateau detection results to store in shared state.
    """
    st = PlateauDetectionState()
    log = HtmlLog(title="Plateau Detection log")
    status = w.HTML("<b>Status:</b> idle")

    # ---- Detection parameters ----
    threshold = w.FloatText(
        value=0.5,
        description="Threshold [A]:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )
    min_length = w.IntText(
        value=50,
        description="Min length:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )
    merge_gap = w.IntText(
        value=100,
        description="Merge gap:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )
    merge_gap_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Merge groups at same I_nom with gap < this (heals noise-split plateaus)."
        "</i>"
    )
    n_blocks = w.IntText(
        value=10,
        description="N blocks:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )

    # ---- Turn trimming parameters ----
    n_skip_start = w.IntText(
        value=0,
        description="Skip start:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )
    n_skip_start_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Turns to skip at start of each plateau (eddy settling). "
        "If N_LAST is set, this is computed automatically."
        "</i>"
    )
    n_last_turns = w.IntText(
        value=0,
        description="N last turns:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )
    n_last_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Use last N turns of each plateau for analysis (0 = all). "
        "e.g. 340 for MC62 2 Hz data. Overrides skip_start."
        "</i>"
    )
    n_skip_end = w.IntText(
        value=0,
        description="Skip end:",
        style={"description_width": "110px"},
        layout=w.Layout(width="180px"),
    )
    n_skip_end_help = w.HTML(
        "<i style='color:#666; font-size:11px;'>"
        "Turns to drop from end of each plateau (ramp contamination guard)."
        "</i>"
    )
    i_nom_tolerance = w.FloatText(
        value=5.0,
        description="I_nom tol [A]:",
        style={"description_width": "110px"},
        layout=w.Layout(width="200px"),
    )

    btn_detect = w.Button(description="Detect plateaus", button_style="primary")
    btn_plot = w.Button(description="Plot classification", button_style="")
    btn_plot.disabled = True

    summary_html = w.HTML("<i>Run detection first.</i>")
    out_plot = w.Output(
        layout=w.Layout(border="1px solid #ddd", padding="6px", width="100%", height="400px")
    )

    def _set_status(msg: str) -> None:
        status.value = f"<b>Status:</b> {msg}"

    def _merge_groups_by_inom(
        groups: list[tuple[int, int]],
        I_mean: np.ndarray,
        gap: int,
        i_tol: float,
    ) -> list[tuple[int, int]]:
        """Merge consecutive groups at the same nominal current (within i_tol).

        Only merges if:
        - The gap between groups is < *gap* turns
        - The mean current of both groups is within *i_tol* A

        This heals small noise-induced splits within a single plateau
        without merging groups at different current levels.
        """
        if not groups:
            return groups

        def _group_inom(gs: int, ge: int) -> float:
            return float(np.mean(I_mean[gs:ge + 1]))

        merged = [groups[0]]
        for s, e in groups[1:]:
            prev_s, prev_e = merged[-1]
            gap_size = s - prev_e - 1
            if gap_size <= gap:
                prev_inom = _group_inom(prev_s, prev_e)
                this_inom = _group_inom(s, e)
                if abs(prev_inom - this_inom) < i_tol:
                    # Same current level, small gap → merge
                    merged[-1] = (prev_s, e)
                    continue
            merged.append((s, e))
        return merged

    def _classify_turns(
        n_turns: int,
        groups: list[tuple[int, int]],
        I_mean: np.ndarray,
        n_last: int,
        skip_end: int,
        skip_start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify every turn and compute boolean mask of "used" turns.

        Categories:
        - "ramp": between plateaus or not in any group
        - "settling": within a plateau but in the eddy-current transient region
        - "used": the settled portion of the plateau (for analysis)

        Returns (turn_category, is_used) arrays.
        """
        cat = np.full(n_turns, "ramp", dtype=object)
        is_used = np.zeros(n_turns, dtype=bool)

        for gs, ge in groups:
            # Effective end after dropping ramp-guard turns
            eff_end = ge - skip_end if skip_end > 0 and (ge - gs + 1) > skip_end else ge

            if n_last > 0:
                # Use last n_last turns of the effective range
                used_start = max(gs, eff_end - n_last + 1)
            elif skip_start > 0:
                # Skip first skip_start turns
                used_start = min(gs + skip_start, eff_end)
            else:
                # Use all turns in the plateau
                used_start = gs

            # Mark categories
            for t in range(gs, ge + 1):
                if t > eff_end:
                    cat[t] = "settling"  # end-trimmed turns
                elif t < used_start:
                    cat[t] = "settling"  # start-trimmed turns (eddy)
                else:
                    cat[t] = "used"
                    is_used[t] = True

        return cat, is_used

    def _on_detect(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("detecting plateaus...")

            segf = get_segmentframe()
            if segf is None:
                log.write("No segment loaded. Load a segment in the Catalog tab first.")
                return
            st.segf = segf

            tb = split_into_turns(segf)
            spt = int(segf.samples_per_turn)
            n_turns = tb.n_turns

            I_mean = tb.I.mean(axis=1)
            I_range, I_blocks = compute_block_averaged_range(
                tb.I, spt, n_blocks=int(n_blocks.value),
            )

            det = detect_plateau_turns(
                I_blocks, I_mean, I_range, float(threshold.value),
            )
            is_plateau = det["is_plateau"]

            # Find contiguous groups, then merge by I_nom
            raw_groups = find_contiguous_groups(is_plateau, min_length=int(min_length.value))
            groups = _merge_groups_by_inom(
                raw_groups, I_mean,
                gap=int(merge_gap.value),
                i_tol=float(i_nom_tolerance.value),
            )

            log.write(f"Raw groups: {len(raw_groups)}, after I_nom-aware merge: {len(groups)}")

            # Classify turns (settling vs used)
            _n_last = int(n_last_turns.value)
            _skip_end = int(n_skip_end.value)
            _skip_start = int(n_skip_start.value)
            cat, is_used = _classify_turns(
                n_turns, groups, I_mean, _n_last, _skip_end, _skip_start,
            )
            st.turn_category = cat

            # Assign plateau IDs and branch
            plateau_id = np.full(n_turns, -1, dtype=int)
            I_nom = np.zeros(n_turns)
            branch = np.full(n_turns, "unknown", dtype=object)
            labels = np.full(n_turns, "ramp", dtype=object)

            prev_inom = 0.0
            for gid, (gs, ge) in enumerate(groups):
                plateau_id[gs:ge + 1] = gid
                grp_inom = float(np.mean(I_mean[gs:ge + 1]))
                I_nom[gs:ge + 1] = grp_inom
                br = "ascending" if grp_inom > prev_inom else "descending"
                branch[gs:ge + 1] = br
                prev_inom = grp_inom
                for t in range(gs, ge + 1):
                    labels[t] = classify_current(I_mean[t])

            # Build summary
            records: list[dict] = []
            for gid, (gs, ge) in enumerate(groups):
                sl = slice(gs, ge + 1)
                I_grp = I_mean[sl]
                n_used = int(np.sum(is_used[sl]))
                n_settling = int(np.sum(cat[sl] == "settling"))
                records.append({
                    "plateau_id": gid,
                    "start": gs,
                    "end": ge,
                    "n_turns": ge - gs + 1,
                    "n_settling": n_settling,
                    "n_used": n_used,
                    "I_mean": float(np.mean(I_grp)),
                    "I_std": float(np.std(I_grp)),
                    "branch": branch[gs],
                    "label": classify_current(float(np.mean(I_grp))),
                })
            summary_df = pd.DataFrame(records) if records else pd.DataFrame()

            # Store state
            st.I_range = I_range
            st.I_blocks = I_blocks
            st.I_mean = I_mean
            st.groups = groups

            # Display summary
            if summary_df.empty:
                summary_html.value = (
                    "<b>No plateaus found.</b> Adjust the detection parameters: "
                    "<b>raise</b> the threshold if the current channel is noisy or is a "
                    "near-constant reference (its within-turn block range exceeds the "
                    "threshold), or <b>lower</b> it only if the current is very clean. "
                    "Also check <b>Min length</b> — plateaus shorter than it are rejected. "
                    "Note: for file-based plateau (staircase) data the segment is already "
                    "split per run via <code>plateau_id</code>, so this tab is usually not needed."
                )
            else:
                n_total_used = int(np.sum(is_used))
                summary_html.value = (
                    f"<b>Found {len(groups)} plateau(s)</b> — "
                    f"{n_total_used} used turns, "
                    f"{int(np.sum(cat == 'settling'))} settling, "
                    f"{int(np.sum(cat == 'ramp'))} ramp<br>"
                    + summary_df.to_html(index=False, classes="table table-sm")
                )

            # Write to shared state
            if set_plateau_info is not None:
                set_plateau_info({
                    "is_plateau": is_plateau,
                    "plateau_id": plateau_id,
                    "I_mean": I_mean,
                    "I_nom": I_nom,
                    "labels": labels,
                    "branch": branch,
                    "groups": groups,
                    "summary_df": summary_df,
                    "turn_category": cat,
                    "is_used": is_used,
                })

            btn_plot.disabled = False
            log.write(f"Detected {len(groups)} plateau(s) from {n_turns} turns.")
            if _n_last > 0:
                log.write(f"Using last {_n_last} turns per plateau "
                          f"(skip_end={_skip_end}).")
            elif _skip_start > 0:
                log.write(f"Skipping first {_skip_start} turns per plateau "
                          f"(skip_end={_skip_end}).")
            _set_status("detection complete")

        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    def _on_plot(_btn) -> None:
        if st.busy:
            return
        st.busy = True
        try:
            _set_status("plotting...")
            import matplotlib.pyplot as plt

            if st.I_mean is None or st.groups is None or st.turn_category is None:
                log.write("Run detection first.")
                return

            I_mean = st.I_mean
            n_turns = len(I_mean)
            cat = st.turn_category

            out_plot.clear_output(wait=True)
            try:
                plt.close("all")
            except Exception:
                pass

            with out_plot:
                fig, ax = plt.subplots(figsize=(9, 3.5))
                st.fig = fig

                turns = np.arange(n_turns)

                # Background: full trace in light grey
                ax.plot(turns, I_mean, "-", color="#ddd", linewidth=0.5, zorder=1)

                # Color by category
                colors = {
                    "used": "tab:green",
                    "settling": "tab:blue",
                    "ramp": "tab:orange",
                }
                labels_plotted = set()
                for category, color in colors.items():
                    mask = cat == category
                    if not np.any(mask):
                        continue
                    idx = np.where(mask)[0]
                    lbl = category if category not in labels_plotted else "_nolegend_"
                    labels_plotted.add(category)
                    ax.scatter(
                        idx, I_mean[idx],
                        c=color, s=1, label=lbl, zorder=2,
                    )

                ax.set_xlabel("Turn index")
                ax.set_ylabel("I [A]")
                ax.set_title("Turn Classification")
                ax.legend(loc="best", markerscale=5)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            _set_status("plot ready")

        except Exception as e:
            log.write(f"ERROR: {e}")
            _set_status("error")
        finally:
            st.busy = False

    btn_detect.on_click(_on_detect)
    btn_plot.on_click(_on_plot)

    # ---- Layout ----
    header = w.HTML(
        "<h3 style='margin:0;'>Plateau Detection</h3>"
        "<div style='color:#555; line-height:1.35; margin-bottom:8px;'>"
        "Detect current plateaus in streaming supercycle data. "
        "Merge gap heals noise-split plateaus at the same I_nom. "
        "N_LAST / skip controls select which turns are 'used' for analysis."
        "</div>"
    )

    detect_row = w.HBox([threshold, min_length, n_blocks])
    merge_row = w.HBox([merge_gap, i_nom_tolerance, merge_gap_help])
    trim_row1 = w.HBox([n_last_turns, n_last_help])
    trim_row2 = w.HBox([n_skip_start, n_skip_start_help])
    trim_row3 = w.HBox([n_skip_end, n_skip_end_help])
    action_row = w.HBox([btn_detect, btn_plot])

    main_panel = w.VBox(
        [
            header,
            w.HTML("<b>Detection</b>"),
            detect_row,
            merge_row,
            w.HTML("<b>Turn trimming (settling / ramp guard)</b>"),
            trim_row1,
            trim_row2,
            trim_row3,
            action_row,
            w.HTML("<hr>"),
            summary_html,
            out_plot,
        ],
        layout=w.Layout(width="70%", min_width="640px"),
    )

    diag_panel = w.VBox(
        [status, log.panel],
        layout=w.Layout(width="30%", min_width="300px"),
    )

    panel = w.HBox([main_panel, diag_panel], layout=w.Layout(width="100%"))
    return panel
