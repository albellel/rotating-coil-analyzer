"""
Phase 1 GUI (ipywidgets).

This module provides an in-notebook GUI to:
- select input files (via directory + glob, or manual list)
- configure parsing/splitting options
- load + validate
- preview plateau segmentation
- export outputs to CSV or Parquet

Limitations
-----------
Pure ipywidgets does not provide a native OS file picker across all Jupyter setups.
This GUI therefore supports:
- directory + glob pattern (recommended for local files)
- manual list of paths (one per line)

Examples
--------
In a notebook (recommended):
>>> from rotating_coil_analyzer.scripts.gui_phase1 import build_phase1_gui
>>> ui = build_phase1_gui()
>>> ui
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from .io_plateau import (
    LoadConfig,
    export_phase1_outputs,
    load_measurements,
    make_plateau_index,
)


def _list_files(directory: str, pattern: str) -> List[str]:
    """Return sorted file list for directory + glob pattern."""
    d = Path(directory).expanduser()
    if not d.exists():
        return []
    return sorted([str(p) for p in d.glob(pattern) if p.is_file()])


def build_phase1_gui() -> widgets.VBox:
    """
    Build and return the Phase 1 ipywidgets GUI.

    Returns
    -------
    widgets.VBox
        The UI container. Display it in the notebook.

    Examples
    --------
    >>> # In a notebook cell:
    >>> # from rotating_coil_analyzer.scripts.gui_phase1 import build_phase1_gui
    >>> # build_phase1_gui()
    """
    # -------------------------
    # Inputs: file selection
    # -------------------------
    dir_text = widgets.Text(
        value="",
        description="Directory",
        placeholder="e.g. C:\\data\\measurements",
        layout=widgets.Layout(width="70%"),
    )
    pattern_text = widgets.Text(
        value="*.txt",
        description="Glob",
        placeholder="e.g. *.txt or *.bin",
        layout=widgets.Layout(width="40%"),
    )
    scan_btn = widgets.Button(description="Scan", button_style="info")
    files_select = widgets.SelectMultiple(
        options=[],
        description="Files",
        layout=widgets.Layout(width="95%", height="150px"),
    )

    manual_paths = widgets.Textarea(
        value="",
        description="Manual",
        placeholder="Optional: one full path per line (overrides selection if non-empty)",
        layout=widgets.Layout(width="95%", height="120px"),
    )

    # -------------------------
    # Config controls
    # -------------------------
    csv_has_header = widgets.Checkbox(value=True, description="CSV has header")

    extra_names = widgets.Text(
        value="",
        description="Extras",
        placeholder="Optional names for columns 5..N, comma-separated (e.g. magnet_voltage_v)",
        layout=widgets.Layout(width="95%"),
    )

    # BIN-specific controls (whole-run)
    bin_n_channels = widgets.IntText(value=0, description="BIN Nchan", tooltip="0 = infer")
    bin_max_try = widgets.IntText(value=8, description="BIN max try")
    bin_chan_names = widgets.Text(
        value="",
        description="BIN names",
        placeholder="Optional names for channels after dphi_cmp, comma-separated (first is forced to current_a)",
        layout=widgets.Layout(width="95%"),
    )

    split_bin = widgets.Checkbox(value=True, description="Split .bin into plateaus")
    gap_threshold = widgets.FloatText(value=2.0, description="Gap dt [s]")
    smooth_window = widgets.IntText(value=2001, description="Smooth win")
    slope_thr = widgets.FloatText(value=1.0, description="|dI/dt| [A/s]")
    level_round = widgets.FloatText(value=10.0, description="Level step [A]")
    min_duration = widgets.FloatText(value=5.0, description="Min dur [s]")

    # Export controls
    export_enable = widgets.Checkbox(value=False, description="Export outputs")
    export_dir = widgets.Text(
        value="processed_data",
        description="Out dir",
        layout=widgets.Layout(width="70%"),
    )
    export_fmt = widgets.Dropdown(options=["csv", "parquet"], value="csv", description="Format")

    # Actions
    load_btn = widgets.Button(description="Load + Validate", button_style="success")
    plot_btn = widgets.Button(description="Preview Plot", button_style="")
    export_btn = widgets.Button(description="Export Now", button_style="warning")

    plateau_dropdown = widgets.Dropdown(options=[], description="Plateau")

    # Output areas
    out_log = widgets.Output(layout=widgets.Layout(border="1px solid #ddd"))
    out_table = widgets.Output(layout=widgets.Layout(border="1px solid #ddd"))
    out_plot = widgets.Output(layout=widgets.Layout(border="1px solid #ddd"))

    # State (captured in closures)
    state = {
        "plateaus": None,
        "raw_items": None,
        "index_df": None,
    }

    def _make_config() -> LoadConfig:
        """Read widget values and build a LoadConfig."""
        extra_list = [s.strip() for s in extra_names.value.split(",") if s.strip()] or None

        nchan = None if bin_n_channels.value in (0, None) else int(bin_n_channels.value)

        bin_names = [s.strip() for s in bin_chan_names.value.split(",") if s.strip()] or None

        return LoadConfig(
            extra_names=extra_list,
            csv_has_header=bool(csv_has_header.value),
            bin_n_currents=nchan,
            bin_max_currents_to_try=int(bin_max_try.value),
            bin_channel_names=bin_names,
            drop_nonfinite=True,
            split_long_run_into_plateaus=bool(split_bin.value),
            gap_threshold_s=float(gap_threshold.value),
            plateau_smooth_window=int(smooth_window.value),
            plateau_slope_threshold_a_per_s=float(slope_thr.value),
            plateau_level_rounding_a=float(level_round.value),
            plateau_min_duration_s=float(min_duration.value),
        )

    def _selected_paths() -> List[str]:
        """Decide file list: manual overrides selection; selection from scan otherwise."""
        manual = [line.strip() for line in manual_paths.value.splitlines() if line.strip()]
        if manual:
            return manual
        return list(files_select.value)

    def _refresh_plateau_dropdown(plateaus: List[pd.DataFrame]) -> None:
        """Update plateau dropdown options from loaded plateaus."""
        if not plateaus:
            plateau_dropdown.options = []
            return
        opts = [(f"{int(p['plateau_id'].iloc[0]):03d} (mean {p['current_a'].mean():.2f} A)", int(p["plateau_id"].iloc[0]))
                for p in plateaus]
        plateau_dropdown.options = opts
        plateau_dropdown.value = opts[0][1]

    def on_scan(_):
        with out_log:
            out_log.clear_output()
            paths = _list_files(dir_text.value, pattern_text.value)
            print(f"Found {len(paths)} files.")
        files_select.options = paths

    def on_load(_):
        with out_log:
            out_log.clear_output()
            paths = _selected_paths()
            if not paths:
                print("No input files selected.")
                return

            cfg = _make_config()
            print("LoadConfig:")
            print(asdict(cfg))

            plateaus, raw_items = load_measurements(paths, cfg)

            # Show validations for each raw input file.
            for item in raw_items:
                v = item.validation
                print(f"\n{item.meta.source_file}: ok={v.ok}, dropped_nonfinite={v.dropped_nonfinite_rows}")
                for w in v.warnings:
                    print("  WARNING:", w)
                for e in v.errors:
                    print("  ERROR:", e)

            # If there are validation errors, stop early.
            if any((not it.validation.ok) for it in raw_items):
                print("\nAt least one input failed validation. Fix inputs/settings and reload.")
                return

            idx = make_plateau_index(plateaus)

            state["plateaus"] = plateaus
            state["raw_items"] = raw_items
            state["index_df"] = idx

            _refresh_plateau_dropdown(plateaus)

        with out_table:
            out_table.clear_output()
            display(idx)

    def on_plot(_):
        plateaus = state.get("plateaus")
        if not plateaus:
            with out_log:
                print("Nothing loaded yet.")
            return

        pid = plateau_dropdown.value
        df = next((p for p in plateaus if int(p["plateau_id"].iloc[0]) == int(pid)), None)
        if df is None:
            with out_log:
                print(f"Plateau {pid} not found.")
            return

        with out_plot:
            out_plot.clear_output()

            # Basic preview plots: current and both delta-flux signals vs time.
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(df["time_s"].to_numpy(), df["current_a"].to_numpy())
            ax1.set_ylabel("Current [A]")
            ax1.grid(True)

            ax2 = fig.add_subplot(3, 1, 2)
            ax2.plot(df["time_s"].to_numpy(), df["dphi_abs"].to_numpy())
            ax2.set_ylabel("ΔΦ_abs")
            ax2.grid(True)

            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(df["time_s"].to_numpy(), df["dphi_cmp"].to_numpy())
            ax3.set_ylabel("ΔΦ_cmp")
            ax3.set_xlabel("time_s [s]")
            ax3.grid(True)

            plt.show()

    def on_export(_):
        plateaus = state.get("plateaus")
        if not plateaus:
            with out_log:
                print("Nothing loaded yet.")
            return

        if not export_enable.value:
            with out_log:
                print("Export is disabled. Tick 'Export outputs' first.")
            return

        with out_log:
            fmt = export_fmt.value
            out_dir = export_dir.value.strip() or "processed_data"
            try:
                export_phase1_outputs(plateaus, out_dir, fmt=fmt)
                print(f"Exported Phase 1 outputs to '{out_dir}' as {fmt}.")
            except Exception as e:
                print(f"Export failed: {e}")

    scan_btn.on_click(on_scan)
    load_btn.on_click(on_load)
    plot_btn.on_click(on_plot)
    export_btn.on_click(on_export)

    # Layout the GUI
    file_box = widgets.VBox([
        widgets.HBox([dir_text, pattern_text, scan_btn]),
        files_select,
        manual_paths,
    ])

    cfg_box = widgets.VBox([
        widgets.HTML("<b>Parsing options</b>"),
        csv_has_header,
        extra_names,
        widgets.HTML("<b>BIN options (whole-run)</b>"),
        widgets.HBox([bin_n_channels, bin_max_try, split_bin]),
        bin_chan_names,
        widgets.HBox([gap_threshold, smooth_window, slope_thr]),
        widgets.HBox([level_round, min_duration]),
    ])

    export_box = widgets.VBox([
        widgets.HTML("<b>Export</b>"),
        widgets.HBox([export_enable, export_fmt]),
        export_dir,
        export_btn,
    ])

    actions_box = widgets.HBox([load_btn, plateau_dropdown, plot_btn])

    ui = widgets.VBox([
        widgets.HTML("<h3>Phase 1: Load / Validate / Split Plateaus</h3>"),
        file_box,
        cfg_box,
        actions_box,
        export_box,
        widgets.HTML("<b>Log</b>"),
        out_log,
        widgets.HTML("<b>Plateau index</b>"),
        out_table,
        widgets.HTML("<b>Preview</b>"),
        out_plot,
    ])

    return ui
