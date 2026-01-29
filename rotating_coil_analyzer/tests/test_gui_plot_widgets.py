from __future__ import annotations

"""Headless smoke tests verifying that GUI plot widgets are created and attached.

These tests run without a display and verify that:
1. w.Output() widgets are created for plot areas
2. Panel builders return valid ipywidgets
3. No exceptions during widget construction
"""

import pytest
import ipywidgets as w


def test_phase1_panel_creates_output_widget():
    """Verify Phase 1 panel creates an Output widget for plots."""
    from rotating_coil_analyzer.gui.app import _build_phase1_panel

    shared = {"catalog": None, "segment_frame": None, "segment_path": None}
    panel = _build_phase1_panel(shared)

    assert isinstance(panel, w.Widget)

    # Find the Output widget in the tree (plot area)
    def find_output(widget):
        if isinstance(widget, w.Output):
            return widget
        if hasattr(widget, "children"):
            for child in widget.children:
                result = find_output(child)
                if result is not None:
                    return result
        return None

    out_plot = find_output(panel)
    assert out_plot is not None, "Phase 1 panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_phase2_panel_creates_output_widget():
    """Verify Phase 2 panel creates an Output widget for plots."""
    from rotating_coil_analyzer.gui.phase2 import build_phase2_panel

    panel = build_phase2_panel(lambda: None)

    assert isinstance(panel, w.Widget)

    # Find the Output widget in the tree
    def find_output(widget):
        if isinstance(widget, w.Output):
            return widget
        if hasattr(widget, "children"):
            for child in widget.children:
                result = find_output(child)
                if result is not None:
                    return result
        return None

    out_plot = find_output(panel)
    assert out_plot is not None, "Phase 2 panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_phase4_panel_creates_output_widget():
    """Verify Phase 4 (Plots) panel creates an Output widget."""
    from rotating_coil_analyzer.gui.phase4_plots import build_phase4_plots_panel

    panel = build_phase4_plots_panel(lambda: None)

    assert isinstance(panel, w.Widget)

    # Find the Output widget in the tree
    def find_output(widget):
        if isinstance(widget, w.Output):
            return widget
        if hasattr(widget, "children"):
            for child in widget.children:
                result = find_output(child)
                if result is not None:
                    return result
        return None

    out_plot = find_output(panel)
    assert out_plot is not None, "Phase 4 panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_build_gui_returns_tab_widget():
    """Verify build_gui returns a Tab widget with all panels."""
    from rotating_coil_analyzer.gui.app import build_gui

    # Note: clear_cell_output=False to avoid IPython dependency
    gui = build_gui(clear_cell_output=False)

    assert isinstance(gui, w.Tab)
    assert len(gui.children) == 5  # Catalog, FFT, Coil Calibration, Harmonic Merge, Plots


def test_backend_init_function_exists_and_callable():
    """Verify _try_enable_interactive_backend_once is available."""
    from rotating_coil_analyzer.gui.app import _try_enable_interactive_backend_once

    ok, msg = _try_enable_interactive_backend_once()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0
