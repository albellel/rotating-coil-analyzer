from __future__ import annotations

"""Headless smoke tests verifying that GUI plot widgets are created and attached.

These tests run without a display and verify that:
1. w.Output() widgets are created for plot areas
2. Panel builders return valid ipywidgets
3. No exceptions during widget construction
4. All 8 tabs are present in the main GUI
"""

import pytest
import ipywidgets as w


def _find_output(widget):
    """Recursively find the first Output widget in a widget tree."""
    if isinstance(widget, w.Output):
        return widget
    if hasattr(widget, "children"):
        for child in widget.children:
            result = _find_output(child)
            if result is not None:
                return result
    return None


def _find_widget_by_type(widget, wtype):
    """Recursively find the first widget of a given type."""
    if isinstance(widget, wtype):
        return widget
    if hasattr(widget, "children"):
        for child in widget.children:
            result = _find_widget_by_type(child, wtype)
            if result is not None:
                return result
    return None


def _find_button_by_description(widget, desc):
    """Recursively find a Button with a matching description."""
    if isinstance(widget, w.Button) and widget.description == desc:
        return widget
    if hasattr(widget, "children"):
        for child in widget.children:
            result = _find_button_by_description(child, desc)
            if result is not None:
                return result
    return None


def test_catalog_panel_creates_output_widget():
    """Verify Catalog panel creates an Output widget for plots."""
    from rotating_coil_analyzer.gui.app import _build_phase1_panel

    shared = {"catalog": None, "segment_frame": None, "segment_path": None}
    panel = _build_phase1_panel(shared)

    assert isinstance(panel, w.Widget)
    out_plot = _find_output(panel)
    assert out_plot is not None, "Catalog panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_harmonics_panel_creates_output_widget():
    """Verify Harmonics panel creates an Output widget for plots."""
    from rotating_coil_analyzer.gui.harmonics import build_phase2_panel

    panel = build_phase2_panel(lambda: None)

    assert isinstance(panel, w.Widget)
    out_plot = _find_output(panel)
    assert out_plot is not None, "Harmonics panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_plots_panel_creates_output_widget():
    """Verify Plots panel creates an Output widget."""
    from rotating_coil_analyzer.gui.plots import build_phase4_plots_panel

    panel = build_phase4_plots_panel(lambda: None)

    assert isinstance(panel, w.Widget)
    out_plot = _find_output(panel)
    assert out_plot is not None, "Plots panel should contain an Output widget for plots"
    assert isinstance(out_plot, w.Output)


def test_build_gui_returns_tab_widget():
    """Verify build_gui returns a Tab widget with 8 panels."""
    from rotating_coil_analyzer.gui.app import build_gui

    # Note: clear_cell_output=False to avoid IPython dependency
    gui = build_gui(clear_cell_output=False)

    assert isinstance(gui, w.Tab)
    assert len(gui.children) == 8, (
        f"Expected 8 tabs (Catalog, Plateau Detection, Harmonics, Coil Calibration, "
        f"Harmonic Merge, Raw Signal Plots, Physics Plots, Comparison), got {len(gui.children)}"
    )


def test_build_gui_tab_titles():
    """Verify all 8 tab titles are correct."""
    from rotating_coil_analyzer.gui.app import build_gui

    gui = build_gui(clear_cell_output=False)

    expected_titles = [
        "Catalog",
        "Plateau Detection",
        "Harmonics",
        "Coil Calibration",
        "Harmonic Merge",
        "Raw Signal Plots",
        "Physics Plots",
        "Comparison",
    ]
    for i, title in enumerate(expected_titles):
        assert gui.get_title(i) == title, (
            f"Tab {i} title should be '{title}', got '{gui.get_title(i)}'"
        )


def test_backend_init_function_exists_and_callable():
    """Verify _try_enable_interactive_backend_once is available."""
    from rotating_coil_analyzer.gui.app import _try_enable_interactive_backend_once

    ok, msg = _try_enable_interactive_backend_once()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


# ---- New panel smoke tests ----

def test_plateau_detection_panel_smoke():
    """Verify Plateau Detection panel builds without error."""
    from rotating_coil_analyzer.gui.plateau_detection import build_plateau_detection_panel

    panel = build_plateau_detection_panel(
        get_segmentframe=lambda: None,
        get_segmentpath=lambda: None,
        set_plateau_info=lambda x: None,
    )

    assert isinstance(panel, w.Widget)
    out = _find_output(panel)
    assert out is not None, "Plateau Detection panel should contain an Output widget"
    btn = _find_button_by_description(panel, "Detect plateaus")
    assert btn is not None, "Plateau Detection panel should have a 'Detect plateaus' button"


def test_physics_plots_panel_smoke():
    """Verify Physics Plots panel builds without error."""
    from rotating_coil_analyzer.gui.physics_plots import build_physics_plots_panel

    panel = build_physics_plots_panel(
        get_merge_result=lambda: None,
        get_plateau_info=lambda: None,
        get_segmentframe=lambda: None,
        set_n_last_recommended=lambda n: None,
    )

    assert isinstance(panel, w.Widget)
    out = _find_output(panel)
    assert out is not None, "Physics Plots panel should contain an Output widget"
    btn = _find_button_by_description(panel, "Compute summary")
    assert btn is not None, "Physics Plots panel should have a 'Compute summary' button"
    btn_eddy = _find_button_by_description(panel, "Fit eddy tau")
    assert btn_eddy is not None, "Physics Plots panel should have a 'Fit eddy tau' button"


def test_comparison_panel_smoke():
    """Verify Comparison panel builds without error."""
    from rotating_coil_analyzer.gui.comparison import build_comparison_panel

    panel = build_comparison_panel()

    assert isinstance(panel, w.Widget)
    out = _find_output(panel)
    assert out is not None, "Comparison panel should contain an Output widget"
    btn = _find_button_by_description(panel, "Load & compare")
    assert btn is not None, "Comparison panel should have a 'Load & compare' button"


def test_harmonic_merge_has_new_widgets():
    """Verify Harmonic Merge panel has the new Wave 1/2 widgets."""
    from rotating_coil_analyzer.gui.harmonic_merge import build_phase3b_harmonic_merge_panel

    panel = build_phase3b_harmonic_merge_panel(
        get_segmentframe_callable=lambda: None,
        get_segmentpath_callable=lambda: None,
        get_kn_bundle_callable=lambda: None,
        set_merge_result_callable=lambda x: None,
    )

    assert isinstance(panel, w.Widget)

    # Check for CEL/FED diagnostic button
    btn = _find_button_by_description(panel, "Diagnose CEL/FED")
    assert btn is not None, "Harmonic Merge should have 'Diagnose CEL/FED' button"

    # Check for MAD outlier button
    btn_mad = _find_button_by_description(panel, "Remove outliers")
    assert btn_mad is not None, "Harmonic Merge should have 'Remove outliers' button"
