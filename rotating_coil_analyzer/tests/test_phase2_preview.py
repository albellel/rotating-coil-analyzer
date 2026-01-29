"""Regression tests for Phase II (Harmonics) preview callback.

Bug B fix: Removed dangling `_init_plot_once()` call that caused NameError.
"""

import pytest


def test_preview_callback_no_init_plot_once_reference():
    """Verify that _init_plot_once is not referenced in _preview_data_quality callback.

    Regression test for Bug B: NameError("name '_init_plot_once' is not defined")
    """
    import inspect
    from rotating_coil_analyzer.gui import phase2

    # Get the source code of the module
    source = inspect.getsource(phase2)

    # The function _init_plot_once should not be called anywhere
    # (it was removed during refactoring)
    assert "_init_plot_once()" not in source, (
        "_init_plot_once() call found in phase2.py - this function was removed"
    )


def test_phase2_panel_import_and_build():
    """Verify that phase2 panel can be imported and built without error."""
    from rotating_coil_analyzer.gui.phase2 import build_phase2_panel

    # Build with a mock callable
    panel = build_phase2_panel(lambda: None)

    # Should return a widget
    import ipywidgets as w
    assert isinstance(panel, w.Widget)


def test_phase2_preview_button_exists():
    """Verify that the Preview data-quality cuts button exists and is wired."""
    from rotating_coil_analyzer.gui.phase2 import build_phase2_panel
    import ipywidgets as w

    panel = build_phase2_panel(lambda: None)

    # Find the preview button by description
    def find_button(widget, description):
        if isinstance(widget, w.Button) and widget.description == description:
            return widget
        if hasattr(widget, 'children'):
            for child in widget.children:
                result = find_button(child, description)
                if result is not None:
                    return result
        return None

    btn = find_button(panel, "Preview data-quality cuts")
    assert btn is not None, "Preview data-quality cuts button not found"

    # Button should have click handlers registered
    assert hasattr(btn, '_click_handlers')
    assert len(btn._click_handlers.callbacks) > 0, "No click handlers registered"
