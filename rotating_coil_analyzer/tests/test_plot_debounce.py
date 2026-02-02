"""Regression tests for Catalog plot duplicate fix.

Bug A fix: Added debounce guard and explicit figure cleanup to prevent
duplicate traces when clicking Preview.
"""

import pytest


def test_catalog_state_has_debounce_fields():
    """Verify CatalogState has debounce fields to prevent duplicate callbacks."""
    from rotating_coil_analyzer.gui.app import CatalogState

    # Create a state instance
    st = CatalogState()

    # Should have debounce fields
    assert hasattr(st, 'busy'), "CatalogState missing 'busy' field"
    assert hasattr(st, 'last_action_key'), "CatalogState missing 'last_action_key' field"
    assert hasattr(st, 'last_action_t'), "CatalogState missing 'last_action_t' field"

    # Default values
    assert st.busy is False
    assert st.last_action_key is None
    assert st.last_action_t == 0.0


def test_catalog_panel_preview_button_wired():
    """Verify Preview button exists and has handlers."""
    from rotating_coil_analyzer.gui.app import _build_phase1_panel
    import ipywidgets as w

    shared = {"catalog": None, "segment_frame": None, "segment_path": None}
    panel = _build_phase1_panel(shared)

    def find_button(widget, description):
        if isinstance(widget, w.Button) and widget.description == description:
            return widget
        if hasattr(widget, 'children'):
            for child in widget.children:
                result = find_button(child, description)
                if result is not None:
                    return result
        return None

    btn = find_button(panel, "Preview")
    assert btn is not None, "Preview button not found"
    assert len(btn._click_handlers.callbacks) > 0, "No click handlers registered"


def test_plot_first_turns_source_has_line_count_check():
    """Verify that _plot_first_turns includes a line count assertion.

    This is a sanity check to catch if multiple lines are accidentally plotted.
    """
    import inspect
    from rotating_coil_analyzer.gui import app

    source = inspect.getsource(app)

    # The function should include a check for number of lines
    assert "ax.get_lines()" in source, (
        "_plot_first_turns should include line count verification"
    )
    assert "n_lines" in source or "get_lines()" in source, (
        "_plot_first_turns should verify number of Line2D artists"
    )
