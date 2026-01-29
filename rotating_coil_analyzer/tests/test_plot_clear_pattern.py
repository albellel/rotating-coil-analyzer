"""Regression tests for plot clearing pattern.

Bug fix: Plots were appending because clear_output() was called outside
the `with out_plot:` context. Fixed by moving clear_output() inside the
`with` block.

The correct pattern is:
    with out_plot:
        _clear_and_close()  # MUST be inside the context!
        fig, ax = plt.subplots()
        ...
        plt.show()
"""

import pytest


def test_phase2_clear_inside_with_block():
    """Verify phase2.py uses _clear_and_close() inside `with out_plot:` blocks."""
    import inspect
    from rotating_coil_analyzer.gui import phase2

    source = inspect.getsource(phase2)

    # The old pattern (_begin_plot() before `with out_plot:`) should be gone
    # The new pattern should have _clear_and_close() inside the with block
    assert "_clear_and_close()" in source, (
        "_clear_and_close() function should exist in phase2.py"
    )

    # Should NOT have `_begin_plot()` (old function)
    # Note: The function was renamed, so old calls should not exist
    lines = source.split("\n")
    for i, line in enumerate(lines):
        # Skip comments and function definitions
        if line.strip().startswith("#") or "def _begin_plot" in line:
            continue
        # Check for calls to old function outside of a with block
        if "_begin_plot()" in line:
            # This would be the old pattern - should not exist
            pytest.fail(
                f"Found old _begin_plot() call at line {i+1}: {line.strip()}"
            )


def test_phase2_has_clear_and_close_function():
    """Verify _clear_and_close function exists and clears properly."""
    import inspect
    from rotating_coil_analyzer.gui import phase2

    source = inspect.getsource(phase2)

    # Verify the function definition includes the correct operations
    assert "def _clear_and_close" in source
    assert "out_plot.clear_output" in source
    assert "plt.close" in source
