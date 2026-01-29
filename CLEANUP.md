# Cleanup Report

This document lists files identified for cleanup/deletion and provides recommendations.

## Recommended for Deletion

### 1. `scratch_kn_smoke.py` (root directory)
- **Status**: DELETE
- **Evidence**: Empty file (1 line), not imported anywhere, name suggests scratch/temporary use
- **Action**: Safe to delete

### 2. `rotating_coil_analyzer/analysis/preprocess.bak`
- **Status**: DELETE
- **Evidence**: Backup file, not imported anywhere
- **Action**: Safe to delete

## Kept for Backward Compatibility

### 3. `rotating_coil_analyzer/gui/phase3_kn.py`
- **Status**: DEPRECATED (keep for now)
- **Evidence**: Imported in app.py but not used in the main GUI flow. The functionality has been split into `phase3a_coil_calibration.py` and `phase3b_harmonic_merge.py`.
- **Recommendation**: Keep as `phase3_kn_legacy.py` or mark with deprecation warnings. May be removed in a future release.
- **Migration**: Users calling `build_phase3_kn_panel()` directly should migrate to the new split tabs.

## File Naming Notes

The following files use "phase" naming but are functional:
- `gui/phase2.py` → Harmonics tab (could be renamed to `tab_harmonics.py` in future)
- `gui/phase3a_coil_calibration.py` → Coil Calibration tab
- `gui/phase3b_harmonic_merge.py` → Harmonic Merge tab
- `gui/phase4_plots.py` → Plots tab (could be renamed to `tab_plots.py` in future)

These have been kept as-is to minimize import breakage. Tab labels in the GUI have been updated to functional names.

## Docstring Updates Applied

The following docstrings were updated to remove "Phase I/II/III/IV" terminology:
- `gui/app.py`: `_build_phase1_panel()`, `build_gui()`
- `gui/phase2.py`: State class renamed to `HarmonicsState`
- `gui/phase4_plots.py`: Module docstring, header HTML
- `analysis/__init__.py`: Module docstring

## Test Files

All test files are active and should be kept:
- `tests/test_gui_plot_widgets.py` - NEW: Verifies plot widgets are correctly created
- `tests/test_kn_bundle.py` - Tests for KnBundle and export/import round-trip
- `tests/test_harmonic_merge.py` - Tests for merge correctness
- Other test files remain unchanged

## Commands to Execute Cleanup

```bash
# Delete scratch file
del scratch_kn_smoke.py
# Or on Unix: rm scratch_kn_smoke.py

# Delete backup file
del rotating_coil_analyzer\analysis\preprocess.bak
# Or on Unix: rm rotating_coil_analyzer/analysis/preprocess.bak
```

## Verification

After cleanup, verify:
```bash
# Run all tests
py -m pytest rotating_coil_analyzer/tests/ -v

# Verify imports
py -c "from rotating_coil_analyzer.gui.app import build_gui; print('OK')"
```
