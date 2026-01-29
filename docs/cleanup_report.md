# Cleanup Report

Generated: 2026-01-29

## Files Recommended for Deletion

### 1. Root directory test/scratch files

| File | Status | Evidence | Action |
|------|--------|----------|--------|
| `kn_identity_20.txt` | DELETE | Test fixture (20 rows of `1 0 1 0`), not imported by any production code | Safe to delete |
| `kn_main_phase_pi4_20.txt` | DELETE | Similar test fixture, not imported | Safe to delete |
| `scratch_kn_smoke.py` | DELETE | Empty or scratch file, not imported | Safe to delete |

### 2. Backup files

| File | Status | Evidence | Action |
|------|--------|----------|--------|
| `rotating_coil_analyzer/analysis/preprocess.bak` | DELETE | Backup file, superseded by `preprocess.py` | Safe to delete |

### 3. Legacy GUI files

| File | Status | Evidence | Action |
|------|--------|----------|--------|
| `rotating_coil_analyzer/gui/phase3_kn.py` | KEEP (deprecated) | Legacy monolithic kn tab. Functionality split into `phase3a_coil_calibration.py` and `phase3b_harmonic_merge.py`. May still be imported by external scripts. | Keep for backward compatibility; mark as deprecated |

## Naming Consistency Updates Applied

All "Phase I/II/III/IV" terminology has been replaced with functional names:

| Old Name | New Name |
|----------|----------|
| Phase I | Catalog |
| Phase II | Harmonics |
| Phase 3A | Coil Calibration |
| Phase 3B | Harmonic Merge |
| Phase IV | Plots |

### Files Updated

- `gui/app.py`: `Phase1State` -> `CatalogState`, messages updated
- `gui/phase2.py`: `HarmonicsState` (already renamed), comments updated
- `gui/phase3a_coil_calibration.py`: `Phase3AState` -> `CoilCalibrationState`, docstrings/messages updated
- `gui/phase3b_harmonic_merge.py`: `Phase3BState` -> `HarmonicMergeState`, docstrings/messages updated
- Various test files: Updated comments referencing Phase terminology

## Golden Standards (kn files)

### Found in `golden_standards/golden_standard_01_LIU_BTP8/COIL_PCB/`

**Central coil (PCB_DQ_5_18_7_250_47x50_Hall):**
```
PCB_DQ_5_18_7_250_47x50_Hall/Kn-Th/Kn_DQ_5_18_7_250_47x50_0001_A_AC.txt      # A-C compensation
PCB_DQ_5_18_7_250_47x50_Hall/Kn-Th/Kn_DQ_5_18_7_250_47x50_0001_A_ABCD.txt    # ABCD compensation
PCB_DQ_5_18_7_250_47x50_Hall/Kn-Th/Kn_DQ_5_18_7_250_47x50_0001_BD_AE.txt     # BD_AE scheme
```

**Integral coil (R45_PCB):**
```
Kn_R45_PCB_N1_0001_A_ABCD.txt
Kn_R45_PCB_N2_0001_A_ABCD.txt
```

**Naming convention interpretation:**
- `_A_AC` = Absolute channel, A-C compensation scheme
- `_A_ABCD` = Absolute channel, ABCD compensation scheme
- `_BD_AE` = BD_AE compensation scheme

## Commands to Execute Cleanup

```bash
# Delete root test files
rm kn_identity_20.txt
rm kn_main_phase_pi4_20.txt
rm scratch_kn_smoke.py

# Delete backup file
rm rotating_coil_analyzer/analysis/preprocess.bak
```

## Verification After Cleanup

```bash
# Run all tests
py -m pytest rotating_coil_analyzer/tests/ -v

# Verify imports
py -c "from rotating_coil_analyzer.gui.app import build_gui; print('OK')"
```
