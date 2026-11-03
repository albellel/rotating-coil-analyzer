# SM18 HCMCBXFB012 Dipole -- Dec 4, 2024

## Overview
Validates the Python pipeline against the SM18 legacy C++/MATLAB analyzer for the HCMCBXFB012 dipole. This is a 5-segment cold magnet with 25 coils.

## Configuration

| Parameter | Value |
|-----------|-------|
| Magnet | HCMCBXFB012 dipole (m=1) |
| Reference | SM18 legacy C++/MATLAB with `dri rot nor cel dit` |
| Reference radius | 0.050 m |
| Samples/turn | 512 |
| Segments | 5 (z = -1, -0.5, 0, +0.5, +1 m) |
| Total turns compared | 285,095 |
| Current range | 0--1,740 A |
| `legacy_rotate_excludes_last` | **True** (SM18 off-by-one in C++ rotation loop) |

## Results

### Best Configuration (from 40-combination sweep)
- Options: `dri`, `rot`, `nor`, `cel` -- `dit` OFF (binary files pre-corrected)
- `legacy_rotate_excludes_last=True` needed for SM18 parity
- Cold geometry required for sub-ppb Kn parity

### Per-Segment Precision (1,740 A plateau)

| Segment | B_main (T) | b3 (ppb) | Worst bn (ppb) |
|---------|-----------|----------|---------------|
| 1 (end) | -5.09e-03 | 0.35 | 2.49 (b15) |
| 2 | 1.743 | 0.09 | 0.28 (b7) |
| **3 (center)** | **2.959** | **0.04** | **0.10 (b9)** |
| 4 | 1.732 | 0.14 | 0.51 (b2) |
| 5 (end) | -4.52e-03 | 2.33 | 9.45 (b2) |

### Verdicts
- **ALL-TURNS: MISMATCH** (expected -- ramp turns have large drift sensitivity)
- **PLATEAU-ONLY: CLOSE** (sub-unit across all harmonics)
- **Central segments (2,3,4)**: worst precision = 46.1 ppb
- **Segment 3 (center)**: all harmonics agree to < 1e-10 relative -- **machine-precision parity**

## Key Caveats
1. `dit` OFF because binary files are pre-corrected; applying dit would double-correct.
2. `legacy_rotate_excludes_last=True` is SM18-specific (off-by-one in legacy C++). BTP8 uses `False`.
3. End segments (1,5) see ~5 mT at 1,740 A, amplifying ppb ratios. In absolute terms, agreement is machine-precision.
4. Ramp turns (~0.6%) show large differences -- expected, not a pipeline defect.

## Key Findings
1. **Sub-ppb parity at the center segment** confirms mathematical identity with C++ code for plateau data.
2. The SM18 off-by-one exists only for legacy compatibility -- no theoretical justification.
3. Cold geometry is essential for Kn parity (warm geometry causes ~0.5% mismatch).
