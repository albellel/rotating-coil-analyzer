# Golden Standard Parity Validation

This document describes the golden standard parity validation workflow for the rotating coil analyzer.

## Overview

The golden standard parity workflow validates our analysis pipeline against reference results from known-good measurements. This ensures:

1. **Correctness**: Our computed harmonics match legacy analyzer outputs within acceptable tolerances
2. **Reproducibility**: Results can be regenerated from raw data + calibration files
3. **Traceability**: Full provenance tracking from raw measurement to final results

## Workflow Components

### Primary Deliverable

**Notebook**: `rotating_coil_analyzer/notebooks/golden_standard_parity.ipynb`

This Jupyter notebook provides an interactive workflow for:
- Dataset introspection and mapping
- Pipeline execution
- Results comparison
- Visualization

### Support Module

**Module**: `rotating_coil_analyzer/validation/golden_runner.py`

Provides reusable APIs for:
- `scan_golden_dataset()`: Scan and analyze golden dataset folders
- `run_pipeline()`: Execute the full analysis pipeline
- `compare_results()`: Compare computed vs reference results
- `export_results()`: Export in canonical schema
- `generate_diff_report()`: Generate human-readable diff reports

## Canonical Schema

### Standard: `example_results` Format

The canonical output schema follows the modern `example_results` standard:

```
Time(s)          - Measurement timestamp
Duration(s)      - Turn duration
Options          - Applied options (dri, rot, nor, cel, fed)
Rref(m)          - Reference radius
Lcoil(m)         - Coil length
I(A)             - Current
Ramprate(A/s)    - Current ramp rate
I1(A)            - Secondary current (if applicable)
Ramprate1(A/s)   - Secondary ramp rate
dx(mm)           - X position offset
dy(mm)           - Y position offset
phi(rad)         - Rotation angle
B_main(T)        - Main field normal component
A_main(T)        - Main field skew component
B_main_TF(T/kA)  - Main field transfer function (normal)
A_main_TF(T/kA)  - Main field transfer function (skew)
B1(T), A1(T)     - First harmonic (Tesla)
b2(Units)...b15(Units) - Normal multipoles (normalized)
a2(Units)...a15(Units) - Skew multipoles (normalized)
```

### Legacy Schema (BTP8 Format)

Legacy reference files use a different format:

```
Date             - Timestamp string
Options          - Applied options
A_MRU Level (rad) - Level sensor A
B_Top Plate Level (rad) - Level sensor B
Rref(m), Lcoil(m), I(A), I FGC(A) - Metadata
x (mm), y (mm)   - Position
B1 (T)...B15 (T) - Normal components (Tesla)
A1 (T)...A15 (T) - Skew components (Tesla)
Angle (rad)      - Rotation angle
```

### Schema Rule

**Our exported results must be a superset of the golden reference columns:**

1. Start with canonical `example_results` column list/order
2. If golden reference has extra columns not in canonical:
   - Append them deterministically
   - Populate if possible, otherwise fill with NaN
3. If computed dataframe has extra internal columns:
   - Keep only if needed for comparison/traceability
   - Otherwise exclude from final export

### Metadata Storage

Provenance metadata is stored in a JSON sidecar file:

```json
{
  "dataset_folder": "path/to/dataset",
  "kn_file": "path/to/kn.txt",
  "magnet_order": 2,
  "r_ref_m": 1.0,
  "l_coil_m": 1.32209,
  "compensation_scheme": "BD_AE",
  "options": ["dri", "rot"],
  "merge_mode": "abs_upto_m_cmp_above",
  "timestamp": "2024-01-29T12:00:00Z",
  "kn_source_type": "segment_txt",
  "kn_source_path": "path/to/kn.txt",
  "merge_per_n_source_map": "abs,abs,cmp,cmp,cmp,..."
}
```

## Output Directory Structure

```
outputs/golden_runs/<dataset_id>/
├── computed_results.csv       # Our computed results
├── computed_results.json      # Provenance metadata
├── reference_<name>.txt       # Copy of golden reference
├── diff_report.md            # Human-readable comparison
├── diff_data.csv             # Numeric differences
└── plots/
    ├── harmonic_comparison_n1.png
    ├── harmonic_comparison_n2.png
    └── ...
```

## How to Run

### Prerequisites

```bash
# Install dependencies
pip install -e .
pip install jupyter matplotlib
```

### Running the Notebook

1. Open the notebook:
   ```bash
   cd rotating_coil_analyzer/notebooks
   jupyter notebook golden_standard_parity.ipynb
   ```

2. Edit the configuration cell with your dataset parameters:
   - `DATASET_FOLDER`: Path to golden dataset
   - `KN_FILE`: Path to kn calibration file
   - `MAGNET_ORDER`: 1=dipole, 2=quadrupole, etc.
   - `OPTIONS`: Tuple of enabled options

3. Run all cells

4. Inspect outputs in `outputs/golden_runs/<dataset_id>/`

### Command-Line Usage

```python
from rotating_coil_analyzer.validation.golden_runner import (
    scan_golden_dataset,
    PipelineConfig,
    run_pipeline,
    compare_results,
    export_results,
)

# Scan dataset
introspection = scan_golden_dataset(Path("golden_standards/golden_standard_01_LIU_BTP8"))

# Configure pipeline
config = PipelineConfig(
    dataset_folder=Path("path/to/data"),
    kn_file=Path("path/to/kn.txt"),
    magnet_order=2,
    r_ref_m=1.0,
    options=("dri", "rot"),
)

# Run pipeline
result = run_pipeline(config)

# Export results
export_results(result.computed_df, Path("output.csv"), result.provenance_metadata)
```

## Adding a New Golden Dataset

1. **Create dataset folder** under `golden_standards/`:
   ```
   golden_standards/golden_standard_XX_<name>/
   ├── <coil_type>/           # Central, Integral, PCB
   │   └── <run_folder>/      # Timestamped run folder
   │       ├── *_Parameters.txt
   │       ├── *_results.txt  # Reference results
   │       ├── *_fluxes_Ascii.txt  # Raw flux data
   │       └── *_current.txt  # Current measurements
   └── kn/                    # Kn calibration files
       └── Kn_*.txt
   ```

2. **Document the dataset** in `golden_standards/README.md`:
   - Magnet type and name
   - Measurement date
   - Compensation scheme
   - Any special considerations

3. **Run the notebook** with updated configuration

4. **Verify tolerances** and adjust if needed

## Interpreting Tolerances

### Typical Error Sources

1. **Floating-point precision**: ~1e-15 relative
2. **Algorithm differences**: Can cause ~1e-6 to 1e-3 differences
3. **Scaling conventions**: May need unit conversion
4. **Time/index alignment**: Row matching critical

### Tolerance Recommendations

| Error Type | Typical Tolerance | Notes |
|------------|-------------------|-------|
| Numeric precision | 1e-9 abs | Floating point |
| Algorithm match | 1e-3 rel | Legacy vs modern |
| Unit conversion | Check scale | T vs mT, etc. |

### Diff Report Interpretation

The diff report shows:

1. **Schema differences**: Missing/extra columns
2. **Max absolute error**: Worst-case per column
3. **RMS error**: Overall spread
4. **Worst rows**: Specific mismatches for debugging

## Constraints

- **No synthetic time**: Time must come from measured data
- **No interpolation**: Downsampling uses decimation only
- **Immutable data models**: All results are frozen dataclasses
- **Full traceability**: Every output carries provenance

## Critical Configuration Parameters

### Samples Per Turn

**This is the most critical parameter!** Using the wrong value causes magnitude errors of 60x or more.

| Data Format | Typical Value | Notes |
|-------------|---------------|-------|
| BTP8 (LIU) | 512 | Standard for CERN BTP8 systems |
| FDI | 1024 or 2048 | Check encoder resolution |
| Custom | Varies | Check acquisition settings |

**How to determine the correct value:**
1. Check the Parameters.txt file for "Samples per turn" or "Encoder resolution"
2. Verify the total samples in a file is divisible by your chosen value
3. Expected turns per run × samples per turn ≈ total samples
4. BTP8 example: 7168 samples ÷ 512 = 14 turns (with 6 saved to results)

### Drift Correction Mode

Two modes are available:
- **legacy** (default): Matches the ffmm C++ analyzer exactly
- **weighted**: Bottura/Pentella style for variable dt

The legacy drift correction formula (from ffmm C++):
```
flux = cumsum(df - mean(df)) - mean(cumsum(df))
```

**Important**: The Python implementation subtracts `mean(cumsum(df))` (the mean of the ORIGINAL cumsum, not the drift-corrected cumsum). This matches the C++ behavior exactly.

## Troubleshooting

### Common Issues

1. **~60x magnitude error**: Wrong SAMPLES_PER_TURN value. Try 512 for BTP8 format.
2. **Schema mismatch**: Check column naming conventions (Bn vs B_n vs Bn(T))
3. **Large errors**: Verify kn file matches compensation scheme
4. **Missing data**: Ensure flux files are in ASCII format
5. **Import errors**: Run from repo root with package installed
6. **Sign flips on odd harmonics**: Check rotation angle convention

### Debug Mode

Enable verbose logging in the notebook:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Key Formulas

The following formulas are implemented to match the legacy ffmm C++ analyzer exactly.

### Drift Correction (legacy mode)
From `ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp` line ~127:
```cpp
fluxAbs = cumsum(df_abs - mean(df_abs)) - mean(cumsum(df_abs));
```

### FFT Normalization
```
f_n = 2 * FFT(flux)[n] / N_samples
```
The factor of 2 accounts for the two-sided FFT spectrum.

### Kn Calibration
From `MatlabAnalyzerRotCoil.cpp` lines 149-156:
```cpp
c_sens_abs[ki] = (1.0 / conj(knAbs[ki])) * pow(Rref, ki);
C_abs[ki] = c_sens_abs[ki] * f_abs[ki];
```
Where `ki` is the harmonic index (starting from 0 for n=1).

### Rotation
```
C_n_rotated = C_n * exp(-i * (n - m) * angle_m)
```
Where:
- `m` is the magnet order (2 for quadrupole)
- `angle_m = arg(C_m)` is the phase of the main harmonic

### Harmonic Merge Strategy
- For n ≤ m: Use absolute channel (ABS)
- For n > m: Use compensated channel (CMP)

This is the "abs_upto_m_cmp_above" merge mode.

## References

- Legacy C++ analyzer: `ffmm/src/core/utils/matlab_analyzer/MatlabAnalyzerRotCoil.cpp`
- Bottura PDF: CERN rotating coil measurement standards
- Measurement head geometry CSV format specification
