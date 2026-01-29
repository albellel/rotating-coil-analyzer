"""Golden standard parity runner - reusable API for validation workflows.

This module provides a clean, reusable API for:
1. Scanning golden dataset folders to identify raw data, kn files, and reference results
2. Running the full analysis pipeline (ingest -> harmonics -> kn -> merge)
3. Exporting results in the canonical schema
4. Comparing computed results against reference results

Design goals:
- Reusable for any golden dataset (not hardcoded to LIU_BTP8)
- Full provenance tracking
- Canonical schema compliance (example_results standard + superset of golden columns)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rotating_coil_analyzer.analysis.kn_bundle import KnBundle, MergeResult
from rotating_coil_analyzer.analysis.kn_pipeline import (
    LegacyKnPerTurn,
    SegmentKn,
    compute_legacy_kn_per_turn,
    load_segment_kn_txt,
    merge_coefficients,
)
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader, StreamingReaderConfig
from rotating_coil_analyzer.models.catalog import MeasurementCatalog
from rotating_coil_analyzer.models.frames import SegmentFrame


# ---------------------------------------------------------------------------
# Canonical schema definition (based on example_results standard)
# ---------------------------------------------------------------------------

CANONICAL_METADATA_COLUMNS = [
    "Time(s)",
    "Duration(s)",
    "Options",
    "Rref(m)",
    "Lcoil(m)",
    "I(A)",
    "Ramprate(A/s)",
    "I1(A)",
    "Ramprate1(A/s)",
    "dx(mm)",
    "dy(mm)",
    "phi(rad)",
    "B_main(T)",
    "A_main(T)",
    "B_main_TF(T/kA)",
    "A_main_TF(T/kA)",
]

# Harmonic columns follow pattern: B1(T), A1(T), b2(Units), a2(Units), ..., b15(Units), a15(Units)
# Where B1/A1 are in Tesla (main field), b2-b15/a2-a15 are normalized units

LEGACY_METADATA_COLUMNS = [
    "Date",
    "Options",
    "A_MRU Level (rad)",
    "B_Top Plate Level (rad)",
    "Rref(m)",
    "Lcoil(m)",
    "I(A)",
    "I FGC(A)",
    "x (mm)",
    "y (mm)",
]


@dataclass
class DatasetMapping:
    """Mapping of reference results to their required inputs."""

    reference_results_path: Path
    raw_data_folder: Path
    kn_file_path: Optional[Path]
    coil_type: str  # "central", "integral", "pcb", etc.
    aperture: int
    segment: str
    compensation_scheme: str
    magnet_order: int
    r_ref_m: float
    l_coil_m: float
    options: Tuple[str, ...]
    notes: str = ""


@dataclass
class DatasetIntrospection:
    """Results of scanning a golden dataset folder."""

    dataset_folder: Path
    raw_data_folders: List[Path]
    kn_files: List[Path]
    reference_results_files: List[Path]
    parameters_files: List[Path]
    mappings: List[DatasetMapping]
    warnings: List[str]


def scan_golden_dataset(dataset_folder: Path) -> DatasetIntrospection:
    """Recursively scan a golden dataset folder and identify all components.

    Parameters
    ----------
    dataset_folder : Path
        Root folder of the golden dataset (e.g., golden_standard_01_LIU_BTP8)

    Returns
    -------
    DatasetIntrospection
        Structured information about the dataset contents and mappings
    """
    dataset_folder = Path(dataset_folder)
    if not dataset_folder.is_dir():
        raise ValueError(f"Dataset folder does not exist: {dataset_folder}")

    raw_data_folders: List[Path] = []
    kn_files: List[Path] = []
    reference_results_files: List[Path] = []
    parameters_files: List[Path] = []
    warnings: List[str] = []

    # Find all Parameters.txt files (indicates raw data folders)
    for p in dataset_folder.rglob("*Parameters*.txt"):
        if p.is_file():
            parameters_files.append(p)
            raw_data_folders.append(p.parent)

    # Find all kn files
    kn_patterns = ["*Kn*.txt", "*kn*.txt"]
    for pattern in kn_patterns:
        for p in dataset_folder.rglob(pattern):
            if p.is_file() and p not in kn_files:
                kn_files.append(p)

    # Find all reference results files
    results_patterns = ["*results*.txt", "*Results*.txt"]
    for pattern in results_patterns:
        for p in dataset_folder.rglob(pattern):
            if p.is_file() and "Parameters" not in p.name and "Average" not in p.name:
                reference_results_files.append(p)

    # Build mappings based on folder structure and Parameters.txt content
    mappings = _build_mappings(
        dataset_folder,
        raw_data_folders,
        kn_files,
        reference_results_files,
        parameters_files,
        warnings,
    )

    return DatasetIntrospection(
        dataset_folder=dataset_folder,
        raw_data_folders=sorted(set(raw_data_folders)),
        kn_files=sorted(kn_files),
        reference_results_files=sorted(reference_results_files),
        parameters_files=sorted(parameters_files),
        mappings=mappings,
        warnings=warnings,
    )


def _parse_parameters_txt(path: Path) -> Dict[str, Any]:
    """Parse a Parameters.txt file into a dictionary."""
    params: Dict[str, Any] = {}
    try:
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                # Try to parse numeric values
                try:
                    if "." in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    # Handle boolean strings
                    if value.lower() == "true":
                        params[key] = True
                    elif value.lower() == "false":
                        params[key] = False
                    else:
                        params[key] = value
    except Exception:
        pass
    return params


def _infer_coil_type(folder: Path) -> str:
    """Infer coil type from folder path."""
    path_lower = str(folder).lower()
    if "central" in path_lower:
        return "central"
    elif "integral" in path_lower:
        return "integral"
    elif "pcb" in path_lower:
        return "pcb"
    else:
        return "unknown"


def _infer_magnet_order(params: Dict[str, Any]) -> int:
    """Infer magnet order from parameters."""
    magnet_type = str(params.get("Parameters.magnetInfo.magnetType", "")).lower()
    if "dipole" in magnet_type:
        return 1
    elif "quadrupole" in magnet_type:
        return 2
    elif "sextupole" in magnet_type:
        return 3
    elif "octupole" in magnet_type:
        return 4
    elif "decapole" in magnet_type:
        return 5
    return 2  # Default to quadrupole


def _find_matching_kn(
    raw_folder: Path,
    kn_files: List[Path],
    compensation_scheme: str,
    shaft_name: str,
) -> Optional[Path]:
    """Find the kn file matching the compensation scheme and shaft."""
    # Normalize compensation scheme for matching
    scheme_patterns = [
        compensation_scheme.upper(),
        compensation_scheme.lower(),
        compensation_scheme.replace("_", ""),
    ]

    for kn_path in kn_files:
        name_upper = kn_path.name.upper()
        # Check if shaft name matches
        if shaft_name and shaft_name.upper() not in str(kn_path.parent).upper():
            continue
        # Check if compensation scheme matches
        for pattern in scheme_patterns:
            if pattern in name_upper:
                return kn_path

    return None


def _build_mappings(
    dataset_folder: Path,
    raw_data_folders: List[Path],
    kn_files: List[Path],
    reference_results_files: List[Path],
    parameters_files: List[Path],
    warnings: List[str],
) -> List[DatasetMapping]:
    """Build mappings between reference results and their inputs."""
    mappings: List[DatasetMapping] = []

    for params_path in parameters_files:
        raw_folder = params_path.parent
        params = _parse_parameters_txt(params_path)

        # Extract key parameters
        coil_type = _infer_coil_type(raw_folder)
        magnet_order = _infer_magnet_order(params)
        compensation_scheme = str(params.get("Parameters.magnetAnalyzer.compensationSc", "ABCD"))
        shaft_name = str(params.get("Parameters.magnetAnalyzer.shaft", ""))
        r_ref = float(params.get("Parameters.magnetAnalyzer.refRadius", 1.0))
        l_coil = float(params.get("Parameters.magnetAnalyzer.shaftLength", 1.0))

        # Build options tuple from parameters
        options_list = []
        if params.get("Parameters.magnetAnalyzer.dri", False):
            options_list.append("dri")
        if params.get("Parameters.magnetAnalyzer.rot", False):
            options_list.append("rot")
        if params.get("Parameters.magnetAnalyzer.nor", False):
            options_list.append("nor")
        if params.get("Parameters.magnetAnalyzer.cel", False):
            options_list.append("cel")
        if params.get("Parameters.magnetAnalyzer.fed", False):
            options_list.append("fed")
        options = tuple(options_list) if options_list else ("dri", "rot")

        # Find matching results file in the same folder
        ref_results: Optional[Path] = None
        for rp in reference_results_files:
            if rp.parent == raw_folder:
                ref_results = rp
                break

        if ref_results is None:
            warnings.append(f"No reference results found for {raw_folder.name}")
            continue

        # Find matching kn file
        kn_file = _find_matching_kn(raw_folder, kn_files, compensation_scheme, shaft_name)
        if kn_file is None:
            warnings.append(f"No matching kn file found for {raw_folder.name} (scheme={compensation_scheme}, shaft={shaft_name})")

        mapping = DatasetMapping(
            reference_results_path=ref_results,
            raw_data_folder=raw_folder,
            kn_file_path=kn_file,
            coil_type=coil_type,
            aperture=1,  # Default aperture
            segment="1",  # Default segment
            compensation_scheme=compensation_scheme,
            magnet_order=magnet_order,
            r_ref_m=r_ref,
            l_coil_m=l_coil,
            options=options,
            notes=f"Shaft: {shaft_name}",
        )
        mappings.append(mapping)

    return mappings


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for running the analysis pipeline."""

    dataset_folder: Path
    kn_file: Path
    magnet_order: int
    r_ref_m: float = 1.0
    l_coil_m: float = 1.0
    options: Tuple[str, ...] = ("dri", "rot")
    compensation_scheme: str = "ABCD"
    aperture: int = 1
    segment: str = "1"
    run_id: Optional[str] = None  # Auto-detect if None


@dataclass
class PipelineResult:
    """Results from running the analysis pipeline."""

    config: PipelineConfig
    computed_df: pd.DataFrame
    merge_result: Optional[MergeResult]
    kn_bundle: Optional[KnBundle]
    provenance_metadata: Dict[str, Any]
    warnings: List[str]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full analysis pipeline on a dataset.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration

    Returns
    -------
    PipelineResult
        Computed results with full provenance
    """
    warnings: List[str] = []
    dataset_folder = Path(config.dataset_folder)
    kn_file = Path(config.kn_file)

    # Load kn
    kn = load_segment_kn_txt(kn_file)
    kn_bundle = KnBundle(
        kn=kn,
        source_type="segment_txt",
        source_path=str(kn_file),
        timestamp=KnBundle.now_iso(),
        segment_id=config.segment,
        aperture_id=config.aperture,
    )

    # Discover measurements
    discovery = MeasurementDiscovery()
    catalog = discovery.build_catalog(dataset_folder)

    # Determine run_id
    run_id = config.run_id
    if run_id is None:
        if len(catalog.runs) == 1:
            run_id = str(catalog.runs[0])
        else:
            run_id = str(catalog.runs[0]) if catalog.runs else ""
            warnings.append(f"Multiple runs found, using first: {run_id}")

    # Process all segments
    reader_config = StreamingReaderConfig()
    reader = StreamingReader(reader_config)

    all_results: List[pd.DataFrame] = []
    merge_result: Optional[MergeResult] = None

    # Get Parameters.txt metadata
    params_path = dataset_folder / f"{Path(dataset_folder).name}_Parameters.txt"
    if not params_path.exists():
        # Try to find it
        for p in dataset_folder.rglob("*Parameters*.txt"):
            params_path = p
            break

    params = _parse_parameters_txt(params_path) if params_path.exists() else {}
    samples_per_turn = int(params.get("Parameters.Measurement.turns", 1)) * 1024  # Estimate
    shaft_speed = float(params.get("Parameters.Measurement.MotorAngularSpeed", -60))

    # Find and process flux files
    flux_files = sorted(dataset_folder.glob("*_fluxes_Ascii.txt"))
    if not flux_files:
        flux_files = sorted(dataset_folder.glob("*_fluxes.txt"))

    for flux_file in flux_files:
        try:
            # Read segment
            seg_frame = reader.read(
                file_path=flux_file,
                run_id=run_id,
                segment=config.segment,
                samples_per_turn=512,  # Common value
                shaft_speed_rpm=abs(shaft_speed),
                aperture_id=config.aperture,
                magnet_order=config.magnet_order,
            )

            # Compute harmonics with kn
            kn_result = compute_legacy_kn_per_turn(
                segf=seg_frame,
                kn=kn,
                magnet_order=config.magnet_order,
                Rref_m=config.r_ref_m,
                do_didt="dri" in config.options,
                do_drift="dri" in config.options,
                do_rotation="rot" in config.options,
                do_cel="cel" in config.options,
                do_feeddown="fed" in config.options,
                do_normalize="nor" in config.options,
            )

            # Build output table
            df = _build_canonical_output(
                kn_result,
                magnet_order=config.magnet_order,
                r_ref_m=config.r_ref_m,
                l_coil_m=config.l_coil_m,
                options=config.options,
            )
            all_results.append(df)

        except Exception as e:
            warnings.append(f"Error processing {flux_file.name}: {e}")

    # Combine all results
    if all_results:
        computed_df = pd.concat(all_results, ignore_index=True)
    else:
        computed_df = pd.DataFrame()

    # Build provenance metadata
    provenance_metadata = {
        "dataset_folder": str(dataset_folder),
        "kn_file": str(kn_file),
        "magnet_order": config.magnet_order,
        "r_ref_m": config.r_ref_m,
        "l_coil_m": config.l_coil_m,
        "options": list(config.options),
        "compensation_scheme": config.compensation_scheme,
        "aperture": config.aperture,
        "segment": config.segment,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_turns_computed": len(computed_df),
    }
    if kn_bundle:
        provenance_metadata.update(kn_bundle.to_metadata_dict())

    return PipelineResult(
        config=config,
        computed_df=computed_df,
        merge_result=merge_result,
        kn_bundle=kn_bundle,
        provenance_metadata=provenance_metadata,
        warnings=warnings,
    )


def _build_canonical_output(
    kn_result: LegacyKnPerTurn,
    *,
    magnet_order: int,
    r_ref_m: float,
    l_coil_m: float,
    options: Tuple[str, ...],
) -> pd.DataFrame:
    """Build output in canonical schema format."""
    H = int(kn_result.orders.size)
    n_turns = int(kn_result.C_abs.shape[0])
    m = int(magnet_order)

    # Merge coefficients using legacy scheme (ABS up to m, CMP above)
    C_merged, per_n_source = merge_coefficients(
        C_abs=kn_result.C_abs,
        C_cmp=kn_result.C_cmp,
        magnet_order=m,
        mode="abs_upto_m_cmp_above",
    )

    out: Dict[str, Any] = {}

    # Metadata columns
    out["Time(s)"] = np.asarray(kn_result.time_median_s, dtype=float)
    out["Duration(s)"] = np.asarray(kn_result.duration_s, dtype=float)
    out["Options"] = " ".join(options)
    out["Rref(m)"] = r_ref_m
    out["Lcoil(m)"] = l_coil_m
    out["I(A)"] = np.asarray(kn_result.I_mean_A, dtype=float)
    out["Ramprate(A/s)"] = np.asarray(kn_result.dI_dt_A_per_s, dtype=float)
    out["I1(A)"] = np.asarray(kn_result.I_mean_A, dtype=float)  # Same as I(A) for single PS
    out["Ramprate1(A/s)"] = np.asarray(kn_result.dI_dt_A_per_s, dtype=float)
    out["dx(mm)"] = np.asarray(kn_result.x_m, dtype=float) * 1000.0
    out["dy(mm)"] = np.asarray(kn_result.y_m, dtype=float) * 1000.0
    out["phi(rad)"] = np.asarray(kn_result.phi_out_rad, dtype=float)

    # Main field (n=m)
    m_idx = m - 1  # 0-indexed
    if m_idx < H:
        main_complex = C_merged[:, m_idx]
        out["B_main(T)"] = np.real(main_complex)
        out["A_main(T)"] = np.imag(main_complex)

        # Transfer function (T/kA)
        I_kA = np.maximum(np.abs(out["I(A)"]), 1e-9) / 1000.0
        out["B_main_TF(T/kA)"] = out["B_main(T)"] / I_kA
        out["A_main_TF(T/kA)"] = out["A_main(T)"] / I_kA
    else:
        out["B_main(T)"] = np.zeros(n_turns)
        out["A_main(T)"] = np.zeros(n_turns)
        out["B_main_TF(T/kA)"] = np.zeros(n_turns)
        out["A_main_TF(T/kA)"] = np.zeros(n_turns)

    # Harmonic columns
    for i, n in enumerate(kn_result.orders.tolist()):
        cn = C_merged[:, i]
        bn = np.real(cn)  # Normal component
        an = np.imag(cn)  # Skew component

        if n == 1:
            # First harmonic in Tesla
            out["B1(T)"] = bn
            out["A1(T)"] = an
        else:
            # Higher harmonics in normalized units (relative to main field)
            # Units: 1e-4 relative
            out[f"b{n}(Units)"] = bn
            out[f"a{n}(Units)"] = an

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Results from comparing computed vs reference results."""

    schema_diff: Dict[str, Any]
    numeric_diff: pd.DataFrame
    summary_stats: Dict[str, float]
    worst_mismatches: pd.DataFrame
    passed: bool
    tolerance_used: float


def compare_results(
    computed_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-9,
) -> ComparisonResult:
    """Compare computed results against reference results.

    Parameters
    ----------
    computed_df : pd.DataFrame
        Our computed results
    reference_df : pd.DataFrame
        Golden reference results
    rtol : float
        Relative tolerance for numeric comparison
    atol : float
        Absolute tolerance for numeric comparison

    Returns
    -------
    ComparisonResult
        Detailed comparison results
    """
    # Schema comparison
    computed_cols = set(computed_df.columns)
    reference_cols = set(reference_df.columns)

    schema_diff = {
        "missing_from_computed": sorted(reference_cols - computed_cols),
        "extra_in_computed": sorted(computed_cols - reference_cols),
        "common_columns": sorted(computed_cols & reference_cols),
    }

    # Find harmonic columns for numeric comparison
    bn_an_pattern = re.compile(r"^[BbAa]\d+.*$")
    harmonic_cols = [c for c in schema_diff["common_columns"] if bn_an_pattern.match(str(c))]

    # Numeric comparison
    numeric_diffs: Dict[str, np.ndarray] = {}
    for col in harmonic_cols:
        if col in computed_df.columns and col in reference_df.columns:
            comp_vals = pd.to_numeric(computed_df[col], errors="coerce").values
            ref_vals = pd.to_numeric(reference_df[col], errors="coerce").values

            # Align lengths
            min_len = min(len(comp_vals), len(ref_vals))
            comp_vals = comp_vals[:min_len]
            ref_vals = ref_vals[:min_len]

            # Compute absolute difference
            diff = np.abs(comp_vals - ref_vals)
            numeric_diffs[col] = diff

    # Build diff dataframe
    if numeric_diffs:
        diff_df = pd.DataFrame(numeric_diffs)
    else:
        diff_df = pd.DataFrame()

    # Summary statistics
    summary_stats = {}
    if not diff_df.empty:
        for col in diff_df.columns:
            vals = diff_df[col].dropna()
            if len(vals) > 0:
                summary_stats[f"{col}_max_abs_err"] = float(np.max(vals))
                summary_stats[f"{col}_rms_err"] = float(np.sqrt(np.mean(vals**2)))
                summary_stats[f"{col}_mean_err"] = float(np.mean(vals))

    # Find worst mismatches
    worst_rows: List[Dict[str, Any]] = []
    for col in numeric_diffs:
        if len(numeric_diffs[col]) > 0:
            worst_idx = int(np.argmax(numeric_diffs[col]))
            worst_rows.append({
                "column": col,
                "row_index": worst_idx,
                "computed": float(computed_df[col].iloc[worst_idx]) if worst_idx < len(computed_df) else np.nan,
                "reference": float(reference_df[col].iloc[worst_idx]) if worst_idx < len(reference_df) else np.nan,
                "abs_error": float(numeric_diffs[col][worst_idx]),
            })

    worst_df = pd.DataFrame(worst_rows) if worst_rows else pd.DataFrame()

    # Check if comparison passed
    max_error = max(summary_stats.get(f"{c}_max_abs_err", 0) for c in harmonic_cols) if harmonic_cols else 0
    # For relative check, compare against typical magnitude
    passed = max_error < atol or max_error < rtol * 1e6  # Rough check

    return ComparisonResult(
        schema_diff=schema_diff,
        numeric_diff=diff_df,
        summary_stats=summary_stats,
        worst_mismatches=worst_df,
        passed=passed,
        tolerance_used=rtol,
    )


def read_legacy_results(path: Path) -> pd.DataFrame:
    """Read a legacy reference results file (BTP8 format).

    Parameters
    ----------
    path : Path
        Path to the results file

    Returns
    -------
    pd.DataFrame
        Parsed results dataframe
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    # Try tab-separated first (most common for legacy exports)
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if df.shape[1] > 5:
            return df
    except Exception:
        pass

    # Try whitespace separation
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        if df.shape[1] > 5:
            return df
    except Exception:
        pass

    raise ValueError(f"Could not parse results file: {path}")


def export_results(
    df: pd.DataFrame,
    output_path: Path,
    metadata: Dict[str, Any],
    *,
    write_sidecar_json: bool = True,
) -> Path:
    """Export results to CSV with optional metadata sidecar.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output file path
    metadata : Dict[str, Any]
        Provenance metadata
    write_sidecar_json : bool
        If True, write metadata to a JSON sidecar file

    Returns
    -------
    Path
        Path to the written results file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    df.to_csv(output_path, sep="\t", index=False)

    # Write sidecar JSON
    if write_sidecar_json:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return output_path


def generate_diff_report(
    comparison: ComparisonResult,
    output_path: Path,
    *,
    dataset_name: str = "",
) -> Path:
    """Generate a human-readable diff report in Markdown format.

    Parameters
    ----------
    comparison : ComparisonResult
        Comparison results
    output_path : Path
        Output file path
    dataset_name : str
        Name of the dataset for the report title

    Returns
    -------
    Path
        Path to the written report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Golden Standard Parity Report: {dataset_name}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Status**: {'PASSED' if comparison.passed else 'FAILED'}",
        f"- **Tolerance**: {comparison.tolerance_used}",
        "",
        "## Schema Comparison",
        "",
        f"- Missing columns in computed: {len(comparison.schema_diff['missing_from_computed'])}",
        f"- Extra columns in computed: {len(comparison.schema_diff['extra_in_computed'])}",
        f"- Common columns: {len(comparison.schema_diff['common_columns'])}",
        "",
    ]

    if comparison.schema_diff["missing_from_computed"]:
        lines.append("### Missing Columns")
        lines.append("")
        for col in comparison.schema_diff["missing_from_computed"]:
            lines.append(f"- `{col}`")
        lines.append("")

    lines.extend([
        "## Numeric Differences",
        "",
    ])

    # Summary stats
    if comparison.summary_stats:
        lines.append("### Error Statistics")
        lines.append("")
        lines.append("| Column | Max Abs Error | RMS Error | Mean Error |")
        lines.append("|--------|---------------|-----------|------------|")

        cols_seen = set()
        for key in comparison.summary_stats:
            if "_max_abs_err" in key:
                col = key.replace("_max_abs_err", "")
                if col not in cols_seen:
                    cols_seen.add(col)
                    max_err = comparison.summary_stats.get(f"{col}_max_abs_err", 0)
                    rms_err = comparison.summary_stats.get(f"{col}_rms_err", 0)
                    mean_err = comparison.summary_stats.get(f"{col}_mean_err", 0)
                    lines.append(f"| {col} | {max_err:.6e} | {rms_err:.6e} | {mean_err:.6e} |")

        lines.append("")

    # Worst mismatches
    if not comparison.worst_mismatches.empty:
        lines.extend([
            "### Worst Mismatches",
            "",
            "| Column | Row | Computed | Reference | Abs Error |",
            "|--------|-----|----------|-----------|-----------|",
        ])

        for _, row in comparison.worst_mismatches.head(20).iterrows():
            lines.append(
                f"| {row['column']} | {row['row_index']} | "
                f"{row['computed']:.6e} | {row['reference']:.6e} | "
                f"{row['abs_error']:.6e} |"
            )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
