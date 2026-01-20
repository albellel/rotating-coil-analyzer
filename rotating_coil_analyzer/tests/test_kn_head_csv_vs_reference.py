from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rotating_coil_analyzer.analysis.kn_head import compute_head_kn_from_csv


def test_head_csv_kn_matches_reference_file() -> None:
    """Validate geometry-based $k_n$ against a legacy reference CSV.

    Reference file is exported by the legacy analyzer for the same measurement head.
    This test is designed to catch convention mismatches early (radius choice,
    warm/cold scaling, alpha handling, etc.).
    """

    data_dir = Path(__file__).parent / "data"
    head_csv = data_dir / "CRMMMMH_AT-00000001.csv"
    ref_kn = data_dir / "CRMMMMH_AT-00000001_kn.csv"

    # This reference file was generated with the "cold" geometry scaling.
    # (The head CSV stores nominal geometry; the legacy analyzer applies a small
    # scaling depending on whether you compute warm or cold calibration.)
    head = compute_head_kn_from_csv(
        head_csv,
        warm_geometry=False,
        n_multipoles=15,
        use_design_radius=True,
        strict_header=True,
    )

    ref = pd.read_csv(ref_kn, sep=";", engine="python")

    # Build reference complex arrays per (array, coil)
    for _, row in ref.iterrows():
        a = int(row["array"])
        c = int(row["coil"])

        re = np.array([float(row[f"real_{k}"]) for k in range(1, 16)], dtype=float)
        im = np.array([float(row[f"imag_{k}"]) for k in range(1, 16)], dtype=float)
        ref_kn_vec = re + 1j * im

        got = head.kn_by_index.get((a, c), None)
        assert got is not None, f"Missing coil ({a},{c}) from computed head k_n"

        # The legacy output is usually float64; tight tolerance is appropriate.
        # Keep a slightly relaxed atol to avoid platform-dependent tiny differences.
        np.testing.assert_allclose(got, ref_kn_vec, rtol=2e-12, atol=2e-12)
