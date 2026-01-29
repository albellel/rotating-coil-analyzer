"""Test that compensation scheme is NOT inferable from MH CSV and must be specified externally.

The MH CSV contains only coil geometry data (radius, angles, turns, magnetic surface, etc.).
It does NOT contain compensation scheme metadata (e.g., "A-C", "ABCD").
The user must specify the compensation scheme explicitly.
"""

import pytest
from pathlib import Path

# Find the golden standard MH CSV file
MH_CSV_PATH = Path(__file__).parent.parent.parent.parent / "golden_standards" / "measurement_heads" / "CRMMMMH_AF-00000001.csv"


@pytest.mark.skipif(not MH_CSV_PATH.exists(), reason="Golden standard MH CSV not available")
def test_mh_csv_does_not_contain_compensation_scheme():
    """Verify that the MH CSV has no compensation scheme field."""
    import pandas as pd

    df = pd.read_csv(MH_CSV_PATH, dtype=str)

    # Check that none of the column names suggest compensation scheme
    columns_lower = [c.lower() for c in df.columns]
    scheme_keywords = ["compensation", "scheme", "bucking", "wiring"]

    for col in columns_lower:
        for kw in scheme_keywords:
            assert kw not in col, (
                f"Unexpected column '{col}' suggests compensation scheme might be in CSV. "
                "If this is true, update the inference logic."
            )


@pytest.mark.skipif(not MH_CSV_PATH.exists(), reason="Golden standard MH CSV not available")
def test_head_kn_data_does_not_include_compensation_scheme():
    """Verify that HeadKnData from MH CSV does not expose compensation scheme."""
    from rotating_coil_analyzer.analysis.kn_head import compute_head_kn_from_csv

    head_kn = compute_head_kn_from_csv(str(MH_CSV_PATH), n_multipoles=5)

    # HeadKnData should not have a compensation_scheme attribute
    assert not hasattr(head_kn, "compensation_scheme"), (
        "HeadKnData should not have compensation_scheme - it's not in the CSV"
    )


def test_kn_bundle_can_store_compensation_scheme_in_extra():
    """Verify KnBundle can store compensation_scheme via the extra dict."""
    import numpy as np
    from rotating_coil_analyzer.analysis.kn_pipeline import SegmentKn
    from rotating_coil_analyzer.analysis.kn_bundle import KnBundle

    # Create a minimal SegmentKn
    orders = np.arange(1, 6)
    kn = SegmentKn(
        orders=orders,
        kn_abs=np.ones(5, dtype=complex),
        kn_cmp=np.ones(5, dtype=complex),
        kn_ext=None,
        source_path="test",
    )

    # Create KnBundle with compensation_scheme in extra
    bundle = KnBundle(
        kn=kn,
        source_type="head_csv",
        source_path="test.csv",
        timestamp=KnBundle.now_iso(),
        extra={"compensation_scheme": "A-C"},
    )

    # Verify it's stored and exported
    meta = bundle.to_metadata_dict()
    assert "kn_extra_compensation_scheme" in meta
    assert meta["kn_extra_compensation_scheme"] == "A-C"


def test_kn_bundle_extra_is_optional():
    """Verify KnBundle works without extra (but compensation_scheme should be documented)."""
    import numpy as np
    from rotating_coil_analyzer.analysis.kn_pipeline import SegmentKn
    from rotating_coil_analyzer.analysis.kn_bundle import KnBundle

    orders = np.arange(1, 6)
    kn = SegmentKn(
        orders=orders,
        kn_abs=np.ones(5, dtype=complex),
        kn_cmp=np.ones(5, dtype=complex),
        kn_ext=None,
        source_path="test",
    )

    # Without extra, should still work
    bundle = KnBundle(
        kn=kn,
        source_type="segment_txt",
        source_path="test.txt",
        timestamp=KnBundle.now_iso(),
    )

    meta = bundle.to_metadata_dict()
    # No kn_extra_ keys should be present
    extra_keys = [k for k in meta.keys() if k.startswith("kn_extra_")]
    assert len(extra_keys) == 0
