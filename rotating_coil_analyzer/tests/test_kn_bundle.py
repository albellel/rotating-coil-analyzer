from __future__ import annotations

"""Unit tests for KnBundle and kn export/import round-trip."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rotating_coil_analyzer.analysis import (
    SegmentKn,
    load_segment_kn_txt,
    write_segment_kn_txt,
    KnBundle,
    MergeResult,
    CHANNEL_ABS,
    CHANNEL_CMP,
    CHANNEL_NAMES,
)


def _make_test_segment_kn(H: int = 10, with_ext: bool = False) -> SegmentKn:
    """Create a test SegmentKn with known values."""
    orders = np.arange(1, H + 1, dtype=int)
    # Use non-trivial complex values for testing round-trip precision
    kn_abs = np.array([complex(0.1 * n, 0.01 * n) for n in orders])
    kn_cmp = np.array([complex(0.2 * n, 0.02 * n) for n in orders])
    kn_ext = np.array([complex(0.3 * n, 0.03 * n) for n in orders]) if with_ext else None
    return SegmentKn(orders=orders, kn_abs=kn_abs, kn_cmp=kn_cmp, kn_ext=kn_ext, source_path="test")


class TestKnExportImportRoundTrip:
    """Test that kn can be exported to TXT and imported back identically."""

    def test_4_column_roundtrip(self) -> None:
        """Test export/import round-trip for 4-column (no ext) kn file."""
        original = _make_test_segment_kn(H=15, with_ext=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kn_test.txt"
            write_segment_kn_txt(original, str(path))

            # Verify file was created
            assert path.exists()

            # Read it back
            loaded = load_segment_kn_txt(str(path))

        # Check all fields match
        assert loaded.orders.shape == original.orders.shape
        np.testing.assert_array_equal(loaded.orders, original.orders)
        np.testing.assert_array_almost_equal(loaded.kn_abs, original.kn_abs, decimal=12)
        np.testing.assert_array_almost_equal(loaded.kn_cmp, original.kn_cmp, decimal=12)
        assert loaded.kn_ext is None

    def test_6_column_roundtrip(self) -> None:
        """Test export/import round-trip for 6-column (with ext) kn file."""
        original = _make_test_segment_kn(H=15, with_ext=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kn_test_ext.txt"
            write_segment_kn_txt(original, str(path))

            loaded = load_segment_kn_txt(str(path))

        np.testing.assert_array_almost_equal(loaded.kn_abs, original.kn_abs, decimal=12)
        np.testing.assert_array_almost_equal(loaded.kn_cmp, original.kn_cmp, decimal=12)
        assert loaded.kn_ext is not None
        np.testing.assert_array_almost_equal(loaded.kn_ext, original.kn_ext, decimal=12)

    def test_roundtrip_preserves_precision(self) -> None:
        """Test that round-trip preserves full double precision."""
        # Use values that require full precision
        H = 5
        orders = np.arange(1, H + 1, dtype=int)
        kn_abs = np.array([1.23456789012345e-10 + 1j * 9.87654321098765e-11 for _ in orders])
        kn_cmp = np.array([5.55555555555555e-5 + 1j * 4.44444444444444e-6 for _ in orders])
        original = SegmentKn(orders=orders, kn_abs=kn_abs, kn_cmp=kn_cmp, kn_ext=None, source_path="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kn_precision.txt"
            write_segment_kn_txt(original, str(path))
            loaded = load_segment_kn_txt(str(path))

        # Check precision to 15 significant figures
        np.testing.assert_array_almost_equal(loaded.kn_abs, original.kn_abs, decimal=15)
        np.testing.assert_array_almost_equal(loaded.kn_cmp, original.kn_cmp, decimal=15)


class TestKnBundleProvenance:
    """Test KnBundle metadata and provenance tracking."""

    def test_kn_bundle_creation_segment_txt(self) -> None:
        """Test creating KnBundle from segment TXT source."""
        kn = _make_test_segment_kn(H=10)
        bundle = KnBundle(
            kn=kn,
            source_type="segment_txt",
            source_path="/path/to/kn.txt",
            timestamp=KnBundle.now_iso(),
            segment_id="Main",
            aperture_id=1,
        )

        assert bundle.source_type == "segment_txt"
        assert bundle.source_path == "/path/to/kn.txt"
        assert bundle.segment_id == "Main"
        assert bundle.aperture_id == 1
        assert bundle.head_abs_connection is None  # Not applicable for segment_txt

    def test_kn_bundle_creation_head_csv(self) -> None:
        """Test creating KnBundle from head CSV source with full provenance."""
        kn = _make_test_segment_kn(H=15)
        bundle = KnBundle(
            kn=kn,
            source_type="head_csv",
            source_path="/path/to/head.csv",
            timestamp=KnBundle.now_iso(),
            segment_id="A",
            aperture_id=2,
            head_abs_connection="1.2",
            head_cmp_connection="1.1-1.3",
            head_ext_connection=None,
            head_warm_geometry=True,
            head_n_multipoles=15,
        )

        assert bundle.source_type == "head_csv"
        assert bundle.head_abs_connection == "1.2"
        assert bundle.head_cmp_connection == "1.1-1.3"
        assert bundle.head_warm_geometry is True
        assert bundle.head_n_multipoles == 15

    def test_kn_bundle_to_metadata_dict(self) -> None:
        """Test exporting KnBundle provenance as dictionary."""
        kn = _make_test_segment_kn(H=10)
        bundle = KnBundle(
            kn=kn,
            source_type="head_csv",
            source_path="/data/head.csv",
            timestamp="2024-01-15T10:30:00Z",
            segment_id="Main",
            aperture_id=1,
            head_abs_connection="1.2",
            head_cmp_connection="1.1-1.3",
            head_warm_geometry=True,
            head_n_multipoles=15,
        )

        meta = bundle.to_metadata_dict()

        assert meta["kn_source_type"] == "head_csv"
        assert meta["kn_source_path"] == "/data/head.csv"
        assert meta["kn_timestamp"] == "2024-01-15T10:30:00Z"
        assert meta["kn_n_harmonics"] == 10
        assert meta["kn_segment_id"] == "Main"
        assert meta["kn_aperture_id"] == 1
        assert meta["kn_head_abs_connection"] == "1.2"
        assert meta["kn_head_cmp_connection"] == "1.1-1.3"
        assert meta["kn_head_warm_geometry"] is True
        assert meta["kn_head_n_multipoles"] == 15

    def test_timestamp_format(self) -> None:
        """Test that timestamp is valid ISO-8601."""
        ts = KnBundle.now_iso()
        assert ts.endswith("Z")
        assert "T" in ts
        # Should be parseable
        from datetime import datetime
        datetime.fromisoformat(ts.replace("Z", "+00:00"))


class TestMergeResultProvenance:
    """Test MergeResult metadata and provenance tracking."""

    def test_merge_result_creation(self) -> None:
        """Test creating MergeResult with full provenance."""
        kn = _make_test_segment_kn(H=5)
        bundle = KnBundle(
            kn=kn,
            source_type="segment_txt",
            source_path="/path/to/kn.txt",
            timestamp=KnBundle.now_iso(),
        )

        n_turns = 10
        H = 5
        C_merged = np.random.randn(n_turns, H) + 1j * np.random.randn(n_turns, H)
        orders = np.arange(1, H + 1)
        per_n_map = np.array([CHANNEL_ABS, CHANNEL_CMP, CHANNEL_CMP, CHANNEL_CMP, CHANNEL_CMP])

        result = MergeResult(
            C_merged=C_merged,
            orders=orders,
            per_n_source_map=per_n_map,
            compensation_scheme="A-C",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="abs_main_cmp_others",
            timestamp=KnBundle.now_iso(),
        )

        assert result.n_turns == n_turns
        assert result.H == H
        assert result.compensation_scheme == "A-C"
        assert result.magnet_order == 1
        assert result.merge_mode == "abs_main_cmp_others"

    def test_source_map_as_names(self) -> None:
        """Test converting per-n source map to channel names."""
        kn = _make_test_segment_kn(H=4)
        bundle = KnBundle(kn=kn, source_type="segment_txt", source_path="test", timestamp="now")

        per_n_map = np.array([CHANNEL_ABS, CHANNEL_CMP, CHANNEL_ABS, CHANNEL_CMP])
        result = MergeResult(
            C_merged=np.zeros((5, 4), dtype=complex),
            orders=np.array([1, 2, 3, 4]),
            per_n_source_map=per_n_map,
            compensation_scheme="test",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="custom",
            timestamp="now",
        )

        names = result.source_map_as_names()
        assert names == ["abs", "cmp", "abs", "cmp"]

    def test_merge_result_to_metadata_dict(self) -> None:
        """Test exporting MergeResult provenance including kn provenance."""
        kn = _make_test_segment_kn(H=3)
        bundle = KnBundle(
            kn=kn,
            source_type="head_csv",
            source_path="/data/head.csv",
            timestamp="2024-01-15T10:00:00Z",
            head_abs_connection="1.2",
            head_cmp_connection="1.1-1.3",
        )

        per_n_map = np.array([CHANNEL_ABS, CHANNEL_CMP, CHANNEL_CMP])
        result = MergeResult(
            C_merged=np.zeros((5, 3), dtype=complex),
            orders=np.array([1, 2, 3]),
            per_n_source_map=per_n_map,
            compensation_scheme="A-C",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="abs_main_cmp_others",
            timestamp="2024-01-15T11:00:00Z",
        )

        meta = result.to_metadata_dict()

        # Check merge-specific fields
        assert meta["merge_timestamp"] == "2024-01-15T11:00:00Z"
        assert meta["merge_mode"] == "abs_main_cmp_others"
        assert meta["merge_compensation_scheme"] == "A-C"
        assert meta["merge_magnet_order"] == 1
        assert meta["merge_n_turns"] == 5
        assert meta["merge_n_harmonics"] == 3
        assert meta["merge_per_n_source_map"] == "abs,cmp,cmp"

        # Check kn provenance is included
        assert meta["kn_source_type"] == "head_csv"
        assert meta["kn_source_path"] == "/data/head.csv"
        assert meta["kn_head_abs_connection"] == "1.2"
        assert meta["kn_head_cmp_connection"] == "1.1-1.3"

    def test_source_map_dataframe(self) -> None:
        """Test converting source map to DataFrame for export."""
        kn = _make_test_segment_kn(H=3)
        bundle = KnBundle(kn=kn, source_type="segment_txt", source_path="test", timestamp="now")

        per_n_map = np.array([CHANNEL_ABS, CHANNEL_CMP, CHANNEL_ABS])
        result = MergeResult(
            C_merged=np.zeros((5, 3), dtype=complex),
            orders=np.array([1, 2, 3]),
            per_n_source_map=per_n_map,
            compensation_scheme="test",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="custom",
            timestamp="now",
        )

        df = result.source_map_dataframe()
        assert list(df.columns) == ["n", "source", "source_code"]
        assert df["n"].tolist() == [1, 2, 3]
        assert df["source"].tolist() == ["abs", "cmp", "abs"]
        assert df["source_code"].tolist() == [0, 1, 0]
