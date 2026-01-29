from __future__ import annotations

"""Unit tests for harmonic merge correctness and per-n selection."""

import numpy as np
import pytest

from rotating_coil_analyzer.analysis import (
    merge_coefficients,
    recommend_merge_choice,
    MergeDiagnostics,
    KnBundle,
    MergeResult,
    SegmentKn,
    CHANNEL_ABS,
    CHANNEL_CMP,
)
from rotating_coil_analyzer.analysis.merge import (
    FLAG_MAIN_FORCED_ABS,
    FLAG_BAD_CHANNEL,
    FLAG_MISMATCH_LARGE,
)


def _make_test_harmonics(n_turns: int = 20, H: int = 8, seed: int = 42):
    """Create test C_abs and C_cmp arrays with known properties."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H))
    # Make cmp slightly noisier than abs for most orders
    C_abs = base + 0.1 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))
    C_cmp = base + 0.3 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))
    return C_abs, C_cmp


class TestMergeCoefficients:
    """Test merge_coefficients function for per-n selection correctness."""

    def test_abs_all_mode(self) -> None:
        """Test that abs_all mode selects all from abs channel."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=5)
        merged, choice = merge_coefficients(C_abs=C_abs, C_cmp=C_cmp, magnet_order=2, mode="abs_all")

        assert choice.shape == (5,)
        assert np.all(choice == 0)  # All abs
        np.testing.assert_array_equal(merged, C_abs)

    def test_cmp_all_mode(self) -> None:
        """Test that cmp_all mode selects all from cmp channel."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=5)
        merged, choice = merge_coefficients(C_abs=C_abs, C_cmp=C_cmp, magnet_order=2, mode="cmp_all")

        assert np.all(choice == 1)  # All cmp
        np.testing.assert_array_equal(merged, C_cmp)

    def test_abs_main_cmp_others_mode(self) -> None:
        """Test that abs_main_cmp_others takes main from abs, rest from cmp."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=5)
        m = 2  # Main order

        merged, choice = merge_coefficients(C_abs=C_abs, C_cmp=C_cmp, magnet_order=m, mode="abs_main_cmp_others")

        # Check choice array
        expected_choice = np.ones(5, dtype=int)  # All cmp
        expected_choice[m - 1] = 0  # Main from abs
        np.testing.assert_array_equal(choice, expected_choice)

        # Check merged values
        np.testing.assert_array_equal(merged[:, m - 1], C_abs[:, m - 1])  # Main from abs
        for j in range(5):
            if j != m - 1:
                np.testing.assert_array_equal(merged[:, j], C_cmp[:, j])  # Others from cmp

    def test_abs_upto_m_cmp_above_mode(self) -> None:
        """Test that abs_upto_m_cmp_above takes 1..m from abs, m+1..H from cmp."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=8)
        m = 3  # Main order

        merged, choice = merge_coefficients(C_abs=C_abs, C_cmp=C_cmp, magnet_order=m, mode="abs_upto_m_cmp_above")

        # Orders 1,2,3 (indices 0,1,2) from abs; orders 4,5,6,7,8 (indices 3,4,5,6,7) from cmp
        expected_choice = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(choice, expected_choice)

        # Check values
        for j in range(m):
            np.testing.assert_array_equal(merged[:, j], C_abs[:, j])
        for j in range(m, 8):
            np.testing.assert_array_equal(merged[:, j], C_cmp[:, j])

    def test_custom_mode_with_explicit_choice(self) -> None:
        """Test custom mode with explicit per-order selection."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=5)
        m = 2

        # Custom pattern: [cmp, abs, cmp, abs, cmp] - but main forced to abs
        custom_choice = [1, 0, 1, 0, 1]

        merged, choice = merge_coefficients(
            C_abs=C_abs, C_cmp=C_cmp, magnet_order=m, mode="custom", per_order_choice=custom_choice
        )

        # Note: main (m=2, index=1) is forced to abs regardless of input
        expected = np.array([1, 0, 1, 0, 1])
        expected[m - 1] = 0  # Main forced to abs
        np.testing.assert_array_equal(choice, expected)

        # Verify values match selections
        for j in range(5):
            if choice[j] == 0:
                np.testing.assert_array_equal(merged[:, j], C_abs[:, j])
            else:
                np.testing.assert_array_equal(merged[:, j], C_cmp[:, j])

    def test_custom_mode_forces_main_to_abs(self) -> None:
        """Test that custom mode always forces main harmonic to abs."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=5)
        m = 3

        # Try to set main to cmp
        custom_choice = [0, 0, 1, 0, 0]  # Trying to set m=3 (index 2) to cmp

        merged, choice = merge_coefficients(
            C_abs=C_abs, C_cmp=C_cmp, magnet_order=m, mode="custom", per_order_choice=custom_choice
        )

        # Main should be forced to abs
        assert choice[m - 1] == 0
        np.testing.assert_array_equal(merged[:, m - 1], C_abs[:, m - 1])

    def test_choice_is_auditable(self) -> None:
        """Test that returned choice array provides full audit trail."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=10, H=6)

        for mode in ["abs_all", "cmp_all", "abs_main_cmp_others", "abs_upto_m_cmp_above"]:
            merged, choice = merge_coefficients(C_abs=C_abs, C_cmp=C_cmp, magnet_order=2, mode=mode)

            # Choice should be reproducible
            assert choice.shape == (6,)
            assert np.all((choice == 0) | (choice == 1))

            # Verify merged matches choice
            for j in range(6):
                if choice[j] == 0:
                    np.testing.assert_array_equal(merged[:, j], C_abs[:, j], err_msg=f"Mode {mode}, order {j+1}")
                else:
                    np.testing.assert_array_equal(merged[:, j], C_cmp[:, j], err_msg=f"Mode {mode}, order {j+1}")


class TestRecommendMergeChoice:
    """Test the merge recommendation engine."""

    def test_recommendation_forces_main_to_abs(self) -> None:
        """Test that recommendation always forces main harmonic to abs."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=20, H=8)
        m = 2

        choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=m)

        assert choice[m - 1] == 0  # Main forced to abs
        assert diag.flags[m - 1] & FLAG_MAIN_FORCED_ABS

    def test_recommendation_returns_diagnostics(self) -> None:
        """Test that recommendation returns full diagnostics."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=20, H=5)

        choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=1)

        assert isinstance(diag, MergeDiagnostics)
        assert diag.orders.shape == (5,)
        assert diag.noise_abs.shape == (5,)
        assert diag.noise_cmp.shape == (5,)
        assert diag.mismatch.shape == (5,)
        assert diag.selected.shape == (5,)
        assert diag.flags.shape == (5,)

    def test_recommendation_prefers_quieter_channel(self) -> None:
        """Test that recommendation prefers channel with lower noise."""
        rng = np.random.default_rng(123)
        n_turns = 50
        H = 5

        # Create data where cmp is clearly quieter for orders 2,3,4,5
        base = 10.0 * np.ones((n_turns, H), dtype=complex)
        C_abs = base + 2.0 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))
        C_cmp = base + 0.1 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))

        choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=1)

        # Main (n=1) forced to abs
        assert choice[0] == 0

        # Others should prefer cmp (significantly quieter)
        for j in range(1, H):
            assert choice[j] == 1, f"Order {j+1} should prefer cmp"

    def test_recommendation_handles_bad_channels(self) -> None:
        """Test that recommendation handles NaN/Inf gracefully."""
        C_abs, C_cmp = _make_test_harmonics(n_turns=20, H=5)

        # Introduce NaN in cmp for order 3
        C_cmp[:, 2] = np.nan

        choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=1)

        # Order 3 (index 2) should fall back to abs and be flagged
        assert choice[2] == 0
        assert diag.flags[2] & FLAG_BAD_CHANNEL


class TestMergeResultRecordsMetadata:
    """Test that MergeResult correctly records all metadata."""

    def test_merge_result_records_compensation_scheme(self) -> None:
        """Test that compensation scheme is recorded in MergeResult."""
        kn = SegmentKn(
            orders=np.arange(1, 6),
            kn_abs=np.ones(5, dtype=complex),
            kn_cmp=np.ones(5, dtype=complex),
            kn_ext=None,
            source_path="test",
        )
        bundle = KnBundle(kn=kn, source_type="segment_txt", source_path="test.txt", timestamp="now")

        result = MergeResult(
            C_merged=np.zeros((10, 5), dtype=complex),
            orders=np.arange(1, 6),
            per_n_source_map=np.array([0, 1, 1, 1, 1]),
            compensation_scheme="A-C",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="abs_main_cmp_others",
            timestamp="now",
        )

        meta = result.to_metadata_dict()
        assert meta["merge_compensation_scheme"] == "A-C"

    def test_merge_result_records_per_n_map(self) -> None:
        """Test that per-n source map is recorded correctly."""
        kn = SegmentKn(
            orders=np.arange(1, 5),
            kn_abs=np.ones(4, dtype=complex),
            kn_cmp=np.ones(4, dtype=complex),
            kn_ext=None,
            source_path="test",
        )
        bundle = KnBundle(kn=kn, source_type="segment_txt", source_path="test.txt", timestamp="now")

        per_n_map = np.array([CHANNEL_ABS, CHANNEL_CMP, CHANNEL_ABS, CHANNEL_CMP])
        result = MergeResult(
            C_merged=np.zeros((10, 4), dtype=complex),
            orders=np.arange(1, 5),
            per_n_source_map=per_n_map,
            compensation_scheme="ABCD",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="custom",
            timestamp="now",
        )

        # Check source_map_as_names
        names = result.source_map_as_names()
        assert names == ["abs", "cmp", "abs", "cmp"]

        # Check metadata dict
        meta = result.to_metadata_dict()
        assert meta["merge_per_n_source_map"] == "abs,cmp,abs,cmp"

    def test_merge_result_includes_kn_provenance(self) -> None:
        """Test that MergeResult includes full kn provenance."""
        kn = SegmentKn(
            orders=np.arange(1, 4),
            kn_abs=np.ones(3, dtype=complex),
            kn_cmp=np.ones(3, dtype=complex),
            kn_ext=None,
            source_path="test",
        )
        bundle = KnBundle(
            kn=kn,
            source_type="head_csv",
            source_path="/data/head.csv",
            timestamp="2024-01-15T10:00:00Z",
            head_abs_connection="1.2",
            head_cmp_connection="1.1-1.3",
            head_warm_geometry=True,
        )

        result = MergeResult(
            C_merged=np.zeros((10, 3), dtype=complex),
            orders=np.arange(1, 4),
            per_n_source_map=np.array([0, 1, 1]),
            compensation_scheme="A-C",
            magnet_order=1,
            kn_provenance=bundle,
            merge_mode="abs_main_cmp_others",
            timestamp="now",
        )

        meta = result.to_metadata_dict()

        # Check kn provenance is included
        assert meta["kn_source_type"] == "head_csv"
        assert meta["kn_source_path"] == "/data/head.csv"
        assert meta["kn_head_abs_connection"] == "1.2"
        assert meta["kn_head_cmp_connection"] == "1.1-1.3"
        assert meta["kn_head_warm_geometry"] is True


class TestMergeEndToEnd:
    """End-to-end tests combining merge_coefficients with MergeResult."""

    def test_full_merge_workflow(self) -> None:
        """Test complete merge workflow from harmonics to MergeResult."""
        # Setup
        C_abs, C_cmp = _make_test_harmonics(n_turns=15, H=6)
        m = 2

        # Get recommendation
        rec_choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=m)

        # Apply merge with recommendation
        merged, applied_choice = merge_coefficients(
            C_abs=C_abs, C_cmp=C_cmp, magnet_order=m, mode="custom", per_order_choice=rec_choice.tolist()
        )

        # Create KnBundle (simulated)
        kn = SegmentKn(
            orders=np.arange(1, 7),
            kn_abs=np.ones(6, dtype=complex),
            kn_cmp=np.ones(6, dtype=complex),
            kn_ext=None,
            source_path="test.txt",
        )
        bundle = KnBundle(
            kn=kn, source_type="segment_txt", source_path="test.txt", timestamp=KnBundle.now_iso()
        )

        # Create MergeResult
        result = MergeResult(
            C_merged=merged,
            orders=np.arange(1, 7),
            per_n_source_map=applied_choice,
            compensation_scheme="A-C",
            magnet_order=m,
            kn_provenance=bundle,
            merge_mode="recommended",
            timestamp=KnBundle.now_iso(),
            diagnostics=diag,
            C_abs=C_abs,
            C_cmp=C_cmp,
        )

        # Verify
        assert result.n_turns == 15
        assert result.H == 6
        assert result.C_abs is not None
        assert result.C_cmp is not None
        assert result.diagnostics is not None

        # Main should be from abs
        np.testing.assert_array_equal(result.C_merged[:, m - 1], C_abs[:, m - 1])

        # Provenance should be complete
        meta = result.to_metadata_dict()
        assert "kn_source_type" in meta
        assert "merge_compensation_scheme" in meta
        assert "merge_per_n_source_map" in meta
