"""Tests for Priority 3 shared channel detection module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rotating_coil_analyzer.ingest.channel_detect import (
    ColumnMapping,
    robust_range,
    detect_flux_channels,
    detect_current_channel,
    validate_channel_assignment,
)


# -----------------------------------------------------------------------
# robust_range
# -----------------------------------------------------------------------


def test_robust_range_basic() -> None:
    x = np.arange(1001, dtype=float)  # 0..1000
    r = robust_range(x)
    # p99.5 ~ 995, p0.5 ~ 5 -> range ~ 990
    assert 985 < r < 995


def test_robust_range_empty() -> None:
    assert np.isnan(robust_range(np.array([])))


def test_robust_range_constant() -> None:
    assert robust_range(np.ones(100)) == 0.0


# -----------------------------------------------------------------------
# detect_flux_channels -- auto
# -----------------------------------------------------------------------


def test_detect_auto_no_swap() -> None:
    """col1 has larger range than col2 -> no swap."""
    mat = np.zeros((1000, 5))
    mat[:, 1] = np.sin(np.linspace(0, 10, 1000))  # range ~ 2
    mat[:, 2] = np.sin(np.linspace(0, 10, 1000)) * 0.01  # range ~ 0.02

    df_abs, df_cmp, abs_col, cmp_col, w = detect_flux_channels(mat)
    assert abs_col == 1
    assert cmp_col == 2
    assert not any("swapped" in s for s in w)


def test_detect_auto_swaps_when_col2_larger() -> None:
    """col2 has larger range than col1 -> swap."""
    mat = np.zeros((1000, 5))
    mat[:, 1] = np.sin(np.linspace(0, 10, 1000)) * 0.01  # small
    mat[:, 2] = np.sin(np.linspace(0, 10, 1000))  # large

    df_abs, df_cmp, abs_col, cmp_col, w = detect_flux_channels(mat)
    assert abs_col == 2
    assert cmp_col == 1
    assert any("swapped" in s for s in w)


# -----------------------------------------------------------------------
# detect_flux_channels -- explicit mapping
# -----------------------------------------------------------------------


def test_detect_explicit_mapping() -> None:
    mat = np.zeros((100, 5))
    mat[:, 1] = 1.0
    mat[:, 2] = 2.0
    mat[:, 3] = 3.0

    mapping = ColumnMapping(flux_abs_col=3, flux_cmp_col=1)
    df_abs, df_cmp, abs_col, cmp_col, w = detect_flux_channels(mat, mapping=mapping)

    assert abs_col == 3
    assert cmp_col == 1
    np.testing.assert_array_equal(df_abs, mat[:, 3])
    np.testing.assert_array_equal(df_cmp, mat[:, 1])
    assert any("explicit" in s for s in w)


# -----------------------------------------------------------------------
# detect_current_channel
# -----------------------------------------------------------------------


def test_detect_current_auto() -> None:
    mat = np.zeros((1000, 6))
    mat[:, 3] = np.linspace(0, 10, 1000)  # range 10
    mat[:, 4] = np.linspace(0, 100, 1000)  # range 100 -> winner
    mat[:, 5] = np.linspace(0, 1, 1000)  # range 1

    I_main, col_idx, w = detect_current_channel(mat, start_col=3)
    assert col_idx == 4
    np.testing.assert_array_equal(I_main, mat[:, 4])


def test_detect_current_explicit() -> None:
    mat = np.zeros((100, 6))
    mat[:, 5] = 42.0

    mapping = ColumnMapping(current_col=5)
    I_main, col_idx, w = detect_current_channel(mat, mapping=mapping)
    assert col_idx == 5
    np.testing.assert_array_equal(I_main, mat[:, 5])


def test_detect_current_no_columns() -> None:
    mat = np.zeros((100, 3))
    I_main, col_idx, w = detect_current_channel(mat, start_col=3)
    assert col_idx == -1
    assert np.all(np.isnan(I_main))


# -----------------------------------------------------------------------
# validate_channel_assignment
# -----------------------------------------------------------------------


def test_validate_warns_cmp_exceeds_abs() -> None:
    abs_data = np.sin(np.linspace(0, 10, 1000)) * 0.01
    cmp_data = np.sin(np.linspace(0, 10, 1000)) * 1.0

    w = validate_channel_assignment(abs_data, cmp_data)
    assert any("Channel assignment may be wrong" in s for s in w)


def test_validate_warns_tiny_abs() -> None:
    abs_data = np.full(1000, 1e-20)
    cmp_data = np.full(1000, 1e-20)

    w = validate_channel_assignment(abs_data, cmp_data, min_abs_range=1e-15)
    assert any("extremely small" in s for s in w)


def test_validate_no_warnings_normal() -> None:
    abs_data = np.sin(np.linspace(0, 10, 1000)) * 1.0
    cmp_data = np.sin(np.linspace(0, 10, 1000)) * 0.01

    w = validate_channel_assignment(abs_data, cmp_data)
    # Should only have the informational "flux ranges" line, no WARNING
    assert not any("WARNING" in s for s in w)


# -----------------------------------------------------------------------
# Integration: StreamingReader with ColumnMapping
# -----------------------------------------------------------------------


def test_streaming_reader_with_column_mapping() -> None:
    """StreamingReaderConfig now accepts column_mapping."""
    from rotating_coil_analyzer.ingest.readers_streaming import (
        StreamingReader,
        StreamingReaderConfig,
    )

    # Create a small ASCII file: 5 columns, 64 rows (4 turns of 16)
    n_rows = 64
    mat = np.zeros((n_rows, 5))
    mat[:, 0] = np.linspace(0, 3.0, n_rows)  # time
    mat[:, 1] = 0.001  # small signal
    mat[:, 2] = 1.0  # large signal (would normally be auto-detected as abs)
    mat[:, 3] = np.linspace(0, 100, n_rows)  # current
    mat[:, 4] = 42.0  # another current

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test.txt"
        np.savetxt(p, mat)

        # Force col1=abs and col2=cmp (override auto-detection which would swap)
        cfg = StreamingReaderConfig(
            strict_time=True,
            column_mapping=ColumnMapping(flux_abs_col=1, flux_cmp_col=2, current_col=4),
        )
        reader = StreamingReader(cfg)
        sf = reader.read(
            p, run_id="test", segment="1",
            samples_per_turn=16, shaft_speed_rpm=60.0,
        )

        # With explicit mapping, col1 should be abs even though col2 has larger range
        assert any("explicit" in w for w in sf.warnings)
        # Current should be from col4
        assert any("explicit current" in w for w in sf.warnings)


# -----------------------------------------------------------------------
# Integration: PlateauReader with ColumnMapping
# -----------------------------------------------------------------------


def test_plateau_reader_with_column_mapping() -> None:
    """PlateauReaderConfig now accepts column_mapping."""
    from rotating_coil_analyzer.ingest.readers_plateau import (
        PlateauReader,
        PlateauReaderConfig,
    )

    n_per_file = 32  # 2 turns of 16
    Ns = 16

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for step, current in [(1, 10.0), (2, 20.0)]:
            fname = f"TEST_Run_{step}_I_{current:.1f}A_Seg1_raw_measurement_data.txt"
            mat = np.zeros((n_per_file, 5))
            mat[:, 0] = np.linspace(0, 1.0, n_per_file)
            mat[:, 1] = 0.001  # small
            mat[:, 2] = 1.0  # large
            mat[:, 3] = current
            mat[:, 4] = 0.0
            np.savetxt(root / fname, mat)

        cfg = PlateauReaderConfig(
            column_mapping=ColumnMapping(flux_abs_col=1, flux_cmp_col=2),
        )
        reader = PlateauReader(cfg)
        rep = root / "TEST_Run_1_I_10.0A_Seg1_raw_measurement_data.txt"
        sf = reader.read(rep, run_id="test", segment="Seg1", samples_per_turn=Ns)

        assert any("explicit" in w for w in sf.warnings)
        assert sf.n_turns == 4  # 2 files * 2 turns each
