"""Tests for native ingest support of the FFMM H/V-steerer plateau format
and the ``*_Parameters.txt`` filename, without breaking the standard layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotating_coil_analyzer.ingest.readers_plateau import (
    parse_plateau_filename,
    PlateauReader,
)
from rotating_coil_analyzer.ingest.discovery import (
    MeasurementDiscovery,
    find_parameters_txt,
)


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

class TestParsePlateauFilename:
    def test_standard_layout(self):
        info = parse_plateau_filename(
            "mag_Run_3_I_-6.667A_Integral_raw_measurement_data.txt"
        )
        assert info == {"base": "mag", "step": 3, "seg": "Integral", "i": pytest.approx(-6.667)}

    def test_hv_layout_active_is_horizontal(self):
        info = parse_plateau_filename(
            "X_Run_0IH_10.000000IV_0.000000_Main_raw_measurement_data.txt"
        )
        assert info["base"] == "X"
        assert info["step"] == 0
        assert info["seg"] == "Main"
        assert info["i"] == pytest.approx(10.0)      # active channel = horizontal
        assert info["ih"] == pytest.approx(10.0)
        assert info["iv"] == pytest.approx(0.0)

    def test_hv_layout_active_is_vertical(self):
        info = parse_plateau_filename(
            "X_Run_40IH_0.000000IV_-6.667000_Main_raw_measurement_data.txt"
        )
        assert info["step"] == 40
        assert info["i"] == pytest.approx(-6.667)    # horizontal is 0 -> use vertical

    def test_hv_layout_both_zero(self):
        info = parse_plateau_filename(
            "X_Run_36IH_0.000000IV_0.000000_Main_raw_measurement_data.txt"
        )
        assert info["i"] == pytest.approx(0.0)

    def test_non_matching_returns_none(self):
        assert parse_plateau_filename("not_a_plateau_file.txt") is None
        assert parse_plateau_filename("X_corr_sigs_Ap_1_SegMain.bin") is None

    def test_extra_pattern_override_wins(self):
        import re
        pat = re.compile(
            r"^(?P<base>.+?)_step(?P<step>\d+)_(?P<seg>[^_]+)_raw_measurement_data\.txt$"
        )
        info = parse_plateau_filename(
            "custom_step7_SegA_raw_measurement_data.txt", extra_pattern=pat
        )
        assert info["base"] == "custom"
        assert info["step"] == 7
        assert info["seg"] == "SegA"


# ---------------------------------------------------------------------------
# find_parameters_txt
# ---------------------------------------------------------------------------

class TestFindParametersTxt:
    def test_exact_name(self, tmp_path):
        (tmp_path / "Parameters.txt").write_text("x: 1\n")
        assert find_parameters_txt(tmp_path).name == "Parameters.txt"

    def test_suffixed_name_fallback(self, tmp_path):
        (tmp_path / "Buckley-steerer_20260623_165352_Parameters.txt").write_text("x: 1\n")
        assert find_parameters_txt(tmp_path).name.endswith("_Parameters.txt")

    def test_exact_preferred_over_suffixed(self, tmp_path):
        (tmp_path / "Parameters.txt").write_text("x: 1\n")
        (tmp_path / "zzz_Parameters.txt").write_text("y: 2\n")
        assert find_parameters_txt(tmp_path).name == "Parameters.txt"

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_parameters_txt(tmp_path, max_up=0)


# ---------------------------------------------------------------------------
# End-to-end: discovery + reader on a synthetic H/V dataset
# ---------------------------------------------------------------------------

def _write_run(folder, base, label, n_rows):
    """Write a tiny 4-column raw_measurement_data file (t, df_abs, df_cmp, I)."""
    t = np.arange(n_rows, dtype=float) * 1e-3
    da = np.cos(2 * np.pi * np.arange(n_rows) / n_rows)
    dc = 0.5 * da
    I = np.full(n_rows, 999.0)  # bogus reference column, like the real degauss data
    mat = np.column_stack([t, da, dc, I])
    path = folder / f"{base}_Run_{label}_Main_raw_measurement_data.txt"
    np.savetxt(path, mat)
    return path


def test_discovery_and_reader_hv_dataset(tmp_path):
    # A *_Parameters.txt (not the exact name) + two H/V-named plateau files.
    (tmp_path / "mag_20260101_000000_Parameters.txt").write_text(
        "Parameters.Measurement.samples: 4\n"
        r"Parameters.MH.FDIs: TABLE{Main\t0\t1}" + "\n"
    )
    base = "mag_20260101_000000"
    _write_run(tmp_path, base, "0IH_10.000000IV_0.000000", n_rows=8)   # +10 A
    _write_run(tmp_path, base, "1IH_-10.000000IV_0.000000", n_rows=8)  # -10 A

    cat = MeasurementDiscovery(strict=False).build_catalog(str(tmp_path))
    assert cat.samples_per_turn == 4
    assert [s.segment_id for s in cat.segments] == ["Main"]
    assert len(cat.segment_files) == 1

    key = next(iter(cat.segment_files))
    rep = cat.segment_files[key]
    # representative is the lowest step (Run_0)
    assert "Run_0IH" in rep.name

    seg = PlateauReader().read(
        rep, run_id=str(key[0]), segment=key[2],
        samples_per_turn=cat.samples_per_turn, magnet_order=1,
    )
    # 2 runs x (8 rows / 4 samples-per-turn) = 4 turns
    assert seg.n_turns == 4
    pid = seg.df["plateau_id"].to_numpy()
    hint = seg.df["plateau_I_hint"].to_numpy()
    hints_per_plateau = {int(p): float(hint[pid == p][0]) for p in sorted(set(pid))}
    # plateau_I_hint carries the TRUE signed set current, not the bogus 999 column
    assert hints_per_plateau == {0: pytest.approx(10.0), 1: pytest.approx(-10.0)}
