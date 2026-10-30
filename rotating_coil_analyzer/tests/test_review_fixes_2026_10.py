"""Regression tests for the fixes of the 2026-10-30 review (docs/11_bugfix_diary.md)."""

from __future__ import annotations

import math
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotating_coil_analyzer.analysis.kn_pipeline import (
    SegmentKn,
    bucked_cmp_orders,
    compute_legacy_kn_per_turn,
    merge_coefficients,
)
from rotating_coil_analyzer.analysis.utility_functions import process_kn_pipeline
from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader

DATA = Path(__file__).parent / "data"
FCCEE_ZIP = DATA / "20231110_140210_FCC-ee_center_DC_PXMQNDI8WC-CR000001.zip"

NS, H, R = 256, 15, 0.02


def _kn(bucked_upto: int = 2) -> SegmentKn:
    r = 0.03
    kn_abs = np.array([100 * 0.5 * r**n / n for n in range(1, H + 1)], dtype=complex)
    kn_cmp = kn_abs * np.where(np.arange(1, H + 1) <= bucked_upto, 1e-9, 0.3)
    return SegmentKn(orders=np.arange(1, H + 1), kn_abs=kn_abs, kn_cmp=kn_cmp, kn_ext=None, source_path="synthetic")


def _signals(C: np.ndarray, kn_vec: np.ndarray, n_turns: int = 4, noise: float = 0.0, seed: int = 0):
    """Forward model of the pipeline: f_n = conj(kn_n) C_n / R^(n-1), flux = Re sum f_n e^{i n theta}."""
    th = 2 * np.pi * np.arange(NS) / NS
    f = np.conj(kn_vec[: C.size]) * C / R ** np.arange(C.size)
    flux = np.real(sum(f[n - 1] * np.exp(1j * n * th) for n in range(1, C.size + 1)))
    df = np.diff(np.concatenate([[flux[-1]], flux]))
    df = np.tile(df, (n_turns, 1))
    if noise:
        df = df + np.random.default_rng(seed).normal(0.0, noise * np.abs(df).max(), df.shape)
    t = np.tile(np.arange(NS) / NS, (n_turns, 1)) + np.arange(n_turns)[:, None]
    return df, t, np.full_like(t, 100.0)


def _run(C, kn, *, m, options, **kw):
    sig_kw = {k: kw.pop(k) for k in ("n_turns", "noise") if k in kw}
    dfa, t, I = _signals(C, kn.kn_abs, **sig_kw)
    dfc, _, _ = _signals(C, kn.kn_cmp, **sig_kw)
    return compute_legacy_kn_per_turn(
        df_abs_turns=dfa, df_cmp_turns=dfc, t_turns=t, I_turns=I, kn=kn, Rref_m=R,
        magnet_order=m, options=options, **kw,
    )


# ---------------------------------------------------------------------------
# skew main harmonic + rotation
# ---------------------------------------------------------------------------


def test_skew_main_with_rotation_normalises_to_A_m() -> None:
    C = np.zeros(H, complex)
    C[1] = 0.5j  # pure skew quadrupole (rotated by 0.1 rad below)
    C[2] = 0.5e-4 * (1 + 1j)
    rot = np.exp(1j * np.arange(1, H + 1) * 0.1)
    kn = _kn()
    r = _run(C * rot, kn, m=2, options=("dri", "rot", "nor"), skew_main=True)
    # C_m rotated onto the imaginary axis, normalised to 1e4 units
    assert abs(r.main_field[0].real) < 1e-12 * abs(r.main_field[0])
    assert np.allclose(r.C_abs[:, 1], 1e4j, rtol=1e-9)
    # c3 = 1e4 * C3 / A2 in the skew frame
    assert np.allclose(r.C_abs[:, 2], 1e4 * C[2] / 0.5, rtol=1e-9)


def test_skew_main_with_rotation_post_merge_normalisation() -> None:
    C = np.zeros(H, complex)
    C[0] = -0.2j
    C[2] = 0.2e-4 * (3 - 1j)
    kn = _kn(bucked_upto=1)
    dfa, t, I = _signals(C, kn.kn_abs)
    dfc, _, _ = _signals(C, kn.kn_cmp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, Cm, Cu, ok = process_kn_pipeline(dfa, dfc, t, I, kn, r_ref=R, magnet_order=1,
                                              options=("dri", "rot"), skew_main=True)
    assert ok.all()
    assert np.allclose(Cu[:, 0].imag, 1e4, rtol=1e-9)
    assert np.all(np.abs(Cu) < 1e5)  # no ~1e16 garbage


def test_normal_main_rotation_unchanged() -> None:
    C = np.zeros(H, complex)
    C[0] = 0.3 * np.exp(0.2j)
    kn = _kn(bucked_upto=1)
    r = _run(C, kn, m=1, options=("dri", "rot"))
    assert np.allclose(r.phi_out_rad, 0.2)
    assert np.allclose(r.C_abs[:, 0], 0.3)


# ---------------------------------------------------------------------------
# diagnostics: wrap boundary, silent cel, divergent fed, implausible field
# ---------------------------------------------------------------------------


def test_wrap_boundary_flagged() -> None:
    C = np.zeros(H, complex)
    C[0] = 0.05j  # skew dipole analysed as normal: angle exactly at +pi/2
    r = _run(C, _kn(bucked_upto=1), m=1, options=("dri", "rot"), n_turns=50, noise=1e-4)
    assert r.phi_near_wrap is not None and r.phi_near_wrap.all()
    assert any("wrap boundary" in w for w in r.warnings)


def test_dipole_cel_without_orders_10_11_is_reported() -> None:
    kn = _kn(bucked_upto=1)
    kn10 = SegmentKn(orders=kn.orders[:10], kn_abs=kn.kn_abs[:10], kn_cmp=kn.kn_cmp[:10], kn_ext=None, source_path="H10")
    C = np.zeros(10, complex)
    C[0] = 1.0
    dfa, t, I = _signals(C, kn10.kn_abs)
    r = compute_legacy_kn_per_turn(df_abs_turns=dfa, df_cmp_turns=dfa, t_turns=t, I_turns=I, kn=kn10,
                                   Rref_m=R, magnet_order=1, options=("dri", "rot", "cel", "fed"))
    assert r.cel_applied is not None and not r.cel_applied.any()
    assert any("cel (and therefore fed) was NOT applied" in w for w in r.warnings)


def test_cel_applied_flag_true_for_quadrupole() -> None:
    C = np.zeros(H, complex)
    C[1] = 1.0
    C[0] = -1e-3  # small offset
    r = _run(C, _kn(), m=2, options=("dri", "rot", "cel", "fed"))
    assert r.cel_applied.all()
    assert r.warnings == ()


def test_divergent_feeddown_and_implausible_field_reported() -> None:
    C = np.zeros(H, complex)
    C[1] = 1.0
    C[0] = 2.0  # centre offset of 2*Rref
    r = _run(C, _kn(), m=2, options=("dri", "rot", "cel", "fed"))
    assert any("|zR| >= 1" in w for w in r.warnings)
    C = np.zeros(H, complex)
    C[1] = 50.0
    r = _run(C, _kn(), m=2, options=("dri", "rot"))
    assert any("implausible" in w for w in r.warnings)


def test_bucked_cmp_orders_and_warning() -> None:
    kn = _kn(bucked_upto=2)
    _, choice_m1 = merge_coefficients(C_abs=np.zeros((1, H), complex), C_cmp=np.zeros((1, H), complex),
                                      magnet_order=1, mode="abs_upto_m_cmp_above")
    assert bucked_cmp_orders(kn, choice_m1) == [2]
    _, choice_m2 = merge_coefficients(C_abs=np.zeros((1, H), complex), C_cmp=np.zeros((1, H), complex),
                                      magnet_order=2, mode="abs_upto_m_cmp_above")
    assert bucked_cmp_orders(kn, choice_m2) == []

    C = np.zeros(H, complex)
    C[0] = 1.0
    dfa, t, I = _signals(C, kn.kn_abs)
    with pytest.warns(UserWarning, match="bucks them"):
        process_kn_pipeline(dfa, dfa, t, I, kn, r_ref=R, magnet_order=1, options=("dri", "rot"))


# ---------------------------------------------------------------------------
# Parameters.txt parsing
# ---------------------------------------------------------------------------


def _write_params(folder: Path, lines: list[str]) -> None:
    (folder / "Parameters.txt").write_text("\n".join(["Parameters.MH.FDIs: TABLE{Main\\t0\\t1\\t}"] + lines) + "\n")


def test_discovery_reads_rref_type_skew_options(tmp_path: Path) -> None:
    _write_params(tmp_path, [
        "Parameters.Measurement.samples: 1024",
        "Parameters.Measurement.MotorAngularSpeed: -60",
        "Parameters.magnetAnalyzer.refRadius: 0.010",
        "Parameters.magnetInfo.magnetType: Quadrupole",
        "Parameters.magnetInfo.magnetNormSkew: skew",
        "Parameters.magnetAnalyzer.dri: true",
        "Parameters.magnetAnalyzer.rot: true",
        "Parameters.magnetAnalyzer.cel: false",
        "Parameters.magnetAnalyzer.nor: true",
    ])
    cat = MeasurementDiscovery().build_catalog(tmp_path)
    assert cat.r_ref_m == pytest.approx(0.010)
    assert cat.magnet_order == 2 and cat.magnet_type == "Quadrupole"
    assert cat.skew_main is True
    assert cat.analyzer_options == ("dri", "rot", "nor")
    assert cat.samples_per_turn == 1024 and cat.shaft_speed_rpm == -60
    assert not any(w.startswith("WARNING") for w in cat.warnings)


def test_discovery_warns_on_missing_settings(tmp_path: Path) -> None:
    _write_params(tmp_path, [])
    cat = MeasurementDiscovery().build_catalog(tmp_path)
    assert cat.r_ref_m is None and cat.magnet_order is None
    text = "\n".join(cat.warnings)
    for needle in ("refRadius", "magnet order unknown", "samples", "shaft speed"):
        assert needle in text


def test_discovery_flags_implausible_rref(tmp_path: Path) -> None:
    _write_params(tmp_path, ["Parameters.magnetAnalyzer.refRadius: 1"])
    cat = MeasurementDiscovery().build_catalog(tmp_path)
    assert cat.r_ref_m == 1.0
    assert any("outside the usual" in w for w in cat.warnings)


# ---------------------------------------------------------------------------
# ASCII streaming files: whole turns + timing check
# ---------------------------------------------------------------------------


def test_ascii_streaming_trimmed_to_whole_turns(tmp_path: Path) -> None:
    Ns = 8
    n = 3 * Ns + 5
    t = np.arange(n) * (1.0 / Ns)  # 60 rpm -> 1 s per turn
    mat = np.column_stack([t, np.sin(t), 0.01 * np.cos(t), np.full(n, 10.0)])
    p = tmp_path / "x.txt"
    np.savetxt(p, mat)
    seg = StreamingReader().read(p, run_id="r", segment="1", samples_per_turn=Ns, shaft_speed_rpm=60.0)
    assert seg.n_turns == 3 and len(seg.df) == 3 * Ns
    assert any("removed 5 rows" in w for w in seg.warnings)
    seg = StreamingReader().read(p, run_id="r", segment="1", samples_per_turn=Ns, shaft_speed_rpm=120.0)
    assert any(w.startswith("WARNING: median dt") for w in seg.warnings)


# ---------------------------------------------------------------------------
# End-to-end on the bundled FCC-ee dataset vs FFMM's own results file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fccee_root(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fccee")
    zipfile.ZipFile(FCCEE_ZIP).extractall(out)
    return next(out.iterdir())


def test_fccee_end_to_end_matches_ffmm(fccee_root: Path) -> None:
    from rotating_coil_analyzer.analysis.kn_pipeline import load_segment_kn_txt
    from rotating_coil_analyzer.analysis.turns import split_into_turns
    from rotating_coil_analyzer.ingest.readers_plateau import PlateauReader
    from rotating_coil_analyzer.models.profile import AnalysisProfile

    cat = MeasurementDiscovery().build_catalog(fccee_root)
    profile = AnalysisProfile.from_catalog(cat, options=("dri", "rot", "cel", "fed"))
    assert profile.magnet_order == 2 and profile.r_ref_m == pytest.approx(0.010)

    key = next(iter(cat.segment_files))
    seg = PlateauReader().read(cat.segment_files[key], run_id=key[0], segment=key[2],
                               samples_per_turn=cat.samples_per_turn)
    tb = split_into_turns(seg)
    kn = load_segment_kn_txt(str(fccee_root / "Kn_values_Seg_Main.txt"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a correct setup must produce no diagnostics
        res, Cm, Cu, ok = process_kn_pipeline(
            tb.df_abs, tb.df_cmp, tb.t, tb.I, kn, r_ref=profile.r_ref_m,
            magnet_order=profile.magnet_order, options=profile.options,
        )
    ref = pd.read_csv(next(fccee_root.rglob("*_Main_results.txt")), sep="\t")
    assert len(ref) == tb.n_turns
    # b3 (~50 units) agrees to ~1e-13 units. The high orders of this
    # quadrupole sit at the numerical floor (|b_n| <~ 1e-3 units for n >= 8)
    # and agree to <~ 1e-5 units (< 1e-9 of the main field); cause not
    # established (not rotation, not ASCII precision). b15 also carries FFMM's
    # rotation off-by-one (exact with legacy_rotate_excludes_last=True).
    # Unchanged by the 2026-10-30 fixes (old and new code bit-identical here).
    for n in range(3, 16):
        assert np.max(np.abs(Cu[:, n - 1].real - ref[f"b{n}(Units)"].to_numpy())) < 2e-5
        assert np.max(np.abs(Cu[:, n - 1].imag - ref[f"a{n}(Units)"].to_numpy())) < 2e-5
    assert np.max(np.abs(Cu[:, 2].real - ref["b3(Units)"].to_numpy())) < 1e-10
    assert np.max(np.abs(res.x_m * 1e3 - ref["dx(mm)"].to_numpy())) < 1e-9
    assert np.max(np.abs(res.y_m * 1e3 - ref["dy(mm)"].to_numpy())) < 1e-9
    assert np.max(np.abs(res.phi_out_rad - ref["phi(rad)"].to_numpy())) < 1e-12


def test_cli_demo_runs_and_writes_csv(tmp_path: Path, capsys) -> None:
    from rotating_coil_analyzer.cli import main

    assert main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "m=2" in out and "Rref=10 mm" in out
    assert "max |b3 - FFMM b3|" in out
    per_turn = next(tmp_path.glob("*_per_turn.csv"))
    df = pd.read_csv(per_turn)
    assert len(df) == 50 and "B2_T" in df.columns and "b3_units" in df.columns


def test_cli_refuses_without_rref(tmp_path: Path) -> None:
    from rotating_coil_analyzer.cli import main

    folder = tmp_path / "meas"
    folder.mkdir()
    _write_params(folder, ["Parameters.Measurement.samples: 8", "Parameters.magnetInfo.magnetType: Dipole"])
    np.savetxt(folder / "x_Run_01_I_10A_Main_raw_measurement_data.txt", np.ones((16, 4)))
    (folder / "kn.txt").write_text("1 0 1 0\n")
    with pytest.raises(SystemExit, match="r_ref_m"):
        main([str(folder), "--kn", str(folder / "kn.txt"), "--out", str(tmp_path / "o")])
