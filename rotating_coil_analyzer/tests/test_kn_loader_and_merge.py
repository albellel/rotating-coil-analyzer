from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from rotating_coil_analyzer.analysis import load_segment_kn_txt
from rotating_coil_analyzer.analysis.merge import recommend_merge_choice, FLAG_MAIN_FORCED_ABS
from rotating_coil_analyzer.analysis.kn_pipeline import merge_coefficients


def test_segment_kn_loader_accepts_4_columns_from_fcc_zip() -> None:
    zpath = Path(__file__).parent / "data" / "20231110_140210_FCC-ee_center_DC_PXMQNDI8WC-CR000001.zip"
    assert zpath.exists()

    with zipfile.ZipFile(zpath, "r") as z:
        members = [m for m in z.namelist() if m.endswith("Kn_values_Seg_Main.txt")]
        assert len(members) == 1
        text = z.read(members[0]).decode("utf-8")

    # Write to a temp file so the loader stays path-based.
    p = Path(__file__).with_suffix(".tmp_kn.txt")
    try:
        p.write_text(text, encoding="utf-8")
        kn = load_segment_kn_txt(str(p))
    finally:
        if p.exists():
            p.unlink()

    assert kn.orders.shape == (15,)
    assert kn.kn_abs.shape == (15,)
    assert kn.kn_cmp.shape == (15,)
    assert kn.kn_ext is None

    # spot-check first row
    # File prints values with ~7 significant digits.
    assert np.isclose(kn.kn_abs[0].real, 0.0311795, rtol=0.0, atol=1e-12)
    assert np.isclose(kn.kn_abs[0].imag, 0.0, rtol=0.0, atol=1e-12)


def test_merge_recommendation_forces_main_order_abs() -> None:
    rng = np.random.default_rng(0)
    n_turns = 20
    H = 8
    m = 2

    # Make abs a bit noisier than cmp for all orders.
    base = rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H))
    C_cmp = base + 0.1 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))
    C_abs = base + 0.3 * (rng.normal(size=(n_turns, H)) + 1j * rng.normal(size=(n_turns, H)))

    choice, diag = recommend_merge_choice(C_abs=C_abs, C_cmp=C_cmp, magnet_order=m)
    assert choice.shape == (H,)
    assert int(choice[m - 1]) == 0
    assert int(diag.flags[m - 1]) & FLAG_MAIN_FORCED_ABS


def test_merge_coefficients_custom_is_traceable() -> None:
    rng = np.random.default_rng(1)
    C_abs = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    C_cmp = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))

    merged, choice = merge_coefficients(
        C_abs=C_abs,
        C_cmp=C_cmp,
        magnet_order=3,
        mode="custom",
        per_order_choice=[1, 1, 1, 0, 0],
    )

    # main order forced to abs
    assert choice.tolist() == [1, 1, 0, 0, 0]
    assert np.allclose(merged[:, 2], C_abs[:, 2])
    assert np.allclose(merged[:, 0], C_cmp[:, 0])
