"""Tests for AnalysisProfile (Priority 2)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from rotating_coil_analyzer.models.profile import AnalysisProfile


# -----------------------------------------------------------------------
# Basic construction
# -----------------------------------------------------------------------


def test_profile_defaults() -> None:
    p = AnalysisProfile(
        magnet_order=2,
        r_ref_m=0.059,
        samples_per_turn=512,
        shaft_speed_rpm=60.0,
    )
    assert p.magnet_order == 2
    assert p.r_ref_m == 0.059
    assert p.samples_per_turn == 512
    assert p.shaft_speed_rpm == 60.0
    assert p.options == ("dri", "rot")
    assert p.drift_mode == "legacy"
    assert p.merge_mode == "abs_upto_m_cmp_above"
    assert p.legacy_rotate_excludes_last is False
    assert p.min_main_field_T == 1e-20
    assert p.abs_calib == 1.0
    assert p.l_coil_m is None
    assert p.skew_main is False


def test_profile_frozen() -> None:
    p = AnalysisProfile(
        magnet_order=1, r_ref_m=0.05,
        samples_per_turn=512, shaft_speed_rpm=60.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.magnet_order = 3  # type: ignore[misc]


def test_profile_replace() -> None:
    p = AnalysisProfile(
        magnet_order=1, r_ref_m=0.05,
        samples_per_turn=512, shaft_speed_rpm=60.0,
    )
    p2 = dataclasses.replace(p, r_ref_m=0.059)
    assert p2.r_ref_m == 0.059
    assert p2.magnet_order == 1  # unchanged


# -----------------------------------------------------------------------
# from_catalog
# -----------------------------------------------------------------------


def _mock_catalog(
    magnet_order: Optional[int] = 2,
    samples_per_turn: int = 512,
    shaft_speed_rpm: float = 60.0,
    r_ref_m: Optional[float] = None,
    skew_main: Optional[bool] = None,
) -> MagicMock:
    cat = MagicMock()
    cat.magnet_order = magnet_order
    cat.samples_per_turn = samples_per_turn
    cat.shaft_speed_rpm = shaft_speed_rpm
    cat.r_ref_m = r_ref_m
    cat.skew_main = skew_main
    cat.parameters_path.name = "Parameters.txt"
    return cat


def test_profile_from_catalog() -> None:
    cat = _mock_catalog(magnet_order=2, samples_per_turn=1024)
    p = AnalysisProfile.from_catalog(cat, r_ref_m=0.02)
    assert p.magnet_order == 2
    assert p.samples_per_turn == 1024
    assert p.r_ref_m == 0.02


def test_profile_from_catalog_takes_r_ref_and_skew_from_parameters() -> None:
    cat = _mock_catalog(r_ref_m=0.010, skew_main=True)
    p = AnalysisProfile.from_catalog(cat)
    assert p.r_ref_m == 0.010
    assert p.skew_main is True


def test_profile_from_catalog_requires_r_ref() -> None:
    """No refRadius in Parameters and no override: refuse (no silent 17 mm)."""
    cat = _mock_catalog(r_ref_m=None)
    with pytest.raises(ValueError, match="r_ref_m"):
        AnalysisProfile.from_catalog(cat)


def test_profile_from_catalog_with_overrides() -> None:
    cat = _mock_catalog(magnet_order=1)
    p = AnalysisProfile.from_catalog(
        cat,
        r_ref_m=0.059,
        options=["dri", "rot", "cel", "fed"],
        legacy_rotate_excludes_last=False,
    )
    assert p.r_ref_m == 0.059
    assert p.options == ("dri", "rot", "cel", "fed")
    assert p.legacy_rotate_excludes_last is False


def test_profile_from_catalog_none_magnet_order() -> None:
    """Unknown magnet order and no override: refuse (no silent m=1)."""
    cat = _mock_catalog(magnet_order=None)
    with pytest.raises(ValueError, match="magnet_order"):
        AnalysisProfile.from_catalog(cat, r_ref_m=0.05)
    p = AnalysisProfile.from_catalog(cat, r_ref_m=0.05, magnet_order=3)
    assert p.magnet_order == 3


# -----------------------------------------------------------------------
# Dict serialization
# -----------------------------------------------------------------------


def test_profile_dict_roundtrip() -> None:
    p = AnalysisProfile(
        magnet_order=2, r_ref_m=0.059,
        samples_per_turn=512, shaft_speed_rpm=60.0,
        options=("dri", "rot", "cel", "fed"),
        l_coil_m=0.47,
    )
    d = p.to_dict()
    assert isinstance(d["options"], list)  # tuple -> list for JSON
    p2 = AnalysisProfile.from_dict(d)
    assert p2 == p


# -----------------------------------------------------------------------
# compute_from_profile parity
# -----------------------------------------------------------------------


def test_compute_from_profile_matches_direct() -> None:
    """compute_from_profile should produce identical results to direct call."""
    from rotating_coil_analyzer.analysis.kn_pipeline import (
        SegmentKn,
        compute_legacy_kn_per_turn,
        compute_from_profile,
    )

    H = 15
    Ns = H + 1
    n_turns = 2
    kn = SegmentKn(
        orders=np.arange(1, H + 1, dtype=int),
        kn_abs=np.ones(H, dtype=complex),
        kn_cmp=np.ones(H, dtype=complex),
        kn_ext=None,
        source_path="<test>",
    )

    theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    sig = np.sin(theta)
    df_abs = np.tile(sig, (n_turns, 1))
    df_cmp = np.tile(sig * 0.1, (n_turns, 1))
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    profile = AnalysisProfile(
        magnet_order=1, r_ref_m=0.05,
        samples_per_turn=Ns, shaft_speed_rpm=60.0,
        options=("dri", "rot"),
        legacy_rotate_excludes_last=True,
    )

    r_direct = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=profile.r_ref_m, magnet_order=profile.magnet_order,
        absCalib=profile.abs_calib, options=profile.options,
        drift_mode=profile.drift_mode, skew_main=profile.skew_main,
        eps_main=profile.min_main_field_T,
        legacy_rotate_excludes_last=profile.legacy_rotate_excludes_last,
    )

    r_profile = compute_from_profile(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        profile=profile,
    )

    np.testing.assert_array_equal(r_direct.C_abs, r_profile.C_abs)
    np.testing.assert_array_equal(r_direct.C_cmp, r_profile.C_cmp)
    np.testing.assert_array_equal(r_direct.phi_out_rad, r_profile.phi_out_rad)
