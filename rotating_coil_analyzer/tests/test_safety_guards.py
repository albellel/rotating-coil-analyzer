"""Tests for Priority 1 safety guards.

Covers:
- CEL division guard (dipole and quadrupole) with near-zero denominator
- Feeddown propagation with near-zero main field
- Rotation angle for bad turns
- safe_normalize_to_units with weak and normal main field
- di/dt eps floor
"""

from __future__ import annotations

import numpy as np
import pytest

from rotating_coil_analyzer.analysis.kn_pipeline import (
    SegmentKn,
    compute_legacy_kn_per_turn,
    safe_normalize_to_units,
)
from rotating_coil_analyzer.analysis.preprocess import di_dt_weights
from rotating_coil_analyzer.analysis.utility_functions import diagnose_cel_fed


def _make_kn(H: int = 15) -> SegmentKn:
    """Create a simple identity-like kn (all ones)."""
    orders = np.arange(1, H + 1, dtype=int)
    kn_abs = np.ones(H, dtype=complex)
    kn_cmp = np.ones(H, dtype=complex)
    return SegmentKn(orders=orders, kn_abs=kn_abs, kn_cmp=kn_cmp, kn_ext=None, source_path="<test>")


def _make_turns(n_turns: int, Ns: int, H: int = 15) -> tuple:
    """Create simple synthetic per-turn arrays.

    Returns (df_abs, df_cmp, t, I) all shaped (n_turns, Ns).
    The signal is a pure sine at harmonic 1 with unit amplitude.
    """
    theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    sig = np.sin(theta)
    df_abs = np.tile(sig, (n_turns, 1))
    df_cmp = np.tile(sig * 0.1, (n_turns, 1))
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)
    return df_abs, df_cmp, t, I


# -----------------------------------------------------------------------
# CEL guard tests
# -----------------------------------------------------------------------


def test_cel_zero_denom_dipole() -> None:
    """For m=1, CEL uses C_cmp[:,10]. If near-zero, zR should be 0, not inf."""
    H = 15
    Ns = H + 1  # minimal
    n_turns = 3
    kn = _make_kn(H)
    df_abs, df_cmp, t, I = _make_turns(n_turns, Ns, H)

    # Make compensated signal nearly zero so C_cmp[:,10] ~ 0
    df_cmp[:, :] = 1e-40

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=1,
        options=("cel",),
    )
    assert np.all(np.isfinite(result.zR)), f"zR has non-finite: {result.zR}"
    assert np.all(np.isfinite(result.x_m))
    assert np.all(np.isfinite(result.y_m))


def test_cel_zero_denom_quadrupole() -> None:
    """For m=2, CEL uses C_abs[:,1]. If near-zero, zR should be 0, not inf."""
    H = 15
    Ns = H + 1
    n_turns = 3
    kn = _make_kn(H)

    # All-zero signal -> all harmonics near zero
    df_abs = np.full((n_turns, Ns), 1e-40)
    df_cmp = np.full((n_turns, Ns), 1e-40)
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=2,
        options=("cel",),
    )
    assert np.all(np.isfinite(result.zR)), f"zR has non-finite: {result.zR}"


def test_cel_feeddown_no_nan() -> None:
    """With cel+fed and near-zero main field, output should have no NaN/inf."""
    H = 15
    Ns = H + 1
    n_turns = 2
    kn = _make_kn(H)
    df_abs = np.full((n_turns, Ns), 1e-40)
    df_cmp = np.full((n_turns, Ns), 1e-40)
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=1,
        options=("cel", "fed"),
    )
    assert np.all(np.isfinite(result.C_abs)), "C_abs has NaN/inf after cel+fed"
    assert np.all(np.isfinite(result.C_cmp)), "C_cmp has NaN/inf after cel+fed"


# -----------------------------------------------------------------------
# Rotation guard tests
# -----------------------------------------------------------------------


def test_rotation_bad_turns_zero() -> None:
    """Turns with zero flux should get phi_out=0.0 (not NaN)."""
    H = 15
    Ns = H + 1
    n_turns = 3
    kn = _make_kn(H)
    df_abs, df_cmp, t, I = _make_turns(n_turns, Ns, H)

    # Zero out one turn
    df_abs[1, :] = 0.0
    df_cmp[1, :] = 0.0

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=1,
        options=("rot",),
    )
    assert np.isfinite(result.phi_out_rad[1]), "phi_out should be finite"
    assert result.phi_out_rad[1] == 0.0, "bad turn should have phi_out=0.0"
    assert result.phi_bad[1], "bad turn should be flagged"


# -----------------------------------------------------------------------
# safe_normalize_to_units tests
# -----------------------------------------------------------------------


def test_safe_normalize_weak_field() -> None:
    """Turns with main field below threshold should get NaN and ok=False."""
    H = 5
    C = np.zeros((4, H), dtype=complex)
    C[:, 0] = [1.0, 1e-25, 1.0, 0.0]  # main field (m=1)
    C[:, 1] = 1.0  # some harmonic

    C_units, ok = safe_normalize_to_units(C, magnet_order=1, min_main_field=1e-20)

    assert ok[0] is np.bool_(True)
    assert ok[1] is np.bool_(False)  # below threshold
    assert ok[2] is np.bool_(True)
    assert ok[3] is np.bool_(False)  # exactly zero

    assert np.isfinite(C_units[0, 1])
    assert np.isnan(C_units[1, 1])
    assert np.isnan(C_units[3, 1])


def test_safe_normalize_normal() -> None:
    """With main field = 1.0, harmonic 2 should become 10000 * C2/C1."""
    H = 5
    C = np.zeros((2, H), dtype=complex)
    C[:, 0] = 1.0  # B1 = 1 T
    C[:, 1] = 0.5  # harmonic 2

    C_units, ok = safe_normalize_to_units(C, magnet_order=1)

    assert np.all(ok)
    np.testing.assert_allclose(C_units[:, 0].real, 10000.0, rtol=1e-12)
    np.testing.assert_allclose(C_units[:, 1].real, 5000.0, rtol=1e-12)


# -----------------------------------------------------------------------
# di/dt eps floor test
# -----------------------------------------------------------------------


def test_didt_eps_floor() -> None:
    """Calling di_dt_weights with eps_I_A=0 should not produce inf weights."""
    n_turns, Ns = 2, 32
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    # Current with a ramp and some near-zero samples
    I = np.tile(np.linspace(0.001, 100.0, Ns), (n_turns, 1))

    result = di_dt_weights(t, I, eps_I_A=0.0)
    assert np.all(np.isfinite(result.weights)), "weights should be finite even with eps_I_A=0"


# -----------------------------------------------------------------------
# max_zR clamp tests
# -----------------------------------------------------------------------


def test_max_zR_clamp_flags_large_zR() -> None:
    """Turns with |zR| > max_zR should be flagged and have zR set to 0."""
    H = 15
    Ns = 128
    n_turns = 5
    kn = _make_kn(H)

    # Create a signal with a strong quadrupole (m=2) and a dipole component
    # that produces a large zR via the absolute-channel formula.
    theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    flux_quad = np.cos(2 * theta)  # strong C_2
    flux_dip = 0.5 * np.cos(theta)  # strong C_1 -> large zR = -C_1/(1*C_2)

    df_quad = np.empty_like(flux_quad)
    df_quad[0] = flux_quad[0]
    df_quad[1:] = flux_quad[1:] - flux_quad[:-1]

    df_dip = np.empty_like(flux_dip)
    df_dip[0] = flux_dip[0]
    df_dip[1:] = flux_dip[1:] - flux_dip[:-1]

    df_abs = np.tile(df_quad + df_dip, (n_turns, 1))
    df_cmp = np.zeros_like(df_abs)
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    # Without clamp — should get large |zR|
    result_no_clamp = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=2,
        options=("cel",),
        max_zR=None,
    )
    assert np.all(np.abs(result_no_clamp.zR) > 0.01), \
        "Expected large |zR| without clamp"
    assert not np.any(result_no_clamp.zR_clamped), \
        "No turns should be flagged without clamp"

    # With clamp — should flag and zero out
    result_clamped = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=2,
        options=("cel",),
        max_zR=0.01,
    )
    assert np.all(result_clamped.zR_clamped), \
        "All turns should be flagged as clamped"
    assert np.all(result_clamped.zR == 0.0), \
        "Clamped zR should be set to 0"
    assert np.all(result_clamped.x_m == 0.0)
    assert np.all(result_clamped.y_m == 0.0)


def test_max_zR_none_backward_compatible() -> None:
    """Default max_zR=None should give identical results to old behavior."""
    H = 15
    Ns = 64
    n_turns = 3
    kn = _make_kn(H)
    df_abs, df_cmp, t, I = _make_turns(n_turns, Ns, H)

    result = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs, df_cmp_turns=df_cmp,
        t_turns=t, I_turns=I, kn=kn,
        Rref_m=0.05, magnet_order=1,
        options=("dri", "rot", "cel", "fed"),
        max_zR=None,
    )

    # zR_clamped should exist and be all-False
    assert result.zR_clamped.shape == (n_turns,)
    assert not np.any(result.zR_clamped)
    # All outputs should be finite
    assert np.all(np.isfinite(result.C_abs))
    assert np.all(np.isfinite(result.zR))


# -----------------------------------------------------------------------
# diagnose_cel_fed tests
# -----------------------------------------------------------------------


def test_diagnose_cel_fed_safe() -> None:
    """Quadrupole with well-centred signal should produce SAFE diagnostic."""
    H = 15
    Ns = 128
    n_turns = 10
    kn = _make_kn(H)

    # Pure quadrupole signal — cel gives zR ~ 0 (no dipole component).
    # Use Rref=1.0 to avoid Rref^n scaling amplifying tiny leakage into
    # a large zR ratio.
    theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    df_abs = np.tile(np.sin(2 * theta) * 10.0, (n_turns, 1))
    df_cmp = np.zeros_like(df_abs)
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    diag = diagnose_cel_fed(
        df_abs, df_cmp, t, I,
        kn=kn, r_ref=1.0, magnet_order=2,
        max_zR=0.01,
    )

    assert diag.recommendation == "SAFE"
    assert diag.n_suspect == 0
    assert diag.n_total == n_turns
    assert diag.max_zR_threshold == 0.01
    assert diag.zR_abs.shape == (n_turns,)
    assert diag.B_main_with_fed.shape == (n_turns,)
    assert diag.B_main_without_fed.shape == (n_turns,)


def test_diagnose_cel_fed_unsafe() -> None:
    """Dipole with noisy compensated channel should produce UNSAFE diagnostic."""
    H = 15
    Ns = 128
    n_turns = 10
    kn = _make_kn(H)

    # Dipole signal in absolute channel + random noise in compensated channel
    # to simulate poor compensated SNR
    rng = np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
    flux = np.cos(theta)  # dipole
    df = np.empty_like(flux)
    df[0] = flux[0]
    df[1:] = flux[1:] - flux[:-1]

    df_abs = np.tile(df, (n_turns, 1))
    # Large noise in compensated channel — cel uses C_cmp[10]/C_cmp[11]
    # which will be pure noise -> large |zR|
    df_cmp = rng.normal(0, 1.0, (n_turns, Ns))
    t = np.tile(np.linspace(0, 1.0, Ns, endpoint=False), (n_turns, 1))
    t += np.arange(n_turns)[:, None]
    I = np.full((n_turns, Ns), 100.0)

    diag = diagnose_cel_fed(
        df_abs, df_cmp, t, I,
        kn=kn, r_ref=0.05, magnet_order=1,
        max_zR=0.01,
    )

    assert diag.recommendation == "UNSAFE"
    assert diag.n_suspect > n_turns // 2
    assert diag.n_total == n_turns
