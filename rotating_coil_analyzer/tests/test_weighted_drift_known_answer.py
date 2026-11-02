"""End-to-end known-answer test of drift_mode="weighted" (Bottura AII.12-14).

Known harmonics are sampled at equal encoder angles, with a constant integrator
drift voltage accumulating over the measured time of each increment. The timing
has the shape measured on the MBB 2025-12-12 plateau data: a pattern repeating
every turn (std/mean about 0.9 %) plus turn-to-turn jitter (about 0.4 %).

This checks the implementation, not the physics: the forward model assumes the
drift is a constant voltage over the turn. See docs/11_bugfix_diary.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotating_coil_analyzer.analysis.kn_pipeline import SegmentKn, compute_legacy_kn_per_turn

NS, H, R = 256, 15, 0.02
PHI_RAD = 0.3
DRIFT_V = 2e-3  # flux units per second; about 0.1 % of the flux amplitude per turn


def _kn() -> SegmentKn:
    r = 0.03
    kn_abs = np.array([100 * 0.5 * r**n / n for n in range(1, H + 1)], dtype=complex)
    return SegmentKn(orders=np.arange(1, H + 1), kn_abs=kn_abs, kn_cmp=0.3 * kn_abs, kn_ext=None, source_path="synthetic")


def _truth() -> np.ndarray:
    C = np.zeros(H, complex)
    C[0] = 0.6
    C[2] = 0.6e-4 * (3.0 - 1.0j)
    C[4] = 0.6e-4 * (0.5 + 2.0j)
    return C


def _timing(n_turns: int = 6, seed: int = 0) -> np.ndarray:
    th = 2 * np.pi * np.arange(NS) / NS
    pattern = 1.0 + 0.012 * np.sin(th) + 0.005 * np.sin(3 * th)
    jitter = 1.0 + 0.004 * np.random.default_rng(seed).standard_normal((n_turns, NS))
    dt = (pattern * jitter / NS).ravel()
    return np.cumsum(dt).reshape(n_turns, NS)


def _durations(t: np.ndarray) -> np.ndarray:
    # Written out here, not imported from the code under test.
    inner = np.diff(t, axis=1)
    first = np.median(inner, axis=1)  # turn 0 has no predecessor
    first[1:] = t[1:, 0] - t[:-1, -1]
    return np.concatenate([first[:, None], inner], axis=1)


def _analyse(drift_mode: str, drift_v: float):
    kn, C, t = _kn(), _truth(), _timing()
    th = 2 * np.pi * np.arange(NS) / NS
    rotated = C * np.exp(1j * np.arange(1, H + 1) * PHI_RAD)
    dur = _durations(t)

    def increments(kn_vec: np.ndarray) -> np.ndarray:
        f = np.conj(kn_vec) * rotated / R ** np.arange(H)
        flux = np.real(sum(f[n - 1] * np.exp(1j * n * th) for n in range(1, H + 1)))
        step = np.diff(np.concatenate([[flux[-1]], flux]))
        return step[None, :] + drift_v * dur

    result = compute_legacy_kn_per_turn(
        df_abs_turns=increments(kn.kn_abs),
        df_cmp_turns=increments(kn.kn_cmp),
        t_turns=t,
        I_turns=np.full_like(t, 100.0),
        kn=kn,
        Rref_m=R,
        magnet_order=1,
        options=("dri", "rot"),
        drift_mode=drift_mode,
    )
    return result.C_abs - C


def test_weighted_recovers_truth_where_legacy_does_not() -> None:
    weighted_error = np.abs(_analyse("weighted", DRIFT_V)).max()
    legacy_error = np.abs(_analyse("legacy", DRIFT_V)).max()
    assert weighted_error < 1e-11
    # The non-uniform timing must matter, or this test could not fail.
    assert legacy_error > 1e3 * weighted_error


@pytest.mark.parametrize("mode", ["legacy", "weighted"])
def test_both_modes_recover_truth_without_drift(mode: str) -> None:
    assert np.abs(_analyse(mode, 0.0)).max() < 1e-11
