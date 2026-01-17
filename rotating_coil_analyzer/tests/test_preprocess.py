from __future__ import annotations

import numpy as np
import pytest

from rotating_coil_analyzer.analysis.preprocess import (
    di_dt_weights,
    integrate_to_flux,
)


def test_di_dt_weights_applied_only_on_ramp_and_high_current() -> None:
    # Two turns, Ns=5
    t = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )

    # Turn 0: ramping current, mean(I)=22 A, slope=1 A/s -> correction should apply.
    # Turn 1: constant current -> correction should NOT apply.
    I = np.array(
        [
            [20.0, 21.0, 22.0, 23.0, 24.0],
            [50.0, 50.0, 50.0, 50.0, 50.0],
        ]
    )

    res = di_dt_weights(t_turns=t, I_turns=I)

    assert res.applied.shape == (2,)
    assert bool(res.applied[0]) is True
    assert bool(res.applied[1]) is False

    I_mean0 = np.mean(I[0, :])
    expected_w0 = I_mean0 / I[0, :]
    assert np.allclose(res.weights[0, :], expected_w0)
    assert np.allclose(res.weights[1, :], 1.0)


def test_integrate_to_flux_legacy_matches_definition() -> None:
    # One turn with a non-zero mean so that drift correction does something.
    df = np.array([[1.0, 2.0, 3.0, 4.0]])

    flux, diag = integrate_to_flux(df, drift=True, drift_mode="legacy")
    assert diag is not None
    assert diag.mode == "legacy"

    df0 = df - np.mean(df, axis=1, keepdims=True)
    expected = np.cumsum(df0, axis=1)
    expected = expected - np.mean(expected, axis=1, keepdims=True)

    assert np.allclose(flux, expected)


def test_integrate_to_flux_weighted_removes_dt_weighted_offset() -> None:
    # Construct a variable-dt turn and a signal proportional to dt.
    # Then the weighted drift correction should subtract the exact offset and yield ~0 increments.
    Ns = 6
    dt = np.array([0.0, 0.5, 1.5, 1.0, 2.0, 0.5])
    t = np.cumsum(dt)

    # Make df = c * dt so that sum(df)/sum(dt) = c.
    c = 3.7
    df = (c * dt)[None, :]
    t_turns = t[None, :]

    flux, diag = integrate_to_flux(df, drift=True, drift_mode="weighted", t_turns=t_turns)
    assert diag is not None
    assert diag.mode == "weighted"
    assert bool(diag.applied[0]) is True

    # The corrected increments are df - c*dt = 0 -> cumulative sum is identically 0.
    assert np.allclose(flux, 0.0, atol=1e-12)


def test_integrate_to_flux_weighted_requires_time() -> None:
    df = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        integrate_to_flux(df, drift=True, drift_mode="weighted", t_turns=None)
