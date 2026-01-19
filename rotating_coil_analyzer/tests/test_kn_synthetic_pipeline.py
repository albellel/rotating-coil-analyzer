from __future__ import annotations

import numpy as np

from rotating_coil_analyzer.analysis.kn_pipeline import SegmentKn, compute_legacy_kn_per_turn


def test_kn_pipeline_synthetic_cosine_gives_expected_main_harmonic_units() -> None:
    """Smoke test for: FFT scaling + kn application + post-kn normalization.

    We synthesize a real flux waveform per turn:

      flux(\theta) = cos(m\theta)

    For a length-N discrete FFT with definition used by numpy, this yields
    FFT(flux)[m] / N = 1/2 (pure real). The legacy scaling uses

      f_m = 2*FFT(flux)[m] / N = 1.

    With kn=1 and rotation reference computed post-kn, C_m should be
    Rref^{m-1}. After normalization, the main harmonic becomes 10000.
    """

    Ns = 128
    n_turns = 3
    m = 2
    H = 8
    Rref = 0.01

    k = np.arange(Ns)
    theta = 2.0 * np.pi * k / float(Ns)
    flux = np.cos(m * theta)  # one turn

    # Build df such that integrate_to_flux(cumsum) reconstructs flux up to a constant.
    df = np.empty_like(flux)
    df[0] = flux[0]
    df[1:] = flux[1:] - flux[:-1]

    df_abs = np.tile(df, (n_turns, 1))
    df_cmp = np.zeros_like(df_abs)

    dt = 1e-3
    t = np.tile(k * dt, (n_turns, 1))
    I = np.tile(np.full(Ns, 100.0), (n_turns, 1))

    orders = np.arange(1, H + 1, dtype=int)
    kn = SegmentKn(
        orders=orders,
        kn_abs=np.ones(H, dtype=complex),
        kn_cmp=np.ones(H, dtype=complex),
        kn_ext=None,
        source_path="synthetic",
    )

    res = compute_legacy_kn_per_turn(
        df_abs_turns=df_abs,
        df_cmp_turns=df_cmp,
        t_turns=t,
        I_turns=I,
        kn=kn,
        Rref_m=Rref,
        magnet_order=m,
        absCalib=1.0,
        options=("rot", "nor"),
        drift_mode="legacy",
    )

    # Main harmonic should normalize to ~10000 (pure real) for each turn.
    main = res.C_abs[:, m - 1]
    assert np.allclose(np.real(main), 10000.0, atol=1e-9, rtol=0.0)
    assert np.allclose(np.imag(main), 0.0, atol=1e-9, rtol=0.0)

    # Rotation reference should be near zero.
    assert np.allclose(res.phi_out_rad, 0.0, atol=1e-9, rtol=0.0)
