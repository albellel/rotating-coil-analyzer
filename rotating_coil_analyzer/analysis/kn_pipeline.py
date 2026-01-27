from __future__ import annotations

"""Legacy-compatible $k_n$ application pipeline.

This module implements the *legacy analyzer* ordering for rotating-coil
analysis when applying complex calibration coefficients (often called $k_n$ or
$S_n$).

Hard constraint (project-wide)
------------------------------
No synthetic time: this module never creates, repairs, interpolates, or
extrapolates timestamps. When time is used (e.g., for di/dt estimation), only
the measured acquisition timestamps are used.

Legacy ordering (per turn)
--------------------------
Given incremental signals ``df_abs`` and ``df_cmp`` and per-turn time/current
arrays ``t`` and ``I``:

1) Optional ``dit`` current-ramp correction (reweight incremental samples).
2) Optional drift correction + integration to flux.
3) FFT on flux, scale as ``f_n = 2*FFT(flux)/Ns`` and drop DC.
4) Apply $k_n$ (complex) and reference-radius scaling.
5) Compute rotation reference from the *post-$k_n$ calibrated* main harmonic,
   then apply rotation (if enabled).
6) Center location (CEL) (if enabled).
7) Feeddown (if enabled).
8) Normalization (if enabled).

The implementation below follows the legacy C++ analyzer semantics in
``MatlabAnalyzerRotCoil.cpp``.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import math

import numpy as np

from .preprocess import (
    DriftMode,
    apply_di_dt_to_channels,
    estimate_linear_slope_per_turn,
    integrate_to_flux,
)


@dataclass(frozen=True)
class SegmentKn:
    """Segment-level $k_n$ values.

    Notes
    -----
    - The arrays are indexed by harmonic order $n$ starting at 1.
      Internally we store arrays as index ``i = n-1``.
    - ``kn_ext`` is optional. Some legacy files include an "external" channel.
    """

    orders: np.ndarray  # (H,) = [1..H]
    kn_abs: np.ndarray  # (H,) complex
    kn_cmp: np.ndarray  # (H,) complex
    kn_ext: Optional[np.ndarray]  # (H,) complex or None
    source_path: str


@dataclass(frozen=True)
class LegacyKnPerTurn:
    """Full per-turn result of the legacy $k_n$ pipeline.

    All complex harmonic arrays are indexed by harmonic order $n$ starting at
    1 (i.e., array index ``i = n-1``).
    """

    orders: np.ndarray  # (H,) = [1..H]

    # Calibrated harmonics after kn (+ optional rotation/CEL/feeddown/nor)
    C_abs: np.ndarray  # (n_turns, H) complex
    C_cmp: np.ndarray  # (n_turns, H) complex

    # "DB" snapshots: after kn, before rotation/feeddown/normalization
    C_abs_db: np.ndarray  # (n_turns, H) complex
    C_cmp_db: np.ndarray  # (n_turns, H) complex

    # Derived per-turn scalars
    phi_out_rad: np.ndarray  # (n_turns,)
    phi_bad: np.ndarray  # (n_turns,) bool

    zR: np.ndarray  # (n_turns,) complex  (dimensionless center)
    z_m: np.ndarray  # (n_turns,) complex  (meters)
    x_m: np.ndarray  # (n_turns,) float
    y_m: np.ndarray  # (n_turns,) float

    main_field: np.ndarray  # (n_turns,) complex (after full pipeline) * absCalib
    main_field_db: np.ndarray  # (n_turns,) complex (after kn, before rot etc) * absCalib

    I_mean_A: np.ndarray  # (n_turns,)
    dI_dt_A_per_s: np.ndarray  # (n_turns,)
    duration_s: np.ndarray  # (n_turns,)
    time_median_s: np.ndarray  # (n_turns,)


    @property
    def H(self) -> int:
        return int(self.orders.size)


def load_segment_kn_txt(path: str) -> SegmentKn:
    """Load a segment $k_n$ text file.

    Supported formats
    -----------------
    - 4 columns per row: ``AbsRe AbsIm CmpRe CmpIm``
    - 6 columns per row: ``AbsRe AbsIm CmpRe CmpIm ExtRe ExtIm``

    Rows correspond to harmonic orders $n=1,2,\\ldots,H$.
    """

    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals = [float(x) for x in parts]
            except Exception as e:
                raise ValueError(f"Invalid numeric row in kn file {path!r}: {line!r}") from e
            if len(vals) not in (4, 6):
                raise ValueError(
                    f"Unsupported kn row with {len(vals)} columns in {path!r}. Expected 4 or 6 columns."
                )
            rows.append(vals)

    if not rows:
        raise ValueError(f"Empty kn file: {path!r}")

    arr = np.asarray(rows, dtype=float)
    H = arr.shape[0]
    orders = np.arange(1, H + 1, dtype=int)

    kn_abs = arr[:, 0] + 1j * arr[:, 1]
    kn_cmp = arr[:, 2] + 1j * arr[:, 3]
    kn_ext: Optional[np.ndarray] = None
    if arr.shape[1] == 6:
        kn_ext = arr[:, 4] + 1j * arr[:, 5]

    if not np.all(np.isfinite(kn_abs)) or not np.all(np.isfinite(kn_cmp)):
        raise ValueError(f"Non-finite values detected in kn file: {path!r}")
    if kn_ext is not None and not np.all(np.isfinite(kn_ext)):
        raise ValueError(f"Non-finite values detected in kn ext channel: {path!r}")

    return SegmentKn(orders=orders, kn_abs=kn_abs.astype(complex), kn_cmp=kn_cmp.astype(complex), kn_ext=kn_ext, source_path=path)


def _wrap_arg_to_pm_pi_over_2(phi: np.ndarray) -> np.ndarray:
    """Wrap angles into [-pi/2, +pi/2] by adding/subtracting pi (legacy convention)."""
    out = np.asarray(phi, dtype=float).copy()
    out[out > (np.pi / 2.0)] -= np.pi
    out[out < (-np.pi / 2.0)] += np.pi
    return out


def _require_shape(x: np.ndarray, name: str, ndim: int) -> None:
    if np.asarray(x).ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {np.asarray(x).shape}")


def compute_legacy_kn_per_turn(
    *,
    df_abs_turns: np.ndarray,
    df_cmp_turns: np.ndarray,
    t_turns: np.ndarray,
    I_turns: np.ndarray,
    kn: SegmentKn,
    Rref_m: float,
    magnet_order: int,
    absCalib: float = 1.0,
    options: Iterable[str] = ("dri", "rot", "nor", "cel", "fed"),
    drift_mode: DriftMode = "legacy",
    skew_main: bool = False,
    eps_main: float = 1e-20,
    legacy_rotate_excludes_last: bool = True,
) -> LegacyKnPerTurn:
    """Compute legacy per-turn harmonics with $k_n$ application.

    Parameters
    ----------
    df_abs_turns, df_cmp_turns:
        Incremental signals, shape (n_turns, Ns).
    t_turns, I_turns:
        Measured time (s) and current (A) arrays, same shape.
    kn:
        Segment-level kn values (Abs/Cmp, optionally Ext).
    Rref_m:
        Reference radius in meters.
    magnet_order:
        Main field order m (e.g. 1 dipole, 2 quadrupole).
    absCalib:
        Absolute calibration factor applied to the main field (as in legacy C++).
    options:
        Iterable of tokens among: {"dit","dri","rot","cel","fed","nor"}.
    drift_mode:
        Drift mode for the "dri" option.
    skew_main:
        If True, main component is taken from Im(main_field) instead of Re(main_field)
        in the normalization step (legacy "skw" option).
    legacy_rotate_excludes_last:
        If True, replicate the legacy loop bounds which do not rotate the last harmonic.

    Returns
    -------
    LegacyKnPerTurn
        Per-turn calibrated complex harmonics and derived quantities.
    """

    opt = {str(x).strip().lower() for x in options}

    df_abs = np.asarray(df_abs_turns, dtype=float)
    df_cmp = np.asarray(df_cmp_turns, dtype=float)
    t = np.asarray(t_turns, dtype=float)
    I = np.asarray(I_turns, dtype=float)

    _require_shape(df_abs, "df_abs_turns", 2)
    if df_cmp.shape != df_abs.shape or t.shape != df_abs.shape or I.shape != df_abs.shape:
        raise ValueError("All per-turn inputs must have the same shape (n_turns, Ns)")

    n_turns, Ns = df_abs.shape
    m = int(magnet_order)
    if m <= 0:
        raise ValueError(f"magnet_order must be >0, got {magnet_order}")

    H = int(kn.orders.size)
    if Ns < (H + 1):
        raise ValueError(f"Need Ns >= H+1 to compute harmonics up to H={H} (DC + harmonics). Got Ns={Ns}.")
    if m > H:
        raise ValueError(f"magnet_order m={m} exceeds available harmonics H={H} from kn file")

    # --- di/dt correction (optional) ---
    if "dit" in opt:
        df_abs, df_cmp, _ = apply_di_dt_to_channels(df_abs, df_cmp, t, I)

    # For reporting (match C++: polyfit slope and mean current)
    I_mean = np.mean(I, axis=1)
    dI_dt = estimate_linear_slope_per_turn(t, I)
    duration = t[:, -1] - t[:, 0]
    time_median = np.median(t, axis=1)

    # --- integrate to flux (drift optional) ---
    drift_enabled = "dri" in opt
    flux_abs, _ = integrate_to_flux(df_abs, drift=drift_enabled, drift_mode=drift_mode, t_turns=t)
    flux_cmp, _ = integrate_to_flux(df_cmp, drift=drift_enabled, drift_mode=drift_mode, t_turns=t)

    # --- FFT and scaling to legacy f_n ---
    f_abs_full = (2.0 * np.fft.fft(flux_abs, axis=1)) / float(Ns)
    f_cmp_full = (2.0 * np.fft.fft(flux_cmp, axis=1)) / float(Ns)
    # drop DC, keep 1..H
    f_abs = f_abs_full[:, 1 : H + 1]
    f_cmp = f_cmp_full[:, 1 : H + 1]

    # --- apply kn ---
    Rref = float(Rref_m)
    if not np.isfinite(Rref) or Rref <= 0.0:
        raise ValueError(f"Rref_m must be finite and > 0, got {Rref_m}")

    idx = np.arange(H, dtype=float)  # 0..H-1 corresponds to n=1..H

    # sensitivity factors (vectorized)
    sens_abs = (1.0 / np.conj(np.asarray(kn.kn_abs, dtype=complex))) * (Rref ** idx)
    sens_cmp = (1.0 / np.conj(np.asarray(kn.kn_cmp, dtype=complex))) * (Rref ** idx)

    # broadcast: (H,) -> (n_turns,H)
    C_abs = f_abs * sens_abs[None, :]
    C_cmp = f_cmp * sens_cmp[None, :]

    # DB snapshot right after kn (legacy)
    C_abs_db = np.array(C_abs, copy=True)
    C_cmp_db = np.array(C_cmp, copy=True)

    # --- rotation reference computed post-kn, pre-rotation (legacy) ---
    c_m = C_abs[:, m - 1]
    mag_m = np.abs(c_m)
    arg_m = np.angle(c_m)
    bad_phi = (~np.isfinite(mag_m)) | (mag_m < float(eps_main)) | (~np.isfinite(arg_m))
    arg_wrapped = _wrap_arg_to_pm_pi_over_2(arg_m)
    phi_out = arg_wrapped / float(m)
    if np.any(bad_phi):
        phi_out = np.array(phi_out, copy=True)
        phi_out[bad_phi] = 0.0

    # --- rotation (optional) ---
    if "rot" in opt:
        # replicate legacy loop bounds (k = 1..H-1) by default
        k_max = H - 1 if legacy_rotate_excludes_last else H
        for k in range(1, k_max + 1):
            # harmonic order k corresponds to column k-1
            col = k - 1
            rot = np.exp(-1j * phi_out * float(k))
            C_abs[:, col] = rot * C_abs[:, col]
            C_cmp[:, col] = rot * C_cmp[:, col]

    # --- center location (optional) ---
    zR = np.zeros(n_turns, dtype=complex)
    z = np.zeros(n_turns, dtype=complex)
    x = np.zeros(n_turns, dtype=float)
    y = np.zeros(n_turns, dtype=float)
    if "cel" in opt:
        if m == 1:
            # legacy dipole special case uses compensated harmonics n=10 and n=11
            if H >= 11:
                Cn_1 = C_cmp[:, 9]
                Cn_2 = C_cmp[:, 10]
                zR = -(Cn_1 / (10.0 * Cn_2))
            else:
                zR[:] = np.nan + 1j * np.nan
        else:
            if m >= 2:
                Cn_1 = C_abs[:, m - 2]
                Cn_2 = C_abs[:, m - 1]
                zR = -(Cn_1 / ((m - 1.0) * Cn_2))
            else:
                zR[:] = np.nan + 1j * np.nan

        z = Rref * zR
        x = np.real(z)
        y = np.imag(z)

    # --- feeddown (optional) ---
    if "fed" in opt:
        # tmp[n] = sum_{k=n..H-1} comb(k,n) * zR^{k-n} * C[k]
        tmp_abs = np.zeros_like(C_abs)
        tmp_cmp = np.zeros_like(C_cmp)
        for n in range(H):
            acc_abs = np.zeros(n_turns, dtype=complex)
            acc_cmp = np.zeros(n_turns, dtype=complex)
            for k in range(n, H):
                coeff = float(math.comb(k, n))
                if k == n:
                    p = 1.0
                else:
                    p = zR ** (k - n)
                acc_abs = acc_abs + coeff * p * C_abs[:, k]
                acc_cmp = acc_cmp + coeff * p * C_cmp[:, k]
            tmp_abs[:, n] = acc_abs
            tmp_cmp[:, n] = acc_cmp
        C_abs = tmp_abs
        C_cmp = tmp_cmp

    # --- normalization (optional) ---
    main_field_db = C_abs_db[:, m - 1] * float(absCalib)
    main_field = C_abs[:, m - 1] * float(absCalib)
    if "nor" in opt:
        main_comp = np.imag(main_field) if bool(skew_main) else np.real(main_field)
        scale = np.full_like(main_comp, np.nan, dtype=float)
        ok = np.isfinite(main_comp) & (np.abs(main_comp) > float(eps_main))
        scale[ok] = 10000.0 / main_comp[ok]
        C_abs = C_abs * scale[:, None]
        C_cmp = C_cmp * scale[:, None]

    return LegacyKnPerTurn(
        orders=np.array(kn.orders, copy=True),
        C_abs=C_abs,
        C_cmp=C_cmp,
        C_abs_db=C_abs_db,
        C_cmp_db=C_cmp_db,
        phi_out_rad=phi_out,
        phi_bad=bad_phi,
        zR=zR,
        z_m=z,
        x_m=x,
        y_m=y,
        main_field=main_field,
        main_field_db=main_field_db,
        I_mean_A=I_mean,
        dI_dt_A_per_s=dI_dt,
        duration_s=duration,
        time_median_s=time_median,
    )


def merge_coefficients(
    *,
    C_abs: np.ndarray,
    C_cmp: np.ndarray,
    magnet_order: int,
    mode: str = "abs_main_cmp_others",
    per_order_choice: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge Abs/Cmp coefficient sets in an explicit, traceable way.

    This function **does not** attempt to decide the best merge. It applies
    exactly the user-selected policy.

    Parameters
    ----------
    C_abs, C_cmp:
        Complex arrays of shape (n_turns, H).
    magnet_order:
        Main field order m.
    mode:
        One of:
          - "abs_all"
          - "cmp_all"
          - "abs_main_cmp_others" (default)
          - "abs_upto_m_cmp_above"
          - "custom" (requires per_order_choice)
    per_order_choice:
        Optional explicit choice array of length H containing 0 (abs) or 1 (cmp).

    Returns
    -------
    C_merged, choice
        choice is length H with 0 (abs) or 1 (cmp), so the merge is auditable.
    """

    A = np.asarray(C_abs, dtype=complex)
    B = np.asarray(C_cmp, dtype=complex)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("C_abs and C_cmp must be the same 2D shape (n_turns, H)")

    n_turns, H = A.shape
    m = int(magnet_order)
    if not (1 <= m <= H):
        raise ValueError(f"magnet_order m must be in [1,{H}], got {magnet_order}")

    mode = str(mode).strip().lower()
    if mode == "abs_all":
        choice = np.zeros(H, dtype=int)
    elif mode == "cmp_all":
        choice = np.ones(H, dtype=int)
    elif mode == "abs_main_cmp_others":
        choice = np.ones(H, dtype=int)
        choice[m - 1] = 0
    elif mode == "abs_upto_m_cmp_above":
        choice = np.ones(H, dtype=int)
        choice[:m] = 0
    elif mode == "custom":
        if per_order_choice is None:
            raise ValueError("mode='custom' requires per_order_choice")
        choice = np.asarray(per_order_choice, dtype=int)
        if choice.shape != (H,):
            raise ValueError(f"per_order_choice must have shape ({H},), got {choice.shape}")
        if not np.all((choice == 0) | (choice == 1)):
            raise ValueError("per_order_choice values must be 0 (abs) or 1 (cmp)")
        # enforce main harmonic as abs by default policy (can be overridden upstream if desired)
        choice = np.array(choice, copy=True)
        choice[m - 1] = 0
    else:
        raise ValueError(f"Unknown merge mode: {mode!r}")

    merged = np.empty_like(A)
    for j in range(H):
        merged[:, j] = B[:, j] if choice[j] == 1 else A[:, j]
    return merged, choice
