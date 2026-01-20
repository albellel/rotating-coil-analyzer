from __future__ import annotations

"""Compute complex calibration coefficients ($k_n$) from a measurement-head CSV.

This implements the same geometry-based computation as the legacy C++ analyzer
(`MatlabAnalyzerRotCoil.cpp`, functions ``loadHeadKn(...)`` and
``calculateHeadKn(...)``).

The measurement-head CSV is expected to follow the CERN-style header used by
the legacy tool. The computation supports:

* Warm vs cold geometry scaling.
* Selecting the *design radius* when the calibrated radius is missing.
* Tangential vs radial coil orientation handling for the complex phase factor.

Hard constraint
---------------
This is purely a geometry transform; it does not touch timestamps.
"""

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---- Legacy header (must match legacy C++ tokenization) ----
LEGACY_HEAD_HEADER: Tuple[str, ...] = (
    "Measurement Head",
    "Array Position",
    "Array Code",
    "Array Name",
    "Coil Position",
    "Coil Code",
    "Coil Name",
    "Number of Turns",
    "Coil Inner Width [m]",
    "Coil Inner Length [m]",
    "Winding Thickness [m]",
    "Magnetic Surface [m]",
    "Radius (calibrated) [m]",
    "Alpha [rad]",
    "Beta [rad]",
    "Tilt Angle [rad]",
    "Radius (design) [m]",
    "Magnetic Surface (Single Coil calibration) [m]",
    "Z position in Measurement Head",
)


KnIndex = Tuple[int, int]  # (array_pos, coil_pos)


@dataclass(frozen=True)
class CoilGeom:
    """Geometry and calibration parameters for a single coil."""

    Nt: float
    Win: float
    Lin: float
    T: float
    S: float
    ro: float
    alpha: float
    beta: float
    fi: float
    pZ: float


@dataclass(frozen=True)
class HeadKnData:
    """Per-coil $k_n$ computed from a measurement-head CSV."""

    kn_by_index: Dict[KnIndex, np.ndarray]  # complex array (H,) per coil
    magnetic_length_by_index: Dict[KnIndex, float]
    array_zpos_by_index: Dict[KnIndex, float]
    source_path: str
    warm_geometry: bool
    use_design_radius: bool
    n_multipoles: int


def _gamma_function(za: complex, dz: complex, n: int) -> complex:
    """Legacy gammaFunction(za, dz, n) with n as harmonic order (1-based)."""
    zeta = dz / za
    zeta_c = np.conj(dz) / za

    nt1 = (1.0 + zeta) ** (n + 2)
    nt2 = -((1.0 + zeta_c) ** (n + 2))
    nt3 = (1.0 - zeta) ** (n + 2)
    nt4 = -((1.0 - zeta_c) ** (n + 2))

    zeta2 = zeta**2
    zeta_c2 = zeta_c**2
    div = complex((n + 1) * (n + 2)) * (zeta2 - zeta_c2)
    return (nt1 + nt2 + nt3 + nt4) / div


def _csi_n(dz: complex, za1: complex, za2: complex, z1: complex, z2: complex, n: int) -> complex:
    """Legacy CsiN(dZ, Za1, Za2, Z1, Z2, n) with n as harmonic order (1-based)."""
    gz1 = _gamma_function(za1, dz, n)
    gz2 = _gamma_function(za2, dz, n)
    mt1 = gz2 * (z2**n)
    r1 = gz1 / gz2
    r2 = (z1 / z2) ** n
    return mt1 * (1.0 - (r1 * r2))


def _param_for_kn_computing(v: CoilGeom, *, warm: bool) -> Tuple[float, float, complex, complex, complex, complex, complex, float, float, float, float]:
    """Legacy paramForKnComputing(...). Returns tuple used by calculateHeadKn."""
    Nt = float(v.Nt)
    T = float(v.T)
    alpha = float(v.alpha)
    beta = float(v.beta)
    fi = float(v.fi)

    if warm:
        Win = float(v.Win) * 0.999999
        Lin = float(v.Lin)
        S = float(v.S)
        ro = float(v.ro)
        pZ = float(v.pZ)
    else:
        Win = float(v.Win) * 0.997
        Lin = float(v.Lin) * 0.997
        S = float(v.S) * 0.995
        ro = float(v.ro) * 0.997
        pZ = float(v.pZ) * 0.997

    w = 0.5 * ((Win - Lin) + math.sqrt((Win - Lin) ** 2 + (4.0 * S / Nt)))
    L = 0.5 * ((Lin - Win) + math.sqrt((Lin - Win) ** 2 + (4.0 * S / Nt)))

    z0 = ro * complex(math.cos(alpha), math.sin(alpha))
    dz0 = 0.5 * w * complex(math.cos(fi), math.sin(fi))

    Z1 = z0 - dz0
    Z2 = z0 + dz0

    dZ = complex(0.5 * (w - Win), 0.5 * T)
    Za1 = Z1 * complex(math.cos(-fi), math.sin(-fi))
    Za2 = Z2 * complex(math.cos(-fi), math.sin(-fi))

    return Nt, float(L), dZ, Za1, Za2, Z1, Z2, alpha, beta, fi, pZ


def compute_head_kn_from_csv(
    csv_path: str,
    *,
    warm_geometry: bool = True,
    n_multipoles: int = 15,
    use_design_radius: bool = True,
    strict_header: bool = True,
) -> HeadKnData:
    """Compute per-coil $k_n$ vectors from a measurement-head CSV.

    Parameters mirror the legacy C++ signature:
    ``loadHeadKn(csvPath, warm, N_multipoles, useDesignRadius, ...)``.
    """

    # Pandas with dtype=str prevents NaN -> float pitfalls for empty cells.
    df = pd.read_csv(csv_path, dtype=str)
    if strict_header:
        got = tuple(df.columns)
        if got[: len(LEGACY_HEAD_HEADER)] != LEGACY_HEAD_HEADER:
            raise ValueError(
                "KN Head file definition changed (header mismatch). "
                "If you want to proceed anyway, disable strict_header."
            )

    # Required columns by name (robust to extra columns appended by future formats)
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            raise ValueError(f"Missing required column: {name}")
        return df[name]

    kn_by_index: Dict[KnIndex, np.ndarray] = {}
    magnetic_length_by_index: Dict[KnIndex, float] = {}
    array_zpos_by_index: Dict[KnIndex, float] = {}

    for i, row in df.iterrows():
        # Parse indices
        try:
            array_pos = int(str(row["Array Position"]).strip())
            coil_pos = int(str(row["Coil Position"]).strip())
        except Exception:
            # Keep legacy behavior: skip ill-defined indices.
            continue

        # Array Z position (optional but present in legacy files)
        try:
            pZ = float(str(row.get("Z position in Measurement Head", "")).strip())
        except Exception:
            pZ = float("nan")

        # Build geometry array (mirrors the C++ arr[0..9])
        def f(name: str) -> float:
            s = str(row.get(name, "")).strip()
            if s == "" or s.lower() == "nan":
                return float("nan")
            return float(s)

        Nt = f("Number of Turns")
        Win = f("Coil Inner Width [m]")
        Lin = f("Coil Inner Length [m]")
        T = f("Winding Thickness [m]")

        # Magnetic surface: if empty, legacy may fall back to single-coil calibration.
        S = f("Magnetic Surface [m]")
        if not np.isfinite(S):
            S_sc = f("Magnetic Surface (Single Coil calibration) [m]")
            if np.isfinite(S_sc):
                S = S_sc

        # Radius selection
        ro = f("Radius (calibrated) [m]")
        if (not np.isfinite(ro)) and use_design_radius:
            ro = f("Radius (design) [m]")

        alpha = f("Alpha [rad]")
        beta = f("Beta [rad]")
        fi = f("Tilt Angle [rad]")

        # Some legacy files use empty strings for beta/fi; treat empty as 0.
        if not np.isfinite(beta):
            beta = 0.0
        if not np.isfinite(fi):
            fi = 0.0

        if not (np.isfinite(Nt) and np.isfinite(Win) and np.isfinite(Lin) and np.isfinite(T) and np.isfinite(S) and np.isfinite(ro) and np.isfinite(alpha)):
            # Skip incomplete rows.
            continue

        geom = CoilGeom(
            Nt=Nt,
            Win=Win,
            Lin=Lin,
            T=T,
            S=S,
            ro=ro,
            alpha=alpha,
            beta=beta,
            fi=fi,
            pZ=pZ,
        )

        Nt2, L, dZ, Za1, Za2, Z1, Z2, alpha2, beta2, fi2, pZ2 = _param_for_kn_computing(geom, warm=warm_geometry)

        nl = Nt2 * L
        vet_kn = np.zeros(int(n_multipoles), dtype=np.complex128)

        eps = 0.1
        for p in range(1, int(n_multipoles) + 1):
            # mult: coil orientation handling
            if abs(alpha2) < eps or abs(math.pi - alpha2) < eps:
                mult = 1.0 + 0.0j
            elif abs(math.pi / 2 - alpha2) < eps or abs(3 * math.pi / 2 - alpha2) < eps:
                mult = complex(math.cos(-(math.pi / 2) * p), math.sin(-(math.pi / 2) * p))
            else:
                raise ValueError(
                    f"Coil is not tangential or radial (alpha={alpha2}). Cannot calculate k_n automatically."
                )

            csin = _csi_n(dZ, Za1, Za2, Z1, Z2, p)
            vet_kn[p - 1] = (nl / float(p)) * csin * mult

        idx = (array_pos, coil_pos)
        kn_by_index[idx] = vet_kn
        magnetic_length_by_index[idx] = float(L)
        array_zpos_by_index[idx] = float(pZ2)

    return HeadKnData(
        kn_by_index=kn_by_index,
        magnetic_length_by_index=magnetic_length_by_index,
        array_zpos_by_index=array_zpos_by_index,
        source_path=csv_path,
        warm_geometry=bool(warm_geometry),
        use_design_radius=bool(use_design_radius),
        n_multipoles=int(n_multipoles),
    )


def parse_connection(conn: str) -> List[Tuple[float, KnIndex]]:
    """Parse a connection string like ``"1.1-1.3+2*1.2"``.

    Each term is ``[coef*]A.C`` where ``A`` is array position and ``C`` is coil
    position. Coef defaults to 1.
    """

    s = conn.strip()
    if not s:
        return []

    # Normalize separators: replace '-' with '+-' but keep leading '-'
    s2 = s.replace(" ", "")
    if s2.startswith("-"):
        s2 = "0" + s2
    s2 = s2.replace("-", "+-")
    parts = [p for p in s2.split("+") if p]

    out: List[Tuple[float, KnIndex]] = []
    for p in parts:
        coef = 1.0
        token = p

        # Optional explicit coefficient: "2*1.2" or "-0.5*3.1"
        if "*" in token:
            c_str, token = token.split("*", 1)
            if c_str in ("", "+"):
                coef = 1.0
            elif c_str == "-":
                coef = -1.0
            else:
                coef = float(c_str)

        # Otherwise a leading sign on the ID is allowed: "-1.3"
        if token.startswith("-"):
            coef *= -1.0
            token = token[1:]
        if token.startswith("+"):
            token = token[1:]

        id_str = token
        if "." not in id_str:
            raise ValueError(f"Connection term must be A.C (got {id_str})")
        a_str, c_str2 = id_str.split(".", 1)
        idx = (int(a_str), int(c_str2))
        out.append((float(coef), idx))

    return out


def compute_segment_kn_from_head(
    head: HeadKnData,
    *,
    abs_connection: str,
    cmp_connection: str,
    ext_connection: Optional[str] = None,
    source_label: Optional[str] = None,
) -> "rotating_coil_analyzer.analysis.kn_pipeline.SegmentKn":
    """Combine per-coil $k_n$ into a segment-level :class:`~SegmentKn`.

    The combination is a simple linear sum of the coils specified by the
    connection strings.
    """

    # Local import to avoid circular dependency.
    from .kn_pipeline import SegmentKn

    H = int(head.n_multipoles)
    orders = np.arange(1, H + 1, dtype=int)

    def combine(conn: str) -> np.ndarray:
        terms = parse_connection(conn)
        if not terms:
            return np.zeros(H, dtype=np.complex128)
        acc = np.zeros(H, dtype=np.complex128)
        for coef, idx in terms:
            if idx not in head.kn_by_index:
                raise KeyError(f"Coil {idx[0]}.{idx[1]} not found in head CSV")
            acc = acc + complex(coef) * head.kn_by_index[idx]
        return acc

    kn_abs = combine(abs_connection)
    kn_cmp = combine(cmp_connection)
    kn_ext = None
    if ext_connection is not None and ext_connection.strip() != "":
        kn_ext = combine(ext_connection)

    src = source_label or f"head_csv:{head.source_path}"
    return SegmentKn(orders=orders, kn_abs=kn_abs, kn_cmp=kn_cmp, kn_ext=kn_ext, source_path=src)


def write_segment_kn_txt(kn: "rotating_coil_analyzer.analysis.kn_pipeline.SegmentKn", path: str) -> None:
    """Write a segment $k_n$ TXT file compatible with :func:`load_segment_kn_txt`."""

    with open(path, "w", encoding="utf-8") as f:
        for i, n in enumerate([int(x) for x in kn.orders]):
            a = kn.kn_abs[i]
            c = kn.kn_cmp[i]
            if kn.kn_ext is not None:
                e = kn.kn_ext[i]
                f.write(f"{a.real:.17g} {a.imag:.17g} {c.real:.17g} {c.imag:.17g} {e.real:.17g} {e.imag:.17g}\n")
            else:
                f.write(f"{a.real:.17g} {a.imag:.17g} {c.real:.17g} {c.imag:.17g}\n")
