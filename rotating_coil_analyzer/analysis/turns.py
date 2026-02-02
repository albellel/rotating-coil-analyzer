from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class TurnBlock:
    """Turn-reshaped view of a :class:`~rotating_coil_analyzer.models.frames.SegmentFrame`.

    Arrays are shaped ``(n_turns, Ns)`` where ``Ns = samples_per_turn``.

    Notes
    -----
    - The project forbids synthetic/modified time. Therefore ``t`` is returned exactly
      as stored in the input data (it may be non-monotonic or contain NaNs for plateau data).
    - For any analysis that depends on the rotation phase, use the implicit angular grid
      per turn (sample index within turn), not time.
    """

    Ns: int
    n_turns: int

    t: np.ndarray  # (n_turns, Ns)
    df_abs: np.ndarray  # (n_turns, Ns)
    df_cmp: np.ndarray  # (n_turns, Ns)
    I: np.ndarray  # (n_turns, Ns)

    plateau_id: Optional[np.ndarray] = None  # (n_turns,)
    plateau_step: Optional[np.ndarray] = None  # (n_turns,)
    plateau_I_hint: Optional[np.ndarray] = None  # (n_turns,)

    warnings: tuple[str, ...] = ()


def _reshape_1d(x: np.ndarray, n_turns: int, Ns: int) -> np.ndarray:
    """Reshape 1D array to (n_turns, Ns) without copying where possible."""
    x = np.asarray(x)
    need = n_turns * Ns
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    if x.size < need:
        raise ValueError(f"Array too short: size={x.size}, need={need}")
    if x.size != need:
        x = x[:need]
    return x.reshape((n_turns, Ns))


def split_into_turns(
    seg: SegmentFrame,
    *,
    columns: Iterable[str] = ("t", "df_abs", "df_cmp", "I"),
    plateau_key: str = "plateau_id",
    strict_plateau_turns: bool = True,
) -> TurnBlock:
    """Convert a SegmentFrame into turn-shaped arrays.

    Parameters
    ----------
    seg:
        Input frame from ingest.
    columns:
        Required sample-wise columns. Defaults are the Catalog contract columns.
    plateau_key:
        Name of the sample-wise plateau id column (plateau data). If absent, plateau metadata
        will be returned as None.
    strict_plateau_turns:
        If True and plateau_key exists, enforce that plateau id is constant within each
        turn. This is the "no turn crosses plateau" invariant.

    Returns
    -------
    TurnBlock
        Turn-shaped arrays and optional per-turn plateau metadata.
    """
    Ns = int(seg.samples_per_turn)
    if Ns <= 0:
        raise ValueError("samples_per_turn must be > 0")

    df = seg.df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in SegmentFrame.df: {missing}")

    n_samples = int(len(df))
    if n_samples != seg.n_turns * Ns:
        raise ValueError(
            f"SegmentFrame length invariant broken: len(df)={n_samples}, "
            f"n_turns*Ns={seg.n_turns * Ns}"
        )

    n_turns = int(seg.n_turns)

    # Sample-wise arrays
    t = _reshape_1d(df["t"].to_numpy(), n_turns, Ns)
    df_abs = _reshape_1d(df["df_abs"].to_numpy(), n_turns, Ns)
    df_cmp = _reshape_1d(df["df_cmp"].to_numpy(), n_turns, Ns)
    I = _reshape_1d(df["I"].to_numpy(), n_turns, Ns)

    warnings = list(getattr(seg, "warnings", ()) or ())

    plateau_id = None
    plateau_step = None
    plateau_I_hint = None

    if plateau_key in df.columns:
        pid_s = _reshape_1d(df[plateau_key].to_numpy(), n_turns, Ns)
        pid0 = pid_s[:, 0]

        if strict_plateau_turns:
            # Check constant within each turn. Use exact equality (ids should be integers stored as float).
            same = np.all(pid_s == pid0[:, None], axis=1)
            if not np.all(same):
                bad_turns = np.where(~same)[0]
                raise ValueError(
                    f"Plateau id changes within a turn (violates contract) in turns: {bad_turns[:20].tolist()}"
                )

        plateau_id = pid0

        # Optional metadata (present in our plateau reader)
        if "plateau_step" in df.columns:
            pstep_s = _reshape_1d(df["plateau_step"].to_numpy(), n_turns, Ns)
            plateau_step = pstep_s[:, 0]
        if "plateau_I_hint" in df.columns:
            pih_s = _reshape_1d(df["plateau_I_hint"].to_numpy(), n_turns, Ns)
            plateau_I_hint = pih_s[:, 0]

    return TurnBlock(
        Ns=Ns,
        n_turns=n_turns,
        t=t,
        df_abs=df_abs,
        df_cmp=df_cmp,
        I=I,
        plateau_id=plateau_id,
        plateau_step=plateau_step,
        plateau_I_hint=plateau_I_hint,
        warnings=tuple(warnings),
    )
