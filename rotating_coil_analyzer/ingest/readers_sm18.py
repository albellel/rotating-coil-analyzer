from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings as pywarnings

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class _FloatMatrixFormat:
    name: str
    dtype: np.dtype
    ncols: int
    # ncols==4: [t, df_abs, df_cmp, I]
    # ncols==3: [df_abs, df_cmp, I] and time reconstructed


@dataclass(frozen=True)
class _StructFormat:
    name: str
    dtype: np.dtype
    # fields: tick, df_abs, df_cmp, I


class Sm18CorrSigsReader:
    """
    Strict reader for corr_sigs segment files.

    Output df columns (always):
      - t      : time [s] (strictly increasing)
      - df_abs : abs delta-flux
      - df_cmp : cmp delta-flux
      - I      : current [A]
    """

    def __init__(self) -> None:
        self._float_candidates = (
            _FloatMatrixFormat("float64_le_4col", np.dtype("<f8"), 4),
            _FloatMatrixFormat("float32_le_4col", np.dtype("<f4"), 4),
            _FloatMatrixFormat("float64_be_4col", np.dtype(">f8"), 4),
            _FloatMatrixFormat("float32_be_4col", np.dtype(">f4"), 4),
            # explicit “no time stored”: reconstruct time from dt_nom
            _FloatMatrixFormat("float32_le_3col_notime", np.dtype("<f4"), 3),
            _FloatMatrixFormat("float64_le_3col_notime", np.dtype("<f8"), 3),
        )

        self._struct_candidates = (
            _StructFormat(
                "tick_i32_le__3xf32_le",
                np.dtype([("tick", "<i4"), ("df_abs", "<f4"), ("df_cmp", "<f4"), ("I", "<f4")]),
            ),
            _StructFormat(
                "tick_u32_le__3xf32_le",
                np.dtype([("tick", "<u4"), ("df_abs", "<f4"), ("df_cmp", "<f4"), ("I", "<f4")]),
            ),
        )

    @staticmethod
    def _dt_nom(samples_per_turn: int, shaft_speed_rpm: Optional[float]) -> Optional[float]:
        if shaft_speed_rpm is None:
            return None
        v = float(shaft_speed_rpm)
        if v == 0.0:
            return None
        Tnom = 60.0 / abs(v)
        return Tnom / float(samples_per_turn)

    @staticmethod
    def _validate_time_axis(t: np.ndarray, dt_nom: Optional[float]) -> Tuple[bool, str]:
        dt = np.diff(t)
        nonpos = int(np.sum(dt <= 0))
        if nonpos > 0:
            return False, f"time not strictly increasing: count(dt<=0)={nonpos}"

        if dt_nom is not None and dt.size > 0:
            dt_med = float(np.median(dt))
            rel = abs(dt_med - dt_nom) / dt_nom
            # identification tolerance (not QC): accept if within ±50%
            if rel > 0.5:
                return False, f"dt median mismatch: dt_med={dt_med:.6g}s vs dt_nom={dt_nom:.6g}s (rel={rel:.3%})"

        return True, "ok"

    @staticmethod
    def _trim_trailing_nonfinite(
        t: np.ndarray, df_abs: np.ndarray, df_cmp: np.ndarray, I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
        finite = np.isfinite(t) & np.isfinite(df_abs) & np.isfinite(df_cmp) & np.isfinite(I)
        if finite.all():
            return t, df_abs, df_cmp, I, None
        if not np.any(finite):
            return t[:0], df_abs[:0], df_cmp[:0], I[:0], "reject (no finite rows)"
        first_bad = int(np.argmax(~finite))
        if not finite[:first_bad].all():
            return t[:0], df_abs[:0], df_cmp[:0], I[:0], "reject (non-finite inside data, not only trailing)"
        n_bad = int((~finite).sum())
        msg = f"note (trim trailing invalid rows: first_bad={first_bad}, n_bad={n_bad})"
        return t[:first_bad], df_abs[:first_bad], df_cmp[:first_bad], I[:first_bad], msg

    @staticmethod
    def _reshape_float_matrix(path: Path, dtype: np.dtype, ncols: int) -> Optional[np.ndarray]:
        b = path.stat().st_size
        item = dtype.itemsize
        row_bytes = ncols * item
        if row_bytes <= 0 or (b % row_bytes) != 0:
            return None
        nrows = b // row_bytes
        data = np.fromfile(path, dtype=dtype, count=nrows * ncols)
        if data.size != nrows * ncols:
            return None
        return data.reshape((nrows, ncols))

    @staticmethod
    def _read_struct(path: Path, dtype: np.dtype) -> Optional[np.ndarray]:
        b = path.stat().st_size
        if (b % dtype.itemsize) != 0:
            return None
        nrows = b // dtype.itemsize
        return np.fromfile(path, dtype=dtype, count=nrows)

    @staticmethod
    def _ticks_to_time_seconds(tick: np.ndarray, dt_nom: Optional[float]) -> Tuple[Optional[np.ndarray], str]:
        if tick.size < 2:
            return None, "reject (tick too short)"

        tick64 = tick.astype(np.int64, copy=False)
        d = np.diff(tick64)

        if np.any(d <= 0):
            return None, f"reject (tick not strictly increasing: count(dtick<=0)={int(np.sum(d<=0))})"

        if dt_nom is None:
            t = tick64 - tick64[0]
            return t.astype(np.float64), "ok (tick units)"

        d_med = float(np.median(d))
        if d_med <= 0:
            return None, "reject (invalid tick median increment)"

        scale = float(dt_nom) / d_med
        t = (tick64 - tick64[0]).astype(np.float64) * scale
        return t, "ok"

    def read(
        self,
        file_path: Path,
        *,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: Optional[float] = None,
    ) -> SegmentFrame:
        p = Path(file_path)
        Ns = int(samples_per_turn)
        dt_nom = self._dt_nom(Ns, shaft_speed_rpm)

        warnings: List[str] = []
        reports: List[str] = []

        if p.suffix.lower() == ".txt":
            arr = np.loadtxt(p)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"{p.name}: txt format unsupported (need >=3 columns).")
            t = arr[:, 0]
            df_abs = arr[:, 1]
            df_cmp = arr[:, 2]
            I = arr[:, 3] if arr.shape[1] >= 4 else np.zeros_like(t)
            df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I})
            return self._postprocess(df, run_id, segment, Ns, shaft_speed_rpm, warnings, source_path=p)

        if p.suffix.lower() != ".bin":
            raise ValueError(f"{p.name}: unsupported extension {p.suffix!r} (expected .bin or .txt).")

        # 1) struct formats (tick + 3 floats)
        for cand in self._struct_candidates:
            try:
                with pywarnings.catch_warnings():
                    pywarnings.simplefilter("error", RuntimeWarning)

                    arr = self._read_struct(p, cand.dtype)
                    if arr is None:
                        reports.append(f"{cand.name}: reject (filesize not divisible by struct size)")
                        continue

                    tick = arr["tick"]
                    df_abs = arr["df_abs"].astype(np.float64, copy=False)
                    df_cmp = arr["df_cmp"].astype(np.float64, copy=False)
                    I = arr["I"].astype(np.float64, copy=False)

                    t, tick_reason = self._ticks_to_time_seconds(tick, dt_nom)
                    if t is None:
                        reports.append(f"{cand.name}: {tick_reason}")
                        continue

                    t, df_abs, df_cmp, I, trim_msg = self._trim_trailing_nonfinite(t, df_abs, df_cmp, I)
                    if trim_msg:
                        if trim_msg.startswith("reject"):
                            reports.append(f"{cand.name}: {trim_msg}")
                            continue
                        reports.append(f"{cand.name}: {trim_msg}")

                    ok, reason = self._validate_time_axis(t, dt_nom)
                    if not ok:
                        reports.append(f"{cand.name}: reject ({reason})")
                        continue

                    reports.append(f"{cand.name}: ACCEPT")
                    df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I})
                    warnings.append(f"Selected binary format: {cand.name}")
                    return self._postprocess(df, run_id, segment, Ns, shaft_speed_rpm, warnings, source_path=p)

            except RuntimeWarning as e:
                reports.append(f"{cand.name}: reject (runtime warning during decode: {e})")
                continue

        # 2) float-matrix formats
        for cand in self._float_candidates:
            try:
                with pywarnings.catch_warnings():
                    pywarnings.simplefilter("error", RuntimeWarning)

                    mat = self._reshape_float_matrix(p, cand.dtype, cand.ncols)
                    if mat is None:
                        reports.append(f"{cand.name}: reject (filesize not divisible by ncols*itemsize)")
                        continue

                    if cand.ncols == 4:
                        t = mat[:, 0].astype(np.float64, copy=False)
                        df_abs = mat[:, 1].astype(np.float64, copy=False)
                        df_cmp = mat[:, 2].astype(np.float64, copy=False)
                        I = mat[:, 3].astype(np.float64, copy=False)
                    elif cand.ncols == 3:
                        if dt_nom is None:
                            reports.append(f"{cand.name}: reject (no time in file and dt_nom unknown)")
                            continue
                        n = mat.shape[0]
                        t = (np.arange(n, dtype=np.float64) * float(dt_nom))
                        df_abs = mat[:, 0].astype(np.float64, copy=False)
                        df_cmp = mat[:, 1].astype(np.float64, copy=False)
                        I = mat[:, 2].astype(np.float64, copy=False)
                    else:
                        reports.append(f"{cand.name}: reject (unsupported ncols)")
                        continue

                    t, df_abs, df_cmp, I, trim_msg = self._trim_trailing_nonfinite(t, df_abs, df_cmp, I)
                    if trim_msg:
                        if trim_msg.startswith("reject"):
                            reports.append(f"{cand.name}: {trim_msg}")
                            continue
                        reports.append(f"{cand.name}: {trim_msg}")

                    ok, reason = self._validate_time_axis(t, dt_nom)
                    if not ok:
                        reports.append(f"{cand.name}: reject ({reason})")
                        continue

                    reports.append(f"{cand.name}: ACCEPT")
                    df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I})
                    warnings.append(f"Selected binary format: {cand.name}")
                    return self._postprocess(df, run_id, segment, Ns, shaft_speed_rpm, warnings, source_path=p)

            except RuntimeWarning as e:
                reports.append(f"{cand.name}: reject (runtime warning during decode: {e})")
                continue

        msg = "No supported binary format validated for this file.\n" + "\n".join(reports[:120])
        raise ValueError(msg)

    @staticmethod
    def _postprocess(
        df: pd.DataFrame,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: Optional[float],
        warnings: List[str],
        *,
        source_path: Path,
    ) -> SegmentFrame:
        Ns = int(samples_per_turn)

        raw = int(len(df))
        rem = raw % Ns
        if rem != 0:
            df = df.iloc[: raw - rem].reset_index(drop=True)
            warnings.append(f"Trimmed to multiple of samples_per_turn: {raw} -> {len(df)} (remainder={rem})")

        n_turns = len(df) // Ns if Ns > 0 else 0

        t = df["t"].to_numpy()
        if t.size >= 2:
            dt = np.diff(t)
            warnings.append(
                f"dt stats [s]: min={float(np.min(dt)):.6g}, median={float(np.median(dt)):.6g}, max={float(np.max(dt)):.6g}"
            )

        if shaft_speed_rpm is not None and float(shaft_speed_rpm) != 0.0 and n_turns >= 1:
            Tnom = 60.0 / abs(float(shaft_speed_rpm))
            t2 = t[: n_turns * Ns].reshape(n_turns, Ns)
            Tturn = t2[:, -1] - t2[:, 0]
            warnings.append(
                f"turn duration T [s]: min={float(np.min(Tturn)):.6g}, median={float(np.median(Tturn)):.6g}, max={float(np.max(Tturn)):.6g}"
            )
            warnings.append(f"nominal T_nom=60/|v|={Tnom:.6g} s (v={shaft_speed_rpm} rpm)")

        # IMPORTANT: pass source_path (your SegmentFrame requires it)
        return SegmentFrame(
            source_path=Path(source_path),
            run_id=run_id,
            segment=segment,
            samples_per_turn=Ns,
            n_turns=int(n_turns),
            df=df,
            warnings=tuple(warnings),
        )
