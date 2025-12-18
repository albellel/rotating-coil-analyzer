from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class Sm18ReaderConfig:
    """
    Reader configuration.

    strict_time:
      - True: require strictly increasing finite time vector after trimming trailing invalid rows.
      - False: allow non-monotonic time (not recommended).
    dt_rel_tol:
      Relative tolerance for dt_median vs dt_nominal (computed from |v| and samples_per_turn).
      Example: 0.2 means ±20%.
    max_currents:
      Try formats with 1..max_currents current channels (total columns = 3 + n_currents).
    """
    strict_time: bool = True
    dt_rel_tol: float = 0.20
    max_currents: int = 4
    dtype_candidates: Tuple[np.dtype, ...] = (np.dtype(np.float64), np.dtype(np.float32))


class Sm18CorrSigsReader:
    """
    Reader for *_corr_sigs_*.bin files.

    IMPORTANT: This reader never synthesizes time. The first column must contain a valid time axis.
    """
    def __init__(self, config: Optional[Sm18ReaderConfig] = None) -> None:
        self.config = config or Sm18ReaderConfig()

    def read(
        self,
        file_path: str | Path,
        *,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: float,
    ) -> SegmentFrame:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))

        Ns = int(samples_per_turn)
        if Ns <= 0:
            raise ValueError("samples_per_turn must be > 0")

        best = self._infer_and_load(path, Ns=Ns, shaft_speed_rpm=float(shaft_speed_rpm))
        mat, ncols, warnings = best

        # Columns: t, df_abs, df_cmp, currents...
        t = mat[:, 0]
        df_abs = mat[:, 1]
        df_cmp = mat[:, 2]
        if ncols >= 4:
            I = mat[:, 3]
        else:
            I = np.full_like(t, np.nan, dtype=np.float64)

        df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I})

        n_turns = int(len(df) // Ns)

        return SegmentFrame(
            source_path=path,
            run_id=run_id,
            segment=segment,
            samples_per_turn=Ns,
            n_turns=n_turns,
            df=df,
            warnings=tuple(warnings),
        )

    def _infer_and_load(
        self,
        path: Path,
        *,
        Ns: int,
        shaft_speed_rpm: float,
    ) -> Tuple[np.ndarray, int, List[str]]:
        """
        Infer dtype and column count from file size and timing consistency.
        Returns (mat_float64, ncols, warnings).
        """
        cfg = self.config
        warnings: List[str] = []

        file_size = path.stat().st_size
        if file_size <= 0:
            raise ValueError("Empty file.")

        v = abs(float(shaft_speed_rpm))
        dt_nom: Optional[float] = None
        if v > 0:
            Tnom = 60.0 / v
            dt_nom = Tnom / float(Ns)

        reports: List[str] = []
        best_score = -1e300
        best: Optional[Tuple[np.ndarray, int, List[str]]] = None

        for dtype in cfg.dtype_candidates:
            bps = int(dtype.itemsize)
            if file_size % bps != 0:
                continue
            n_scalars = file_size // bps

            # total columns = 3 + n_currents
            for n_curr in range(1, max(1, cfg.max_currents) + 1):
                ncols = 3 + n_curr
                if n_scalars % ncols != 0:
                    continue

                # Load and reshape
                arr = np.fromfile(path, dtype=dtype)
                if arr.size != n_scalars:
                    # Extremely unlikely; but keep safe.
                    arr = arr[:n_scalars]
                mat = arr.reshape(-1, ncols).astype(np.float64, copy=False)

                ok, score, rep, local_warnings, mat_out = self._validate_candidate(
                    mat,
                    Ns=Ns,
                    dt_nom=dt_nom,
                    strict_time=cfg.strict_time,
                    dt_rel_tol=cfg.dt_rel_tol,
                )
                reports.append(f"{dtype.name}_le_{ncols}col: {rep}")
                if ok and score > best_score:
                    best_score = score
                    warnings2 = warnings + local_warnings + [f"Selected binary format: {dtype.name}_le_{ncols}col"]
                    best = (mat_out, ncols, warnings2)

        if best is None:
            msg = "No supported binary format validated for this file.\n" + "\n".join(reports[:80])
            raise ValueError(msg)

        return best

    def _validate_candidate(
        self,
        mat: np.ndarray,
        *,
        Ns: int,
        dt_nom: Optional[float],
        strict_time: bool,
        dt_rel_tol: float,
    ) -> Tuple[bool, float, str, List[str], np.ndarray]:
        """
        Validate candidate by:
        - trimming trailing invalid rows (non-finite in t/df_abs/df_cmp)
        - enforcing strictly increasing time (if strict_time)
        - enforcing dt_median close to dt_nom (if provided)

        Returns (ok, score, report, warnings, mat_trimmed)
        """
        warnings: List[str] = []

        if mat.shape[1] < 3:
            return False, -1e300, "reject (ncols<3)", warnings, mat

        t = mat[:, 0]
        df_abs = mat[:, 1]
        df_cmp = mat[:, 2]

        finite = np.isfinite(t) & np.isfinite(df_abs) & np.isfinite(df_cmp)
        if not finite.all():
            bad_idx = np.flatnonzero(~finite)
            first_bad = int(bad_idx[0])
            # Do not accept NaNs/Infs in the middle
            if not finite[:first_bad].all():
                return False, -1e300, "reject (non-finite in middle)", warnings, mat
            n_bad = int(len(bad_idx))
            warnings.append(f"Trimmed trailing invalid rows: first_bad_index={first_bad}, n_bad={n_bad}")
            mat = mat[:first_bad, :]

        if mat.shape[0] < Ns:
            return False, -1e300, "reject (too few rows after trim)", warnings, mat

        # Trim to a multiple of Ns (drop partial last turn)
        remainder = int(mat.shape[0] % Ns)
        if remainder != 0:
            warnings.append(f"Trimmed to multiple of samples_per_turn: {mat.shape[0]} -> {mat.shape[0] - remainder}")
            mat = mat[: mat.shape[0] - remainder, :]

        t = mat[:, 0]
        if not np.isfinite(t).all():
            return False, -1e300, "reject (non-finite time after trim)", warnings, mat

        dt = np.diff(t)
        if not np.isfinite(dt).all():
            return False, -1e300, "reject (non-finite dt)", warnings, mat

        n_nonpos = int(np.count_nonzero(dt <= 0.0))
        if strict_time and n_nonpos > 0:
            return False, -1e300, f"reject (time not strictly increasing: count(dt<=0)={n_nonpos})", warnings, mat

        dt_med = float(np.median(dt)) if dt.size else float("nan")
        dt_min = float(np.min(dt)) if dt.size else float("nan")
        dt_max = float(np.max(dt)) if dt.size else float("nan")

        # Score: closeness to dt_nom if available; otherwise prefer strict monotonic
        score = 0.0
        rep = f"note (dt stats [s]: min={dt_min:.6g}, median={dt_med:.6g}, max={dt_max:.6g})"

        if dt_nom is not None and math.isfinite(dt_med) and dt_med > 0:
            ratio = dt_med / dt_nom
            # ratio close to 1 is best (score 0); penalize log-distance
            score = -abs(math.log(ratio))
            if strict_time and abs(ratio - 1.0) > dt_rel_tol:
                return False, -1e300, f"reject (dt_med/dt_nom={ratio:.3g} outside ±{dt_rel_tol:.3g})", warnings, mat
            rep += f", dt_nom={dt_nom:.6g}, ratio={ratio:.6g}"
        else:
            if strict_time and n_nonpos > 0:
                return False, -1e300, f"reject (time not strictly increasing: count(dt<=0)={n_nonpos})", warnings, mat
            # Slightly prefer strictly increasing
            score = 0.0 if n_nonpos == 0 else -10.0

        # Also compute turn-duration stats from turn start times (for reporting)
        t0 = t[::Ns]
        if t0.size >= 2:
            Tturn = np.diff(t0)
            Tmin = float(np.min(Tturn))
            Tmed = float(np.median(Tturn))
            Tmax = float(np.max(Tturn))
            rep += f", turn duration T [s]: min={Tmin:.6g}, median={Tmed:.6g}, max={Tmax:.6g}"

        return True, float(score), rep, warnings, mat
