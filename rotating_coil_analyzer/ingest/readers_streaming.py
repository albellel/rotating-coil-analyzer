from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class StreamingReaderConfig:
    """
    Reader configuration for streaming (continuous) acquisition data.

    Streaming acquisition produces a single large binary file from hours-long
    continuous measurements.

    strict_time:
      - True: require strictly increasing finite time vector after trimming trailing invalid rows.
      - False: allow non-monotonic time (not recommended).
    dt_rel_tol:
      Relative tolerance for dt_median vs dt_nominal (computed from |v| and samples_per_turn).
      Example: 0.2 means +/-20%.
    max_currents:
      Try formats with 0..max_currents current channels (total columns = 3 + n_currents).
    dtype_candidates:
      Candidate numpy dtypes to test for the binary content (little-endian only).
    """
    strict_time: bool = True
    dt_rel_tol: float = 0.25
    max_currents: int = 3
    dtype_candidates: Tuple[np.dtype, ...] = (np.dtype("<f8"), np.dtype("<f4"))


class StreamingReader:
    """
    Reader for streaming (continuous) acquisition segment files (bin/txt/csv).

    Streaming acquisition typically produces large binary files from hours-long
    continuous measurements with the rotating coil.

    HARD REQUIREMENT:
      - time is always taken from the file (column 0)
      - no synthetic time is ever generated
    """

    def __init__(self, config: Optional[StreamingReaderConfig] = None):
        self.config = config or StreamingReaderConfig()

    def read(
        self,
        file_path: str | Path,
        *,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: float,
        aperture_id: Optional[int] = None,
        magnet_order: Optional[int] = None,
    ) -> SegmentFrame:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))

        Ns = int(samples_per_turn)
        if Ns <= 0:
            raise ValueError("samples_per_turn must be > 0")

        if path.suffix.lower() in {".txt", ".csv"}:
            mat, ncols, warnings = self._load_ascii(path)
        else:
            mat, ncols, warnings = self._infer_and_load(path, Ns=Ns, shaft_speed_rpm=float(shaft_speed_rpm))

        # Column semantics:
        #   col0: time [s]
        #   col1/col2: flux channels (abs/cmp, order can vary -> we auto-assign by robust amplitude)
        #   col3..: one or more current traces (potentially from different sources)
        t = mat[:, 0]

        # Assign df_abs and df_cmp robustly between col1 and col2
        if ncols >= 3:
            c1 = mat[:, 1]
            c2 = mat[:, 2]
            r1 = float(np.nanpercentile(c1, 99.5) - np.nanpercentile(c1, 0.5))
            r2 = float(np.nanpercentile(c2, 99.5) - np.nanpercentile(c2, 0.5))
            if np.isfinite(r1) and np.isfinite(r2) and (r2 > r1):
                df_abs = c2
                df_cmp = c1
                warnings.append("swapped flux columns: treated col2 as abs and col1 as cmp (by robust range)")
                abs_col, cmp_col = 2, 1
            else:
                df_abs = c1
                df_cmp = c2
                abs_col, cmp_col = 1, 2
        else:
            df_abs = mat[:, 1] if ncols >= 2 else np.full((len(t),), np.nan, dtype=np.float64)
            df_cmp = mat[:, 2] if ncols >= 3 else np.full((len(t),), np.nan, dtype=np.float64)
            abs_col, cmp_col = 1, 2

        # Collect current candidates and select a "main" current.
        I_main = np.full((len(t),), np.nan, dtype=np.float64)
        curr_cols = []
        curr_mat = None
        if ncols >= 4:
            curr_cols = list(range(3, ncols))
            curr_mat = mat[:, 3:ncols]

            # Robust dynamic range (avoid spikes).
            ranges: List[Tuple[float, int]] = []
            for k in range(curr_mat.shape[1]):
                c = curr_mat[:, k]
                finite = np.isfinite(c)
                if finite.sum() < max(10, int(0.9 * len(c))):
                    ranges.append((float("-inf"), k))
                    continue
                lo = float(np.nanpercentile(c, 0.5))
                hi = float(np.nanpercentile(c, 99.5))
                ranges.append((hi - lo, k))

            # Select by max range; tie-breaker: smallest column index (deterministic)
            best_range = max(r for r, _ in ranges) if ranges else float("-inf")
            best_ks = [k for r, k in ranges if np.isfinite(r) and abs(r - best_range) <= 0.0]
            best_k = min(best_ks) if best_ks else (ranges[0][1] if ranges else 0)
            I_main = curr_mat[:, best_k]

            # Emit mapping summary.
            abs_r = float(np.nanpercentile(df_abs, 99.5) - np.nanpercentile(df_abs, 0.5)) if len(df_abs) else float("nan")
            cmp_r = float(np.nanpercentile(df_cmp, 99.5) - np.nanpercentile(df_cmp, 0.5)) if len(df_cmp) else float("nan")
            warnings.append(f"column map: t=0, df_abs={abs_col}, df_cmp={cmp_col}, current cols=3..{ncols-1}")
            warnings.append(f"flux ranges (p99.5-p0.5): abs={abs_r:.6g}, cmp={cmp_r:.6g}")

            # Report current candidates
            ranges_sorted = sorted(ranges, key=lambda rk: (-rk[0], rk[1]))
            rng_txt = ", ".join([f"col{3+k}:{r:.6g}" for r, k in ranges_sorted if np.isfinite(r)])
            warnings.append(f"current candidate ranges (p99.5-p0.5): {rng_txt}")

        df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I_main})

        # Preserve all current traces (if present) for debugging/selection downstream.
        for j, col_idx in enumerate(curr_cols):
            df[f"I{j}"] = mat[:, col_idx]

        # Duplicate current detection (exact sample equality)
        if len(curr_cols) >= 2:
            for a in range(len(curr_cols)):
                for b in range(a + 1, len(curr_cols)):
                    ia = df[f"I{a}"].to_numpy()
                    ib = df[f"I{b}"].to_numpy()
                    if np.allclose(ia, ib, atol=0.0, rtol=0.0, equal_nan=True):
                        warnings.append(f"duplicate current detected: I{a} == I{b} (exact match). Canonical will use lowest-index among ties.")

        n_turns = int(len(df) // Ns)

        return SegmentFrame(
            source_path=path,
            run_id=run_id,
            segment=str(segment),
            samples_per_turn=Ns,
            n_turns=n_turns,
            df=df.astype(np.float64, copy=False),
            warnings=tuple(warnings),
            aperture_id=aperture_id,
            magnet_order=magnet_order,
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
            raise ValueError("Empty file")

        # Nominal dt from |v|
        v = abs(float(shaft_speed_rpm)) if shaft_speed_rpm is not None else 0.0
        if v <= 0:
            warnings.append("shaft_speed_rpm <= 0; nominal dt check disabled")
            dt_nom = None
        else:
            T_nom = 60.0 / v
            dt_nom = T_nom / float(Ns)

        reports: List[str] = []
        best: Optional[Tuple[np.ndarray, int, List[str]]] = None
        best_score = -1e99

        for dtype in cfg.dtype_candidates:
            bps = int(dtype.itemsize)
            if file_size % bps != 0:
                continue
            n_scalars = file_size // bps

            # total columns = 3 + n_currents  (allow 0 currents)
            for n_curr in range(0, max(0, int(cfg.max_currents)) + 1):
                ncols = 3 + n_curr
                if ncols <= 0:
                    continue
                if n_scalars % ncols != 0:
                    continue

                arr = np.fromfile(path, dtype=dtype)
                if arr.size != n_scalars:
                    arr = arr[:n_scalars]
                with np.errstate(all="ignore"):
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
        Validate a candidate matrix:
          - finite time
          - trim trailing invalid rows
          - strict monotonic time if strict_time
          - dt median close to dt_nom if dt_nom available

        Returns (ok, score, report, warnings, mat_out).
        """
        warnings: List[str] = []

        if mat.shape[1] < 3:
            return False, -1e9, "reject: ncols < 3", warnings, mat

        # Trim trailing rows with any non-finite in t/flux columns
        finite_mask = np.isfinite(mat[:, 0]) & np.isfinite(mat[:, 1]) & np.isfinite(mat[:, 2])
        if not np.all(finite_mask):
            first_bad = int(np.argmax(~finite_mask))
            if first_bad < max(10, int(0.01 * len(mat))):
                return False, -1e9, f"reject: invalid rows too early (first_bad={first_bad})", warnings, mat
            warnings.append(f"trim trailing invalid rows: first_bad={first_bad}, n_bad={(~finite_mask).sum()}")
            mat = mat[:first_bad, :]

        # Trim to integer turns
        rem = mat.shape[0] % int(Ns)
        if rem != 0:
            warnings.append(f"trim to multiple of samples_per_turn: removed {rem} rows")
            mat = mat[:-rem, :]

        t = mat[:, 0]
        dt = np.diff(t)
        dt_f = dt[np.isfinite(dt)]
        if dt_f.size < 10:
            return False, -1e9, "reject: too few finite dt", warnings, mat
        dt_med = float(np.median(dt_f))
        if not np.isfinite(dt_med) or dt_med <= 0:
            return False, -1e9, f"reject: dt_med={dt_med}", warnings, mat

        noninc = int(np.sum(dt_f <= 0))
        if strict_time and noninc > 0:
            return False, -1e9, f"reject: time not strictly increasing (dt<=0 count={noninc})", warnings, mat

        # Nominal dt check
        score = 0.0
        if dt_nom is not None and np.isfinite(dt_nom) and dt_nom > 0:
            rel = abs(dt_med - dt_nom) / dt_nom
            if rel > float(dt_rel_tol):
                return False, -1e9, f"reject: dt_med {dt_med:.6g} not within {dt_rel_tol:.3g} of dt_nom {dt_nom:.6g}", warnings, mat
            score += (1.0 - rel) * 100.0
            warnings.append(f"dt nominal check: dt_med={dt_med:.6g}, dt_nom={dt_nom:.6g}, rel_err={rel:.3g}")

        # Prefer more rows (more turns)
        score += mat.shape[0] / float(Ns)

        rep = f"ok rows={mat.shape[0]} ncols={mat.shape[1]} dt_med={dt_med:.6g} noninc={noninc} score={score:.3f}"
        return True, score, rep, warnings, mat

    def _load_ascii(self, path: Path) -> Tuple[np.ndarray, int, List[str]]:
        """
        Load ASCII (txt/csv) into a matrix.

        Policy:
          - time must be present as column 0
          - accept >=3 columns
          - return float64 matrix
        """
        warnings: List[str] = []
        # Try whitespace first, then CSV
        df = None
        try:
            df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, comment="#")
        except Exception:
            df = pd.read_csv(path, sep=",", engine="python", header=None, comment="#")
        if df is None or df.shape[1] < 3:
            raise ValueError(f"ASCII file must have >=3 columns (t, flux1, flux2, ...): {path}")
        mat = df.to_numpy(dtype=np.float64, copy=False)
        return mat, int(mat.shape[1]), warnings


# Backward compatibility aliases (deprecated)
Sm18ReaderConfig = StreamingReaderConfig
Sm18CorrSigsReader = StreamingReader
