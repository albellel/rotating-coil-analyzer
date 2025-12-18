from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame


def _dt_nominal(samples_per_turn: int, shaft_speed_rpm: float) -> Optional[float]:
    v = abs(float(shaft_speed_rpm))
    if v <= 0 or samples_per_turn <= 0:
        return None
    T_nom = 60.0 / v
    return T_nom / float(samples_per_turn)


def _time_is_plausible(t: np.ndarray, dt_nom: Optional[float], samples_per_turn: int) -> Tuple[bool, float, float]:
    """
    Returns (ok, dt_median, neg_frac)
    Accepts:
      - strictly increasing time
      - per-turn wrapped time (resets at turn boundaries)
    """
    if t.size < 10:
        return False, np.nan, np.nan

    dt = np.diff(t.astype(np.float64, copy=False))
    if dt.size == 0:
        return False, np.nan, np.nan

    neg = np.count_nonzero(dt <= 0)
    neg_frac = neg / float(dt.size)

    # median of positive dt only (avoid turn resets)
    dt_pos = dt[dt > 0]
    if dt_pos.size == 0:
        return False, np.nan, neg_frac
    dt_med = float(np.median(dt_pos))

    # Basic plausibility: time step should be in a sane range
    if not (1e-7 < dt_med < 0.5):
        return False, dt_med, neg_frac

    # If we know expected dt_nom, require same order-of-magnitude (loose)
    if dt_nom is not None:
        if not (0.05 * dt_nom <= dt_med <= 20.0 * dt_nom):
            return False, dt_med, neg_frac

    # If there are negatives, they should mostly occur at turn boundaries
    if neg > 0 and samples_per_turn > 0:
        boundary_idx = np.arange(dt.size) % samples_per_turn == (samples_per_turn - 1)
        neg_idx = dt <= 0
        if np.count_nonzero(neg_idx & boundary_idx) / float(neg) < 0.8:
            # too many negative dt away from boundaries
            return False, dt_med, neg_frac

    return True, dt_med, neg_frac


def _choose_format(data: np.ndarray, dt_nom: Optional[float], samples_per_turn: int) -> float:
    """
    Score candidate interpretation, higher is better.
    data shape: (n_rows, n_cols)
    We consider candidate 'time' in col 0.
    """
    t = data[:, 0]
    ok, dt_med, neg_frac = _time_is_plausible(t, dt_nom=dt_nom, samples_per_turn=samples_per_turn)
    if not ok:
        return -1.0

    score = 0.0
    score += 5.0

    if dt_nom is not None:
        score += 3.0 * (1.0 / (1.0 + abs(dt_med - dt_nom) / dt_nom))
    score += 1.0 * (1.0 - neg_frac)  # prefer fewer resets
    return score


def _find_raw_time_file(corr_sigs_path: Path, aperture: int, segment: str) -> Optional[Path]:
    """
    If corr_sigs is an older format without embedded FDI time, time may exist in a sibling file.
    We only use REAL time from file; we never synthesize time.
    """
    root = corr_sigs_path.parent

    patterns = [
        f"*raw_time*Ap_{aperture}*Seg{segment}*.bin",
        f"*raw_time*Ap_{aperture}*Seg_{segment}*.bin",
        f"*fdi_raw_time*Ap_{aperture}*Seg{segment}*.bin",
        f"*time*Ap_{aperture}*Seg{segment}*.bin",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(root.glob(pat))

    # De-dup
    uniq = []
    seen = set()
    for p in hits:
        if p.name not in seen:
            uniq.append(p)
            seen.add(p.name)

    if len(uniq) == 1:
        return uniq[0]
    return None


@dataclass
class Sm18CorrSigsReader:
    """
    Reads SM18 *_corr_sigs_*.bin.

    Script documentation indicates (newer) format is:
      [t_fdi, flux_abs, flux_cmp, I1..IN]  (double)  :contentReference[oaicite:5]{index=5}

    Older runs may omit 't' in corr_sigs; if so we try a separate raw_time file.
    """
    max_currents: int = 4  # try up to 4 current columns in corr_sigs

    def read(
        self,
        file_path: Path,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: float,
        aperture: int = 1,
    ) -> SegmentFrame:
        file_path = Path(file_path)
        warnings: List[str] = []

        if samples_per_turn <= 0:
            raise ValueError("samples_per_turn must be > 0")

        dt_nom = _dt_nominal(samples_per_turn=samples_per_turn, shaft_speed_rpm=shaft_speed_rpm)

        # Candidate interpretations:
        # dtype: float64 and float32 (little endian), columns: 3..(3+max_currents+1 for time)
        candidates: List[Tuple[str, np.dtype, int, bool]] = []

        # with time
        for n_curr in range(1, self.max_currents + 1):
            n_cols = 3 + n_curr  # t, abs, cmp, currents...
            candidates.append((f"float64_le_{n_cols}col", np.dtype("<f8"), n_cols, True))
            candidates.append((f"float32_le_{n_cols}col", np.dtype("<f4"), n_cols, True))

        # without time (older): abs, cmp, currents...
        for n_curr in range(1, self.max_currents + 1):
            n_cols = 2 + n_curr  # abs, cmp, currents...
            candidates.append((f"float64_le_{n_cols}col_noT", np.dtype("<f8"), n_cols, False))
            candidates.append((f"float32_le_{n_cols}col_noT", np.dtype("<f4"), n_cols, False))

        best = None
        best_score = -1.0
        reports: List[str] = []

        file_size = file_path.stat().st_size

        for name, dt, n_cols, has_time in candidates:
            itemsize = dt.itemsize
            if file_size % itemsize != 0:
                continue
            n_vals = file_size // itemsize
            if n_vals % n_cols != 0:
                continue
            n_rows = n_vals // n_cols

            # Read ONLY first chunk for scoring (fast, robust)
            n_probe = min(n_rows, max(5000, 5 * samples_per_turn))
            raw = np.fromfile(file_path, dtype=dt, count=n_probe * n_cols)
            if raw.size != n_probe * n_cols:
                continue
            mat = raw.reshape(-1, n_cols)

            if has_time:
                score = _choose_format(mat, dt_nom=dt_nom, samples_per_turn=samples_per_turn)
                reports.append(f"{name}: score={score:.3f}")
                if score > best_score:
                    best_score = score
                    best = (name, dt, n_cols, True)
            else:
                # No-time candidates: accept only if values look like flux+current (heuristic)
                # flux columns should be relatively small compared to current columns.
                flux_mag = float(np.nanmedian(np.abs(mat[:, 0:2])))
                curr_mag = float(np.nanmedian(np.abs(mat[:, -1])))
                ok = (np.isfinite(flux_mag) and np.isfinite(curr_mag) and curr_mag > 0.01 and flux_mag < 10.0)
                score = 1.0 if ok else -1.0
                reports.append(f"{name}: score={score:.3f} (no-time heuristic)")
                if score > best_score:
                    best_score = score
                    best = (name, dt, n_cols, False)

        if best is None or best_score < 0:
            msg = "No supported binary format validated for this file.\n" + "\n".join(reports[:80])
            raise ValueError(msg)

        name, dt, n_cols, has_time = best
        warnings.append(f"Selected binary format: {name}")

        # Load full file with chosen interpretation
        raw = np.fromfile(file_path, dtype=dt)
        mat = raw.reshape(-1, n_cols)

        # Trim to multiple of samples_per_turn (never invent time; just drop trailing partial turn)
        rem = mat.shape[0] % samples_per_turn
        if rem != 0:
            warnings.append(f"Trimmed to multiple of samples_per_turn: {mat.shape[0]} -> {mat.shape[0] - rem}")
            mat = mat[: mat.shape[0] - rem, :]

        n_rows = int(mat.shape[0])
        n_turns = n_rows // samples_per_turn

        if has_time:
            t = mat[:, 0].astype(np.float64, copy=False)
            df_abs = mat[:, 1].astype(np.float64, copy=False)
            df_cmp = mat[:, 2].astype(np.float64, copy=False)
            currents = mat[:, 3:].astype(np.float64, copy=False)
        else:
            # Try to obtain REAL FDI time from a separate file if present
            tf = _find_raw_time_file(file_path, aperture=aperture, segment=segment)
            if tf is None:
                raise ValueError(
                    "corr_sigs appears to have no embedded FDI time column, and no unique raw_time file was found.\n"
                    "Per your requirement, we will NOT synthesize time.\n"
                    f"corr_sigs: {file_path}"
                )
            warnings.append(f"Using time from separate file: {tf.name}")

            t_raw = np.fromfile(tf, dtype=dt)
            # time file is typically single-column
            if t_raw.size != n_rows:
                raise ValueError(f"raw_time length mismatch: {t_raw.size} != {n_rows} (corr_sigs rows)")
            t = t_raw.astype(np.float64, copy=False)

            df_abs = mat[:, 0].astype(np.float64, copy=False)
            df_cmp = mat[:, 1].astype(np.float64, copy=False)
            currents = mat[:, 2:].astype(np.float64, copy=False)

        # Build dataframe
        df_dict: Dict[str, np.ndarray] = {"t": t, "df_abs": df_abs, "df_cmp": df_cmp}
        if currents.shape[1] == 1:
            df_dict["I"] = currents[:, 0]
        else:
            for k in range(currents.shape[1]):
                df_dict[f"I{k+1}"] = currents[:, k]

        df = pd.DataFrame(df_dict)

        # Add a minimal warning about nominal speed computation (abs(v))
        if shaft_speed_rpm < 0:
            warnings.append(f"Note: Parameters.Measurement.v is negative; using |v|={abs(shaft_speed_rpm)} rpm.")

        return SegmentFrame(
            source_path=file_path,
            run_id=run_id,
            segment=str(segment),
            samples_per_turn=int(samples_per_turn),
            n_turns=int(n_turns),
            df=df,
            warnings=tuple(warnings),
        )
