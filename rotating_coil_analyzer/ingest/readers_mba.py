from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class MbaReaderConfig:
    """
    Reader configuration for MBA plateau text files (*_raw_measurement_data.txt).

    align_time:
      - True: if a subsequent plateau file has a time vector that resets/overlaps,
              apply a constant offset to the whole block so time becomes strictly increasing.
              This does NOT synthesize per-sample time; it only aligns file boundaries.
      - False: keep raw time values and allow discontinuities.

    strict_time:
      - True: require final concatenated time to be strictly increasing (after optional alignment).
    """
    align_time: bool = True
    strict_time: bool = True
    max_rows_preview_warning: int = 2_000_000


class MbaRawMeasurementReader:
    """
    Reads and concatenates MBA plateau files of the form:

      <base>_Run_<step>_I_<current>A_<segment>_raw_measurement_data.txt

    The discovery layer stores one representative file per (base, aperture, segment).
    This reader, given that representative file, finds all matching plateau files for the
    same base+segment in the same directory, sorts them by step, and concatenates them.
    """

    _PAT = re.compile(
        r"^(?P<base>.+?)_Run_(?P<step>\d+)_I_(?P<i>[-\d.]+)A_(?P<seg>[^_]+)_raw_measurement_data\.txt$",
        flags=re.IGNORECASE,
    )

    def __init__(self, config: Optional[MbaReaderConfig] = None):
        self.config = config or MbaReaderConfig()

    @staticmethod
    def _robust_range(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        try:
            return float(np.nanpercentile(x, 99.5) - np.nanpercentile(x, 0.5))
        except Exception:
            return float("nan")

    def _find_plateau_files(self, representative: Path) -> Tuple[str, str, List[Path]]:
        m = self._PAT.match(representative.name)
        if not m:
            raise ValueError(f"Not an MBA raw_measurement_data file: {representative.name}")
        base = m.group("base")
        seg = m.group("seg")
        glob_pat = f"{base}_Run_*_I_*A_{seg}_raw_measurement_data.txt"
        files = sorted(representative.parent.glob(glob_pat))

        def step_key(p: Path) -> Tuple[int, float, str]:
            mm = self._PAT.match(p.name)
            if not mm:
                return (10**9, float("nan"), p.name)
            step = int(mm.group("step"))
            try:
                curr = float(mm.group("i"))
            except Exception:
                curr = float("nan")
            return (step, curr, p.name)

        files.sort(key=step_key)
        if not files:
            raise FileNotFoundError(f"No MBA plateau files matched {glob_pat} in {representative.parent}")
        return base, seg, files

    def _read_one(self, path: Path) -> np.ndarray:
        # whitespace-separated numeric file, no header
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        return df.to_numpy(dtype=np.float64, copy=False)

    def read(
        self,
        path: str | Path,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        aperture_id: Optional[int] = None,
    ) -> SegmentFrame:
        p = Path(path).expanduser().resolve()
        base, seg, files = self._find_plateau_files(p)

        # Concatenate
        blocks: List[np.ndarray] = []
        warnings: List[str] = []
        warnings.append(f"MBA reader: concatenating {len(files)} plateau files for base='{base}', segment='{seg}'")
        median_dt: Optional[float] = None
        last_t: Optional[float] = None
        applied_offsets = 0

        for f in files:
            mat = self._read_one(f)
            if mat.ndim != 2 or mat.shape[1] < 3:
                raise ValueError(f"File {f.name} has invalid shape {mat.shape}; expected >=3 columns.")

            t = mat[:, 0].astype(np.float64, copy=False)

            # estimate dt from this block
            if median_dt is None and t.size >= 3:
                dt = np.diff(t)
                dt = dt[np.isfinite(dt)]
                if dt.size:
                    median_dt = float(np.median(dt))

            if self.config.align_time and last_t is not None and t.size:
                if not np.isfinite(last_t):
                    pass
                else:
                    first_t = float(t[0])
                    if (not np.isfinite(first_t)) or first_t <= last_t:
                        # apply constant offset so that this block starts after last_t
                        dt_nom = median_dt if (median_dt is not None and np.isfinite(median_dt) and median_dt > 0) else 0.0
                        offset = (last_t + dt_nom) - first_t
                        t = t + offset
                        mat = mat.copy()
                        mat[:, 0] = t
                        applied_offsets += 1
                        warnings.append(f"time overlap/reset detected at {f.name}; applied offset={offset:.6g} s")

            if t.size:
                last_t = float(t[-1])

            blocks.append(mat)

        if applied_offsets:
            warnings.append(f"applied time-alignment offsets to {applied_offsets} plateau files (boundary alignment only).")

        mat = np.vstack(blocks)

        ncols = mat.shape[1]
        t = mat[:, 0].astype(np.float64, copy=False)

        # Choose abs/cmp between col1 and col2 by robust range (larger -> abs)
        c1 = mat[:, 1].astype(np.float64, copy=False)
        c2 = mat[:, 2].astype(np.float64, copy=False)
        r1 = self._robust_range(c1)
        r2 = self._robust_range(c2)
        if np.isfinite(r1) and np.isfinite(r2) and r2 > r1:
            df_abs = c2
            df_cmp = c1
            warnings.append("swapped flux columns: treated col2 as abs and col1 as cmp (by robust range)")
        else:
            df_abs = c1
            df_cmp = c2

        # Current candidates: columns >=3 (may include extras)
        curr_cols = list(range(3, ncols))
        I_main = np.zeros_like(t)
        if curr_cols:
            ranges = []
            for idx in curr_cols:
                ranges.append((self._robust_range(mat[:, idx]), idx))
            # pick by largest robust range; tie-breaker: lowest index
            ranges_sorted = sorted(ranges, key=lambda ri: (-ri[0], ri[1]))
            best_idx = ranges_sorted[0][1]
            I_main = mat[:, best_idx].astype(np.float64, copy=False)
            rng_txt = ", ".join([f"col{idx}:{r:.6g}" for r, idx in ranges_sorted if np.isfinite(r)])
            warnings.append(f"current candidate ranges (p99.5-p0.5): {rng_txt}")
            warnings.append(f"selected main current column: col{best_idx} (stored as df['I'])")

        df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I_main})

        # Preserve all candidate currents as I0, I1, ...
        for j, idx in enumerate(curr_cols):
            df[f"I{j}"] = mat[:, idx].astype(np.float64, copy=False)

        # Duplicate current detection (exact sample equality)
        if len(curr_cols) >= 2:
            for a in range(len(curr_cols)):
                for b in range(a + 1, len(curr_cols)):
                    ia = df[f"I{a}"].to_numpy()
                    ib = df[f"I{b}"].to_numpy()
                    if np.allclose(ia, ib, atol=0.0, rtol=0.0, equal_nan=True):
                        warnings.append(f"duplicate current detected: I{a} == I{b} (exact match).")

        # Strict time check
        if self.config.strict_time:
            dt = np.diff(df["t"].to_numpy())
            if not np.all(np.isfinite(dt)):
                warnings.append("non-finite dt encountered in time vector after concatenation.")
            if np.any(dt <= 0):
                raise ValueError("time is not strictly increasing after concatenation (and optional alignment).")

        Ns = int(samples_per_turn)
        n_turns = int(len(df) // Ns)

        return SegmentFrame(
            source_path=p,
            run_id=run_id,
            segment=str(segment),
            samples_per_turn=Ns,
            n_turns=n_turns,
            df=df.astype(np.float64, copy=False),
            warnings=tuple(warnings),
            aperture_id=aperture_id,
        )
