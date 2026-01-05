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
    """Reader configuration for MBA plateau text files (``*_raw_measurement_data.txt``).

    Project-wide hard constraint:
        **No synthetic/modified time is allowed anywhere in this project.**

    For MBA plateau concatenation this means:
        - The time column ``t`` is always the raw time read from each plateau file.
        - Time is never offset/shifted/aligned across plateau boundaries.
        - The concatenated time vector may therefore contain discontinuities and resets.

    Notes on legacy flags:
        ``align_time`` and ``strict_time`` are kept only for backward compatibility with
        older notebooks, but they are **disallowed**. If either is set to ``True``, the
        reader will raise.

    max_rows_preview_warning:
        If the concatenated trace exceeds this many rows, a warning is emitted (GUI usability).
    """

    align_time: bool = False
    strict_time: bool = False
    max_rows_preview_warning: int = 2_000_000


class MbaRawMeasurementReader:
    """Reads and concatenates MBA plateau files.

    MBA acquisition consists of many plateau files (one per current level), e.g.:

        ``<base>_Run_<step>_I_<current>A_<segment>_raw_measurement_data.txt``

    The discovery layer stores one representative file per (base, aperture, segment).
    This reader, given that representative file, finds all matching plateau files for the
    same base+segment in the same directory, sorts them by step, and concatenates them.

    Hard constraints enforced here:
        - No synthetic time: the ``t`` column is always the raw time stored in each file.
        - Plateau-safe turns: each plateau is trimmed independently to a whole number of turns,
          so turns never cross plateau boundaries.

    Output columns:
        - ``t``: raw time from the plateau files (may reset between plateaus)
        - ``df_abs``, ``df_cmp``: inferred flux channels (abs is chosen as the larger-range one)
        - ``I``: selected main current channel
        - ``I0``, ``I1``, ...: all current candidates (if present)
        - ``plateau_id``: 0,1,2,... in concatenation order (float in df due to global cast)
        - ``plateau_step``: parsed Run_XX step (float in df)
        - ``plateau_I_hint``: current parsed from filename, if parseable (float in df)
        - ``sample_in_plateau``: 0..(n_keep-1) within each plateau (float in df)
        - ``k``: global sample index 0..N-1 across the concatenated trace (float in df)

    Important:
        - ``k`` is not time; it exists only as an ordering axis for plotting.
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
        files = list(representative.parent.glob(glob_pat))

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
        magnet_order: Optional[int] = None,
    ) -> SegmentFrame:
        if self.config.align_time or self.config.strict_time:
            raise ValueError(
                "MBA reader: align_time/strict_time are disallowed because they imply modifying or "
                "enforcing a stitched time axis. This project forbids synthetic/modified time." 
            )

        Ns = int(samples_per_turn)
        if Ns <= 0:
            raise ValueError("samples_per_turn must be > 0")

        p = Path(path).expanduser().resolve()
        base, seg, files = self._find_plateau_files(p)

        blocks: List[np.ndarray] = []
        plateau_id_blocks: List[np.ndarray] = []
        plateau_step_blocks: List[np.ndarray] = []
        plateau_i_hint_blocks: List[np.ndarray] = []
        sample_in_plateau_blocks: List[np.ndarray] = []

        warnings: List[str] = []
        warnings.append(f"MBA reader: concatenating {len(files)} plateau files for base='{base}', segment='{seg}'")

        last_plateau_end_t: Optional[float] = None

        for pid, f in enumerate(files):
            mm = self._PAT.match(f.name)
            step = int(mm.group("step")) if mm else pid
            try:
                i_hint = float(mm.group("i")) if mm else float("nan")
            except Exception:
                i_hint = float("nan")

            mat = self._read_one(f)
            if mat.ndim != 2 or mat.shape[1] < 3:
                raise ValueError(f"File {f.name} has invalid shape {mat.shape}; expected >=3 columns.")

            # Raw time from file (never modified).
            t = mat[:, 0].astype(np.float64, copy=False)

            # Plateau boundary diagnostic (no correction).
            if last_plateau_end_t is not None and t.size:
                first_t = float(t[0])
                if np.isfinite(first_t) and np.isfinite(last_plateau_end_t) and first_t <= last_plateau_end_t:
                    warnings.append(
                        f"time reset/overlap across plateaus at {f.name}: prev_end_t={last_plateau_end_t:.6g}, first_t={first_t:.6g} "
                        "(expected for MBA; time is kept raw by design)"
                    )

            # Intra-plateau time diagnostics (warning-level only; no correction).
            if t.size >= 3:
                n_bad_t = int(np.sum(~np.isfinite(t)))
                if n_bad_t:
                    warnings.append(f"non-finite time values within plateau file {f.name}: {n_bad_t} samples")
                dt = np.diff(t)
                n_bad_dt = int(np.sum(~np.isfinite(dt)))
                if n_bad_dt:
                    warnings.append(f"non-finite dt values within plateau file {f.name}: {n_bad_dt} intervals")
                dt_f = dt[np.isfinite(dt)]
                if dt_f.size:
                    dt_med = float(np.median(dt_f))
                    dt_max = float(np.max(dt_f))
                    n_nonpos = int(np.sum(dt_f <= 0))
                    if n_nonpos:
                        warnings.append(
                            f"non-increasing time within plateau file {f.name}: {n_nonpos} non-positive finite dt values"
                        )
                    if dt_med > 0 and (dt_max / dt_med) > 10.0:
                        warnings.append(
                            f"large dt spread within plateau file {f.name}: median={dt_med:.6g}, max={dt_max:.6g}"
                        )
                else:
                    warnings.append(f"all dt are non-finite within plateau file {f.name}")
            else:
                n_bad_t = int(np.sum(~np.isfinite(t)))
                if n_bad_t:
                    warnings.append(f"non-finite time values within short plateau file {f.name}: {n_bad_t} samples")
                warnings.append(f"short time vector in plateau file {f.name}: n={t.size}")
            # Plateau-safe trimming: do not allow turns to cross plateau boundaries.
            n_rows = int(mat.shape[0])
            n_keep = (n_rows // Ns) * Ns
            if n_keep <= 0:
                raise ValueError(
                    f"Plateau file {f.name} shorter than one turn: rows={n_rows}, samples_per_turn={Ns}"
                )
            if n_keep < n_rows:
                removed = n_rows - n_keep
                warnings.append(f"trim plateau to full turns: {f.name} removed {removed} rows (kept {n_keep})")
                mat = mat[:n_keep, :]

            blocks.append(mat)
            plateau_id_blocks.append(np.full((n_keep,), float(pid), dtype=np.float64))
            plateau_step_blocks.append(np.full((n_keep,), float(step), dtype=np.float64))
            plateau_i_hint_blocks.append(np.full((n_keep,), float(i_hint), dtype=np.float64))
            sample_in_plateau_blocks.append(np.arange(n_keep, dtype=np.float64))

            if t.size:
                last_plateau_end_t = float(t[min(len(t), n_keep) - 1])

        mat = np.vstack(blocks)
        plateau_id = np.concatenate(plateau_id_blocks)
        plateau_step = np.concatenate(plateau_step_blocks)
        plateau_i_hint = np.concatenate(plateau_i_hint_blocks)
        sample_in_plateau = np.concatenate(sample_in_plateau_blocks)

        if len(mat) > self.config.max_rows_preview_warning:
            warnings.append(
                f"large concatenated MBA trace: {len(mat)} rows (preview/plotting may be slow)."
            )

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

        # Global sample index for ordering/plotting (NOT time)
        k = np.arange(len(t), dtype=np.float64)

        df = pd.DataFrame(
            {
                "t": t,
                "df_abs": df_abs,
                "df_cmp": df_cmp,
                "I": I_main,
                "plateau_id": plateau_id,
                "plateau_step": plateau_step,
                "plateau_I_hint": plateau_i_hint,
                "sample_in_plateau": sample_in_plateau,
                "k": k,
            }
        )

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

        # Final trimming safeguard (should be exact because each plateau is trimmed).
        n_total = int(len(df))
        n_keep_total = (n_total // Ns) * Ns
        if n_keep_total < n_total:
            removed = n_total - n_keep_total
            warnings.append(f"trim concatenated trace to full turns: removed {removed} rows (kept {n_keep_total})")
            df = df.iloc[:n_keep_total, :].reset_index(drop=True)

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
            magnet_order=magnet_order,
        )
