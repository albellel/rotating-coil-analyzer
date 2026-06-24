"""Readers for DC-plateau (staircase) rotating-coil measurement files.

Each plateau run is a separate text file with per-turn rows.  This module
loads individual runs and concatenates them into a single turn array with
run metadata (run ID, nominal current, branch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.ingest.channel_detect import (
    ColumnMapping,
    detect_flux_channels,
    detect_current_channel,
    validate_channel_assignment,
)


# ---------------------------------------------------------------------------
# Plateau filename parsing (shared by the reader and the discovery layer)
# ---------------------------------------------------------------------------

#: Standard staircase naming: ``<base>_Run_<step>_I_<current>A_<seg>_raw_measurement_data.txt``
_PAT_PLATEAU_STD = re.compile(
    r"^(?P<base>.+?)_Run_(?P<step>\d+)_I_(?P<i>[-\d.]+)A_(?P<seg>[^_]+)_raw_measurement_data\.txt$",
    flags=re.IGNORECASE,
)

#: H/V steerer naming (e.g. FFMM degauss): a run number glued to ``IH`` followed
#: by the horizontal and vertical set currents:
#: ``<base>_Run_<step>IH_<ih>IV_<iv>_<seg>_raw_measurement_data.txt``
_PAT_PLATEAU_HV = re.compile(
    r"^(?P<base>.+?)_Run_(?P<step>\d+)IH_(?P<ih>[-0-9.]+)IV_(?P<iv>[-0-9.]+)_(?P<seg>[^_]+)_raw_measurement_data\.txt$",
    flags=re.IGNORECASE,
)


def _safe_float(s: object) -> float:
    try:
        return float(s)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def parse_plateau_filename(name: str, extra_pattern: Optional["re.Pattern"] = None) -> Optional[dict]:
    """Parse a plateau ``*_raw_measurement_data.txt`` filename.

    Supported layouts (tried in order):

    1. ``extra_pattern`` (optional caller override) — must expose at least the
       named groups ``base``, ``step``, ``seg`` (and optionally ``i``).
    2. **Standard staircase**:
       ``<base>_Run_<step>_I_<current>A_<seg>_raw_measurement_data.txt``
    3. **H/V steerer** (e.g. FFMM degauss):
       ``<base>_Run_<step>IH_<ih>IV_<iv>_<seg>_raw_measurement_data.txt``

    Returns a dict with keys ``base``, ``step`` (int), ``seg``, and ``i`` (the
    representative current). For the H/V form ``i`` is the **active channel**
    current (signed ``ih`` if non-zero, else ``iv``), and ``ih``/``iv`` are also
    included. Returns ``None`` if no layout matches.
    """
    if extra_pattern is not None:
        m = extra_pattern.match(name)
        if m:
            gd = m.groupdict()
            return {
                "base": gd.get("base", ""),
                "step": int(gd["step"]) if gd.get("step") not in (None, "") else 0,
                "seg": gd.get("seg", ""),
                "i": _safe_float(gd.get("i")) if gd.get("i") not in (None, "") else float("nan"),
            }

    m = _PAT_PLATEAU_STD.match(name)
    if m:
        return {
            "base": m.group("base"),
            "step": int(m.group("step")),
            "seg": m.group("seg"),
            "i": _safe_float(m.group("i")),
        }

    m = _PAT_PLATEAU_HV.match(name)
    if m:
        ih = _safe_float(m.group("ih"))
        iv = _safe_float(m.group("iv"))
        i_active = ih if (np.isfinite(ih) and ih != 0.0) else iv
        return {
            "base": m.group("base"),
            "step": int(m.group("step")),
            "seg": m.group("seg"),
            "i": i_active,
            "ih": ih,
            "iv": iv,
        }

    return None


@dataclass(frozen=True)
class PlateauReaderConfig:
    """Reader configuration for plateau-based text files (``*_raw_measurement_data.txt``).

    Plateau-based acquisition collects data at discrete DC current levels, producing
    one text file per plateau (current step).

    Project-wide hard constraint:
        **No synthetic/modified time is allowed anywhere in this project.**

    For plateau concatenation this means:
        - The time column ``t`` is always the raw time read from each plateau file.
        - Time is never offset/shifted/aligned across plateau boundaries.
        - The concatenated time vector may therefore contain discontinuities and resets.

    Notes on legacy flags:
        ``align_time`` and ``strict_time`` are kept only for backward compatibility with
        older notebooks, but they are **disallowed**. If either is set to ``True``, the
        reader will raise.

    max_rows_preview_warning:
        If the concatenated trace exceeds this many rows, a warning is emitted (GUI usability).
    column_mapping:
        Optional explicit column assignment.  ``None`` means auto-detect.
    filename_pattern:
        Optional regex override for plateau filename matching.  ``None`` uses
        the default ``_Run_XX_I_XXA_<seg>_raw_measurement_data.txt`` pattern.
    """

    align_time: bool = False
    strict_time: bool = False
    max_rows_preview_warning: int = 2_000_000
    column_mapping: Optional[ColumnMapping] = None
    filename_pattern: Optional[str] = None


class PlateauReader:
    """Reads and concatenates plateau-based acquisition files.

    Plateau-based acquisition consists of many files (one per current level), e.g.:

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

    #: Standard pattern (kept for backward compatibility / introspection).
    _PAT = _PAT_PLATEAU_STD

    def __init__(self, config: Optional[PlateauReaderConfig] = None):
        self.config = config or PlateauReaderConfig()
        # Optional caller-supplied filename pattern; otherwise the shared parser
        # (standard + H/V layouts) is used.
        if self.config.filename_pattern is not None:
            self._extra_pat: Optional[re.Pattern] = re.compile(
                self.config.filename_pattern, flags=re.IGNORECASE
            )
        else:
            self._extra_pat = None

    def _parse(self, name: str) -> Optional[dict]:
        return parse_plateau_filename(name, extra_pattern=self._extra_pat)

    def _find_plateau_files(self, representative: Path) -> Tuple[str, str, List[Path]]:
        info = self._parse(representative.name)
        if not info:
            raise ValueError(f"Not a plateau raw_measurement_data file: {representative.name}")
        base = info["base"]
        seg = info["seg"]

        # Broad glob on base+suffix (layout-agnostic), then filter by a successful
        # parse that yields the SAME segment. This matches both the standard
        # ``_I_<cur>A_`` layout and the H/V ``..IH_..IV_..`` layout.
        glob_pat = f"{base}_Run_*_raw_measurement_data.txt"
        matched: List[Tuple[int, float, str, Path]] = []
        for p in representative.parent.glob(glob_pat):
            ci = self._parse(p.name)
            if ci and ci["seg"] == seg:
                step = int(ci["step"])
                curr = ci.get("i", float("nan"))
                curr_key = curr if np.isfinite(curr) else float("inf")
                matched.append((step, curr_key, p.name, p))

        if not matched:
            raise FileNotFoundError(
                f"No plateau files matched base='{base}', segment='{seg}' in {representative.parent}"
            )
        matched.sort(key=lambda x: (x[0], x[1], x[2]))
        return base, seg, [m[3] for m in matched]

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
        """Read plateau (DC) measurement files and return a SegmentFrame.

        Discovers all plateau files matching the representative filename
        pattern, reads each one, trims to integer turns, and concatenates
        them with plateau metadata (plateau_id, plateau_step, plateau_I_hint).

        Parameters
        ----------
        path : str or Path
            Path to any one of the plateau raw_measurement_data files.
            All matching files in the same directory are auto-discovered.
        run_id : str
            Run identifier (propagated to SegmentFrame metadata).
        segment : str
            Segment identifier (e.g. "Integral", "Central").
        samples_per_turn : int
            Number of samples per coil revolution.
        aperture_id : int, optional
            Physical aperture id (default None).
        magnet_order : int, optional
            Main harmonic order (default None).

        Returns
        -------
        SegmentFrame
            Concatenated segment with columns t, df_abs, df_cmp, I,
            plateau_id, plateau_step, plateau_I_hint.
        """
        if self.config.align_time or self.config.strict_time:
            raise ValueError(
                "Plateau reader: align_time/strict_time are disallowed because they imply modifying or "
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
        warnings.append(f"Plateau reader: concatenating {len(files)} plateau files for base='{base}', segment='{seg}'")

        last_plateau_end_t: Optional[float] = None

        for pid, f in enumerate(files):
            fi = self._parse(f.name)
            step = int(fi["step"]) if fi else pid
            i_hint = float(fi["i"]) if fi else float("nan")

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
                        "(expected for plateau data; time is kept raw by design)"
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
                f"large concatenated plateau trace: {len(mat)} rows (preview/plotting may be slow)."
            )

        ncols = mat.shape[1]
        t = mat[:, 0].astype(np.float64, copy=False)

        # Choose abs/cmp between col1 and col2 (shared detection with optional override)
        mapping = self.config.column_mapping
        df_abs, df_cmp, abs_col, cmp_col, detect_w = detect_flux_channels(
            mat, mapping=mapping,
        )
        warnings.extend(detect_w)
        warnings.extend(validate_channel_assignment(df_abs, df_cmp))

        # Current channel (shared detection with optional override)
        I_main, best_curr_col, curr_w = detect_current_channel(
            mat, start_col=3, mapping=mapping,
        )
        warnings.extend(curr_w)
        curr_cols = list(range(3, ncols))

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


# Backward compatibility aliases (deprecated)
MbaReaderConfig = PlateauReaderConfig
MbaRawMeasurementReader = PlateauReader
