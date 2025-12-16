"""
Phase 1: input readers + plateau handling.

This module ingests raw measurement files and returns a list of plateau tables
with canonical columns.

Supported inputs
----------------
1) TXT plateau-per-file:
   - whitespace-delimited numeric columns
   - no header
   - each file is a single current plateau (your workflow)
   - multiple files are sorted chronologically (heuristic based on time or filename)

2) CSV plateau-per-file:
   - headered or headerless numeric CSV
   - if headered, we accept canonical names or common aliases

3) BIN whole-run:
   - raw little-endian float64 stream, reshaped into (3 + N) columns:
     [time_s, dphi_abs, dphi_cmp, current_a, (optional extra channels...)]
   - one file contains many plateaus (stair-step)
   - optionally split into plateaus by current_a

Important semantic note
-----------------------
In FFMM online analysis, the time vector used is sourced from the ABS device "ftime"
channel. :contentReference[oaicite:1]{index=1}
Here we name it time_s to avoid implying UTC time.

Examples
--------
TXT plateau files:
>>> from rotating_coil_analyzer.scripts.io_plateau import load_measurements, LoadConfig
>>> plateaus, report = load_measurements(["run1.txt", "run2.txt"], LoadConfig())
>>> len(plateaus)  # two plateaus
2

BIN whole-run:
>>> plateaus, report = load_measurements(["cycle.bin"], LoadConfig(split_long_run_into_plateaus=True))
>>> len(plateaus) >= 1
True
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Literal, Tuple

import numpy as np
import pandas as pd

from .validate_plateau import (
    CORE_COLS,
    ValidationResult,
    default_column_names,
    drop_nonfinite_rows,
    validate_core_table,
)


ExportFormat = Literal["csv", "parquet"]
FileKind = Literal["txt", "csv", "bin"]


# Best-effort filename parsing for ordering and metadata display.
_RE_TS = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")
_RE_RUN = re.compile(r"_Run_(?P<run>\d+)", re.IGNORECASE)
_RE_SETPOINT = re.compile(r"_I_(?P<i>[-+]?\d+(?:\.\d+)?)A", re.IGNORECASE)


@dataclass(frozen=True)
class PlateauMeta:
    """
    Metadata for one plateau or one raw input item.

    Examples
    --------
    >>> PlateauMeta(source_file="x.txt", source_path="x.txt", file_kind="txt",
    ...             parsed_timestamp=None, run_index=None, current_setpoint=None)
    PlateauMeta(source_file='x.txt', source_path='x.txt', file_kind='txt', parsed_timestamp=None, run_index=None, current_setpoint=None)
    """
    source_file: str
    source_path: str
    file_kind: FileKind
    parsed_timestamp: Optional[pd.Timestamp]
    run_index: Optional[int]
    current_setpoint: Optional[float]


@dataclass
class PlateauData:
    """
    Container for one loaded dataset (either a single plateau file, or a whole-run before splitting).

    Attributes
    ----------
    meta:
        Filename-derived metadata.
    df:
        Standardized dataframe with canonical columns.
    validation:
        Validation result (after optional cleaning).

    Notes
    -----
    For plateau-per-file inputs, df is already a plateau.
    For whole-run inputs (.bin), df may later be split into multiple plateaus.
    """
    meta: PlateauMeta
    df: pd.DataFrame
    validation: ValidationResult


@dataclass(frozen=True)
class LoadConfig:
    """
    Configuration controlling reading, cleaning, splitting, and export.

    Attributes
    ----------
    extra_names:
        Names to assign to columns 5..N for headerless TXT/CSV. If None, uses extra_1, extra_2, ...
    csv_has_header:
        If True, treat CSV as headered; otherwise as headerless numeric.
    bin_n_currents:
        For BIN: override number of additional channels after the first 3 columns.
        Minimum is 1 (current_a). If None, infer by divisibility.
    bin_max_currents_to_try:
        Max channels to try when inferring.
    bin_channel_names:
        Optional names for BIN channels after the first 3 columns (length must equal inferred/provided N).
        If None, uses ["current_a", "extra_1", ...].
    drop_nonfinite:
        If True, drop rows where required columns have NaN/Inf.
    split_long_run_into_plateaus:
        If True, split a long run into plateaus (intended for BIN runs).
    gap_threshold_s:
        Detect large time gaps and split run segments before plateau detection.
    plateau_smooth_window:
        Rolling median window (samples) for current smoothing.
    plateau_slope_threshold_a_per_s:
        Points are considered "plateau-like" when |dI/dt| below this value.
    plateau_level_rounding_a:
        Quantize plateau levels to this step (A) for robust grouping.
    plateau_min_duration_s:
        Minimum plateau duration (seconds) to keep.
    """
    extra_names: Optional[Sequence[str]] = None
    csv_has_header: bool = True

    bin_n_currents: Optional[int] = None
    bin_max_currents_to_try: int = 8
    bin_channel_names: Optional[Sequence[str]] = None  # after first 3 columns

    drop_nonfinite: bool = True

    split_long_run_into_plateaus: bool = True
    gap_threshold_s: float = 2.0

    plateau_smooth_window: int = 2001
    plateau_slope_threshold_a_per_s: float = 1.0
    plateau_level_rounding_a: float = 10.0
    plateau_min_duration_s: float = 5.0


def _parse_ts(p: Path) -> Optional[pd.Timestamp]:
    """Parse YYYYMMDD_HHMMSS from filename, if present."""
    m = _RE_TS.search(p.name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group("date") + m.group("time"), format="%Y%m%d%H%M%S")
    except Exception:
        return None


def _parse_run_index(p: Path) -> Optional[int]:
    """Parse _Run_<n> from filename, if present."""
    m = _RE_RUN.search(p.name)
    if not m:
        return None
    try:
        return int(m.group("run"))
    except Exception:
        return None


def _parse_setpoint(p: Path) -> Optional[float]:
    """Parse _I_<value>A from filename, if present."""
    m = _RE_SETPOINT.search(p.name)
    if not m:
        return None
    try:
        return float(m.group("i"))
    except Exception:
        return None


def _peek_first_time_value(path: Path, *, sep_regex: str = r"\s+") -> Optional[float]:
    """
    Read the first numeric value in the first non-empty line of a text file.
    Used only to order plateau-per-file inputs.
    """
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("%"):
                    continue
                parts = re.split(sep_regex, s)
                return float(parts[0])
    except Exception:
        return None
    return None


def order_plateau_files(paths: Sequence[str]) -> list[str]:
    """
    Order plateau-per-file inputs.

    Strategy
    --------
    1) If the first time value differs significantly across files, sort by that.
    2) Otherwise sort by parsed filename timestamp.
    3) Otherwise lexicographic.

    Examples
    --------
    >>> order_plateau_files(["b_20250101_000000.txt", "a_20240101_000000.txt"])[0].startswith("a_")
    True
    """
    ps = [Path(p) for p in paths]

    first_times = [(_peek_first_time_value(p), p) for p in ps]
    ft_vals = [x for x, _ in first_times if x is not None]

    # If time bases across files appear global, use them.
    if len(ft_vals) >= 2 and (max(ft_vals) - min(ft_vals)) > 5.0:
        return [str(p) for _, p in sorted(first_times, key=lambda x: float("inf") if x[0] is None else x[0])]

    # Otherwise rely on filename timestamp if available.
    dated = [(_parse_ts(p), p) for p in ps]
    if any(ts is not None for ts, _ in dated):
        return [str(p) for ts, p in sorted(dated, key=lambda x: pd.Timestamp.min if x[0] is None else x[0])]

    return sorted([str(p) for p in ps])


def _standardize_headerless(df: pd.DataFrame, extra_names: Optional[Sequence[str]]) -> pd.DataFrame:
    """
    Apply canonical names to headerless numeric table.
    """
    df.columns = default_column_names(df.shape[1], extra_names=extra_names)
    return df


def read_txt_plateau(path: str, cfg: LoadConfig) -> PlateauData:
    """
    Read a whitespace-delimited headerless TXT plateau file.

    Expected column order (minimum 4):
      time_s, dphi_abs, dphi_cmp, current_a, [extras...]

    Examples
    --------
    >>> # df, validation are returned in PlateauData
    >>> # item = read_txt_plateau("some_plateau.txt", LoadConfig())
    """
    p = Path(path)
    raw = pd.read_csv(p, sep=r"\s+", header=None, comment="#", engine="python")
    df = _standardize_headerless(raw, cfg.extra_names)
    df.insert(0, "source_file", p.name)

    dropped = 0
    if cfg.drop_nonfinite:
        df2, dropped = drop_nonfinite_rows(df, CORE_COLS)
        df = df2

    vr0 = validate_core_table(df.drop(columns=["source_file"]))
    vr = ValidationResult(ok=vr0.ok, errors=vr0.errors, warnings=vr0.warnings, dropped_nonfinite_rows=dropped)

    meta = PlateauMeta(
        source_file=p.name,
        source_path=str(p),
        file_kind="txt",
        parsed_timestamp=_parse_ts(p),
        run_index=_parse_run_index(p),
        current_setpoint=_parse_setpoint(p),
    )
    return PlateauData(meta=meta, df=df, validation=vr)


def read_csv_plateau(path: str, cfg: LoadConfig) -> PlateauData:
    """
    Read a CSV plateau file (headered or headerless).

    If headered, we accept canonical names or common aliases:
      - abstime/time -> time_s
      - abs         -> dphi_abs
      - cmp         -> dphi_cmp
      - current     -> current_a

    Examples
    --------
    >>> # item = read_csv_plateau("plateau.csv", LoadConfig(csv_has_header=True))
    """
    p = Path(path)

    if cfg.csv_has_header:
        df = pd.read_csv(p)

        # Remap common legacy names into canonical names.
        rename_map = {
            "abstime": "time_s",
            "time": "time_s",
            "abs": "dphi_abs",
            "cmp": "dphi_cmp",
            "current": "current_a",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        missing = [c for c in CORE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{p.name}: CSV header mode requires columns {CORE_COLS}. Missing={missing}. Present={list(df.columns)}"
            )
    else:
        raw = pd.read_csv(p, header=None)
        df = _standardize_headerless(raw, cfg.extra_names)

    df.insert(0, "source_file", p.name)

    dropped = 0
    if cfg.drop_nonfinite:
        df2, dropped = drop_nonfinite_rows(df, CORE_COLS)
        df = df2

    vr0 = validate_core_table(df.drop(columns=["source_file"]))
    vr = ValidationResult(ok=vr0.ok, errors=vr0.errors, warnings=vr0.warnings, dropped_nonfinite_rows=dropped)

    meta = PlateauMeta(
        source_file=p.name,
        source_path=str(p),
        file_kind="csv",
        parsed_timestamp=_parse_ts(p),
        run_index=_parse_run_index(p),
        current_setpoint=_parse_setpoint(p),
    )
    return PlateauData(meta=meta, df=df, validation=vr)


def _infer_bin_n_channels(n_doubles: int, max_channels: int) -> int:
    """
    Infer number of channels after the first 3 columns for BIN files.

    BIN layout assumed:
      [time_s, dphi_abs, dphi_cmp, channel_1, ..., channel_N]
    where channel_1 is current_a, channel_2+ are extras.

    Returns N (>=1).
    """
    candidates = [n for n in range(1, max_channels + 1) if n_doubles % (3 + n) == 0]
    if not candidates:
        raise ValueError(
            f"Cannot infer N channels: {n_doubles} float64 values not divisible by (3+N) for N=1..{max_channels}."
        )
    return min(candidates)


def read_bin_run(path: str, cfg: LoadConfig) -> PlateauData:
    """
    Read a whole-run BIN file.

    BIN is assumed to be raw little-endian float64 values with columns:
      time_s, dphi_abs, dphi_cmp, current_a, (optional extra channels...)

    Examples
    --------
    >>> # item = read_bin_run("cycle.bin", LoadConfig())
    """
    p = Path(path)
    raw = np.fromfile(p, dtype="<f8")  # little-endian float64

    if raw.size == 0:
        raise ValueError(f"{p.name}: empty bin file.")

    n_channels = cfg.bin_n_currents
    if n_channels is None:
        n_channels = _infer_bin_n_channels(raw.size, cfg.bin_max_currents_to_try)

    ncols = 3 + n_channels
    if raw.size % ncols != 0:
        raise ValueError(f"{p.name}: raw.size={raw.size} not divisible by ncols={ncols} (3+{n_channels}).")

    mat = raw.reshape((-1, ncols))

    # Name columns: first 3 are fixed, then channel_1 is current_a by definition.
    base_cols = ["time_s", "dphi_abs", "dphi_cmp"]

    if cfg.bin_channel_names is None:
        chan_cols = ["current_a"] + [f"extra_{i}" for i in range(1, n_channels)]
    else:
        chan_cols = list(cfg.bin_channel_names)
        if len(chan_cols) != n_channels:
            raise ValueError(f"{p.name}: bin_channel_names length must equal n_channels={n_channels}.")

        # Ensure the first channel is current_a for downstream consistency.
        # If the user named it differently, we still force rename to current_a.
        if chan_cols[0] != "current_a":
            chan_cols = ["current_a"] + chan_cols[1:]

    cols = base_cols + chan_cols
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "source_file", p.name)

    dropped = 0
    if cfg.drop_nonfinite:
        df2, dropped = drop_nonfinite_rows(df, CORE_COLS)
        df = df2

    vr0 = validate_core_table(df.drop(columns=["source_file"]))
    vr = ValidationResult(ok=vr0.ok, errors=vr0.errors, warnings=vr0.warnings, dropped_nonfinite_rows=dropped)

    meta = PlateauMeta(
        source_file=p.name,
        source_path=str(p),
        file_kind="bin",
        parsed_timestamp=_parse_ts(p),
        run_index=_parse_run_index(p),
        current_setpoint=_parse_setpoint(p),
    )
    return PlateauData(meta=meta, df=df, validation=vr)


def _split_by_time_gaps(df: pd.DataFrame, gap_threshold_s: float) -> List[pd.DataFrame]:
    """
    Split a run into segments if there are large time gaps.
    """
    t = df["time_s"].to_numpy()
    if t.size < 2:
        return [df]

    dt = np.diff(t)
    cuts = np.where(dt > gap_threshold_s)[0] + 1
    if cuts.size == 0:
        return [df]

    bounds = [0] + cuts.tolist() + [len(df)]
    return [df.iloc[a:b].copy() for a, b in zip(bounds[:-1], bounds[1:])]


def split_run_into_plateaus_by_current(
    df: pd.DataFrame,
    *,
    smooth_window: int,
    slope_threshold_a_per_s: float,
    level_rounding_a: float,
    min_duration_s: float,
) -> List[pd.DataFrame]:
    """
    Split a long run into current plateaus.

    The logic is intentionally conservative:
    - smooth current with a rolling median
    - compute dI/dt
    - mark points as "plateau-like" when |dI/dt| <= threshold
    - group contiguous plateau-like regions
    - compute a robust plateau level (median current), quantize to level_rounding_a
    - drop short regions by duration
    - merge consecutive regions with the same quantized level

    Parameters
    ----------
    df:
        Whole-run table with columns including time_s and current_a.
    smooth_window:
        Rolling median window in samples.
    slope_threshold_a_per_s:
        |dI/dt| threshold to consider a point part of a plateau.
    level_rounding_a:
        Quantization step in amperes for grouping plateau levels.
    min_duration_s:
        Minimum duration (seconds) to keep a plateau.

    Returns
    -------
    list[pd.DataFrame]
        Each dataframe includes plateau_id and plateau_level_a.

    Examples
    --------
    >>> # plateaus = split_run_into_plateaus_by_current(run_df, smooth_window=2001,
    ... #     slope_threshold_a_per_s=1.0, level_rounding_a=10.0, min_duration_s=5.0)
    """
    if len(df) == 0:
        return []

    # Smooth current for robust derivative and level estimation.
    cur = pd.Series(df["current_a"].to_numpy())
    cur_s = cur.rolling(window=smooth_window, center=True, min_periods=1).median().to_numpy()

    t = df["time_s"].to_numpy()
    # Derivative with respect to time (handles non-uniform sampling).
    dcur_dt = np.gradient(cur_s, t, edge_order=1)

    is_plateau = np.abs(dcur_dt) <= slope_threshold_a_per_s

    # Find contiguous True regions.
    idx = np.arange(len(df))
    true_idx = idx[is_plateau]
    if true_idx.size == 0:
        return []

    # Build segments from consecutive indices.
    cuts = np.where(np.diff(true_idx) > 1)[0]
    starts = np.r_[true_idx[0], true_idx[cuts + 1]]
    ends = np.r_[true_idx[cuts], true_idx[-1]]

    segments: List[pd.DataFrame] = []
    for a, b in zip(starts, ends):
        seg = df.iloc[int(a) : int(b) + 1].copy()
        dur = float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0])
        if dur < min_duration_s:
            continue

        level = float(np.median(seg["current_a"].to_numpy()))
        level_q = float(np.round(level / level_rounding_a) * level_rounding_a)

        seg["plateau_level_a"] = level_q
        segments.append(seg)

    # Merge consecutive segments with same quantized level (common if a brief glitch splits a plateau).
    merged: List[pd.DataFrame] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        if float(seg["plateau_level_a"].iloc[0]) == float(merged[-1]["plateau_level_a"].iloc[0]):
            merged[-1] = pd.concat([merged[-1], seg], ignore_index=True)
        else:
            merged.append(seg)

    # Assign plateau_id
    for i, seg in enumerate(merged, start=1):
        seg["plateau_id"] = i

    return merged


def load_measurements(
    paths: Sequence[str],
    cfg: LoadConfig = LoadConfig(),
) -> tuple[List[pd.DataFrame], List[PlateauData]]:
    """
    Phase 1 main entry point.

    Parameters
    ----------
    paths:
        Either:
        - multiple .txt/.csv plateau files, OR
        - a single .bin run file.
    cfg:
        LoadConfig controlling parsing, cleaning, and splitting.

    Returns
    -------
    (plateaus, raw_items)
    plateaus:
        List of per-plateau DataFrames (each includes plateau_id).
    raw_items:
        List of PlateauData containing validation and metadata for each input file.

    Examples
    --------
    >>> # TXT plateau-per-file:
    >>> # plateaus, items = load_measurements(["p1.txt","p2.txt"], LoadConfig())
    >>> # BIN whole-run:
    >>> # plateaus, items = load_measurements(["cycle.bin"], LoadConfig(split_long_run_into_plateaus=True))
    """
    paths = list(paths)
    if not paths:
        raise ValueError("No input paths provided.")

    exts = {Path(p).suffix.lower() for p in paths}

    # Plateau-per-file mode: multiple TXT/CSV files.
    if exts <= {".txt", ".csv"}:
        ordered = order_plateau_files(paths)
        raw_items: List[PlateauData] = []

        for p in ordered:
            ext = Path(p).suffix.lower()
            if ext == ".txt":
                item = read_txt_plateau(p, cfg)
            else:
                item = read_csv_plateau(p, cfg)
            raw_items.append(item)

        # Assign plateau_id sequentially after ordering.
        plateaus: List[pd.DataFrame] = []
        for i, item in enumerate(raw_items, start=1):
            df = item.df.copy()
            df["plateau_id"] = i
            plateaus.append(df)

        return plateaus, raw_items

    # Whole-run mode: single BIN.
    if exts == {".bin"} and len(paths) == 1:
        item = read_bin_run(paths[0], cfg)
        raw_items = [item]

        if not cfg.split_long_run_into_plateaus:
            df = item.df.copy()
            df["plateau_id"] = 1
            return [df], raw_items

        # Split by time gaps first, then plateau-split inside each segment.
        segments = _split_by_time_gaps(item.df, cfg.gap_threshold_s)

        plateaus_all: List[pd.DataFrame] = []
        plateau_counter = 0
        for seg in segments:
            plateaus_seg = split_run_into_plateaus_by_current(
                seg,
                smooth_window=cfg.plateau_smooth_window,
                slope_threshold_a_per_s=cfg.plateau_slope_threshold_a_per_s,
                level_rounding_a=cfg.plateau_level_rounding_a,
                min_duration_s=cfg.plateau_min_duration_s,
            )
            for pseg in plateaus_seg:
                plateau_counter += 1
                pseg["plateau_id"] = plateau_counter
                plateaus_all.append(pseg)

        return plateaus_all, raw_items

    raise ValueError(
        f"Unsupported input mix: extensions={sorted(exts)}, count={len(paths)}. "
        "Provide multiple .txt/.csv plateau files, or one .bin run file."
    )


def make_plateau_index(plateaus: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Create a compact plateau index table for reporting/export.

    Returns columns like:
    - plateau_id
    - n_samples
    - time_start_s, time_end_s, duration_s
    - current_mean_a, current_std_a
    - plateau_level_a (if present)
    - source_files
    - extras

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"time_s":[0,1], "dphi_abs":[1,1], "dphi_cmp":[0,0], "current_a":[10,10], "plateau_id":[1,1], "source_file":["x","x"]})
    >>> make_plateau_index([df]).iloc[0]["plateau_id"]
    1
    """
    rows = []
    for df in plateaus:
        pid = int(df["plateau_id"].iloc[0])
        rows.append(
            {
                "plateau_id": pid,
                "n_samples": int(len(df)),
                "time_start_s": float(df["time_s"].iloc[0]),
                "time_end_s": float(df["time_s"].iloc[-1]),
                "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
                "current_mean_a": float(df["current_a"].mean()),
                "current_std_a": float(df["current_a"].std()),
                "plateau_level_a": float(df["plateau_level_a"].iloc[0]) if "plateau_level_a" in df.columns else np.nan,
                "source_files": ",".join(sorted(set(df["source_file"].astype(str).tolist()))),
                "extras": ",".join([c for c in df.columns if c.startswith("extra_")]),
            }
        )
    return pd.DataFrame(rows)


def export_dataframe(df: pd.DataFrame, path: str, fmt: ExportFormat) -> None:
    """
    Export a DataFrame to CSV or Parquet.

    Parquet requires `pyarrow` (recommended) or `fastparquet`.

    Examples
    --------
    >>> # export_dataframe(df, "processed_data/plateau_index.csv", "csv")
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(out, index=False)
        return

    if fmt == "parquet":
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            raise RuntimeError(
                "Parquet export failed. Install 'pyarrow' (recommended) or 'fastparquet'. "
                f"Original error: {e}"
            ) from e
        return

    raise ValueError(f"Unknown export format: {fmt}")


def export_phase1_outputs(
    plateaus: Sequence[pd.DataFrame],
    out_dir: str,
    *,
    fmt: ExportFormat,
    index_name: str = "plateau_index",
) -> None:
    """
    Export Phase 1 intermediate outputs:
    - plateau index table
    - each plateau table

    Examples
    --------
    >>> # export_phase1_outputs(plateaus, "processed_data", fmt="csv")
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_df = make_plateau_index(plateaus)
    export_dataframe(index_df, str(out / f"{index_name}.{fmt}"), fmt)

    for df in plateaus:
        pid = int(df["plateau_id"].iloc[0])
        export_dataframe(df, str(out / f"plateau_{pid:03d}.{fmt}"), fmt)
