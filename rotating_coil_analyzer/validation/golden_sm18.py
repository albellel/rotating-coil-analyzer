from __future__ import annotations

"""Golden-standard validation helper for SM18 *corr_sigs* measurements.

This module provides a small CLI-oriented workflow:

1) Discover a measurement folder (Parameters.txt + corr_sigs files).
2) Load each segment via the standard readers.
3) Run the legacy-compatible $k_n$ pipeline.
4) Export analyzer outputs in a stable CSV schema.
5) Optionally, parse a reference "results_*.txt" export and compute diffs.

The intent is to support a validation campaign without touching the GUI.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import glob

from rotating_coil_analyzer.analysis.kn_pipeline import (
    LegacyKnPerTurn,
    SegmentKn,
    compute_legacy_kn_per_turn,
    load_segment_kn_txt,
    merge_coefficients,
)
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig
from rotating_coil_analyzer.models.catalog import MeasurementCatalog
from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class GoldenRunConfig:
    """Configuration for a validation run."""

    run_id: Optional[str] = None
    apertures: Optional[Sequence[int]] = None
    segments: Optional[Sequence[str]] = None

    magnet_order: Optional[int] = None
    Rref_m: Optional[float] = None
    absCalib: float = 1.0

    options: Tuple[str, ...] = ("dri", "rot", "nor", "cel", "fed")

    # If provided, explicitly select a kn file per (ap, seg).
    kn_override: Optional[Dict[Tuple[int, str], Path]] = None

    # If provided, use this kn file for all processed segments (CLI convenience).
    kn_file: Optional[Path] = None

    # If True, attempt to find and compare against reference results exports.
    compare_to_reference: bool = True


def _find_single_run(cat: MeasurementCatalog) -> str:
    if len(cat.runs) == 1:
        return str(cat.runs[0])
    raise ValueError(
        f"Catalog contains multiple runs: {cat.runs}. Provide run_id explicitly to disambiguate."
    )


def _find_kn_file(folder: Path, *, ap: int, seg: str, run_id: str) -> Optional[Path]:
    """Find a Kn_values file for (ap, seg).

    The SM18 script ecosystem has seen multiple naming conventions. This helper tries
    a few patterns and returns the first match.
    """

    seg_esc = re.escape(str(seg))
    ap_esc = re.escape(str(ap))
    run_esc = re.escape(str(run_id))

    patterns = [
        # Example (as in your screenshot): *_Kn_values_Ap_1_Seg_1.txt
        rf"^{run_esc}.*_Kn_values_Ap_{ap_esc}_Seg_?{seg_esc}\.txt$",
        # Sometimes "Seg1" without underscore
        rf"^{run_esc}.*_Kn_values_Ap_{ap_esc}_Seg{seg_esc}\.txt$",
        # Generic match if run_id not embedded
        rf".*_Kn_values_Ap_{ap_esc}_Seg_?{seg_esc}\.txt$",
    ]

    for p in sorted(folder.rglob("*.txt")):
        name = p.name
        for pat in patterns:
            if re.match(pat, name, flags=re.IGNORECASE):
                return p
    return None


def _find_reference_results_file(folder: Path, *, ap: int, seg: str, run_id: str) -> Optional[Path]:
    seg_esc = re.escape(str(seg))
    ap_esc = re.escape(str(ap))
    run_esc = re.escape(str(run_id))
    patterns = [
        rf"^{run_esc}.*_results_Ap_{ap_esc}_Seg_?{seg_esc}\.txt$",
        rf"^{run_esc}.*_results_Ap_{ap_esc}_Seg{seg_esc}\.txt$",
        rf".*_results_Ap_{ap_esc}_Seg_?{seg_esc}\.txt$",
    ]
    for p in sorted(folder.rglob("*.txt")):
        name = p.name
        for pat in patterns:
            if re.match(pat, name, flags=re.IGNORECASE):
                return p
    return None


def _read_reference_results_txt(path: Path) -> pd.DataFrame:
    """Best-effort parser for legacy/reference "results_*.txt" exports.

    This parser is intentionally permissive:
    - ignores comment lines (starting with '#')
    - accepts whitespace or tab separation
    - if a header row exists, it is used; otherwise numeric columns are auto-named

    The comparison layer matches common column naming conventions (bN/aN, Bn/An, etc.).
    """

    import io

    def _guess_sep_from_header(header_line: str) -> str:
        """Guess a delimiter for SM18 legacy exports.

        Practical issue
        ---------------
        The reference ``results_*.txt`` files are often *fixed-width* tables.
        The ``Options`` column can contain tokens separated by *single* spaces
        (e.g. ``dri rot nor cel dit``), while true column boundaries are
        represented by *multiple* spaces or by tabs.

        Therefore we prefer:
        - tab, if present
        - otherwise, 2+ spaces as delimiter (keeps single-space tokens intact)
        """

        if "\t" in header_line:
            return "\t"
        if ";" in header_line:
            return ";"
        if "," in header_line:
            return ","
        return r"\s{2,}"

    def _is_numeric_series(x: pd.Series, *, frac_ok: float = 0.95) -> bool:
        v = pd.to_numeric(x, errors="coerce")
        if v.size == 0:
            return False
        return float(np.isfinite(v).mean()) >= float(frac_ok)

    lines = path.read_text(errors="ignore").splitlines()

    # Prefer the standard SM18 results header (Time(s) ...).
    header_idx: Optional[int] = None
    header_line = ""
    for i, raw in enumerate(lines[:500]):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if ("Time(s)" in s) and ("Duration(s)" in s):
            header_idx = i
            header_line = s
            break
    # Fall back: some exports have a 'turn' header.
    if header_idx is None:
        for i, raw in enumerate(lines[:500]):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if re.search(r"\bturn\b", s, flags=re.IGNORECASE):
                header_idx = i
                header_line = s
                break

    # If we found a header line, **prefer** fixed-width parsing.
    # Rationale:
    # - The legacy exports are typically aligned tables.
    # - The "Options" column can contain single-space tokens.
    # - Delimiter-based parsing can silently shift columns, leading to
    #   pathological cases where Time(s) becomes a token like 'nor'.
    if header_idx is not None:
        try:
            # Infer column start positions from the header text itself.
            matches = list(re.finditer(r"\S+", header_line.rstrip("\n")))
            if len(matches) >= 2:
                cols = [m.group(0).strip() for m in matches]
                colspecs = []
                for i, m in enumerate(matches):
                    start = int(m.start())
                    end = int(matches[i + 1].start()) if i + 1 < len(matches) else None
                    colspecs.append((start, end))

                data_lines = []
                for raw in lines[int(header_idx) + 1 :]:
                    s = raw.rstrip("\n")
                    if not s.strip() or s.lstrip().startswith("#"):
                        continue
                    data_lines.append(s)
                buf = io.StringIO("\n".join(data_lines))

                df_fwf = pd.read_fwf(buf, names=cols, colspecs=colspecs)
                if df_fwf.shape[0] > 0 and df_fwf.shape[1] > 1:
                    if "Time(s)" in df_fwf.columns:
                        df_fwf["Time(s)"] = pd.to_numeric(df_fwf["Time(s)"], errors="coerce")
                    # If Time(s) is not numeric, fall back to delimiter-based parsing.
                    if ("Time(s)" not in df_fwf.columns) or _is_numeric_series(df_fwf.get("Time(s)", pd.Series([], dtype=float))):
                        return df_fwf
        except Exception:
            pass

        # Delimiter-based fallback (kept for non-aligned exports).
        sep = _guess_sep_from_header(header_line)
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                comment="#",
                skiprows=int(header_idx),
                engine="python",
            )
            if df.shape[0] > 0 and df.shape[1] > 1:
                if "Time(s)" in df.columns and _is_numeric_series(df["Time(s)"]):
                    df["Time(s)"] = pd.to_numeric(df["Time(s)"], errors="coerce")
                    return df
        except Exception:
            pass

    # Last resort: pandas inference with common separators.
    for sep in ("\t", r"\s{2,}", r"\s+", ",", ";"):
        for header in (0, None):
            try:
                df = pd.read_csv(path, sep=sep, comment="#", header=header, engine="python")
                if df.shape[0] > 0 and df.shape[1] > 1:
                    return df
            except Exception:
                continue

    raise ValueError(f"Empty or unparsable reference results file: {path}")


def _build_output_table(knr: LegacyKnPerTurn, *, magnet_order: int) -> pd.DataFrame:
    """Build a stable, comparison-friendly per-turn table."""

    H = int(knr.orders.size)
    n_turns = int(knr.C_abs.shape[0])

    out: Dict[str, np.ndarray] = {
        "turn": np.arange(n_turns, dtype=int),
        "I_mean_A": np.asarray(knr.I_mean_A, dtype=float),
        "time_median_s": np.asarray(knr.time_median_s, dtype=float),
        "dI_dt_A_per_s": np.asarray(knr.dI_dt_A_per_s, dtype=float),
        "duration_s": np.asarray(knr.duration_s, dtype=float),
        "phi_out_rad": np.asarray(knr.phi_out_rad, dtype=float),
        "phi_bad": np.asarray(knr.phi_bad, dtype=bool),
        "x_m": np.asarray(knr.x_m, dtype=float),
        "y_m": np.asarray(knr.y_m, dtype=float),
    }

    # Convenience aliases (will be overwritten in the golden-run wrapper if a more
    # appropriate time reference is available, e.g. turn start time).
    out["Time(s)"] = out["time_median_s"]
    out["Duration(s)"] = out["duration_s"]

    # Mixed (legacy) channel used by SM18 results files:
    #   - ABS for orders <= m
    #   - CMP for orders > m
    # This is what is usually meant by the "final reported harmonics".
    m = int(magnet_order)
    if not (1 <= m <= H):
        raise ValueError(f"magnet_order must be in [1, {H}], got {m}")
    
    # SM18 “results_*.txt” legacy mixed channel:
    #   ABS for orders <= m, CMP for orders > m
    C_mix, _choice_mix = merge_coefficients(
        C_abs=knr.C_abs,
        C_cmp=knr.C_cmp,
        magnet_order=m,
        mode="abs_upto_m_cmp_above",
    )
    C_mix_db, _choice_mix_db = merge_coefficients(
        C_abs=knr.C_abs_db,
        C_cmp=knr.C_cmp_db,
        magnet_order=m,
        mode="abs_upto_m_cmp_above",
    )

    # Export harmonics as Re/Im to avoid downstream CSV complex parsing issues.
    for i, n in enumerate(knr.orders.tolist()):
        out[f"abs_re_n{n}"] = np.real(knr.C_abs[:, i])
        out[f"abs_im_n{n}"] = np.imag(knr.C_abs[:, i])
        out[f"cmp_re_n{n}"] = np.real(knr.C_cmp[:, i])
        out[f"cmp_im_n{n}"] = np.imag(knr.C_cmp[:, i])
        out[f"mix_re_n{n}"] = np.real(C_mix[:, i])
        out[f"mix_im_n{n}"] = np.imag(C_mix[:, i])

        # DB snapshot: useful when reference exports correspond to "deb" stage.
        out[f"abs_db_re_n{n}"] = np.real(knr.C_abs_db[:, i])
        out[f"abs_db_im_n{n}"] = np.imag(knr.C_abs_db[:, i])
        out[f"cmp_db_re_n{n}"] = np.real(knr.C_cmp_db[:, i])
        out[f"cmp_db_im_n{n}"] = np.imag(knr.C_cmp_db[:, i])
        out[f"mix_db_re_n{n}"] = np.real(C_mix_db[:, i])
        out[f"mix_db_im_n{n}"] = np.imag(C_mix_db[:, i])

    return pd.DataFrame(out)


def _infer_bn_an_columns(df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Infer (b_n, a_n) column mapping from a reference dataframe.

    IMPORTANT
    ---------
    SM18 reference exports often contain BOTH:
      - physical-field columns in Tesla: e.g. 'B1(T)', 'A1(T)', 'B_main(T)', '...TF(T/kA)'
      - normalized multipoles in 1e-4 units: e.g. 'b2(Units)', 'a2(Units)', ...

    For validation against the analyzer's 'nor' option, we must compare ONLY
    the normalized multipole columns (bN/aN), and MUST NOT treat Tesla columns
    as bN/aN. Otherwise, 'B1(T)' gets mis-identified as 'b1' and comparisons
    explode by orders of magnitude.

    Returns
    -------
    b_map, a_map:
        dict mapping harmonic order n -> column name.
    """

    cols = list(df.columns)
    b_map: Dict[int, str] = {}
    a_map: Dict[int, str] = {}

    def _canon(name: object) -> str:
        s = str(name).strip().lower()
        # Remove bracketed unit hints but keep the base token.
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"\[.*?\]", "", s)
        s = re.sub(r"\{.*?\}", "", s)
        # Remove separators.
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def _is_tesla_like(raw: str) -> bool:
        s = raw.strip().lower()
        # Physical-field / transfer-function columns we must not use as b_n/a_n.
        if "(t" in s or "tesla" in s:
            return True
        if "tf" in s or "t/ka" in s or "tka" in s:
            return True
        # Common SM18 physical columns:
        if "b_main" in s or "a_main" in s:
            return True
        return False

    # Accept a broad set of b/a conventions, but ONLY for non-Tesla columns.
    b_re = re.compile(r"^(?:b|bn|normal)(?P<n>\d+)$", flags=re.IGNORECASE)
    a_re = re.compile(r"^(?:a|an|skew)(?P<n>\d+)$", flags=re.IGNORECASE)

    for c in cols:
        c0 = str(c)
        if _is_tesla_like(c0):
            continue

        cc = _canon(c0)
        mb = b_re.match(cc)
        if mb:
            b_map[int(mb.group("n"))] = c0
        ma = a_re.match(cc)
        if ma:
            a_map[int(ma.group("n"))] = c0

    return b_map, a_map


@dataclass(frozen=True)
class ComparisonSummary:
    n_turns_analyzer: int
    n_turns_reference: int
    n_matched: int
    per_column: pd.DataFrame


def compare_units_table(
    *,
    analyzer: pd.DataFrame,
    reference: pd.DataFrame,
    channel: str = "abs",
    stage: str = "final",
    max_order: Optional[int] = None,
    align_on_time: bool = True,
) -> ComparisonSummary:
    """Compare analyzer outputs to a reference results export in "units".

    Assumptions
    -----------
    - The reference export contains columns like b3/a3 (normal/skew) in $10^{-4}$ units.
    - The analyzer table is produced by :func:`_build_output_table`.

    Parameters
    ----------
    channel:
        "abs" or "cmp".
    stage:
        "final" compares abs_re_nN/abs_im_nN (or cmp_*) from the analyzer table.
        "db" compares *_db_* columns.
    """

    channel = str(channel).strip().lower()
    if channel not in {"abs", "cmp"}:
        raise ValueError("channel must be 'abs' or 'cmp'")
    stage = str(stage).strip().lower()
    if stage not in {"final", "db"}:
        raise ValueError("stage must be 'final' or 'db'")

    b_map, a_map = _infer_bn_an_columns(reference)
    if not b_map:
        cols = [str(c) for c in list(reference.columns)]
        preview = ", ".join(cols[:40])
        raise ValueError(
            "Could not infer any b_n columns from reference results. "
            "Expected column names like b3 and a3 (or B3/A3, bn3/an3, normal3/skew3), "
            f"possibly with unit suffixes. Available columns (first 40): {preview}"
        )

    n_turns_a = int(analyzer.shape[0])
    n_turns_r = int(reference.shape[0])

    # Preferred alignment: by measured time.
    # This avoids misleading diffs when one side contains a few extra/missing turns.
    merged: Optional[pd.DataFrame] = None
    n_matched = 0
    used_tol: Optional[float] = None
    used_dt: Optional[float] = None
    if bool(align_on_time) and ("Time(s)" in analyzer.columns) and ("Time(s)" in reference.columns):
        a = analyzer.copy()
        r = reference.copy()
        a["__t"] = pd.to_numeric(a["Time(s)"], errors="coerce")
        r["__t"] = pd.to_numeric(r["Time(s)"], errors="coerce")
        a = a.dropna(subset=["__t"]).sort_values("__t")
        r = r.dropna(subset=["__t"]).sort_values("__t")
        if a.shape[0] > 2 and r.shape[0] > 2:
            # Tolerance based on the median inter-turn spacing.
            # We deliberately keep this conservative to avoid "off-by-one-turn"
            # nearest-neighbor matches when the chosen time anchor differs
            # (e.g. start-of-turn vs median-of-turn).
            # Use a robust estimate of inter-turn spacing. If either side has
            # duplicated/degenerate time stamps, fall back to the other side.
            dt_a = float(np.nanmedian(np.diff(a["__t"].to_numpy())))
            dt_r = float(np.nanmedian(np.diff(r["__t"].to_numpy())))
            dt = dt_a
            if not np.isfinite(dt) or dt <= 0:
                dt = dt_r
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0

            # Tolerance: wide enough to tolerate different time anchors
            # (start vs median) but still narrow enough to avoid systematic
            # off-by-one-turn matching.
            tol = max(0.20, 0.30 * dt)
            used_dt = float(dt)
            used_tol = float(tol)
            merged = pd.merge_asof(
                a,
                r,
                on="__t",
                direction="nearest",
                tolerance=tol,
                suffixes=("_ana", "_ref"),
            )
            # Count how many analyzer rows found a reference match.
            # (We use a reference column that must exist: the first b-column we inferred.)
            probe_col = b_map[min(b_map.keys())]
            n_matched = int(pd.to_numeric(merged.get(probe_col), errors="coerce").notna().sum())

            # If the time alignment clearly failed, fall back to positional alignment.
            # This avoids producing all-NaN diffs (and meaningless RMS/"worst" metrics)
            # when the two time bases differ by a constant offset or use different anchors.
            n_expected = int(min(a.shape[0], r.shape[0]))
            if n_expected >= 100 and n_matched < int(0.50 * n_expected):
                merged = None
                n_matched = 0

    if merged is None:
        # Fallback: positional truncation.
        n = min(n_turns_a, n_turns_r)
        merged = None
    else:
        n = int(merged.shape[0])

    rows: List[Dict[str, object]] = []
    orders = sorted(b_map.keys())
    if max_order is not None:
        orders = [k for k in orders if k <= int(max_order)]

    for k in orders:
        if merged is not None and ("__t" in merged.columns):
            ref_b = pd.to_numeric(merged[b_map[k]], errors="coerce").to_numpy()
            a_col_name = a_map.get(k)
            if a_col_name is not None and a_col_name in merged.columns:
                ref_a = pd.to_numeric(merged[a_col_name], errors="coerce").to_numpy()
            else:
                ref_a = np.full((n,), np.nan, dtype=float)
        else:
            ref_b = pd.to_numeric(reference[b_map[k]], errors="coerce").to_numpy()[:n]
            a_col_name = a_map.get(k)
            if a_col_name is not None and a_col_name in reference.columns:
                ref_a = pd.to_numeric(reference[a_col_name], errors="coerce").to_numpy()[:n]
            else:
                ref_a = np.full((n,), np.nan, dtype=float)

        if stage == "final":
            b_col = f"{channel}_re_n{k}"
            a_col = f"{channel}_im_n{k}"
        else:
            b_col = f"{channel}_db_re_n{k}"
            a_col = f"{channel}_db_im_n{k}"

        if b_col not in analyzer.columns or a_col not in analyzer.columns:
            continue

        if merged is not None and ("__t" in merged.columns):
            ana_b = pd.to_numeric(merged[b_col], errors="coerce").to_numpy()
            ana_a = pd.to_numeric(merged[a_col], errors="coerce").to_numpy()
        else:
            ana_b = pd.to_numeric(analyzer[b_col], errors="coerce").to_numpy()[:n]
            ana_a = pd.to_numeric(analyzer[a_col], errors="coerce").to_numpy()[:n]

        db = ana_b - ref_b
        da = ana_a - ref_a

        # Avoid RuntimeWarnings from nanmax/nanmean on all-NaN arrays.
        if np.all(np.isnan(db)):
            b_abs_max = float("nan")
            b_rms = float("nan")
        else:
            b_abs_max = float(np.nanmax(np.abs(db)))
            b_rms = float(np.sqrt(np.nanmean(db**2)))
        if np.all(np.isnan(da)):
            a_abs_max = float("nan")
            a_rms = float("nan")
        else:
            a_abs_max = float(np.nanmax(np.abs(da)))
            a_rms = float(np.sqrt(np.nanmean(da**2)))

        rows.append(
            {
                "order_n": int(k),
                "b_col_ref": b_map[k],
                "a_col_ref": a_map.get(k, ""),
                "b_abs_max": b_abs_max,
                "b_rms": b_rms,
                "a_abs_max": a_abs_max,
                "a_rms": a_rms,
            }
        )

    per_col = pd.DataFrame(rows)
    return ComparisonSummary(
        n_turns_analyzer=n_turns_a,
        n_turns_reference=n_turns_r,
        n_matched=int(n_matched) if n_matched else int(min(n_turns_a, n_turns_r)),
        per_column=per_col,
    )


def _summary_metric(summary: ComparisonSummary) -> float:
    """Scalar metric to rank comparisons (lower is better).

    We use the median RMS over b and a columns to suppress outlier turns.
    """

    if summary.per_column is None or summary.per_column.empty:
        return float("inf")
    b = pd.to_numeric(summary.per_column.get("b_rms"), errors="coerce")
    a = pd.to_numeric(summary.per_column.get("a_rms"), errors="coerce")
    v = pd.concat([b, a], axis=0)
    return float(np.nanmedian(v.to_numpy()))


def _resolve_catalog_root(folder: Path, *, prefer_ap: int = 1, max_down: int = 3) -> Path:
    """Resolve a user-provided folder to a catalog root that contains Parameters.txt.

    The GUI workflow typically points Phase I to the *aperture* folder (e.g. ...\aperture1),
    which contains Parameters.txt. For validation, users often pass the parent run folder.
    This helper keeps MeasurementDiscovery strict while providing a controlled downward search.

    Strategy
    --------
    1) Try MeasurementDiscovery.build_catalog(folder) directly.
    2) If Parameters.txt is not found in folder or its parents, search for Parameters.txt in
       descendant folders up to `max_down` levels.
    3) Prefer the candidate whose path contains 'aperture{prefer_ap}' (if any), then the
       shallowest depth. If ambiguous, raise with a deterministic message.

    Returns
    -------
    Path
        Folder that should be passed to MeasurementDiscovery.build_catalog().
    """
    disc = MeasurementDiscovery(strict=True)
    try:
        disc.build_catalog(folder)
        return folder
    except FileNotFoundError as e:
        msg = str(e)
        if "Parameters.txt not found" not in msg:
            raise

    candidates: List[Path] = []
    for p in folder.rglob("Parameters.txt"):
        try:
            rel = p.relative_to(folder)
        except Exception:
            continue
        depth = len(rel.parts) - 1  # number of directory levels below `folder`
        if depth <= int(max_down):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Parameters.txt not found in {folder} (parents or descendants depth<={max_down}).")

    ap_token = f"aperture{int(prefer_ap)}"

    def _score(p: Path) -> Tuple[int, int, str]:
        rel_parent = str(p.parent).lower()
        ap_bonus = -1 if ap_token in rel_parent else 0  # prefer matching aperture
        depth = len(p.relative_to(folder).parts) - 1
        return (depth, ap_bonus, str(p))

    candidates_sorted = sorted(candidates, key=_score)
    best = candidates_sorted[0]
    best_score = _score(best)

    # Detect ambiguity among the best-scoring candidates.
    tied = [p for p in candidates_sorted if _score(p) == best_score]
    tied_parents = {p.parent for p in tied}
    if len(tied_parents) > 1:
        raise ValueError(
            "Ambiguous Parameters.txt discovery under "
            f"{folder} (depth<={max_down}). Candidates: "
            + ", ".join(str(pp) for pp in sorted(tied_parents))
            + ". Please pass the specific aperture folder (e.g. ...\\aperture1) "
            + "or set --ap/--seg to disambiguate."
        )

    return best.parent



def run_golden_folder(
    folder: str | Path,
    *,
    cfg: GoldenRunConfig,
    out_dir: Optional[str | Path] = None,
) -> Dict[Tuple[int, str], Dict[str, object]]:
    """Run the analyzer on a golden-standard folder.

    Returns a dict keyed by (aperture, segment_id) containing:
      - segment_frame: SegmentFrame
      - kn: SegmentKn
      - legacy: LegacyKnPerTurn
      - table: pandas DataFrame (export table)
      - reference_path (optional)
      - comparison (optional)
    """

    folder = Path(folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(str(folder))

    prefer_ap = 1
    if cfg.apertures is not None and len(list(cfg.apertures)) == 1:
        prefer_ap = int(list(cfg.apertures)[0])

    folder = _resolve_catalog_root(folder, prefer_ap=prefer_ap, max_down=3)
    cat = MeasurementDiscovery(strict=True).build_catalog(folder)
    run_id = str(cfg.run_id) if cfg.run_id is not None else _find_single_run(cat)

    aps = list(cat.enabled_apertures)
    if cfg.apertures is not None:
        aps = [int(a) for a in cfg.apertures]

    seg_filter = None
    if cfg.segments is not None:
        seg_filter = {str(s) for s in cfg.segments}

    magnet_order = int(cfg.magnet_order) if cfg.magnet_order is not None else cat.magnet_order
    if magnet_order is None or int(magnet_order) <= 0:
        raise ValueError("magnet_order is not defined (Parameters.magnetAnalyzer.magnetOrder <= 0). Provide cfg.magnet_order.")

    Rref_m = cfg.Rref_m
    if Rref_m is None or not np.isfinite(Rref_m) or float(Rref_m) <= 0.0:
        raise ValueError("Rref_m must be provided and > 0 for a golden run.")

    options = tuple(str(x).strip().lower() for x in cfg.options)

    reader = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=5))

    out_root = Path(out_dir).expanduser().resolve() if out_dir is not None else (folder / "validation_out")
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[Tuple[int, str], Dict[str, object]] = {}

    for ap in aps:
        seg_specs = cat.segments_for_aperture(ap)
        for ss in seg_specs:
            seg_id = str(ss.segment_id)
            if seg_filter is not None and seg_id not in seg_filter:
                continue

            seg_path = cat.get_segment_file(run_id, ap, seg_id)
            segf: SegmentFrame = reader.read(
                seg_path,
                run_id=run_id,
                segment=seg_id,
                samples_per_turn=cat.samples_per_turn,
                shaft_speed_rpm=cat.shaft_speed_rpm,
                aperture_id=ap,
                magnet_order=int(magnet_order),
            )

            tb = split_into_turns(segf)

            # kn: forced file, override-map, or auto-find
            kn_path = None
            if cfg.kn_file is not None:
                kn_path = Path(cfg.kn_file)
            if kn_path is None and cfg.kn_override is not None:
                kn_path = cfg.kn_override.get((ap, seg_id))
            if kn_path is None:
                kn_path = _find_kn_file(folder, ap=ap, seg=seg_id, run_id=run_id)
            if kn_path is None:
                raise FileNotFoundError(
                    f"Could not locate a kn file for ap={ap} seg={seg_id}. "
                    f"Expected a '*Kn_values_Ap_{ap}_Seg_{seg_id}.txt' style file (or provide cfg.kn_override)."
                )
            kn: SegmentKn = load_segment_kn_txt(str(kn_path))

            legacy: LegacyKnPerTurn = compute_legacy_kn_per_turn(
                df_abs_turns=tb.df_abs,
                df_cmp_turns=tb.df_cmp,
                t_turns=tb.t,
                I_turns=tb.I,
                kn=kn,
                Rref_m=float(Rref_m),
                magnet_order=int(magnet_order),
                absCalib=float(cfg.absCalib),
                options=options,
            )

            table = _build_output_table(legacy, magnet_order=int(magnet_order))
            # Export start/end timestamps as additional diagnostics, but keep
            # the legacy-facing alias "Time(s)" as the **turn median**. This
            # matches the common golden-standard export convention where
            # "Time(s)" ~= turn midpoint (approximately half the turn duration
            # after the start).
            t_start = np.asarray(tb.t[:, 0], dtype=float)
            t_end = np.asarray(tb.t[:, -1], dtype=float)
            table["time_start_s"] = t_start
            table["time_end_s"] = t_end
            out_csv = out_root / f"analyzer_Ap_{ap}_Seg_{seg_id}.csv"
            table.to_csv(out_csv, index=False)

            bundle: Dict[str, object] = {
                "segment_path": seg_path,
                "segment_frame": segf,
                "kn_path": kn_path,
                "kn": kn,
                "legacy": legacy,
                "table": table,
                "out_csv": out_csv,
            }

            if cfg.compare_to_reference:
                ref_path = _find_reference_results_file(folder, ap=ap, seg=seg_id, run_id=run_id)
                if ref_path is not None:
                    ref_df = _read_reference_results_txt(ref_path)
                    # Try the 4 plausible legacy interpretations and keep the best.
                    # (Some reference exports correspond to compensated channel and/or DB stage.)
                    cmp_candidates: Dict[str, object] = {}
                    best_key: Optional[str] = None
                    best_metric: float = float("inf")
                    for ch in ("mix", "abs", "cmp"):
                        for st in ("final", "db"):
                            key = f"{ch}/{st}"
                            try:
                                s = compare_units_table(analyzer=table, reference=ref_df, channel=ch, stage=st)
                                # Robust metric: median RMS over available orders (b and a).
                                if s.per_column.shape[0] == 0:
                                    raise ValueError("no comparable b/a columns")
                                m = float(np.nanmedian(np.r_[s.per_column["b_rms"].to_numpy(), s.per_column["a_rms"].to_numpy()]))
                                cmp_candidates[key] = (s, m)
                                if np.isfinite(m) and m < best_metric:
                                    best_metric = m
                                    best_key = key
                            except Exception as e:
                                cmp_candidates[key] = e

                    if best_key is None:
                        cmp_sum = {
                            "best": None,
                            "best_metric_median_rms": float("nan"),
                            "candidates": cmp_candidates,
                            "note": (
                                "No candidate produced a finite median-RMS metric. "
                                "This typically indicates a time-alignment failure (0 matches) "
                                "or missing/non-numeric b_n/a_n columns on one side."
                            ),
                        }
                    else:
                        cmp_sum = {
                            "best": best_key,
                            "best_metric_median_rms": best_metric,
                            "candidates": cmp_candidates,
                        }
                    bundle["reference_path"] = ref_path
                    bundle["reference_df"] = ref_df
                    bundle["comparison"] = cmp_sum

            results[(ap, seg_id)] = bundle

    return results


def _parse_options_csv(s: str) -> Tuple[str, ...]:
    toks = [t.strip() for t in re.split(r"[\s,;]+", s.strip()) if t.strip()]
    return tuple(t.lower() for t in toks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import textwrap

    p = argparse.ArgumentParser(
        prog="python -m rotating_coil_analyzer.validation.golden_sm18",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Run the rotating-coil analyzer on an SM18 golden-standard folder.

            The folder must contain Parameters.txt and corr_sigs/generic_corr_sigs files.
            If 'compare' is enabled, the folder should also contain reference exports
            named '*_results_Ap_X_Seg_Y.txt'.
            """
        ),
    )

    p.add_argument("folder", help="Measurement folder (contains Parameters.txt and corr_sigs files)")
    p.add_argument("--run-id", default=None, help="Run id (if folder contains multiple runs)")
    p.add_argument("--ap", default=None, help="Comma-separated apertures to process (e.g. '1' or '1,2')")
    p.add_argument("--seg", default=None, help="Comma-separated segment ids to process (e.g. 'NCS,CS' or '1,2,3')")
    p.add_argument("--magnet-order", type=int, default=None, help="Main order m (1 dipole, 2 quadrupole, ...)")
    p.add_argument("--rref-m", type=float, required=True, help="Reference radius Rref in meters")
    p.add_argument("--abs-calib", type=float, default=1.0, help="absCalib factor (legacy convention)")
    p.add_argument(
        "--kn-file",
        default=None,
        help=(
            "Optional explicit Kn_values file path to use for all processed segments. "
            "If omitted, the script auto-discovers '*_Kn_values_Ap_X_Seg_Y.txt' under the folder."
        ),
    )
    p.add_argument(
        "--options",
        default="dri,rot,nor,cel,fed",
        help="Comma/space separated legacy options among: dit,dri,rot,cel,fed,nor",
    )
    p.add_argument("--no-compare", action="store_true", help="Do not attempt to parse/compare reference results_*.txt")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <folder>/validation_out)")

    ns = p.parse_args(list(argv) if argv is not None else None)

    aps = None
    if ns.ap:
        aps = [int(x) for x in ns.ap.split(",") if x.strip()]
    segs = None
    if ns.seg:
        segs = [s.strip() for s in ns.seg.split(",") if s.strip()]

    kn_file = None
    if ns.kn_file:
        # Users sometimes paste shortened paths with "..." (UI elision). Treat that as a glob
        # and fall back to auto-discovery when the provided path cannot be resolved.
        kn_arg = str(ns.kn_file).strip()
        candidate = Path(kn_arg).expanduser()

        if candidate.exists() and candidate.is_file():
            kn_file = candidate.resolve()
        else:
            # 1) Try glob expansion (also handles "..." -> "*").
            pattern = kn_arg.replace("...", "*")
            matches = [Path(m) for m in glob.glob(pattern)]
            matches = [m for m in matches if m.exists() and m.is_file()]

            if len(matches) == 1:
                kn_file = matches[0].resolve()
                print(f"[info] resolved --kn-file via glob: {kn_file}")
            else:
                # 2) Fall back to auto-discovery under the selected folder.
                print(f"[warn] --kn-file not found: {kn_arg!r}. Falling back to auto-discovery in folder.")
                kn_file = None

    cfg = GoldenRunConfig(
        run_id=ns.run_id,
        apertures=aps,
        segments=segs,
        magnet_order=ns.magnet_order,
        Rref_m=float(ns.rref_m),
        absCalib=float(ns.abs_calib),
        options=_parse_options_csv(ns.options),
        kn_file=kn_file,
        compare_to_reference=(not bool(ns.no_compare)),
    )

    bundles = run_golden_folder(ns.folder, cfg=cfg, out_dir=ns.out_dir)

    # Print a compact summary.
    for (ap, seg), b in bundles.items():
        knp = b.get("kn_path")
        knp_s = str(knp) if knp is not None else "<none>"
        print(f"[Ap {ap} Seg {seg}] wrote: {b['out_csv']} (kn: {knp_s})")
        if "comparison" in b:
            cmp_obj = b["comparison"]
            if isinstance(cmp_obj, dict) and "best" in cmp_obj:
                best = str(cmp_obj.get("best"))
                best_m = float(cmp_obj.get("best_metric_median_rms", float("nan")))
                print(f"  compare: best={best} (median RMS over b/a, units) = {best_m:.3g}")
                cand = cmp_obj.get("candidates", {})
                for key in ("abs/final", "abs/db", "cmp/final", "cmp/db"):
                    v = cand.get(key)
                    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], ComparisonSummary):
                        s: ComparisonSummary = v[0]
                        m = float(v[1])
                        df = s.per_column
                        ncols = int(df.shape[0])
                        worst_b = float(df["b_abs_max"].max()) if ncols else float("nan")
                        worst_a = float(df["a_abs_max"].max()) if ncols else float("nan")
                        print(
                            f"    {key}: n_turns ana/ref/matched={s.n_turns_analyzer}/{s.n_turns_reference}/{s.n_matched}; "
                            f"median RMS={m:.3g}; worst |Δb|={worst_b:.3g}, worst |Δa|={worst_a:.3g}"
                        )
                    elif isinstance(v, Exception):
                        print(f"    {key}: failed ({type(v).__name__}: {v})")
                    elif v is not None:
                        print(f"    {key}: {v!r}")
            elif isinstance(cmp_obj, ComparisonSummary):
                # Backward compatibility
                df = cmp_obj.per_column
                ncols = int(df.shape[0])
                worst_b = float(df["b_abs_max"].max()) if ncols else float("nan")
                worst_a = float(df["a_abs_max"].max()) if ncols else float("nan")
                print(
                    f"  compare: n_turns analyzer/ref/matched = "
                    f"{cmp_obj.n_turns_analyzer}/{cmp_obj.n_turns_reference}/{cmp_obj.n_matched}; "
                    f"worst |Δb|={worst_b:.3g}, worst |Δa|={worst_a:.3g} (units)"
                )
            else:
                print(f"  compare: could not compare ({cmp_obj!r})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
