from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from rotating_coil_analyzer.models.catalog import MeasurementCatalog, SegmentSpec


_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


def _parse_kv_parameters(path: Path) -> Dict[str, str]:
    """
    Minimal Parameters.txt parser.

    Preserves raw values. TABLE payloads are decoded in _parse_table().
    """
    d: Dict[str, str] = {}
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    return d


def _parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    vv = v.strip().lower()
    if vv in _BOOL_TRUE:
        return True
    if vv in _BOOL_FALSE:
        return False
    return default


def _parse_int(v: Optional[str], default: int) -> int:
    if v is None:
        return default
    try:
        return int(float(v.strip()))
    except Exception:
        return default


def _parse_float(v: Optional[str], default: float) -> float:
    if v is None:
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


def _decode_table_payload(payload: str) -> str:
    """
    Parameters TABLE payloads may include literal escape sequences:
      '\\t' for tabs, '\\n' for newlines.
    Decode them conservatively.
    """
    # Replace escaped sequences. Do not touch other backslashes.
    payload = payload.replace("\\t", "\t").replace("\\n", "\n")
    return payload


def _parse_table(value: str) -> List[List[str]]:
    """
    Parse a Parameters TABLE{...} into rows/columns (tab-separated, newline-separated).

    Accepts both:
      - TABLE{1\\t0\\t1\\n2\\t2\\t3}   (escaped)
      - TABLE{...} where the payload already contains real newlines/tabs
    """
    m = re.match(r"^TABLE\{(.*)\}\s*$", value, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Not a TABLE{{...}} value: {value[:60]}...")
    payload = _decode_table_payload(m.group(1).strip())
    rows = [r for r in payload.splitlines() if r.strip()]
    return [r.split("\t") for r in rows]


def find_parameters_txt(selected_dir: Path, max_up: int = 2) -> Path:
    """
    Search Parameters.txt in selected_dir or up to max_up parent levels.
    Returns the first match (nearest to selected_dir).
    """
    p = Path(selected_dir).expanduser().resolve()
    for up in range(max_up + 1):
        cand = p / "Parameters.txt"
        if cand.exists() and cand.is_file():
            return cand
        p = p.parent
    raise FileNotFoundError(f"Parameters.txt not found in {selected_dir} or up to {max_up} parent folders.")


def _find_fdis_table_key(params: Dict[str, str], ap: int, strict: bool = True) -> Tuple[str, List[str]]:
    """
    Find the Parameters key that defines the FDIs table for a given aperture.

    Supported variants (checked in order):
      - Measurement.AP{ap}.FDIs
      - Parameters.AP{ap}.FDIs
    Additionally, for aperture 1 we accept:
      - Measurement.MH.FDIs
      - Parameters.MH.FDIs
      - Any single remaining *.FDIs TABLE (non-AP*) as a last-resort disambiguation
    """
    tried: List[str] = []
    candidates: List[str] = []
    direct = [f"Measurement.AP{ap}.FDIs", f"Parameters.AP{ap}.FDIs"]
    if ap == 1:
        direct += ["Measurement.MH.FDIs", "Parameters.MH.FDIs"]
    for k in direct:
        tried.append(k)
        if k in params:
            return k, tried

    # If AP-specific key not found, try to infer (only for ap=1)
    if ap == 1:
        # collect all keys ending with .FDIs that look like TABLE{...}
        for k, v in params.items():
            if not k.endswith(".FDIs"):
                continue
            if k in tried:
                continue
            if not isinstance(v, str):
                continue
            if v.strip().startswith("TABLE{"):
                candidates.append(k)

        # Remove AP2 key candidates if any exist (ap=1 inference should not steal ap=2 tables)
        candidates_no_ap2 = [k for k in candidates if not re.search(r"\.AP2\.FDIs$", k)]
        # Prefer non-AP keys if available
        if len(candidates_no_ap2) == 1:
            return candidates_no_ap2[0], tried + candidates_no_ap2
        if len(candidates_no_ap2) > 1 and strict:
            raise ValueError(
                "Ambiguous FDIs table for aperture 1. Found multiple candidates: "
                + ", ".join(sorted(candidates_no_ap2))
                + ". Provide Measurement.AP1.FDIs (or Parameters.AP1.FDIs) to disambiguate."
            )

    if strict:
        raise ValueError(f"Missing required FDIs table for aperture {ap}. Tried: {', '.join(tried)}")
    # Non-strict fallback: return empty and let caller decide
    return "", tried


@dataclass
class MeasurementDiscovery:
    """
    Phase-1 discovery: build a catalog for a measurement folder.

    STRICT POLICY (per your requirement): no degraded mode
      - Parameters.txt must be found (selected folder or up to 2 parents)
      - An FDIs table must be present for each enabled aperture
      - We do not invent segments if no FDIs table exists
    """
    strict: bool = True

    def build_catalog(self, selected_dir: str | Path) -> MeasurementCatalog:
        selected = Path(selected_dir).expanduser().resolve()
        if not selected.exists() or not selected.is_dir():
            raise FileNotFoundError(f"Not a directory: {selected}")

        parameters_path = find_parameters_txt(selected, max_up=2)
        parameters_root = parameters_path.parent
        params = _parse_kv_parameters(parameters_path)

        warnings: List[str] = []

        samples_per_turn = _parse_int(params.get("Parameters.Measurement.samples"), default=512)
        shaft_speed_rpm = _parse_float(params.get("Parameters.Measurement.v"), default=60.0)

        # Main field order used for phase-referenced multipole conventions (legacy 'magnetOrder').
        magnet_order_raw = _parse_int(params.get("Parameters.magnetAnalyzer.magnetOrder"), default=0)
        magnet_order: Optional[int] = None
        if magnet_order_raw is not None and int(magnet_order_raw) > 0:
            magnet_order = int(magnet_order_raw)
        else:
            warnings.append(
                "Parameters.magnetAnalyzer.magnetOrder not provided or <=0; defaulting to m=1 for phase reference (can be overridden in GUI)."
            )

        enabled_aps: List[int] = []
        # If keys missing: default AP1 enabled, AP2 disabled.
        ap1_en = _parse_bool(params.get("Measurement.AP1.enabled"), default=True)
        ap2_en = _parse_bool(params.get("Measurement.AP2.enabled"), default=False)
        if ap1_en:
            enabled_aps.append(1)
        if ap2_en:
            enabled_aps.append(2)

        segments: List[SegmentSpec] = []
        for ap in enabled_aps:
            key, tried = _find_fdis_table_key(params, ap=ap, strict=self.strict)
            if key not in params:
                raise ValueError(f"Missing required key '{key}' in Parameters.txt")
            if key not in [f"Measurement.AP{ap}.FDIs", f"Parameters.AP{ap}.FDIs"]:
                warnings.append(f"FDIs table for aperture {ap} taken from '{key}' (fallback).")

            table = _parse_table(params[key])  # may raise (strict)
            for row in table:
                if len(row) < 3:
                    raise ValueError(f"Malformed {key} row (need >=3 columns): {row}")
                seg_id = row[0].strip()
                fdi_abs = int(float(row[1]))
                fdi_cmp = int(float(row[2]))
                length_m: Optional[float] = None
                if len(row) >= 4:
                    try:
                        length_m = float(row[3])
                    except Exception:
                        length_m = None
                segments.append(
                    SegmentSpec(aperture_id=ap, segment_id=seg_id, fdi_abs=fdi_abs, fdi_cmp=fdi_cmp, length_m=length_m)
                )

        # Discover segment files. Two supported families:
        # A) SM18 corr_sigs / generic_corr_sigs
        # B) MBA raw_measurement_data plateau text files (multiple per segment; reader concatenates)
        segment_files: Dict[Tuple[str, int, str], Path] = {}

        # Build fast lookup for allowed segments per aperture
        segs_by_ap: Dict[int, set[str]] = {}
        for s in segments:
            segs_by_ap.setdefault(s.aperture_id, set()).add(s.segment_id)

        # A) SM18
        pat_sm18 = re.compile(
            r"^(?P<run>.+?)_(?:(?P<generic>generic)_)?corr_sigs_Ap_(?P<ap>\d+)_Seg(?P<seg>[^.]+)\.(?P<ext>bin|txt|csv)$",
            flags=re.IGNORECASE,
        )

        # B) MBA plateau (no aperture token in filename typically)
        pat_mba = re.compile(
            r"^(?P<base>.+?)_Run_(?P<step>\d+)_I_(?P<i>[-\d.]+)A_(?P<seg>[^_]+)_raw_measurement_data\.txt$",
            flags=re.IGNORECASE,
        )

        for p in parameters_root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name

            m = pat_sm18.match(name)
            if m:
                run_id = m.group("run")
                ap = int(m.group("ap"))
                seg_id = m.group("seg")
                if ap not in enabled_aps:
                    continue
                if seg_id not in segs_by_ap.get(ap, set()):
                    warnings.append(
                        f"Found SM18 segment file not defined in FDIs table: ap={ap} seg='{seg_id}' ({p.name})"
                    )
                    continue

                key = (run_id, ap, seg_id)

                prev = segment_files.get(key)
                if prev is None:
                    segment_files[key] = p
                else:
                    # Prefer .bin, then non-generic over generic
                    prev_ext = prev.suffix.lower()
                    new_ext = p.suffix.lower()
                    prev_is_bin = prev_ext == ".bin"
                    new_is_bin = new_ext == ".bin"
                    prev_is_generic = ("_generic_corr_sigs_" in prev.name.lower())
                    new_is_generic = ("_generic_corr_sigs_" in p.name.lower())

                    choose_new = False
                    if (not prev_is_bin) and new_is_bin:
                        choose_new = True
                    elif prev_is_bin == new_is_bin:
                        if prev_is_generic and (not new_is_generic):
                            choose_new = True
                    if choose_new:
                        segment_files[key] = p
                continue

            m2 = pat_mba.match(name)
            if m2:
                base = m2.group("base")
                seg_id = m2.group("seg")
                # determine aperture: prefer folder hint if multiple apertures enabled
                ap = 1
                if 2 in enabled_aps:
                    lower_parts = [pp.name.lower() for pp in p.parents]
                    if any("aperture2" in s for s in lower_parts) or any("ap_2" in s for s in lower_parts):
                        ap = 2

                if ap not in enabled_aps:
                    continue
                if seg_id not in segs_by_ap.get(ap, set()):
                    warnings.append(
                        f"Found MBA raw file with segment='{seg_id}' not defined in FDIs table (ignored): {p.name}"
                    )
                    continue

                key = (base, ap, seg_id)
                # store the first (lowest step) file as representative; reader will concatenate all
                prev = segment_files.get(key)
                if prev is None:
                    segment_files[key] = p
                else:
                    # keep the one with the smallest step number for determinism
                    def step_of(path: Path) -> int:
                        mm = pat_mba.match(path.name)
                        return int(mm.group("step")) if mm else 10**9
                    if step_of(p) < step_of(prev):
                        segment_files[key] = p
                continue

        # Runs present in discovered files
        runs = sorted({k[0] for k in segment_files.keys()})

        # Sanity: warn about missing files for defined segments
        for ap in enabled_aps:
            for seg_id in sorted(segs_by_ap.get(ap, set())):
                present = any((run, ap, seg_id) in segment_files for run in runs)
                if runs and not present:
                    warnings.append(f"No files discovered for ap={ap} seg='{seg_id}' in this folder.")

        return MeasurementCatalog(
            root_dir=selected,
            parameters_path=parameters_path,
            parameters_root=parameters_root,
            samples_per_turn=samples_per_turn,
            shaft_speed_rpm=shaft_speed_rpm,
            magnet_order=magnet_order,
            enabled_apertures=tuple(enabled_aps),
            segments=tuple(segments),
            runs=tuple(runs),
            segment_files=segment_files,
            warnings=tuple(warnings),
        )
