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


def find_parameters_txt(selected_dir: Path, max_up: int = 2) -> Path:
    """
    Search Parameters.txt in selected_dir or up to max_up parent levels.
    Returns the first match (nearest to selected_dir).
    """
    p = Path(selected_dir).expanduser().resolve()
    for k in range(max_up + 1):
        cand = p / "Parameters.txt"
        if cand.exists() and cand.is_file():
            return cand
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(f"Parameters.txt not found in '{selected_dir}' or up to {max_up} parent levels.")


def _decode_table_payload(payload: str) -> str:
    """
    Decode TABLE payload that may contain escaped sequences like '\\t' and '\\n'.
    Also accept payloads that already contain real tabs/newlines.
    """
    # Convert escaped backslash sequences to actual characters
    payload = payload.replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\n")
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


def _segment_sort_key(seg_id: str) -> Tuple[int, str]:
    """
    Sort numeric segment ids numerically, then fallback to lexical for non-numeric.
    """
    try:
        return (0, f"{int(seg_id):06d}")
    except Exception:
        return (1, seg_id)


@dataclass
class MeasurementDiscovery:
    """
    Phase-1 discovery: build a catalog for a measurement folder.

    STRICT POLICY (per your requirement): no degraded mode
      - Parameters.txt must be found (selected folder or up to 2 parents)
      - Measurement.AP*.FDIs table must parse for enabled apertures
      - We do not invent segments from filenames if Parameters do not define them
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
            key = f"Measurement.AP{ap}.FDIs"
            if key not in params:
                raise ValueError(f"Missing required key '{key}' in Parameters.txt")
            table = _parse_table(params[key])  # may raise (strict)
            for row in table:
                if len(row) < 3:
                    raise ValueError(f"Malformed {key} row (need >=3 columns): {row}")
                seg_id = row[0].strip()
                fdi_abs = int(float(row[1]))
                fdi_cmp = int(float(row[2]))
                length_m: Optional[float] = None
                if len(row) >= 4 and row[3].strip() != "":
                    try:
                        length_m = float(row[3])
                    except Exception:
                        length_m = None
                        warnings.append(f"AP{ap} seg {seg_id}: could not parse length '{row[3]}', set None")
                segments.append(SegmentSpec(aperture_id=ap, segment_id=seg_id, fdi_abs=fdi_abs, fdi_cmp=fdi_cmp, length_m=length_m))

        # Scan for segment files (bin/txt/csv), include both corr_sigs and generic_corr_sigs.
        # Typical filename:
        #   <run_id>_corr_sigs_Ap_<ap>_Seg<seg>.bin
        #   <run_id>_generic_corr_sigs_Ap_<ap>_Seg<seg>.bin
        pat = re.compile(
            r"^(?P<run>.+?)_(?:(?P<generic>generic)_)?corr_sigs_Ap_(?P<ap>\d+)_Seg(?P<seg>[^.]+)\.(?P<ext>bin|txt|csv)$",
            re.IGNORECASE,
        )

        segment_files: Dict[Tuple[str, int, str], Path] = {}
        run_ids: set[str] = set()

        for p in parameters_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".bin", ".txt", ".csv"}:
                continue
            m = pat.match(p.name)
            if not m:
                continue
            ap = int(m.group("ap"))
            if ap not in enabled_aps:
                continue
            run_id = m.group("run")
            seg_id = m.group("seg")

            key = (run_id, ap, seg_id)

            # If multiple files for the same key, prefer:
            # 1) .bin over .txt/.csv
            # 2) non-generic over generic
            prev = segment_files.get(key)
            if prev is None:
                segment_files[key] = p
            else:
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
                    # same "bin-ness": prefer non-generic
                    if prev_is_generic and (not new_is_generic):
                        choose_new = True

                if choose_new:
                    segment_files[key] = p

            run_ids.add(run_id)

        runs = sorted(run_ids)

        # Strict: warn about files for segments not defined in Parameters (do not invent SegmentSpec).
        defined = {(s.aperture_id, s.segment_id) for s in segments}
        extras = []
        for (run_id, ap, seg_id), p in segment_files.items():
            if (ap, seg_id) not in defined:
                extras.append((run_id, ap, seg_id, p.name))
        if extras:
            warnings.append("Found segment files not defined in Parameters.AP*.FDIs (ignored by UI selection):")
            for run_id, ap, seg_id, name in extras[:20]:
                warnings.append(f"  run={run_id} ap={ap} seg={seg_id} file={name}")

        # Sort segments
        segments = sorted(segments, key=lambda s: (s.aperture_id, _segment_sort_key(s.segment_id)))

        return MeasurementCatalog(
            root_dir=selected,
            parameters_path=parameters_path,
            parameters_root=parameters_root,
            samples_per_turn=int(samples_per_turn),
            shaft_speed_rpm=float(shaft_speed_rpm),
            enabled_apertures=enabled_aps,
            segments=segments,
            runs=runs,
            segment_files=segment_files,
            warnings=tuple(warnings),
        )
