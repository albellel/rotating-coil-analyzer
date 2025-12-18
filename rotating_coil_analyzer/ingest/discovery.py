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

    The SM18 Parameters.txt often contains TABLE{...} payloads where \t and \n are *escaped*
    (i.e. literal backslash characters). We keep values raw here and decode TABLE payloads later.
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


def _get_str(d: Dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    return d.get(key, default)


def _get_int(d: Dict[str, str], key: str, default: Optional[int] = None) -> Optional[int]:
    v = d.get(key, None)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _get_float(d: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key, None)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _get_bool(d: Dict[str, str], key: str, default: Optional[bool] = None) -> Optional[bool]:
    v = d.get(key, None)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return default


def _decode_table_payload(payload: str) -> str:
    """
    Decode a TABLE payload that may contain escaped '\t' and '\n' sequences.
    We intentionally do NOT attempt to "fix" malformed payloads; strict mode will fail.
    """
    # Convert literal backslash sequences to actual control characters.
    # Example in Parameters.txt: "TABLE{1\\t0\\t1\\n2\\t2\\t3}"
    try:
        return payload.encode("utf-8").decode("unicode_escape")
    except Exception:
        # Fallback: minimal replacements
        return payload.replace("\\t", "\t").replace("\\n", "\n")


def _parse_table(value: str) -> List[List[str]]:
    """
    Parse a Parameters TABLE{...} into rows/columns (tab-separated, newline-separated).

    Accepts both:
      - TABLE{1\t0\t1\n2\t2\t3}   (escaped)
      - TABLE{...} where the payload already contains real newlines/tabs
    """
    m = re.match(r"^TABLE\{(.*)\}\s*$", value, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Not a TABLE{{...}} value: {value[:60]}...")
    payload = _decode_table_payload(m.group(1))
    rows = [r for r in payload.splitlines() if r.strip()]
    return [r.split("\t") for r in rows]


@dataclass
class MeasurementDiscovery:
    """
    Phase-1 discovery: build a catalog for a measurement folder.
    """
    strict: bool = True

    # Keep this method name: the GUI expects it.
    def build_catalog(self, root_dir: str | Path) -> MeasurementCatalog:
        root = Path(root_dir).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Folder not found or not a directory: {root}")

        params_path = root / "Parameters.txt"
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters.txt not found in: {root}")

        d = _parse_kv_parameters(params_path)
        warnings: List[str] = []

        Ns = _get_int(d, "Parameters.Measurement.samples", None)
        if Ns is None or Ns <= 0:
            raise ValueError("Missing or invalid Parameters.Measurement.samples (samples per turn).")

        # 'v' can be negative (direction). We keep the sign in catalog, but downstream timing uses |v|.
        v_rpm = _get_float(d, "Parameters.Measurement.v", None)
        if v_rpm is None:
            raise ValueError("Missing Parameters.Measurement.v (shaft speed, rpm).")

        # Determine enabled apertures
        enabled_aps: List[int] = []
        for ap in (1, 2):
            key = f"Measurement.AP{ap}.enabled"
            b = _get_bool(d, key, None)
            if b is True:
                enabled_aps.append(ap)
        if not enabled_aps:
            # Some datasets omit Measurement.AP#.enabled. In strict mode, raise.
            if self.strict:
                raise ValueError("No enabled apertures found (Measurement.AP#.enabled).")
            warnings.append("No enabled apertures found; defaulting to [1].")
            enabled_aps = [1]

        # Parse segment table(s) from enabled apertures. In Phase-1 we only need segment IDs and FDIs.
        segments: List[SegmentSpec] = []
        for ap in enabled_aps:
            key = f"Measurement.AP{ap}.FDIs"
            raw = _get_str(d, key, None)
            if raw is None:
                if self.strict:
                    raise ValueError(f"Strict mode: failed to parse segments from {key} (key missing).")
                warnings.append(f"{key} missing; no segments for AP{ap}.")
                continue

            try:
                rows = _parse_table(raw)
            except Exception as e:
                if self.strict:
                    raise ValueError(f"Strict mode: failed to parse segments from {key} TABLE{{...}}.") from e
                warnings.append(f"{key} parse failed: {e}")
                continue

            for r in rows:
                # Allowed: 3 columns (segment, abs_fdi, cmp_fdi) or 4 columns (+length)
                if len(r) < 3:
                    if self.strict:
                        raise ValueError(f"{key} row has <3 columns: {r!r}")
                    warnings.append(f"{key} row has <3 columns; skipped: {r!r}")
                    continue

                seg_id = str(r[0]).strip()
                try:
                    fdi_abs = int(str(r[1]).strip())
                    fdi_cmp = int(str(r[2]).strip())
                except Exception as e:
                    if self.strict:
                        raise ValueError(f"{key} row has non-integer FDI indices: {r!r}") from e
                    warnings.append(f"{key} row has non-integer FDI indices; skipped: {r!r}")
                    continue

                length_m: Optional[float] = None
                if len(r) >= 4 and str(r[3]).strip():
                    try:
                        length_m = float(str(r[3]).strip())
                    except Exception:
                        length_m = None

                if len(r) < 4:
                    warnings.append(f"AP{ap} segment {seg_id}: length missing in FDIs table; set to None.")

                segments.append(SegmentSpec(segment_id=seg_id, fdi_abs=fdi_abs, fdi_cmp=fdi_cmp, length_m=length_m))

        if not segments and self.strict:
            raise ValueError("Strict mode: no segments parsed from any enabled aperture FDIs tables.")

        # Discover segment files
        segment_files: Dict[Tuple[str, str], Path] = {}
        run_ids: set[str] = set()

        # Typical filename:
        #   <run_id>_corr_sigs_Ap_<ap>_Seg<seg>.bin
        pat = re.compile(r"^(?P<run>.+?)_corr_sigs_Ap_(?P<ap>\d+)_Seg(?P<seg>[^.]+)\.bin$", re.IGNORECASE)

        for p in root.rglob("*.bin"):
            m = pat.match(p.name)
            if not m:
                continue
            ap = int(m.group("ap"))
            if ap not in enabled_aps:
                continue
            run_id = m.group("run")
            seg = m.group("seg")
            run_ids.add(run_id)
            segment_files[(run_id, seg)] = p

        runs = sorted(run_ids)

        if not runs and self.strict:
            raise FileNotFoundError("Strict mode: no *_corr_sigs_*.bin files found under the selected folder.")

        return MeasurementCatalog(
            root_dir=root,
            parameters_path=params_path,
            samples_per_turn=int(Ns),
            shaft_speed_rpm=float(v_rpm),
            enabled_apertures=enabled_aps,
            segments=segments,
            runs=runs,
            segment_files=segment_files,
            warnings=tuple(warnings),
        )
