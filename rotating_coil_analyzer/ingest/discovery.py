from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rotating_coil_analyzer.models.catalog import MeasurementCatalog, RunCatalog, SegmentDef


_TABLE_RE = re.compile(r"TABLE\{(.*)\}\s*$", re.DOTALL)


def _read_parameters_txt(path: Path) -> Dict[str, str]:
    """
    Read CERN-style Parameters.txt key/value pairs.
    Keeps raw values (including TABLE{...} blocks).
    """
    out: Dict[str, str] = {}
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Simple "Key: Value" parsing; TABLE blocks stay on same logical line in these files.
    # We therefore split by lines and parse at first ':'.
    for line in raw.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_table(value: str) -> List[List[str]]:
    """
    Accept both:
      - real tab/newline inside TABLE{...}
      - escaped \t and \n sequences inside TABLE{...}
    """
    m = _TABLE_RE.search(value.strip())
    if not m:
        raise ValueError(f"Not a TABLE{{...}} value: {value[:80]}")

    inside = m.group(1)

    # Handle escaped sequences if present
    inside = inside.replace("\\t", "\t").replace("\\n", "\n")

    rows: List[List[str]] = []
    for row in inside.splitlines():
        row = row.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split("\t") if p.strip() != ""]
        rows.append(parts)
    return rows


def _parse_fdis_segments(params: Dict[str, str], aperture: int, strict: bool) -> Tuple[List[SegmentDef], List[str]]:
    """
    Supports BOTH FDIs formats:
      4-col: seg_name, abs_idx, cmp_idx, length_m   (new)  :contentReference[oaicite:3]{index=3}
      3-col: seg_id,   abs_idx, cmp_idx             (old)  :contentReference[oaicite:4]{index=4}
    """
    warnings: List[str] = []
    key = f"Measurement.AP{aperture}.FDIs"
    if key not in params:
        if strict:
            raise ValueError(f"Missing {key} in Parameters.txt")
        return [], [f"Missing {key}; no segments declared."]

    table_rows = _parse_table(params[key])

    segs: List[SegmentDef] = []
    for r in table_rows:
        if len(r) < 3:
            if strict:
                raise ValueError(f"Bad FDIs row (need >=3 cols): {r}")
            warnings.append(f"Skipping bad FDIs row (need >=3 cols): {r}")
            continue

        seg_name = str(r[0])
        try:
            fdi_abs = int(float(r[1]))
            fdi_cmp = int(float(r[2]))
        except Exception as e:
            if strict:
                raise ValueError(f"Cannot parse FDI indices in row {r}: {e}")
            warnings.append(f"Skipping FDIs row (cannot parse indices): {r}")
            continue

        length_m: Optional[float] = None
        if len(r) >= 4:
            try:
                length_m = float(r[3])
            except Exception:
                length_m = None
                warnings.append(f"AP{aperture} segment {seg_name}: length column present but not numeric; set to None.")
        else:
            warnings.append(f"AP{aperture} segment {seg_name}: length missing in FDIs table; set to None.")

        segs.append(SegmentDef(name=seg_name, fdi_abs=fdi_abs, fdi_cmp=fdi_cmp, length_m=length_m))

    if strict and not segs:
        raise ValueError(f"Strict mode: failed to parse segments from {key} TABLE{{...}}.")
    return segs, warnings


def _extract_segment_token_from_filename(p: Path) -> str:
    """
    From ..._corr_sigs_Ap_1_SegNCS.bin   -> 'NCS'
         ..._corr_sigs_Ap_1_Seg1.bin     -> '1'
    """
    m = re.search(r"_Seg([^._]+)\.bin$", p.name, flags=re.IGNORECASE)
    if not m:
        return p.stem
    token = m.group(1)
    # normalize "Seg1" cases: token is already "1"
    return token


def _discover_corr_sigs_files(root_dir: Path, aperture: int) -> Dict[str, Path]:
    patterns = [
        f"*corr_sigs*Ap_{aperture}_Seg*.bin",
        f"*corr_sigs*AP{aperture}*Seg*.bin",
        f"*corr_sigs*ap_{aperture}_Seg*.bin",
    ]
    found: Dict[str, Path] = {}
    for pat in patterns:
        for p in root_dir.glob(pat):
            seg = _extract_segment_token_from_filename(p)
            found[seg] = p
    return found


@dataclass
class MeasurementDiscovery:
    strict: bool = True

    def discover(self, root_dir: Path, aperture: int = 1) -> Tuple[MeasurementCatalog, List[str]]:
        """
        root_dir is typically .../aperture1 or .../aperture2.
        """
        warnings: List[str] = []

        root_dir = Path(root_dir)
        params_path = root_dir / "Parameters.txt"
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters.txt not found in {root_dir}")

        params = _read_parameters_txt(params_path)

        # Samples per turn
        Ns = int(float(params.get("Parameters.Measurement.samples", "0")))
        if Ns <= 0 and self.strict:
            raise ValueError("Parameters.Measurement.samples missing or invalid.")

        # Speed (rpm) – ALWAYS use abs(v)
        v = float(params.get("Parameters.Measurement.v", "0"))
        shaft_speed_rpm = abs(v)

        # Enabled aperture
        ap_enabled = params.get(f"Measurement.AP{aperture}.enabled", "true").strip().lower() == "true"

        seg_defs, seg_warn = _parse_fdis_segments(params, aperture=aperture, strict=self.strict)
        warnings.extend(seg_warn)

        corr_files = _discover_corr_sigs_files(root_dir, aperture=aperture)
        if not corr_files:
            warnings.append(f"No corr_sigs files found for aperture {aperture} in {root_dir}")

        # Run id: simplest stable default = folder name
        run_id = root_dir.parent.name + "/" + root_dir.name

        run = RunCatalog(
            run_id=run_id,
            root_dir=root_dir,
            parameters_path=params_path,
            samples_per_turn=Ns,
            shaft_speed_rpm=shaft_speed_rpm,
            segments=seg_defs if ap_enabled else [],
            corr_sigs_files=corr_files if ap_enabled else {},
        )

        cat = MeasurementCatalog(root_dir=root_dir, runs={run_id: run})
        return cat, warnings
