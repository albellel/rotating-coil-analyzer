from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rotating_coil_analyzer.models.catalog import MeasurementCatalog, RunSpec, SegmentSpec


_PARAM_SAMPLES = "Parameters.Measurement.samples"
_PARAM_SPEED = "Parameters.Measurement.v"


@dataclass(frozen=True)
class _ParsedParameters:
    samples_per_turn: int
    shaft_speed_rpm: float
    enabled_apertures: tuple[int, ...]
    segments: tuple[SegmentSpec, ...]
    warnings: tuple[str, ...]


class MeasurementDiscovery:
    """
    Phase-1: filesystem discovery + Parameters parsing (strict about having FDIs mapping).
    No multipoles here; only catalog + file mapping.
    """

    def __init__(self, max_parent_hops_for_parameters: int = 2) -> None:
        self._max_hops = int(max_parent_hops_for_parameters)

    def build_catalog(self, selected_folder: Path) -> MeasurementCatalog:
        p = Path(selected_folder).expanduser().resolve()

        params_path = self._find_parameters(p)
        lines = params_path.read_text(encoding="utf-8", errors="replace").splitlines()

        parsed = self._parse_parameters(lines)

        # Discover corrected-signal files (.bin) and map to (run_id, segment)
        seg_files, runs = self._discover_segment_files(p)

        # Strict requirement: we must have segments parsed from Parameters.
        if len(parsed.segments) == 0:
            raise ValueError("Strict mode: failed to parse segments from Measurement.AP#.FDIs TABLE{...}.")

        warnings = list(parsed.warnings)
        if len(seg_files) == 0:
            warnings.append("No corr_sigs *.bin files discovered under selected folder.")

        return MeasurementCatalog(
            root_dir=p,
            parameters_path=params_path,
            samples_per_turn=parsed.samples_per_turn,
            shaft_speed_rpm=parsed.shaft_speed_rpm,
            enabled_apertures=parsed.enabled_apertures,
            segments=parsed.segments,
            runs=tuple(RunSpec(r) for r in sorted(runs)),
            segment_files=seg_files,
            warnings=tuple(warnings),
        )

    def _find_parameters(self, selected_folder: Path) -> Path:
        cur = selected_folder
        for _ in range(self._max_hops + 1):
            candidate = cur / "Parameters.txt"
            if candidate.exists() and candidate.is_file():
                return candidate
            if cur.parent == cur:
                break
            cur = cur.parent
        raise FileNotFoundError(f"Parameters.txt not found in {selected_folder} or up to {self._max_hops} parent folders.")

    def _parse_parameters(self, lines: list[str]) -> _ParsedParameters:
        kv: dict[str, str] = {}
        for ln in lines:
            if ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            kv[k.strip()] = v.strip()

        warnings: list[str] = []

        # samples_per_turn
        if _PARAM_SAMPLES not in kv:
            raise ValueError(f"Missing {_PARAM_SAMPLES} in Parameters.txt")
        samples_per_turn = int(float(kv[_PARAM_SAMPLES]))

        # shaft speed (can be negative -> take sign elsewhere later, but store the numeric)
        if _PARAM_SPEED not in kv:
            raise ValueError(f"Missing {_PARAM_SPEED} in Parameters.txt")
        shaft_speed_rpm = float(kv[_PARAM_SPEED])

        # enabled apertures
        enabled: list[int] = []
        for ap in (1, 2, 3, 4):
            k = f"Measurement.AP{ap}.enabled"
            if k in kv and kv[k].lower() == "true":
                enabled.append(ap)
        if len(enabled) == 0:
            # If not specified, assume single aperture AP1 enabled (common legacy behavior)
            enabled = [1]
            warnings.append("No Measurement.AP#.enabled flags found; assuming AP1 enabled.")

        # segments from Measurement.AP#.FDIs: TABLE{...}
        segs: list[SegmentSpec] = []
        for ap in enabled:
            table_key = f"Measurement.AP{ap}.FDIs"
            if table_key not in kv:
                warnings.append(f"Missing {table_key}; no segments parsed for AP{ap}.")
                continue
            table_str = kv[table_key]
            segs_ap, w_ap = self._parse_fdis_table(ap, table_str)
            segs.extend(segs_ap)
            warnings.extend(w_ap)

        return _ParsedParameters(
            samples_per_turn=samples_per_turn,
            shaft_speed_rpm=shaft_speed_rpm,
            enabled_apertures=tuple(enabled),
            segments=tuple(segs),
            warnings=tuple(warnings),
        )

    def _parse_fdis_table(self, ap: int, table_str: str) -> tuple[list[SegmentSpec], list[str]]:
        """
        Supports both formats:

        (A) with length:
            TABLE{NCS\t0\t1\t0.47\nCS\t2\t3\t0.47}

        (B) without length:
            TABLE{1\t0\t1\n2\t2\t3\n...}

        Returns SegmentSpec(aperture_id=ap, segment=<token0>, fdi_abs=<token1>, fdi_cmp=<token2>, length_m=<optional>).
        """
        warnings: list[str] = []
        m = re.search(r"TABLE\s*\{(.*)\}\s*$", table_str)
        if not m:
            return [], [f"AP{ap}: FDIs entry is not a TABLE{{...}}: {table_str!r}"]

        body = m.group(1)
        rows = body.split("\\n")
        out: list[SegmentSpec] = []

        for r in rows:
            r = r.strip()
            if not r:
                continue
            # Accept both tab and spaces
            parts = re.split(r"[\t ]+", r)
            if len(parts) < 3:
                warnings.append(f"AP{ap}: malformed FDIs row (need >=3 fields): {r!r}")
                continue

            seg_name = str(parts[0])
            try:
                fdi_abs = int(float(parts[1]))
                fdi_cmp = int(float(parts[2]))
            except Exception:
                warnings.append(f"AP{ap}: cannot parse FDIs indices in row: {r!r}")
                continue

            length_m: Optional[float] = None
            if len(parts) >= 4:
                try:
                    length_m = float(parts[3])
                except Exception:
                    length_m = None
                    warnings.append(f"AP{ap} segment {seg_name}: length present but not numeric; set to None.")
            else:
                warnings.append(f"AP{ap} segment {seg_name}: length missing in FDIs table; set to None.")

            out.append(SegmentSpec(aperture_id=ap, segment=seg_name, fdi_abs=fdi_abs, fdi_cmp=fdi_cmp, length_m=length_m))

        return out, warnings

    def _discover_segment_files(self, selected_folder: Path) -> tuple[dict[tuple[str, str], Path], set[str]]:
        """
        Maps corr_sigs files:
          <run>_corr_sigs_Ap_1_SegCS.bin
          <run>_corr_sigs_Ap_1_Seg2.bin
        to keys: (run_id, segment)

        run_id keeps a suffix "_Ap1" if Ap_1 is present.
        """
        seg_files: dict[tuple[str, str], Path] = {}
        runs: set[str] = set()

        rx = re.compile(r"^(?P<run>.+?)_corr_sigs_(?:(?P<ap>Ap_\d+)_)?Seg(?P<seg>[^.]+)\.bin$", re.IGNORECASE)

        for f in selected_folder.glob("*.bin"):
            m = rx.match(f.name)
            if not m:
                continue
            run = m.group("run")
            ap = m.group("ap")
            seg = m.group("seg")

            run_id = run
            if ap is not None:
                run_id = f"{run}_Ap{ap.split('_')[1]}"

            runs.add(run_id)
            seg_files[(run_id, str(seg))] = f

        return seg_files, runs
