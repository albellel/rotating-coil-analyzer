from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rotating_coil_analyzer.models.catalog import (
    ChannelSpec,
    MeasurementCatalog,
    RunDescriptor,
    SegmentSpec,
)

# File discovery: allow underscores/hyphens/etc in the cycle token by stopping at "_corr_sigs_"
_RUN_RE = re.compile(
    r"""
    ^
    (?P<magnet>.+?)
    _(?P<date>\d{8})
    _(?P<time>\d{6})
    _(?P<cycle>.+?)
    _corr_sigs_
    (?:Ap_(?P<ap>\d+)_)?     # optional
    Seg(?P<seg>[A-Za-z0-9]+)
    \.(?P<ext>bin|txt)
    $
    """,
    re.VERBOSE,
)

# Parameters keys: support both
#   Measurement.AP1.FDIs
#   Parameters.Measurement.AP1.FDIs
_FDI_KEY_RE = re.compile(r"^(?:.*\.)?Measurement\.AP(?P<ap>\d+)\.FDIs$")
_AP_ENABLED_KEY_RE = re.compile(r"^(?:.*\.)?Measurement\.AP(?P<ap>\d+)\.enabled$")


@dataclass
class DiscoveryOptions:
    max_parent_search_depth: int = 2
    strict: bool = True


class MeasurementDiscovery:
    def __init__(self, options: Optional[DiscoveryOptions] = None) -> None:
        self.options = options or DiscoveryOptions()

    def locate_parameters(self, selected_dir: Path) -> Path:
        d = selected_dir.resolve()
        for _ in range(self.options.max_parent_search_depth + 1):
            candidate = d / "Parameters.txt"
            if candidate.exists():
                return candidate
            d = d.parent
        raise FileNotFoundError(
            f"Parameters.txt not found within {self.options.max_parent_search_depth} parent levels of {selected_dir}"
        )

    def parse_parameters(self, parameters_path: Path) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for line in parameters_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            params[key.strip()] = val.strip()
        return params

    @staticmethod
    def _parse_bool(v: str) -> bool:
        s = str(v).strip().lower()
        return s in ("true", "1", "yes", "y", "on")

    @staticmethod
    def _parse_table_value(raw: str) -> List[str]:
        """
        Accept TABLE{...} with either real tabs/newlines or escaped \\t \\n.
        Returns list of row strings.
        """
        s = str(raw).strip()
        if not (s.startswith("TABLE{") and s.endswith("}")):
            raise ValueError(f"FDIs value is not TABLE{{...}}: {s[:80]!r}")

        inner = s[len("TABLE{") : -1]
        inner = inner.replace("\\t", "\t").replace("\\n", "\n")
        rows = [r.strip() for r in inner.splitlines() if r.strip()]
        return rows

    @staticmethod
    def _split_row(row: str) -> List[str]:
        """
        Support either tab-separated or whitespace-separated rows.
        """
        if "\t" in row:
            cols = [c.strip() for c in row.split("\t") if c.strip() != ""]
        else:
            cols = [c.strip() for c in re.split(r"\s+", row) if c.strip() != ""]
        return cols

    def _discover_fdi_table_keys(self, params: Dict[str, Any]) -> List[Tuple[int, str]]:
        found: List[Tuple[int, str]] = []
        for k in params.keys():
            m = _FDI_KEY_RE.match(k)
            if m:
                found.append((int(m.group("ap")), k))
        found.sort(key=lambda x: x[0])
        return found

    def _discover_enabled_keys(self, params: Dict[str, Any]) -> Dict[int, str]:
        enabled: Dict[int, str] = {}
        for k in params.keys():
            m = _AP_ENABLED_KEY_RE.match(k)
            if m:
                enabled[int(m.group("ap"))] = k
        return enabled

    def parse_segments(self, params: Dict[str, Any]) -> Tuple[List[int], List[SegmentSpec], List[str]]:
        warnings: List[str] = []
        segments: List[SegmentSpec] = []

        fdi_keys = self._discover_fdi_table_keys(params)
        enabled_keys = self._discover_enabled_keys(params)

        if not fdi_keys:
            if self.options.strict:
                keys_hint = [k for k in params.keys() if "FDI" in k or "FDIs" in k or "AP" in k]
                raise ValueError(
                    "Strict mode: no Measurement.AP#.FDIs keys found. "
                    f"Nearby keys: {keys_hint[:40]}"
                )
            return [], [], warnings

        # Determine enabled apertures.
        enabled_apertures: List[int] = []
        for ap, k_fdi in fdi_keys:
            k_en = enabled_keys.get(ap)
            if k_en is None:
                # Deterministic rule: if FDIs table exists, treat aperture as enabled but warn.
                enabled_apertures.append(ap)
                warnings.append(f"AP{ap}: enabled key missing; assuming enabled because {k_fdi} exists.")
            else:
                if self._parse_bool(params[k_en]):
                    enabled_apertures.append(ap)

        if self.options.strict and not enabled_apertures:
            raise ValueError("Strict mode: no enabled apertures (Measurement.AP#.enabled all false/missing).")

        multi_ap = len(enabled_apertures) > 1

        for ap in enabled_apertures:
            # Find the FDIs key for this aperture
            k_fdi = next((k for ap2, k in fdi_keys if ap2 == ap), None)
            if k_fdi is None:
                continue

            rows = self._parse_table_value(params[k_fdi])

            for row in rows:
                cols = self._split_row(row)

                # Accept 3 or 4 columns:
                # (segment, abs, cmp) or (segment, abs, cmp, length_m)
                if len(cols) not in (3, 4):
                    raise ValueError(
                        f"Strict mode: AP{ap} FDIs row has {len(cols)} columns; expected 3 or 4. Row={row!r}"
                    )

                seg_name = cols[0]
                fdi_abs = int(cols[1])
                fdi_cmp = int(cols[2])

                if len(cols) == 4:
                    length_m = float(cols[3])
                else:
                    length_m = float("nan")
                    warnings.append(f"AP{ap} segment {seg_name}: length missing in FDIs table; set to NaN.")

                segments.append(
                    SegmentSpec(
                        segment=str(seg_name),
                        fdi_abs=fdi_abs,
                        fdi_cmp=fdi_cmp,
                        length_m=length_m,
                        aperture_id=(ap if multi_ap else None),
                    )
                )

        if self.options.strict and not segments:
            raise ValueError("Strict mode: failed to parse any segments from Measurement.AP#.FDIs TABLE{...}.")

        return enabled_apertures, segments, warnings

    def discover_segment_files(self, root_dir: Path) -> List[Path]:
        keep: List[Path] = []
        for ext in ("*.bin", "*.txt"):
            for p in root_dir.rglob(ext):
                if _RUN_RE.match(p.name):
                    keep.append(p)
        return sorted(keep)

    def build_catalog(self, selected_dir: Path) -> MeasurementCatalog:
        parameters_path = self.locate_parameters(selected_dir)
        root_dir = parameters_path.parent
        params = self.parse_parameters(parameters_path)

        # Required
        if "Parameters.Measurement.samples" not in params:
            raise ValueError("Strict mode: missing Parameters.Measurement.samples")
        samples_per_turn = int(params["Parameters.Measurement.samples"])

        # Optional (may be negative: direction)
        shaft_speed_rpm = None
        if "Parameters.Measurement.v" in params:
            try:
                shaft_speed_rpm = float(params["Parameters.Measurement.v"])
            except ValueError:
                shaft_speed_rpm = None

        enabled_apertures, segments, warn_seg = self.parse_segments(params)
        files = self.discover_segment_files(root_dir)

        runs: Dict[str, RunDescriptor] = {}
        segment_files: Dict[Tuple[str, str], Path] = {}

        for f in files:
            m = _RUN_RE.match(f.name)
            if not m:
                continue

            magnet = m.group("magnet")
            date = m.group("date")
            time = m.group("time")
            cycle = m.group("cycle")
            ap_tok = m.group("ap")
            seg = m.group("seg")

            aperture_token = int(ap_tok) if ap_tok else None
            run_id = f"{magnet}_{date}_{time}_{cycle}" + (f"_Ap{aperture_token}" if aperture_token else "")

            if run_id not in runs:
                runs[run_id] = RunDescriptor(
                    run_id=run_id,
                    magnet_name=magnet,
                    date_yyyymmdd=date,
                    time_hhmmss=time,
                    cycle_name=cycle,
                    aperture_token=aperture_token,
                )

            segment_files[(run_id, seg)] = f

        channels: List[ChannelSpec] = []
        for s in segments:
            channels.append(
                ChannelSpec(
                    segment=s.segment,
                    mode="abs",
                    fdi_id=s.fdi_abs,
                    length_m=s.length_m,
                    aperture_id=s.aperture_id,
                    aperture_token=None,
                )
            )
            channels.append(
                ChannelSpec(
                    segment=s.segment,
                    mode="cmp",
                    fdi_id=s.fdi_cmp,
                    length_m=s.length_m,
                    aperture_id=s.aperture_id,
                    aperture_token=None,
                )
            )

        if self.options.strict:
            if not segment_files:
                raise ValueError(
                    "Strict mode: no corr_sigs files found matching expected pattern "
                    "(check filenames or broaden discovery for this measurement type)."
                )

        return MeasurementCatalog(
            root_dir=root_dir,
            parameters_path=parameters_path,
            parameters=params,
            samples_per_turn=samples_per_turn,
            shaft_speed_rpm=shaft_speed_rpm,
            enabled_apertures=enabled_apertures,
            segments=segments,
            channels=channels,
            runs=sorted(runs.values(), key=lambda r: r.run_id),
            segment_files=segment_files,
            warnings=warn_seg,
        )
