from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame


@dataclass(frozen=True)
class _Candidate:
    name: str
    dtype: np.dtype


class Sm18CorrSigsReader:
    """
    STRICT reader for SM18-style corrected signals binaries (*.bin).

    Contract:
      - Time MUST be present in the file (first column).
      - File MUST be 4 columns: t, df_abs, df_cmp, I (extras not supported here).
      - Time is accepted if either:
          A) global monotone increasing (relative time since start), OR
          B) per-turn sawtooth: within each turn time increases, and resets between turns.
      - No synthetic time generation, ever.
    """

    # --- Strict validation tolerances (tune here if needed) ---
    # expected dt_nom = (60/|v|)/Ns
    _DT_MED_FACTOR_MIN = 0.25   # dt_median must be >= 0.25 * dt_nom
    _DT_MED_FACTOR_MAX = 4.0    # dt_median must be <= 4.0 * dt_nom
    _PER_TURN_MONO_MIN_FRAC = 0.95  # at least 95% turns must be strictly increasing within turn
    _GLOBAL_MONO_MIN_FRAC = 0.999   # at least 99.9% diffs must be positive for global monotone

    # for turn duration: Tturn median must be close to Tnom (loose here; strict QC belongs to Phase-2)
    _TTURN_FACTOR_MIN = 0.5
    _TTURN_FACTOR_MAX = 2.0

    def read(
        self,
        file_path: Path,
        run_id: str,
        segment: str,
        samples_per_turn: int,
        shaft_speed_rpm: float,
    ) -> SegmentFrame:
        fp = Path(file_path).expanduser().resolve()
        Ns = int(samples_per_turn)
        v = float(shaft_speed_rpm)
        vabs = abs(v)
        if vabs <= 0:
            raise ValueError("shaft_speed_rpm is zero; cannot validate time axis.")

        Tnom = 60.0 / vabs
        dt_nom = Tnom / float(Ns)

        candidates = [
            _Candidate("float32_le_4col", np.dtype("<f4")),
            _Candidate("float64_le_4col", np.dtype("<f8")),
        ]

        reports: list[str] = []
        best: Optional[tuple[pd.DataFrame, list[str], str]] = None
        best_score = -1.0

        for cand in candidates:
            try:
                df, warnings, score = self._try_candidate(fp, cand, Ns, Tnom, dt_nom)
                reports.append(f"{cand.name}: ok(score={score:.3f})")
                if score > best_score:
                    best_score = score
                    best = (df, warnings, cand.name)
            except Exception as e:
                reports.append(f"{cand.name}: reject ({type(e).__name__}: {e})")

        if best is None:
            msg = "No supported binary format validated for this file.\n" + "\n".join(reports[:80])
            raise ValueError(msg)

        df, warnings, best_name = best
        warnings = [f"Selected binary format: {best_name}"] + warnings

        n = len(df)
        n_turns = n // Ns
        return SegmentFrame(
            source_path=fp,
            run_id=run_id,
            segment=segment,
            samples_per_turn=Ns,
            n_turns=int(n_turns),
            df=df,
            warnings=tuple(warnings),
        )

    def _try_candidate(
        self,
        fp: Path,
        cand: _Candidate,
        Ns: int,
        Tnom: float,
        dt_nom: float,
    ) -> tuple[pd.DataFrame, list[str], float]:
        arr = np.fromfile(fp, dtype=cand.dtype)
        if arr.size < 4 * Ns:
            raise ValueError("file too small for one turn (need at least 4*Ns samples)")

        nrow = arr.size // 4
        rem = arr.size - nrow * 4
        if rem != 0:
            # ignore remainder (partial row)
            arr = arr[: nrow * 4]

        mat = arr.reshape((nrow, 4))

        t = mat[:, 0].astype(np.float64, copy=False)
        df_abs = mat[:, 1].astype(np.float64, copy=False)
        df_cmp = mat[:, 2].astype(np.float64, copy=False)
        I = mat[:, 3].astype(np.float64, copy=False)

        # --- trailing invalid trimming (strict: only allowed at the end) ---
        finite_mask = np.isfinite(t) & np.isfinite(df_abs) & np.isfinite(df_cmp) & np.isfinite(I)
        if not np.any(finite_mask):
            raise ValueError("no finite rows")

        if np.any(~finite_mask):
            first_bad = int(np.argmax(~finite_mask))
            # strict: after first_bad everything must be non-finite (no holes)
            if np.any(finite_mask[first_bad:]):
                raise ValueError("non-finite values appear inside data (not only trailing)")
            n_bad = len(finite_mask) - first_bad
            t = t[:first_bad]
            df_abs = df_abs[:first_bad]
            df_cmp = df_cmp[:first_bad]
            I = I[:first_bad]
            warn_trim = f"Trimmed trailing invalid rows: first_bad_index={first_bad}, n_bad={n_bad}"
        else:
            warn_trim = ""

        # --- trim to full turns ---
        n = len(t)
        n_turns = n // Ns
        if n_turns < 1:
            raise ValueError("less than one full turn after trimming")
        n_keep = n_turns * Ns
        if n_keep != n:
            t = t[:n_keep]
            df_abs = df_abs[:n_keep]
            df_cmp = df_cmp[:n_keep]
            I = I[:n_keep]

        # --- detect time mode: global monotone vs per-turn sawtooth ---
        warnings: list[str] = []
        if warn_trim:
            warnings.append(warn_trim)

        tR = t.reshape((n_turns, Ns))
        dR = np.diff(tR, axis=1)

        per_turn_mono_frac = float(np.mean(np.all(dR > 0, axis=1)))
        # global monotone check
        dt = np.diff(t)
        global_mono_frac = float(np.mean(dt > 0))

        time_mode: Optional[str] = None
        if global_mono_frac >= self._GLOBAL_MONO_MIN_FRAC:
            time_mode = "global_monotone"
        elif per_turn_mono_frac >= self._PER_TURN_MONO_MIN_FRAC:
            time_mode = "per_turn_sawtooth"
        else:
            raise ValueError(
                f"time axis invalid: global_mono_frac={global_mono_frac:.6f}, per_turn_mono_frac={per_turn_mono_frac:.3f}"
            )

        warnings.append(f"time_mode={time_mode}, global_mono_frac={global_mono_frac:.6f}, per_turn_mono_frac={per_turn_mono_frac:.3f}")

        # --- sanity vs nominal dt ---
        if time_mode == "global_monotone":
            dt_med = float(np.median(dt))
        else:
            dt_med = float(np.median(dR))  # within-turn dt

        if not (self._DT_MED_FACTOR_MIN * dt_nom <= dt_med <= self._DT_MED_FACTOR_MAX * dt_nom):
            raise ValueError(f"dt_median={dt_med:.6g} s inconsistent with dt_nom={dt_nom:.6g} s")

        # --- turn duration sanity (from t_last - t_first per turn) ---
        Tturn = tR[:, -1] - tR[:, 0]
        Tmed = float(np.median(Tturn))
        if not (self._TTURN_FACTOR_MIN * Tnom <= Tmed <= self._TTURN_FACTOR_MAX * Tnom):
            raise ValueError(f"median turn duration {Tmed:.6g} s inconsistent with Tnom={Tnom:.6g} s")

        # build dataframe (time as acquired; no modification)
        df = pd.DataFrame({"t": t, "df_abs": df_abs, "df_cmp": df_cmp, "I": I})

        # informative timing stats
        warnings.append(f"dt_nom={dt_nom:.6g} s, dt_median={dt_med:.6g} s")
        warnings.append(f"Tnom={Tnom:.6g} s, median(Tturn)={Tmed:.6g} s")

        # scoring: prefer candidates where dt_med closer to dt_nom and Tmed closer to Tnom
        score = 1.0
        score -= abs(dt_med / dt_nom - 1.0)
        score -= abs(Tmed / Tnom - 1.0)
        if time_mode == "global_monotone":
            score += 0.1  # slight preference (easier diagnostics), but both are valid
        return df, warnings, float(score)
