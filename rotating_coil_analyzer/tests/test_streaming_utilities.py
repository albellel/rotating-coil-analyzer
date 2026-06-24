"""Tests for the streaming-analysis utilities that were previously untested:

- ``process_kn_pipeline`` (the notebook/GUI entry point)
- plateau detection helpers (``compute_block_averaged_range``,
  ``detect_plateau_turns``, ``classify_current``, ``find_contiguous_groups``)
- ``diagnose_fdi_transitions`` (FDI stuck-channel diagnostic)

Plus a minimal import smoke test for the validation package.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotating_coil_analyzer.analysis.kn_pipeline import (
    SegmentKn,
    compute_legacy_kn_per_turn,
    merge_coefficients,
    safe_normalize_to_units,
)
from rotating_coil_analyzer.analysis.utility_functions import (
    process_kn_pipeline,
    compute_block_averaged_range,
    detect_plateau_turns,
    classify_current,
    find_contiguous_groups,
    diagnose_fdi_transitions,
)


def _make_kn(H: int) -> SegmentKn:
    """Trivial unit kn (abs == cmp == 1+0j) for H harmonic orders."""
    orders = np.arange(1, H + 1, dtype=int)
    ones = np.ones(H, dtype=complex)
    return SegmentKn(orders=orders, kn_abs=ones, kn_cmp=ones.copy(),
                     kn_ext=None, source_path="synthetic")


# ---------------------------------------------------------------------------
# process_kn_pipeline
# ---------------------------------------------------------------------------

class TestProcessKnPipeline:
    def _inputs(self):
        H, nt, Ns = 6, 5, 32
        m, rref = 1, 0.02
        kn = _make_kn(H)
        # Strong harmonic-1 content so the main field is well above min_b1_T.
        th = 2 * np.pi * np.arange(Ns) / Ns
        rng = np.random.default_rng(42)
        base = np.cos(th)[None, :] * np.linspace(1.0, 2.0, nt)[:, None]
        df_abs = base + 0.01 * rng.standard_normal((nt, Ns))
        df_cmp = base + 0.01 * rng.standard_normal((nt, Ns))
        t = np.tile(np.arange(Ns) * 1e-3, (nt, 1))
        I = np.full((nt, Ns), 100.0)
        return kn, df_abs, df_cmp, t, I, m, rref

    def test_shapes_and_ok_main(self):
        kn, df_abs, df_cmp, t, I, m, rref = self._inputs()
        result, C_merged, C_units, ok_main = process_kn_pipeline(
            df_abs, df_cmp, t, I, kn=kn, r_ref=rref, magnet_order=m,
        )
        nt, H = df_abs.shape[0], kn.orders.size
        assert C_merged.shape == (nt, H)
        assert C_units.shape == (nt, H)
        assert ok_main.shape == (nt,)
        # Strong main field -> all turns normalisable.
        assert ok_main.all()

    def test_wrapper_equals_composition(self):
        """process_kn_pipeline must equal compute -> merge -> normalise."""
        kn, df_abs, df_cmp, t, I, m, rref = self._inputs()
        opts = ("dri", "rot", "cel", "fed")

        _, C_merged, C_units, ok_main = process_kn_pipeline(
            df_abs, df_cmp, t, I, kn=kn, r_ref=rref, magnet_order=m, options=opts,
        )

        res = compute_legacy_kn_per_turn(
            df_abs_turns=df_abs, df_cmp_turns=df_cmp, t_turns=t, I_turns=I,
            kn=kn, Rref_m=rref, magnet_order=m, options=opts,
        )
        C_merged_ref, _ = merge_coefficients(
            C_abs=res.C_abs, C_cmp=res.C_cmp, magnet_order=m,
            mode="abs_upto_m_cmp_above",
        )
        C_units_ref, ok_ref = safe_normalize_to_units(
            C_merged_ref, magnet_order=m, min_main_field=1e-4,
        )

        assert np.allclose(C_merged, C_merged_ref, equal_nan=True)
        assert np.allclose(C_units, C_units_ref, equal_nan=True)
        assert np.array_equal(ok_main, ok_ref)

    def test_weak_field_flags_not_ok(self):
        kn, df_abs, df_cmp, t, I, m, rref = self._inputs()
        # Scale the signal far below min_b1_T so normalisation is flagged.
        _, _, C_units, ok_main = process_kn_pipeline(
            df_abs * 1e-12, df_cmp * 1e-12, t, I,
            kn=kn, r_ref=rref, magnet_order=m, min_b1_T=1e-4,
        )
        assert not ok_main.any()
        assert np.isnan(C_units).all()


# ---------------------------------------------------------------------------
# Plateau detection helpers
# ---------------------------------------------------------------------------

class TestPlateauHelpers:
    def test_block_averaged_range(self):
        nt, Ns = 3, 100
        I = np.zeros((nt, Ns))
        I[0] = 5.0                              # flat -> range 0
        I[1] = np.linspace(0.0, 10.0, Ns)       # ramp -> large range
        I[2, :50], I[2, 50:] = 0.0, 8.0         # step -> large range
        I_range, I_blocks = compute_block_averaged_range(I, Ns, n_blocks=10)
        assert I_range.shape == (nt,)
        assert I_blocks.shape == (nt, 10)
        assert I_range[0] == pytest.approx(0.0)
        assert I_range[1] > 8.0
        assert I_range[2] > 7.0

    def test_detect_plateau_turns_rules(self):
        I_blocks = np.array([
            [1.0, 1.0, 1.0, 1.0],   # flat -> plateau
            [0.0, 0.0, 0.0, 5.0],   # ends high
            [5.0, 0.0, 0.0, 0.0],   # starts high
            [0.0, 1.0, 2.0, 3.0],   # ramp
        ])
        I_mean = I_blocks.mean(axis=1)
        I_range = I_blocks.max(axis=1) - I_blocks.min(axis=1)
        info = detect_plateau_turns(I_blocks, I_mean, I_range, threshold=2.0)
        assert bool(info["is_plateau"][0]) is True
        assert bool(info["is_plateau"][3]) is False
        # The flat turn passes all three rules.
        assert info["range_ok"][0] and info["start_ok"][0] and info["end_ok"][0]
        # range_ok implies start/end ok (mean lies within the block min/max).
        assert np.all(~info["range_ok"] | (info["start_ok"] & info["end_ok"]))

    def test_classify_current_defaults(self):
        assert classify_current(10.0) == "zero"
        assert classify_current(100.0) == "pre-ramp"
        assert classify_current(300.0) == "injection"
        assert classify_current(50_000.0) == "flat-high"

    def test_classify_current_custom_thresholds(self):
        th = {"a": 10, "b": 100}
        assert classify_current(5, th) == "a"
        assert classify_current(50, th) == "b"
        assert classify_current(500, th) == "flat-high"

    def test_find_contiguous_groups(self):
        mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
        assert find_contiguous_groups(mask, min_length=1) == [(1, 2), (4, 4), (6, 8)]
        assert find_contiguous_groups(mask, min_length=2) == [(1, 2), (6, 8)]
        assert find_contiguous_groups(np.zeros(5, dtype=bool)) == []
        assert find_contiguous_groups(np.ones(3, dtype=bool), min_length=1) == [(0, 2)]


# ---------------------------------------------------------------------------
# FDI stuck-channel diagnostic
# ---------------------------------------------------------------------------

class TestDiagnoseFdiTransitions:
    @staticmethod
    def _flux_from_ranges(R: np.ndarray, Ns: int = 64) -> np.ndarray:
        """Build turn samples whose robust range equals ~R[t]."""
        return np.stack([np.linspace(0.0, float(r), Ns) for r in R])

    def test_responsive_transition_is_ok(self):
        # run0: turns 0..9 (range ~1); run1: turns 20..49 (range ~11 throughout).
        R = np.empty(50)
        R[0:10] = 1.0
        R[10:20] = np.linspace(1.0, 11.0, 10)   # gap
        R[20:50] = 11.0
        flux = self._flux_from_ranges(R)
        run_info = [
            {"run_id": 0, "start": 0, "end": 9, "I_nom": 0.0},
            {"run_id": 1, "start": 20, "end": 49, "I_nom": 100.0},
        ]
        checks = diagnose_fdi_transitions(flux, np.zeros(50), run_info)
        assert len(checks) == 1
        assert checks[0].severity == "OK"
        assert checks[0].is_stuck is False
        assert checks[0].response_ratio == pytest.approx(1.0, abs=0.05)

    def test_stuck_transition_is_flagged(self):
        # run1 starts at the OLD range (~1) and only settles to ~11 later.
        R = np.empty(50)
        R[0:10] = 1.0
        R[10:20] = 1.0                          # gap, no response
        R[20:30] = 1.0                          # plateau start: still stuck
        R[30:40] = np.linspace(1.0, 11.0, 10)   # belated settling
        R[40:50] = 11.0                         # settled
        flux = self._flux_from_ranges(R)
        run_info = [
            {"run_id": 0, "start": 0, "end": 9, "I_nom": 0.0},
            {"run_id": 1, "start": 20, "end": 49, "I_nom": 100.0},
        ]
        checks = diagnose_fdi_transitions(flux, np.zeros(50), run_info)
        assert len(checks) == 1
        assert checks[0].severity == "STUCK"
        assert checks[0].is_stuck is True
        assert abs(checks[0].response_ratio) < 0.3

    def test_small_delta_I_is_skipped(self):
        R = np.full(30, 5.0)
        flux = self._flux_from_ranges(R)
        run_info = [
            {"run_id": 0, "start": 0, "end": 9, "I_nom": 100.0},
            {"run_id": 1, "start": 20, "end": 29, "I_nom": 101.0},  # |dI|=1 < 5
        ]
        checks = diagnose_fdi_transitions(flux, np.zeros(30), run_info, min_delta_I=5.0)
        assert checks == []


# ---------------------------------------------------------------------------
# Validation package smoke test
# ---------------------------------------------------------------------------

def test_validation_package_imports():
    """The validation package and its live module import cleanly."""
    import rotating_coil_analyzer.validation as validation  # noqa: F401
    from rotating_coil_analyzer.validation import golden_streaming

    # Key public entry points exist.
    assert callable(golden_streaming.run_golden_folder)
    assert callable(golden_streaming.compare_units_table)

    # The dead golden_runner module has been removed.
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rotating_coil_analyzer.validation.golden_runner")
