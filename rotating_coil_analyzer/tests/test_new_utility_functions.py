"""Tests for utility functions added during the deduplication refactor."""

import numpy as np
import pandas as pd
import pytest

from rotating_coil_analyzer.analysis.utility_functions import (
    compute_level_stats,
    diff_sigma,
    discover_runs,
    eddy_model,
    mad_sigma_clip,
    plateau_summary,
    plot_hysteresis,
)


# =====================================================================
#  mad_sigma_clip
# =====================================================================

class TestMadSigmaClip:
    def test_removes_known_outliers(self):
        rng = np.random.default_rng(42)
        n = 100
        vals = rng.normal(0, 1, n)
        vals[0] = 50.0  # obvious outlier
        vals[1] = -50.0
        df = pd.DataFrame({"val": vals, "label": "inj"})
        clean, removed = mad_sigma_clip(df, "val", n_sigma=5, label_col="label")
        assert len(clean) == n - 2
        assert removed == {"inj": 2}

    def test_no_outliers(self):
        df = pd.DataFrame({"val": np.ones(20), "label": "flat"})
        clean, removed = mad_sigma_clip(df, "val", n_sigma=5, label_col="label")
        # all identical -> MAD = 0 -> sigma < 1e-15 -> skip
        assert len(clean) == 20
        assert removed == {}

    def test_per_label(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "val": np.concatenate([rng.normal(0, 1, 50),
                                   rng.normal(100, 1, 50)]),
            "label": ["A"] * 50 + ["B"] * 50,
        })
        # Inject one outlier in each group
        df.loc[0, "val"] = 100  # outlier in A
        df.loc[50, "val"] = -100  # outlier in B
        clean, removed = mad_sigma_clip(df, "val", n_sigma=5, label_col="label")
        assert "A" in removed
        assert "B" in removed
        assert len(clean) == 98

    def test_small_group_skipped(self):
        df = pd.DataFrame({"val": [1, 2, 100, 200], "label": "tiny"})
        clean, removed = mad_sigma_clip(df, "val", n_sigma=3, label_col="label")
        assert len(clean) == 4  # group too small (<5), not clipped


# =====================================================================
#  discover_runs
# =====================================================================

class TestDiscoverRuns:
    def test_parses_filenames(self, tmp_path):
        for rid, i_nom in [(1, 0.0), (2, 100.0), (3, -200.0)]:
            fname = f"Run_{rid:02d}_I_{i_nom}A_Integral_raw_measurement_data.txt"
            (tmp_path / fname).write_text("dummy")
        runs = discover_runs(tmp_path, "Integral")
        assert len(runs) == 3
        assert runs[0]["run_id"] == 1
        assert runs[0]["I_nom"] == 0.0
        assert runs[2]["I_nom"] == -200.0

    def test_empty_dir(self, tmp_path):
        assert discover_runs(tmp_path, "Central") == []

    def test_custom_pattern(self, tmp_path):
        fname = "Run_05_I_42.0A_NCS_raw_measurement_data.txt"
        (tmp_path / fname).write_text("dummy")
        runs = discover_runs(tmp_path, "NCS",
                             file_pattern="*_NCS_raw_measurement_data.txt")
        assert len(runs) == 1
        assert runs[0]["run_id"] == 5


# =====================================================================
#  plateau_summary
# =====================================================================

class TestPlateauSummary:
    @pytest.fixture()
    def sample_df(self):
        rng = np.random.default_rng(99)
        rows = []
        for run_id in [1, 2]:
            i_nom = 100.0 * run_id
            branch = "ascending" if run_id == 1 else "descending"
            for t in range(20):
                rows.append({
                    "run_id": run_id,
                    "I_nom": i_nom,
                    "branch": branch,
                    "turn_in_run": t,
                    "ok_main": True,
                    "B1_T": 0.1 * run_id + rng.normal(0, 1e-4),
                    "b2_units": 5.0 + rng.normal(0, 0.1),
                    "a2_units": 0.1 + rng.normal(0, 0.01),
                })
        return pd.DataFrame(rows)

    def test_basic_summary(self, sample_df):
        summ = plateau_summary(sample_df, n_last=10, harmonics_range=range(2, 3))
        assert len(summ) == 2
        assert "B1_mean" in summ.columns
        assert "b2_units_mean" in summ.columns
        assert "a2_units_mean" in summ.columns
        assert "TF" in summ.columns
        assert "quality" in summ.columns
        assert (summ["quality"] == "good").all()

    def test_n_last_selects_tail(self, sample_df):
        summ = plateau_summary(sample_df, n_last=5, harmonics_range=range(2, 3))
        assert summ["n_selected"].iloc[0] == 5


# =====================================================================
#  eddy_model
# =====================================================================

class TestEddyModel:
    def test_known_values(self):
        # At t=0: B_inf + A
        assert eddy_model(0, 1.0, 0.5, 10.0) == pytest.approx(1.5)
        # At t=inf: B_inf
        assert eddy_model(1e6, 1.0, 0.5, 10.0) == pytest.approx(1.0)
        # At t=tau: B_inf + A*exp(-1)
        expected = 1.0 + 0.5 * np.exp(-1)
        assert eddy_model(10.0, 1.0, 0.5, 10.0) == pytest.approx(expected)

    def test_vectorised(self):
        t = np.array([0, 10, 20])
        result = eddy_model(t, 2.0, -1.0, 10.0)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0)  # 2 + (-1)*1


# =====================================================================
#  compute_level_stats
# =====================================================================

class TestComputeLevelStats:
    def test_basic(self):
        df = pd.DataFrame({
            "label": ["inj"] * 10 + ["flat"] * 5,
            "ok_main": [True] * 15,
            "I_mean_A": [300.0] * 10 + [4800.0] * 5,
            "B1_T": [0.5] * 10 + [1.2] * 5,
            "b2_units": [1.0] * 10 + [2.0] * 5,
            "b3_units": [0.1] * 10 + [0.2] * 5,
        })
        s = compute_level_stats(df, "inj")
        assert s["N"] == 10
        assert s["B1_mean"] == pytest.approx(0.5)
        assert s["I_mean"] == pytest.approx(300.0)

    def test_empty_returns_empty(self):
        df = pd.DataFrame({
            "label": ["flat"] * 5,
            "ok_main": [True] * 5,
            "I_mean_A": [100.0] * 5,
            "B1_T": [0.5] * 5,
            "b2_units": [1.0] * 5,
            "b3_units": [0.1] * 5,
        })
        assert compute_level_stats(df, "inj") == {}

    def test_ok_filter(self):
        df = pd.DataFrame({
            "label": ["inj"] * 10,
            "ok_main": [True] * 8 + [False] * 2,
            "I_mean_A": [300.0] * 10,
            "B1_T": [0.5] * 10,
            "b2_units": [1.0] * 10,
            "b3_units": [0.1] * 10,
        })
        s = compute_level_stats(df, "inj")
        assert s["N"] == 8


# =====================================================================
#  diff_sigma
# =====================================================================

class TestDiffSigma:
    def test_known_values(self):
        s1 = {"B1_mean": 0.500, "B1_std": 0.010, "N": 100}
        s2 = {"B1_mean": 0.490, "B1_std": 0.010, "N": 100}
        d, err, sig = diff_sigma(s1, s2, "B1")
        assert d == pytest.approx(0.010)
        expected_err = np.sqrt(0.01**2 / 100 + 0.01**2 / 100)
        assert err == pytest.approx(expected_err)
        assert sig == pytest.approx(abs(d) / expected_err)

    def test_zero_std(self):
        s1 = {"B1_mean": 1.0, "B1_std": 0.0, "N": 10}
        s2 = {"B1_mean": 1.0, "B1_std": 0.0, "N": 10}
        d, err, sig = diff_sigma(s1, s2, "B1")
        assert d == pytest.approx(0.0)
        assert sig == 0.0


# =====================================================================
#  plot_hysteresis (smoke test)
# =====================================================================

class TestPlotHysteresis:
    def test_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        summ = pd.DataFrame({
            "run_id": [1, 2, 3, 4],
            "I_nom": [0, 100, 200, 100],
            "B1_mean": [0.0, 0.1, 0.2, 0.1],
            "B1_std": [0.001, 0.001, 0.001, 0.001],
            "branch": ["ascending", "ascending", "descending", "descending"],
            "quality": ["good", "good", "good", "good"],
        })
        fig, ax = plt.subplots()
        plot_hysteresis(ax, summ, "I_nom", "B1_mean", yerr_col="B1_std")
        plt.close(fig)
