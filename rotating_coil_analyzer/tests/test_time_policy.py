import tempfile
import unittest
from pathlib import Path

import numpy as np

from rotating_coil_analyzer.ingest.readers_sm18 import Sm18CorrSigsReader, Sm18ReaderConfig
from rotating_coil_analyzer.ingest.readers_mba import MbaRawMeasurementReader, MbaReaderConfig


class TestTimePolicy(unittest.TestCase):
    def test_sm18_binary_time_is_exactly_file_time(self):
        """
        SM18: 't' must be taken from file column 0 and never synthesized/shifted.
        We create a tiny synthetic binary file with a known time vector and verify
        that the reader returns it unchanged.
        """
        Ns = 8
        n_turns = 3
        n_rows = Ns * n_turns

        shaft_speed_rpm = 60.0  # T_nom = 1 s, so dt_nom = 1/Ns = 0.125 s
        dt = 1.0 / Ns
        t = (np.arange(n_rows, dtype=np.float64) * dt).astype(np.float64)

        # 6 columns: t, flux1, flux2, I0, I1, I2 (float64 little-endian)
        f1 = 1e-3 * np.sin(np.linspace(0, 2 * np.pi, n_rows))
        f2 = 1e-5 * np.cos(np.linspace(0, 2 * np.pi, n_rows))
        I0 = np.linspace(0.0, 10.0, n_rows)
        I1 = I0.copy()
        I2 = I0.copy()
        mat = np.column_stack([t, f1, f2, I0, I1, I2]).astype("<f8")

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "seg.bin"
            mat.tofile(p)

            reader = Sm18CorrSigsReader(Sm18ReaderConfig(strict_time=True, dt_rel_tol=0.25, max_currents=3))
            seg = reader.read(
                p,
                run_id="R",
                segment="S",
                samples_per_turn=Ns,
                shaft_speed_rpm=shaft_speed_rpm,
                aperture_id=1,
            )

            t_out = seg.df["t"].to_numpy()
            self.assertEqual(len(t_out), n_rows)
            # Exact equality is expected (no offsets, no re-sampling, no synthetic time).
            self.assertTrue(np.allclose(t_out, t, atol=0.0, rtol=0.0))

    def test_mba_disallows_time_alignment_options(self):
        """
        MBA: configuration options that imply time modification (align_time/strict_time)
        must be rejected, because NO synthetic/modified time is allowed in the project.
        """
        reader = MbaRawMeasurementReader(MbaReaderConfig(align_time=True, strict_time=False))
        with tempfile.TemporaryDirectory() as d:
            # We only need a path that *looks* like an MBA plateau file name.
            dummy = Path(d) / "X_Run_01_I_1.00A_NCS_raw_measurement_data.txt"
            dummy.write_text("0 0 0 0 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                reader.read(dummy, run_id="X", segment="NCS", samples_per_turn=8, aperture_id=1)


if __name__ == "__main__":
    unittest.main()
