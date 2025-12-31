import unittest
import tempfile
from pathlib import Path

import numpy as np

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery
from rotating_coil_analyzer.ingest.readers_mba import MbaRawMeasurementReader, MbaReaderConfig


class TestMBA(unittest.TestCase):
    def test_mh_fdis_and_plateau_concatenation(self):
        """Acceptance test for MBA ingest:

        Requirements validated here:
          - Plateau files are discovered and concatenated in Run_XX order.
          - No synthetic/modified time: raw per-file time is preserved in df['t'] (resets allowed).
          - Plateau-safe trimming: each plateau is trimmed to an integer number of turns.
        """
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            # Parameters uses MH.FDIs
            params = "\n".join(
                [
                    "Parameters.Measurement.samples: 8",
                    "Parameters.Measurement.v: 60",
                    r"Parameters.MH.FDIs: TABLE{NCS\t0\t1\t0.470\nCS\t2\t3\t0.470}",
                ]
            )
            (root / "Parameters.txt").write_text(params, encoding="utf-8")

            base = "20251212_171620_MBA"

            # Create two plateau files per segment (Run_01 and Run_02), with constant currents 10 and 20.
            # Each file has 10 samples; with samples_per_turn=8 each plateau must be trimmed to 8 samples.
            def write_plateau(seg: str, run_no: int, current: float):
                t = np.arange(10) * 0.1  # raw time resets per file (MBA reality)
                absf = np.sin(np.linspace(0, 1, 10)) * 1e-3
                cmpf = absf * 1e-2
                I = np.full(10, current)
                aux = np.zeros(10)
                mat = np.column_stack([t, absf, cmpf, I, aux])
                fn = f"{base}_Run_{run_no:02d}_I_{current:.2f}A_{seg}_raw_measurement_data.txt"
                np.savetxt(root / fn, mat)

            write_plateau("NCS", 1, 10.0)
            write_plateau("NCS", 2, 20.0)
            write_plateau("CS", 1, 10.0)
            write_plateau("CS", 2, 20.0)

            cat = MeasurementDiscovery(strict=True).build_catalog(root)

            # One run (base), 2 segments
            self.assertIn(base, cat.runs)
            f_ncs = cat.get_segment_file(base, 1, "NCS")
            self.assertTrue(f_ncs.name.endswith("_NCS_raw_measurement_data.txt"))

            # Reader concatenates both plateau files WITHOUT time stitching.
            reader = MbaRawMeasurementReader(MbaReaderConfig())
            seg_frame = reader.read(f_ncs, run_id=base, segment="NCS", samples_per_turn=8, aperture_id=1)

            # 2 plateaus * (10 samples trimmed to 8) = 16
            self.assertEqual(len(seg_frame.df), 16)

            # Raw time resets at the start of the second plateau (no synthetic alignment)
            self.assertAlmostEqual(float(seg_frame.df["t"].iloc[0]), 0.0, places=12)
            self.assertAlmostEqual(float(seg_frame.df["t"].iloc[8]), 0.0, places=12)

            # Plateau metadata present and boundary is exactly at a multiple of samples_per_turn
            self.assertIn("plateau_id", seg_frame.df.columns)
            self.assertEqual(int(seg_frame.df["plateau_id"].iloc[7]), 0)
            self.assertEqual(int(seg_frame.df["plateau_id"].iloc[8]), 1)

            # Current spans 10 -> 20
            self.assertGreater(np.nanpercentile(seg_frame.df["I"], 99), 15.0)


if __name__ == "__main__":
    unittest.main()
