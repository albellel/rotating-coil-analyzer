import unittest
import tempfile
from pathlib import Path
import numpy as np

from rotating_coil_analyzer.ingest.readers_streaming import StreamingReader, StreamingReaderConfig


class TestReaderBinaryInference(unittest.TestCase):
    def _write_bin(self, path: Path, mat: np.ndarray):
        mat.astype("<f8", copy=False).tofile(path)

    def test_dt_nominal_accepts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            Ns = 512
            n_turns = 5
            n = Ns * n_turns
            dt = 1.0 / Ns  # v=60 rpm => T=1s
            t = np.arange(n) * dt
            df_abs = np.sin(2*np.pi*(np.arange(n)/Ns))
            df_cmp = 1e-3 * df_abs
            I = np.linspace(0, 100, n)
            mat = np.column_stack([t, df_abs, df_cmp, I])
            self._write_bin(p, mat)

            cfg = StreamingReaderConfig(strict_time=True, dt_rel_tol=0.1, max_currents=2)
            reader = StreamingReader(cfg)
            seg = reader.read(p, run_id="RUN", segment="3", samples_per_turn=Ns, shaft_speed_rpm=60.0)
            self.assertEqual(seg.n_turns, n_turns)

    def test_dt_nominal_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            Ns = 512
            n_turns = 5
            n = Ns * n_turns
            dt = 0.01  # far from 1/Ns
            t = np.arange(n) * dt
            df_abs = np.sin(2*np.pi*(np.arange(n)/Ns))
            df_cmp = 1e-3 * df_abs
            I = np.linspace(0, 100, n)
            mat = np.column_stack([t, df_abs, df_cmp, I])
            self._write_bin(p, mat)

            cfg = StreamingReaderConfig(strict_time=True, dt_rel_tol=0.1, max_currents=2)
            reader = StreamingReader(cfg)
            with self.assertRaises(ValueError):
                reader.read(p, run_id="RUN", segment="3", samples_per_turn=Ns, shaft_speed_rpm=60.0)


if __name__ == "__main__":
    unittest.main()
