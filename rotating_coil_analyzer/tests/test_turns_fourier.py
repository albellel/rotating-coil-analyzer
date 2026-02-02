import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rotating_coil_analyzer.models.frames import SegmentFrame
from rotating_coil_analyzer.analysis.turns import split_into_turns
from rotating_coil_analyzer.analysis.fourier import dft_per_turn


class TestTurnsFourier(unittest.TestCase):
    def test_split_into_turns_shapes_and_plateau_id(self):
        Ns = 8
        n_turns = 2
        n = Ns * n_turns

        t = np.arange(n, dtype=float)
        df_abs = np.arange(n, dtype=float) * 0.1
        df_cmp = df_abs * 0.01
        I = np.concatenate([np.full(Ns, 10.0), np.full(Ns, 20.0)])

        plateau_id = np.concatenate([np.zeros(Ns), np.ones(Ns)])
        plateau_step = np.concatenate([np.full(Ns, 0.0), np.full(Ns, 1.0)])
        plateau_I_hint = I.copy()

        df = pd.DataFrame(
            {
                "t": t,
                "df_abs": df_abs,
                "df_cmp": df_cmp,
                "I": I,
                "plateau_id": plateau_id,
                "plateau_step": plateau_step,
                "plateau_I_hint": plateau_I_hint,
            }
        )

        seg = SegmentFrame(
            source_path=Path("dummy"),
            run_id="R",
            segment="S",
            samples_per_turn=Ns,
            n_turns=n_turns,
            df=df.astype(np.float64, copy=False),
            warnings=(),
            aperture_id=1,
        )

        tb = split_into_turns(seg)
        self.assertEqual(tb.t.shape, (n_turns, Ns))
        self.assertTrue(np.allclose(tb.plateau_id, np.array([0.0, 1.0])))
        self.assertTrue(np.allclose(tb.I[:, 0], np.array([10.0, 20.0])))

    def test_dft_per_turn_complex_exponential(self):
        Ns = 16
        n_turns = 3
        n_h = 5

        k = np.arange(Ns)
        theta = 2.0 * np.pi * k / float(Ns)
        x = np.exp(1j * n_h * theta)  # one turn
        X = np.tile(x, (n_turns, 1))

        harm = dft_per_turn(X, n_max=Ns - 1)

        c = harm.coeff
        self.assertTrue(np.allclose(c[:, n_h], 1.0 + 0j, atol=1e-12, rtol=0.0))
        for m in [0, 1, 2, 3, 4, 6, 7]:
            self.assertTrue(np.allclose(c[:, m], 0.0 + 0j, atol=1e-12, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
