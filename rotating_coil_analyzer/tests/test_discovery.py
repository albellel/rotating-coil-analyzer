import unittest
import tempfile
from pathlib import Path

from rotating_coil_analyzer.ingest.discovery import MeasurementDiscovery


class TestDiscovery(unittest.TestCase):
    def _write_params(self, root: Path, ap2: bool = False):
        p = root / "Parameters.txt"
        # Escaped TABLE payload (single-line)
        ap1_tbl = r"TABLE{3\t0\t1\t0.47\n4\t2\t3\t0.47}"
        ap2_tbl = r"TABLE{3\t4\t5\t0.47\n4\t6\t7\t0.47}"
        txt = []
        txt.append("Parameters.Measurement.samples: 512")
        txt.append("Parameters.Measurement.v: -60")
        txt.append("Measurement.AP1.enabled: true")
        txt.append(f"Measurement.AP1.FDIs: {ap1_tbl}")
        if ap2:
            txt.append("Measurement.AP2.enabled: true")
            txt.append(f"Measurement.AP2.FDIs: {ap2_tbl}")
        else:
            txt.append("Measurement.AP2.enabled: false")
        p.write_text("\n".join(txt), encoding="utf-8")
        return p

    def test_regex_corr_and_generic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_params(root, ap2=False)

            # Two files, same run/ap/seg but different token; non-generic should win if both exist.
            (root / "RUN_corr_sigs_Ap_1_Seg3.bin").write_bytes(b"\x00" * 8 * 4 * 512)  # dummy size multiple
            (root / "RUN_generic_corr_sigs_Ap_1_Seg4.bin").write_bytes(b"\x00" * 8 * 4 * 512)

            cat = MeasurementDiscovery(strict=True).build_catalog(root)
            self.assertIn("RUN", cat.runs)
            # Ensure keys include aperture
            self.assertTrue(any(k[1] == 1 for k in cat.segment_files.keys()))

    def test_two_aperture_collision_prevented(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_params(root, ap2=True)

            (root / "RUN_corr_sigs_Ap_1_Seg3.bin").write_bytes(b"\x00" * 8 * 4 * 512)
            (root / "RUN_corr_sigs_Ap_2_Seg3.bin").write_bytes(b"\x00" * 8 * 4 * 512)

            cat = MeasurementDiscovery(strict=True).build_catalog(root)
            k1 = ("RUN", 1, "3")
            k2 = ("RUN", 2, "3")
            self.assertIn(k1, cat.segment_files)
            self.assertIn(k2, cat.segment_files)

    def test_table_parsing_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_params(root, ap2=False)
            (root / "RUN_corr_sigs_Ap_1_Seg3.bin").write_bytes(b"\x00" * 8 * 4 * 512)
            cat = MeasurementDiscovery(strict=True).build_catalog(root)
            segs = cat.segments_for_aperture(1)
            self.assertEqual([s.segment_id for s in segs], ["3", "4"])
            self.assertEqual(segs[0].fdi_abs, 0)
            self.assertEqual(segs[0].fdi_cmp, 1)


if __name__ == "__main__":
    unittest.main()
