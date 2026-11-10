"""CPU tests of scripts/window_detail.py (no ffmpeg: the helpers only)."""
import argparse
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("window_detail", Path(__file__).resolve().parents[1] / "scripts" / "window_detail.py")
wd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wd)


class BoundsTests(unittest.TestCase):
    def test_serving_geometry(self):
        self.assertEqual(wd.window_bounds(1552, 277, 22), [(0, 277), (277, 532), (532, 787), (787, 1042), (1042, 1297), (1297, 1552)])
        self.assertEqual(len(wd.window_bounds(2827, 277, 22)), 11)

    def test_single_and_partial(self):
        self.assertEqual(wd.window_bounds(121, 277, 22), [(0, 121)])
        self.assertEqual(wd.window_bounds(300, 277, 22), [(0, 277), (277, 300)])

    def test_bad_overlap(self):
        with self.assertRaises(ValueError):
            wd.window_bounds(100, 22, 22)


class BoxTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(wd.parse_box("woman=520,80,700,240"), ("woman", (520, 80, 700, 240)))

    def test_bad(self):
        for text in ("520,80,700,240", "a=1,2,3", "a=10,10,5,20", "a=10,10,20,5"):
            with self.assertRaises((argparse.ArgumentTypeError, ValueError)):
                wd.parse_box(text)


class MeasureTests(unittest.TestCase):
    def test_flat_frame_has_no_detail_and_noise_has(self):
        import numpy as np

        flat = np.full((2, 16, 16), 100.0, dtype=np.float32)
        self.assertEqual(wd.laplacian_variance(flat).tolist(), [0.0, 0.0])
        noisy = np.random.default_rng(1).normal(100, 10, (1, 32, 32)).astype(np.float32)
        blurred = (noisy[:, :-1, :-1] + noisy[:, 1:, :-1] + noisy[:, :-1, 1:] + noisy[:, 1:, 1:]) / 4
        self.assertGreater(wd.laplacian_variance(noisy)[0], 2 * wd.laplacian_variance(blurred)[0])

    def test_per_window_means_are_plain_floats_even_from_numpy(self):
        import json

        import numpy as np

        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        out = wd.per_window(values, [0.25, 0.75, 1.25, 1.75], [(0, 24), (24, 48)], 24.0)
        self.assertEqual(json.loads(json.dumps(out)), [15.0, 35.0])

    def test_per_window_means(self):
        values = [10.0, 20.0, 30.0, 40.0]
        times = [0.25, 0.75, 1.25, 1.75]
        self.assertEqual(wd.per_window(values, times, [(0, 24), (24, 48), (48, 72)], 24.0), [15.0, 35.0, None])


if __name__ == "__main__":
    unittest.main()
