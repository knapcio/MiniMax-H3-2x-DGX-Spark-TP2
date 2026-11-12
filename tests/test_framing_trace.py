"""CPU tests of scripts/framing_trace.py: the settling rule and the shot sizes (no ffmpeg, no faces)."""
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("framing_trace", Path(__file__).resolve().parents[1] / "scripts" / "framing_trace.py")
ft = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ft)


class SizeTests(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(ft.shot_size(0.6), "close-up")
        self.assertEqual(ft.shot_size(0.35), "close-up")
        self.assertEqual(ft.shot_size(0.28), "medium")
        self.assertEqual(ft.shot_size(0.18), "wide")
        self.assertEqual(ft.shot_size(None), "no face")


class CutRuleTests(unittest.TestCase):
    def test_exact_frame_step(self):
        self.assertEqual(ft.sample_step(0.25, 24.0), 6)
        self.assertEqual(ft.sample_step(1.0, 24.0), 24)
        self.assertEqual(ft.sample_step(0.01, 24.0), 1)

    def test_failed_match_is_a_cut(self):
        self.assertTrue(ft.is_cut(5, None, 0.9))
        self.assertTrue(ft.is_cut(9, 1.0, 0.99))

    def test_low_correlation_with_an_impossible_zoom_is_a_cut(self):
        # the nine cuts of queues 147/148: 14-76 inliers, corr 0.13-0.38, "zoom" 0.35-0.76
        for inl, z, c in ((72, 0.76, 0.27), (29, 0.49, 0.26), (76, 0.75, 0.38), (41, 0.35, -0.05), (14, 0.39, 0.13)):
            self.assertTrue(ft.is_cut(inl, z, c), (inl, z, c))

    def test_moves_are_not_cuts(self):
        # a pull-back (zoom 0.92 per step, corr 0.51) and a tilt from the hands to the face (42 inliers, corr 0.19, zoom 1.03)
        self.assertFalse(ft.is_cut(332, 0.92, 0.51))
        self.assertFalse(ft.is_cut(42, 1.03, 0.19))
        self.assertFalse(ft.is_cut(73, 0.92, 0.30))
        self.assertFalse(ft.is_cut(400, 1.0, None))

    def test_correlation_of_images(self):
        from PIL import Image
        import random

        rnd = random.Random(1)
        a = Image.frombytes("L", (64, 36), bytes(rnd.randrange(256) for _ in range(64 * 36))).convert("RGB")
        self.assertAlmostEqual(ft.correlation(a, a), 1.0, places=6)
        b = Image.frombytes("L", (64, 36), bytes(rnd.randrange(256) for _ in range(64 * 36))).convert("RGB")
        self.assertLess(abs(ft.correlation(a, b)), 0.2)
        flat = Image.new("RGB", (64, 36), (40, 40, 40))
        self.assertEqual(ft.correlation(flat, flat), 1.0)


class HeadEndTests(unittest.TestCase):
    def test_static_shot_settles_at_zero(self):
        times = [i * 0.25 for i in range(16)]
        scales = [0.40 + (0.03 if i % 2 else -0.03) for i in range(16)]   # a talking head wobbles
        zooms = [None] + [1.0] * 15
        self.assertEqual(ft.head_end(times, scales, zooms), 0.0)

    def test_cut_from_wide_to_close_up(self):
        # four samples wide (face 0.12), a cut (no zoom match), then close-up at 0.38
        times = [i * 0.25 for i in range(16)]
        scales = [0.12] * 4 + [0.38] * 12
        zooms = [None] + [1.0] * 3 + [None] + [1.0] * 11
        self.assertEqual(ft.head_end(times, scales, zooms, cuts=[1.0]), 1.0)
        self.assertEqual(ft.head_end(times, scales, zooms), 1.0)   # the scale alone finds it too

    def test_push_in_settles_when_the_zoom_stops(self):
        # a 1.5 s push-in: the face grows and the zoom per step is 1.15, then static
        times = [i * 0.25 for i in range(16)]
        scales = [0.15 * 1.15 ** i for i in range(6)] + [0.15 * 1.15 ** 6] * 10
        zooms = [None] + [1.15] * 6 + [1.0] * 9
        self.assertAlmostEqual(ft.head_end(times, scales, zooms), 1.5)

    def test_push_in_to_the_last_frame_never_arrives(self):
        times = [i * 0.25 for i in range(8)]
        self.assertIsNone(ft.head_end(times, [0.4] * 8, [None] + [1.15] * 7))
        # a slow push-in (5 % per sample) has under 10 % of zoom left two samples before the end: it arrives there
        self.assertEqual(ft.head_end(times, [0.4] * 8, [None] + [1.05] * 7), 1.5)
        # a drift of 1 % in total is static
        self.assertEqual(ft.head_end(times, [0.4] * 8, [None] + [1.0014] * 7), 0.0)

    def test_missing_faces_do_not_break_the_rule(self):
        times = [i * 0.25 for i in range(8)]
        self.assertEqual(ft.head_end(times, [None, 0.3, None, 0.31, 0.3, None, 0.3, 0.3], [None] + [1.0] * 7), 0.25)
        self.assertIsNone(ft.head_end(times, [None] * 8, [None] + [1.0] * 7))
        self.assertIsNone(ft.head_end([], [], []))

    def test_cut_back_at_the_end_starts_the_final_framing(self):
        # close-up, then a cut back to the wide shot for the last third: the final framing is the wide tail
        times = [i * 0.25 for i in range(12)]
        scales = [0.38] * 8 + [0.12] * 4
        zooms = [None] + [1.0] * 7 + [None] + [1.0] * 3
        self.assertEqual(ft.head_end(times, scales, zooms, cuts=[2.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
