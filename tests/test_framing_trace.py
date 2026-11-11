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
