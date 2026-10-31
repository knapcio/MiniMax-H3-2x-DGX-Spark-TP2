import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/estimate_continuation.py'
spec = importlib.util.spec_from_file_location('estimate_continuation', SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ContinuationEstimateTests(unittest.TestCase):
    def test_alignment_matches_recorded_durations(self):
        for seconds, frames in ((10, 243), (15, 362), (16, 396), (30, 736), (60, 1450), (300, 7213)):
            with self.subTest(seconds=seconds):
                self.assertEqual(subject.aligned_frames(seconds), frames)

    def test_300_second_window_counts_and_last_window(self):
        expected = {141: (61, 60, [73]), 175: (47, 47, []), 277: (29, 28, [73])}
        for window, (count, full, short) in expected.items():
            with self.subTest(window=window):
                result = subject.summarize(7213, window, 22, 960, 544, 100.0)
                self.assertEqual((result['windows'], result['full_windows'], result['short_window_frames']),
                                 (count, full, short))
                self.assertEqual(result['full_windows_seconds_at_observed_rate'], full * 100.0)
                self.assertAlmostEqual(result['final_cumulative_video_mib_float32'], 396.32, places=2)
                self.assertIn('excludes short windows', result['caveat'])

    def test_single_short_window_and_invalid_inputs(self):
        result = subject.summarize(107, 277, 22, 960, 544)
        self.assertEqual(result['short_window_frames'], [107])
        self.assertEqual(result['handovers'], 0)
        self.assertIsNone(result['full_windows_seconds_at_observed_rate'])
        for args in ((7213, 140, 22), (7213, 141, 23), (7200, 141, 22), (7213, 141, 141)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                subject.plan(*args)
        with self.assertRaises(ValueError): subject.summarize(7213, 277, 22, 960, 545)
        with self.assertRaises(ValueError): subject.aligned_frames(float('nan'))
        with self.assertRaises(ValueError): subject.aligned_frames(3)
        with self.assertRaises(ValueError): subject.aligned_frames(True)


if __name__ == '__main__':
    unittest.main()
