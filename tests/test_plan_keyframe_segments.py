"""Exact frame/sample arithmetic for optional two-keyframe story segments."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import plan_keyframe_segments as planner
import snapshot_first_last


class KeyframeSegmentPlanTests(unittest.TestCase):
    def test_300_seconds_has_minimal_balanced_exact_plan(self):
        result = planner.plan_segments(7200)
        self.assertTrue(result['possible_preserving_endpoints'])
        self.assertEqual(result['segment_count'], 29)
        self.assertEqual(Counter(x['frames'] for x in result['segments']), {243: 20, 260: 9})
        self.assertEqual(result['target_seconds_exact'], '300')
        self.assertEqual(result['target_audio_samples'], 9_600_000)
        self.assertEqual(result['segments'][0]['start_frame'], 0)
        self.assertEqual(result['segments'][-1]['end_frame'], 7200)
        self.assertEqual(result['segments'][0]['timeline_start_sample'], 0)
        self.assertEqual(result['segments'][-1]['timeline_end_sample'], 9_600_000)
        for left, right in zip(result['segments'], result['segments'][1:]):
            self.assertEqual(left['end_frame'], right['start_frame'])
            self.assertEqual(left['timeline_end_sample'], right['timeline_start_sample'])
        for part in result['segments']:
            error = abs(Fraction(part['timeline_end_sample']) - Fraction(part['end_frame'] * 32000, 24))
            self.assertLessEqual(error, Fraction(1, 2))
            info = snapshot_first_last.validate_request(
                {'task': 'fl2va', 'frame_indices': [0, -1]}, part['request_duration_seconds'])
            self.assertEqual((info['requested_frames'], info['aligned_frames']), (part['frames'], part['frames']))

    def test_300_seconds_with_each_later_first_frame_discarded(self):
        result = planner.plan_segments(7200, drop_boundary_frame=True)
        self.assertTrue(result['possible_preserving_endpoints'])
        self.assertEqual(result['segment_count'], 36)
        self.assertEqual(Counter(x['frames'] for x in result['segments']), {192: 17, 209: 19})
        self.assertEqual(sum(x['frames'] for x in result['segments']), 7235)
        self.assertEqual(sum(x['retained_frames'] for x in result['segments']), 7200)
        self.assertEqual(sum(x['discard_prefix_frames'] for x in result['segments']), 35)
        self.assertEqual(result['segments'][1]['source_audio_cut_start_samples_exact'], '4000/3')
        self.assertEqual(result['segments'][-1]['timeline_end_sample'], 9_600_000)
        for left, right in zip(result['segments'], result['segments'][1:]):
            self.assertEqual(left['end_frame'], right['start_frame'])
            self.assertEqual(left['timeline_end_sample'], right['timeline_start_sample'])
        sixty = planner.plan_segments(1440, drop_boundary_frame=True)
        self.assertEqual(sixty['segment_count'], 7)
        self.assertEqual(sum(x['retained_frames'] for x in sixty['segments']), 1440)

    def test_exact_short_targets_can_be_impossible_without_trimming(self):
        for seconds in (10, 15, 30, 60):
            with self.subTest(seconds=seconds):
                result = planner.plan_segments(seconds * 24)
                self.assertFalse(result['possible_preserving_endpoints'])
                self.assertNotIn('segments', result)
        one = planner.plan_segments(243)
        self.assertEqual(one['segment_count'], 1)
        self.assertEqual(one['segments'][0]['frames'], 243)

    def test_counts_match_independent_dynamic_program(self):
        # Independent coin-change reference checks both impossibility and the minimal count.
        limit = 1200
        for window in (107, 175, 277):
            for drop in (False, True):
                with self.subTest(window=window, drop=drop):
                    allowed = range(107, window + 1, 17)
                    best = [None] * (limit + 1)
                    best[0] = 0
                    for n in range(1, limit + 1):
                        candidates = [1] if n in allowed else []
                        candidates += [best[n - f + int(drop)] + 1 for f in allowed
                                       if 0 < n - f + int(drop) <= limit and best[n - f + int(drop)] is not None]
                        if candidates:
                            best[n] = min(candidates)
                    for frames in range(1, limit + 1, 7):
                        result = planner.plan_segments(frames, window=window, drop_boundary_frame=drop)
                        self.assertEqual(result['possible_preserving_endpoints'], best[frames] is not None, frames)
                        if best[frames] is not None:
                            self.assertEqual(result['segment_count'], best[frames], frames)

    def test_decimal_validation_and_invalid_bounds(self):
        self.assertEqual(planner.seconds_to_frames('10.125'), 243)
        self.assertEqual(planner.seconds_to_frames('300.000'), 7200)
        for raw in ('0', '-1', 'NaN', 'Infinity', '0.01', '10.833333333333334', '1e-999999999', '9' * 129):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                planner.seconds_to_frames(raw)
        for kwargs in ({'window': 240}, {'overlap': 175, 'window': 175}, {'sample_rate': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                planner.plan_segments(7200, **kwargs)
        for value in (True, 0, -1, 1_000_001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                planner.plan_segments(value)
        with self.assertRaises(ValueError):
            planner.plan_segments(7200, drop_boundary_frame=1)

    def test_cli_reports_infeasible_without_fallback(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/plan_keyframe_segments.py'), '--seconds', '10'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stdout)['possible_preserving_endpoints'])


if __name__ == '__main__':
    unittest.main()
