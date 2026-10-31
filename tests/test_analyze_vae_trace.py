import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_vae_trace.py'
spec = importlib.util.spec_from_file_location('analyze_vae_trace', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VaeTraceAnalysisTests(unittest.TestCase):
    def test_maps_same_phase_drops_to_approximate_overlap(self):
        text = ('INFO h3_tp2 VAE temporal blend 1 approx-frame 17: old [97, 98] new [101, 100] blended [97, 99]\n'
                'INFO h3_tp2 VAE temporal blend 2 approx-frame 34: old [97, 98] new [101, 100] blended [97, 99]\n')
        means = [100.0] * 50
        means[17] = means[34] = 97.0
        report = module.compare(module.parse_trace(text), means)
        self.assertEqual(len(report['periodic_global_drop_runs']), 1)
        self.assertEqual([row['approx_frame'] for row in report['overlaps']], [17, 34])
        self.assertTrue(all(row['encoded_repeated_drop'] for row in report['overlaps']))
        self.assertEqual(report['overlaps'][0]['new_minus_old_first'], 4)
        self.assertEqual(report['overlaps'][0]['old_rebound_first'], 1)
        self.assertEqual(report['overlaps'][0]['new_rebound_first'], -1)
        self.assertEqual(report['overlaps'][0]['blend_rebound_first'], 2)
        self.assertEqual(report['overlaps'][0]['encoded_rebound_after'], 3)
        boundary = report['encoded_changes_by_phase'][0]
        self.assertEqual(boundary['frames'], 2)
        self.assertEqual(boundary['median_change'], -3)
        self.assertEqual(boundary['negative_fraction'], 1)
        self.assertEqual(boundary['median_next_change'], 3)
        self.assertEqual(report['encoded_changes_by_phase'][1]['median_change'], 3)
        self.assertIn('not proof of cause', report['caveat'])

    def test_duplicate_missing_or_nonfinite_trace_is_rejected(self):
        line = 'h3_tp2 VAE temporal blend 1 approx-frame 17: old [1] new [2] blended [1]\n'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.parse_trace(line + line)
        with self.assertRaisesRegex(ValueError, 'no temporal'):
            module.parse_trace('different log')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            module.parse_trace(line.replace('[2]', '[NaN]'))

    def test_exploratory_offset_can_find_shifted_periodic_drops(self):
        text = ('h3_tp2 VAE temporal blend 1 approx-frame 17: old [97] new [101] blended [97]\n'
                'h3_tp2 VAE temporal blend 2 approx-frame 34: old [97] new [101] blended [97]\n')
        means = [100.0] * 50
        means[18] = means[35] = 97.0
        report = module.compare(module.parse_trace(text), means)
        self.assertEqual(report['drop_matches_by_candidate_offset'][1], 2)
        self.assertEqual(report['candidate_offsets_with_most_matches'], [1])
        self.assertTrue(all(row['nearest_repeated_drop_offset'] == 1 for row in report['overlaps']))
        self.assertFalse(any(row['encoded_repeated_drop'] for row in report['overlaps']))
        self.assertIsNone(report['overlaps'][0]['old_rebound_first'])
        self.assertIn('by chance', report['caveat'])

    def test_encoded_frames_are_sampled_once_each(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x64:rate=24:duration=1',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
            self.assertEqual(len(module.encoded_means(path)), 24)

    def test_short_sequence_leaves_unseen_phases_empty(self):
        text = 'h3_tp2 VAE temporal blend 1 approx-frame 1: old [99] new [99] blended [99]\n'
        report = module.compare(module.parse_trace(text), [100.0, 99.0])
        self.assertIsNone(report['encoded_changes_by_phase'][0]['median_change'])
        self.assertIsNone(report['encoded_changes_by_phase'][1]['median_next_change'])
        self.assertEqual(report['encoded_changes_by_phase'][1]['negative_fraction'], 1)

    def test_phase_stats_honor_interval_and_reject_invalid_bounds(self):
        values = [100.0] * 52
        values[17] = values[34] = values[51] = 97.0
        result = module.phase_changes(values, start=18, end=50)
        self.assertEqual(result[0]['frames'], 1)
        self.assertEqual(result[0]['median_change'], -3)
        for kwargs in ({'start': 0}, {'start': 10, 'end': 10}, {'end': 53}, {'period': 0}):
            with self.assertRaises(ValueError):
                module.phase_changes(values, **kwargs)


if __name__ == '__main__': unittest.main()
