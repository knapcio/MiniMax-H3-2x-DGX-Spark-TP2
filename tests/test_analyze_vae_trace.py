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
        self.assertIn('not proof of cause', report['caveat'])

    def test_duplicate_missing_or_nonfinite_trace_is_rejected(self):
        line = 'h3_tp2 VAE temporal blend 1 approx-frame 17: old [1] new [2] blended [1]\n'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.parse_trace(line + line)
        with self.assertRaisesRegex(ValueError, 'no temporal'):
            module.parse_trace('different log')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            module.parse_trace(line.replace('[2]', '[NaN]'))

    def test_encoded_frames_are_sampled_once_each(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x64:rate=24:duration=1',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
            self.assertEqual(len(module.encoded_means(path)), 24)


if __name__ == '__main__': unittest.main()
