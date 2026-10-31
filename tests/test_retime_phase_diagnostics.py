import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/retime_phase_diagnostics.py'
spec = importlib.util.spec_from_file_location('retime_phase', SCRIPT)
phase = importlib.util.module_from_spec(spec); spec.loader.exec_module(phase)


class RetimePhaseTests(unittest.TestCase):
    def test_phase_statistics_use_transition_frame_numbers(self):
        values = [0] + [1 if n % 5 else 3 for n in range(1, 26)]
        result = phase.phase_summary(values, 5)
        self.assertEqual([p['median'] for p in result], [3, 1, 1, 1, 1])
        self.assertEqual(sum(p['count'] for p in result), 25)

    def test_insufficient_data_and_invalid_period_fail(self):
        for values, period in (([0] * 20, 1), ([0] * 20, 61), ([0] * 5, 5)):
            with self.subTest(period=period):
                with self.assertRaises(ValueError): phase.phase_summary(values, period)

    def test_real_video_is_fully_decoded_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=128x96:rate=24:duration=1', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(path)], check=True)
            before = phase.digest(path)
            report = phase.inspect(path, 5)
            self.assertEqual(report['frames'], 24)
            self.assertEqual(report['sha256'], before)
            self.assertEqual(phase.digest(path), before)
            self.assertEqual(len(report['regions']['lower-middle']), 5)
            self.assertIn('not by themselves prove', report['caveat'])


if __name__ == '__main__': unittest.main()
