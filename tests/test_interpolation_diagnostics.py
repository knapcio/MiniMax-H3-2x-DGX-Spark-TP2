import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('interpolation_diagnostics', ROOT / 'scripts/interpolation_diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
spec = importlib.util.spec_from_file_location('slow_video_2x', ROOT / 'scripts/slow_video_2x.py')
slow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slow)


class InterpolationDiagnosticsTests(unittest.TestCase):
    def frame(self, value):
        return Image.new('L', (diag.WIDTH, diag.HEIGHT), value)

    def test_linear_triplet_has_no_excess_and_even_halves(self):
        row = diag.triplet(self.frame(0), self.frame(5), self.frame(10))['whole']
        self.assertEqual((row['first_half'], row['second_half'], row['direct']), (5, 5, 10))
        self.assertEqual(row['excess'], 0)
        self.assertEqual(row['imbalance'], 0)

    def test_middle_only_flash_has_excess(self):
        row = diag.triplet(self.frame(0), self.frame(20), self.frame(0))['ground']
        self.assertEqual((row['first_half'], row['second_half'], row['direct'], row['excess']), (20, 20, 0, 40))
        with self.assertRaises(ValueError):
            diag.triplet(Image.new('RGB', (diag.WIDTH, diag.HEIGHT)), self.frame(0), self.frame(0))

    def test_real_encoded_retiming_has_expected_triplet_count_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, candidate = root / 'source.mp4', root / 'slow.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=24:duration=1',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            slow.create(source, candidate)
            report = diag.inspect(candidate, top=3)
            self.assertEqual(report['frames'], 48)
            self.assertEqual(report['triplets'], 23)
            self.assertEqual(len(report['highest_excess_triplets']), 3)
            self.assertEqual(report['video_sha256'], hashlib.sha256(candidate.read_bytes()).hexdigest())
            self.assertIn('not a quality', report['caveat'])
            with self.assertRaises(ValueError):
                diag.inspect(candidate, top=0)


if __name__ == '__main__':
    unittest.main()
