import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ground_light_stats.py'
spec = importlib.util.spec_from_file_location('ground_light_stats', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GroundLightStatsTests(unittest.TestCase):
    def test_samples_one_frame_per_second_without_timestamp_duplication(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for index in range(49):
                level = 20 if index < 24 else 100 if index < 48 else 200
                Image.new('RGB', (64, 64), (level, level, level)).save(root / f'frame-{index:03d}.png')
            video = root / 'sample.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-framerate', '24', '-i', str(root / 'frame-%03d.png'),
                            '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', str(video)], check=True)
            result = module.inspect(video)
            self.assertEqual(result['sample_seconds'], [0, 1, 2])
            self.assertEqual(len(result['ground_gray_mean']), 3)
            for actual, expected in zip(result['ground_gray_mean'], (20, 100, 200), strict=True):
                self.assertAlmostEqual(actual, expected, delta=2)
            self.assertEqual(len(result['ground_thirds_gray_mean']['left']), 3)
            self.assertIn('does not track clouds', result['caveat'])

    def test_rejects_invalid_region(self):
        for value in (-0.1, 1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    module.inspect(Path('unused.mp4'), value)


if __name__ == '__main__':
    unittest.main()
