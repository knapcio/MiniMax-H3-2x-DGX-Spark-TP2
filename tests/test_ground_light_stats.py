import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image, ImageDraw


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ground_light_stats.py'
spec = importlib.util.spec_from_file_location('ground_light_stats', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GroundLightStatsTests(unittest.TestCase):
    def test_spatial_spread_distinguishes_uniform_and_local_light_change(self):
        reference = Image.new('L', (20, 20), 100)
        uniform = Image.new('L', (20, 20), 120)
        local = reference.copy()
        ImageDraw.Draw(local).rectangle((0, 0, 9, 19), fill=80)
        uniform_stats = module.spatial_change(reference, uniform, radius=0)
        local_stats = module.spatial_change(reference, local, radius=0)
        self.assertEqual(uniform_stats['signed_mean'], 20)
        self.assertEqual(uniform_stats['spatial_std'], 0)
        self.assertEqual(local_stats['signed_mean'], -10)
        self.assertEqual(local_stats['spatial_std'], 10)
        self.assertEqual(local_stats['boundary_or_clipped_pixels'], 0)

    def test_extreme_difference_reports_clipping(self):
        reference = Image.new('L', (4, 4), 0)
        current = Image.new('L', (4, 4), 255)
        self.assertEqual(module.spatial_change(reference, current, radius=0)['boundary_or_clipped_pixels'], 16)

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
            self.assertEqual(result['ground_change_spatial_std'], [0, 0, 0])
            self.assertEqual(result['adjacent_ground_change_spatial_std'], [0, 0])
            self.assertEqual(result['mean_adjacent_ground_change_spatial_std'], 0)
            self.assertIn('does not track clouds', result['caveat'])

    def test_adjacent_spread_detects_local_change_after_the_first_second(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for index in range(72):
                frame = Image.new('RGB', (336, 192), (100, 100, 100))
                if index >= 48:
                    ImageDraw.Draw(frame).rectangle((0, 120, 167, 191), fill=(60, 60, 60))
                frame.save(root / f'frame-{index:03d}.png')
            video = root / 'sample.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-framerate', '24', '-i', str(root / 'frame-%03d.png'),
                            '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', str(video)], check=True)
            result = module.inspect(video)
            self.assertEqual(result['adjacent_ground_change_spatial_std'][0], 0)
            self.assertGreater(result['adjacent_ground_change_spatial_std'][1], 10)
            self.assertEqual(result['max_adjacent_ground_change_spatial_std'],
                             result['adjacent_ground_change_spatial_std'][1])

    def test_rejects_invalid_region(self):
        for value in (-0.1, 1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    module.inspect(Path('unused.mp4'), value)


if __name__ == '__main__':
    unittest.main()
