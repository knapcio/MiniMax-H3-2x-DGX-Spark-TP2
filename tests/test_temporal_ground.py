import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load('temporal_ground', ROOT / 'results/20261030-painting-fl2va-continuation/experimental-temporal-ground.py')
verify = load('verify_luma', ROOT / 'scripts/verify_luma_repair.py')


class TemporalGroundTests(unittest.TestCase):
    def test_centered_filter_is_symmetric_and_preserves_linear_change_away_from_edges(self):
        kernel = subject.weights(3, 1.5)
        np.testing.assert_allclose(kernel, kernel[::-1])
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)
        impulse = np.zeros((11, 1, 1), dtype=np.float32); impulse[5] = 10
        before = subject.smoothed_difference(impulse, 4, kernel)[0, 0]
        after = subject.smoothed_difference(impulse, 6, kernel)[0, 0]
        self.assertAlmostEqual(float(before), float(after), places=6)
        linear = np.arange(11, dtype=np.float32).reshape(11, 1, 1)
        self.assertAlmostEqual(float(subject.smoothed_difference(linear, 5, kernel)[0, 0]), 0, places=5)
        self.assertEqual(subject.boundary_weight(0, 48, 12), 0)
        self.assertEqual(subject.boundary_weight(47, 48, 12), 0)
        self.assertEqual(subject.boundary_weight(24, 48, 12), 1)

    def test_correction_respects_mask_and_luma_limit(self):
        frame = np.full((20, 20), 100, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32); mask[10:] = 1
        delta = np.full((5, 5), 50, dtype=np.float32)
        out = subject.adjust_luma(frame, delta, mask, 1.0, 6)
        np.testing.assert_array_equal(out[:10], frame[:10])
        self.assertEqual(int(np.max(np.abs(out.astype(np.int16) - frame.astype(np.int16)))), 6)

    def test_real_media_preserves_audio_chroma_timing_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(48):
                frame = Image.new('RGB', (96, 64), (100, 100, 100))
                level = 40 if index % 6 < 3 else 180
                ImageDraw.Draw(frame).rectangle((0, 42, 95, 63), fill=(level, level, level))
                frame.save(root / f'frame-{index:03d}.png')
            source, output = root / 'source.mp4', root / 'output.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-framerate', '24', '-i', str(root / 'frame-%03d.png'),
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=2', '-c:v', 'libx264',
                            '-crf', '0', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
            with source.open('rb') as stream: before = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.assertEqual(subject.render(source, output), 48)
            result = verify.inspect(source, output, limit=6, unchanged_above=40)
            self.assertEqual(result['source_sha256'], before)
            self.assertGreater(len(result['changed_luma_frames']), 0)
            self.assertNotIn(0, result['changed_luma_frames'])
            self.assertNotIn(47, result['changed_luma_frames'])
            self.assertLessEqual(result['max_luma_change'], 6)
            self.assertTrue(result['chroma_identical'] and result['video_timing_identical'] and result['aac_packets_identical'])
            with self.assertRaises(ValueError): subject.render(source, output)


if __name__ == '__main__':
    unittest.main()
