import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest
import wave


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/syncnet_probe.py'
spec = importlib.util.spec_from_file_location('syncnet_probe', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyncnetCropTests(unittest.TestCase):
    def write_report(self, root, samples):
        path = Path(root) / 'faces.json'
        path.write_text(json.dumps({'samples': samples}))
        return path

    def test_stable_face_produces_even_square_and_counts_multiple_faces(self):
        rows = [{'time_s_approx': i / 4, 'detected': True, 'face_count': 2 if i == 3 else 1,
                 'face_box': [.43, .31, .22, .38]} for i in range(40)]
        with tempfile.TemporaryDirectory() as root:
            crop = module.crop_from_landmarks(self.write_report(root, rows), 960, 544, 6.5)
        self.assertEqual(crop['samples'], 26)
        self.assertEqual(crop['detections'], 26)
        self.assertEqual(crop['samples_with_multiple_faces'], 1)
        self.assertEqual(crop['x'] % 2, 0)
        self.assertEqual(crop['y'] % 2, 0)
        self.assertEqual(crop['side'] % 2, 0)
        self.assertLessEqual(crop['x'] + crop['side'], 960)
        self.assertLessEqual(crop['y'] + crop['side'], 544)

    def test_missing_or_unstable_face_cannot_create_a_silent_verdict(self):
        rows = [{'time_s_approx': i / 4, 'detected': True, 'face_count': 1,
                 'face_box': [.2 if i % 2 else .7, .3, .2, .3]} for i in range(12)]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'varies too much'):
                module.crop_from_landmarks(self.write_report(root, rows), 960, 544, 3)
            rows[0]['detected'] = rows[1]['detected'] = False
            with self.assertRaisesRegex(ValueError, '90%'):
                module.crop_from_landmarks(self.write_report(root, rows), 960, 544, 3)

    def test_invalid_landmark_boxes_are_rejected(self):
        rows = [{'time_s_approx': 0, 'detected': True, 'face_box': [.9, .3, .2, .3]}]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'outside frame'):
                module.crop_from_landmarks(self.write_report(root, rows), 960, 544, 3)

    def test_a_later_interval_uses_only_its_own_face_samples(self):
        rows = [{'time_s_approx': i / 4, 'detected': True, 'face_count': 1,
                 'face_box': [.1 if i < 20 else .6, .3, .2, .3]} for i in range(40)]
        with tempfile.TemporaryDirectory() as root:
            report = self.write_report(root, rows)
            early = module.crop_from_landmarks(report, 960, 544, 3, start=0)
            later = module.crop_from_landmarks(report, 960, 544, 3, start=6)
        self.assertEqual(early['samples'], 12)
        self.assertEqual(later['samples'], 12)
        self.assertGreater(later['x'] - early['x'], 400)

    def test_delayed_control_really_delays_a_mid_file_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav, source = root / 'source.wav', root / 'source.mp4'
            rng = random.Random(42)
            samples = [rng.randrange(-10000, 10001) for _ in range(16000 * 5)]
            with wave.open(str(wav), 'wb') as f:
                f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                f.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=64x64:r=25:d=5',
                            '-i', str(wav), '-c:v', 'libx264', '-threads', '1', '-c:a', 'aac',
                            '-t', '5', str(source)], check=True)
            ref, delayed = root / 'reference.mkv', root / 'delayed.mkv'
            crop = {'x': 0, 'y': 0, 'side': 64}
            module.make_crop(source, ref, crop, 3, 0, start=1)
            module.make_crop(source, delayed, crop, 3, 200, start=1)
            def decoded(path):
                return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                                                '-ac', '1', '-ar', '16000', '-f', 's16le', '-'])
            original, shifted = decoded(ref), decoded(delayed)
            delay_bytes = 3200 * 2
            self.assertEqual(len(original), 3 * 16000 * 2)
            self.assertEqual(len(shifted), len(original))
            self.assertEqual(shifted[:delay_bytes], bytes(delay_bytes))
            self.assertEqual(shifted[delay_bytes:], original[:-delay_bytes])


if __name__ == '__main__':
    unittest.main()
