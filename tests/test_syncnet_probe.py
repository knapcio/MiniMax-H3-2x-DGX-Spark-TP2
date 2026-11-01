import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


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


if __name__ == '__main__':
    unittest.main()
