import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('track_faces', SCRIPTS / 'track_face_detections.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def face(cx, cy=.5, width=.1, height=.2):
    return {'face_box': [cx - width / 2, cy - height / 2, width, height], 'confidence': .9}


class FaceTracks(unittest.TestCase):
    def test_reordered_detections_occlusion_and_background(self):
        left, right, tiny = face(.3), face(.7), face(.9, width=.01, height=.01)
        frames = [{'faces': [right, left, tiny]}, {'faces': [left]}, {'faces': [right]}, {'faces': []}]
        tracks = module.track(frames, [(.3, .5), (.7, .5)], fps=2)
        self.assertEqual([[r['detected'] for r in t] for t in tracks],
                         [[True, True, False, False], [True, False, True, False]])
        self.assertEqual(tracks[0][0]['face_box'], left['face_box'])
        self.assertEqual(tracks[1][0]['face_box'], right['face_box'])
        self.assertEqual(tracks[0][0]['time_s_approx'], .25)

    def test_assignment_maximizes_tracks_instead_of_stealing_a_face(self):
        near_both, left_only = face(.4), face(.2)
        frames = [{'faces': [near_both, left_only]}]
        left, right = module.track(frames, [(.3, .5), (.5, .5)], max_distance=.18)
        self.assertEqual(left[0]['face_box'], left_only['face_box'])
        self.assertEqual(right[0]['face_box'], near_both['face_box'])

    def test_cli_writes_separate_syncnet_compatible_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); detections = root / 'faces.json'; output = root / 'tracks'
            detections.write_text(json.dumps([{'faces': [face(.3), face(.7)]}]))
            command = [sys.executable, str(SCRIPTS / 'track_face_detections.py'), str(detections),
                       '--anchor', '.3', '.5', '--anchor', '.7', '.5', '--output-dir', str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads((output / 'track-01.json').read_text())['samples'][0]['detected'])
            self.assertTrue(json.loads((output / 'track-02.json').read_text())['samples'][0]['detected'])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        with self.assertRaises(ValueError):
            module.track([{'faces': []}], [(float('nan'), .5)])


if __name__ == '__main__':
    unittest.main()
