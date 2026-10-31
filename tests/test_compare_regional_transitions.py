import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/compare_regional_transitions.py'
spec = importlib.util.spec_from_file_location('regional_transitions', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
class RegionalTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.make_video('source', 90)
        cls.candidate = cls.make_video('candidate', 95)
        cls.constant = cls.make_video('constant', 100)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def make_video(cls, name, dip):
        folder = cls.root / name
        folder.mkdir()
        for frame in range(70):
            value = dip if frame in (17, 34, 51) else 100
            Image.new('RGB', (64, 48), (value, value, value)).save(folder / f'{frame:03d}.png')
        output = cls.root / (name + '.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-framerate', '24', '-i', str(folder / '%03d.png'),
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-pix_fmt', 'yuv420p',
                        str(output)], check=True)
        return output

    def test_signed_changes_distinguish_darkening_and_partial_repair(self):
        report = module.compare(self.source, self.candidate)
        rows = {row['frame']: row for row in report['selected_transitions']}
        self.assertTrue({17, 34, 51}.issubset(rows))
        for frame in (17, 34, 51):
            for region in module.regional.LABELS:
                before = rows[frame]['source_signed_gray'][region]
                after = rows[frame]['candidate_signed_gray'][region]
                self.assertLess(before, after)
                self.assertLess(after, 0)
        self.assertIn('do not prove a decoder cause', report['caveat'])

    def test_no_periodic_changes_produces_empty_result(self):
        report = module.compare(self.constant, self.constant)
        self.assertEqual(report['selected_transitions'], [])
        self.assertIsNone(report['source_median_signed_gray'])

    def test_invalid_region_threshold_is_rejected(self):
        for count in (0, 7):
            with self.assertRaises(ValueError):
                module.compare(self.source, self.candidate, count)


if __name__ == '__main__':
    unittest.main()
