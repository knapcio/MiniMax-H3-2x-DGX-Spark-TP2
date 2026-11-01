import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('join', SCRIPTS / 'screen_segment_join.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class SegmentJoinTests(unittest.TestCase):
    def make_video(self, path, color):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={color}:s=160x96:r=24:d=2',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-threads', '1',
                        '-pix_fmt', 'yuv420p', str(path)], check=True)

    def test_matching_and_different_color_joins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / 'left.mp4'; same = root / 'same.mp4'; different = root / 'different.mp4'
            self.make_video(left, 'red'); self.make_video(same, 'red'); self.make_video(different, 'green')
            matching, sheet = module.assess(left, same)
            self.assertEqual(sheet.shape, (6, 180, 320, 3))
            self.assertFalse(matching['attention_suggested'])
            self.assertLess(matching['boundary_rgb_mae_255'], 1)
            changed, _ = module.assess(left, different)
            self.assertTrue(changed['attention_suggested'])
            self.assertGreater(changed['boundary_rgb_mae_255'], 15)
            report, contact = root / 'report.json', root / 'contact.png'
            cmd = [sys.executable, str(SCRIPTS / 'screen_segment_join.py'), str(left), str(different),
                   '--report', str(report), '--contact', str(contact)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(report.read_text())['attention_suggested'])
            self.assertTrue(contact.is_file())
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            with self.assertRaises(ValueError): module.assess(left, same, frames=1)


if __name__ == '__main__':
    unittest.main()
