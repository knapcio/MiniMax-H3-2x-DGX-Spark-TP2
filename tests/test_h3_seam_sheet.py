import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/h3_seam_sheet.py'
spec = importlib.util.spec_from_file_location('h3_seam_sheet', SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class SeamSheetTests(unittest.TestCase):
    def test_frame_indices_and_no_overwrite_on_real_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.mp4'; output = root / 'sheet.png'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x72:rate=24:duration=5',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            before = subject.digest(source)
            result = subject.make(source, output, seconds=5, window=107, overlap=22)
            self.assertEqual(result['boundaries'], [107])
            self.assertEqual(result['sample_frames'], [104, 107, 110])
            self.assertEqual(result['source_sha256'], before)
            self.assertEqual(result['output_sha256'], subject.digest(output))
            with Image.open(output) as image:
                self.assertEqual(image.width, 960)
            with self.assertRaises(ValueError): subject.make(source, output, seconds=5, window=107, overlap=22)
            self.assertEqual(subject.digest(source), before)

    def test_invalid_frame_offset_is_rejected_before_reading_source(self):
        for offset in (0, 25, True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                subject.make(Path('missing.mp4'), Path('out.png'), seconds=5, delta=offset)


if __name__ == '__main__':
    unittest.main()
