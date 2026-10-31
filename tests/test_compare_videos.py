import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/compare_videos.py'
spec = importlib.util.spec_from_file_location('compare_videos', SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class CompareVideosTests(unittest.TestCase):
    def test_comparison_verifies_media_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); left, right, output = root / 'left.mp4', root / 'right.mp4', root / 'comparison.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=24:duration=2',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=2', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(left)], check=True)
            shutil.copyfile(left, right)
            before = (subject.digest(left), subject.digest(right))
            result = subject.create(left, right, output, frames=48, left_title='Przed', right_title='Po')
            self.assertEqual(result['frames'], 48)
            self.assertEqual(result['sha256'], subject.digest(output))
            self.assertEqual((subject.digest(left), subject.digest(right)), before)
            self.assertEqual(subject.exact.audio_packets(output), subject.exact.audio_packets(left))
            with self.assertRaises(ValueError):
                subject.create(left, right, output, frames=48, left_title='Przed', right_title='Po')
            self.assertEqual(result['sha256'], subject.digest(output))

    def test_invalid_frame_count_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                subject.create(root / 'a', root / 'b', root / 'out', frames=0, left_title='A', right_title='B')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
