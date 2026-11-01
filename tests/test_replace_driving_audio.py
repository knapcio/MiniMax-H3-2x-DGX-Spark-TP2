import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('replace_audio', SCRIPTS / 'replace_driving_audio.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ReplaceDrivingAudio(unittest.TestCase):
    def test_replaces_audio_without_changing_any_video_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); video = root / 'input.mp4'; source = root / 'voice.wav'; output = root / 'output.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x96:r=24:d=3',
                            '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=mono', '-t', '3', '-c:v', 'libx264',
                            '-preset', 'ultrafast', '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'anoisesrc=seed=42:r=32000:d=3',
                            '-c:a', 'pcm_s16le', str(source)], check=True)
            before = module.sha(video); result = module.replace(video, source, output)
            self.assertEqual(result['frames'], 72)
            self.assertTrue(result['decoded_video_frames_identical'])
            self.assertEqual(module.sha(video), before)
            ref = module.audio.decode(source, 8000, 3); got = module.audio.decode(output, 8000, 3)
            measured = module.audio.compare_segment(ref, got, 4000, 16000, 4000)
            self.assertGreater(measured['correlation'], .9)
            self.assertLess(abs(measured['shift_samples']), 10)
            with self.assertRaises(ValueError): module.replace(video, source, output)
            with self.assertRaises(ValueError): module.replace(video, source, video)


if __name__ == '__main__':
    unittest.main()
