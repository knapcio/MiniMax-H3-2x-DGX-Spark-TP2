import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('normalize_audio', ROOT / 'scripts/normalize_ambient_audio.py')
normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalize)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
class AudioNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def clip(self, name, *, silent=False, volume=-25):
        path = self.root / name
        audio = 'anullsrc=r=32000:cl=mono' if silent else 'sine=frequency=440:sample_rate=32000'
        command = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=24',
                   '-f', 'lavfi', '-i', audio, '-t', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac']
        if not silent:
            command += ['-af', f'volume={volume}dB']
        subprocess.run(command + [str(path)], check=True, capture_output=True)
        return path

    def test_levelled_copy_preserves_video_and_refuses_overwrite(self):
        source = self.clip('source.mp4')
        before = source.read_bytes()
        output = self.root / 'levelled.mp4'
        report = normalize.create(source, output)
        self.assertTrue(output.is_file())
        self.assertTrue(output.with_suffix('.audio-normalization.json').is_file())
        self.assertTrue(report['video_packets_and_timestamps_identical'])
        self.assertLess(abs(float(report['after']['input_i']) + 20), 1)
        self.assertEqual(before, source.read_bytes())
        with self.assertRaises(FileExistsError):
            normalize.create(source, output)

    def test_silence_and_excessive_boost_are_not_published(self):
        for source in (self.clip('silent.mp4', silent=True), self.clip('very-quiet.mp4', volume=-35)):
            output = self.root / (source.stem + '-result.mp4')
            with self.assertRaises(ValueError):
                normalize.create(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.audio-normalization.json').exists())


if __name__ == '__main__':
    unittest.main()
