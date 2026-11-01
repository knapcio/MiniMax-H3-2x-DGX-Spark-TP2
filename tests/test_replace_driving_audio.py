import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('replace_audio', SCRIPTS / 'replace_driving_audio.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ReplaceDrivingAudio(unittest.TestCase):
    def test_private_snapshot_survives_transient_change_and_refuses_changed_inputs(self):
        for change in ('source_restored', 'source_persistent', 'snapshot'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                video, source, other, output = (root / name for name in ('video.mp4', 'source.wav', 'other.wav', 'out.mp4'))
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x96:r=24:d=2',
                                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', '-pix_fmt', 'yuv420p', str(video)], check=True)
                for path, seed in ((source, 1), (other, 2)):
                    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'anoisesrc=seed={seed}:r=32000:d=2',
                                    '-c:a', 'pcm_s16le', str(path)], check=True)
                original = source.read_bytes()
                video_sha, source_sha = module.sha(video), module.sha(source)
                real_run = subprocess.run
                seen = []

                def mutate(command, *args, **kwargs):
                    if command[0] != 'ffmpeg' or '-filter_complex' not in command:
                        return real_run(command, *args, **kwargs)
                    inputs = [Path(command[i + 1]) for i, arg in enumerate(command[:-1]) if arg == '-i']
                    snapshot = inputs[1]
                    self.assertNotEqual(snapshot, source)
                    self.assertEqual(snapshot.suffix, source.suffix)
                    self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o700)
                    seen.append(snapshot)
                    if change.startswith('source_'):
                        source.write_bytes(other.read_bytes())
                    try:
                        result = real_run(command, *args, **kwargs)
                    finally:
                        if change == 'source_restored':
                            source.write_bytes(original)
                    if change == 'snapshot':
                        snapshot.write_bytes(other.read_bytes())
                    return result

                with patch.object(module.subprocess, 'run', side_effect=mutate):
                    if change == 'source_restored':
                        result = module.replace(video, source, output)
                    else:
                        with self.assertRaisesRegex(ValueError, 'changed during processing'):
                            module.replace(video, source, output)
                self.assertEqual(len(seen), 1)
                self.assertFalse(list(root.glob('.driving-*')))
                self.assertEqual(module.sha(video), video_sha)
                if change == 'source_restored':
                    self.assertEqual(result['driving_audio_sha256'], source_sha)
                    ref = module.audio.decode(source, 8000, 2)
                    got = module.audio.decode(output, 8000, 2)
                    measured = module.audio.compare_segment(ref, got, 2000, 8000, 2000)
                    self.assertGreater(measured['correlation'], .9)
                    self.assertLess(abs(measured['shift_samples']), 10)
                else:
                    self.assertFalse(output.exists())

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
            self.assertTrue(result['source_audio_frame_timing']['continuous'])
            self.assertTrue(result['output_audio_frame_timing']['continuous'])
            self.assertEqual(module.sha(video), before)
            ref = module.audio.decode(source, 8000, 3); got = module.audio.decode(output, 8000, 3)
            measured = module.audio.compare_segment(ref, got, 4000, 16000, 4000)
            self.assertGreater(measured['correlation'], .9)
            self.assertLess(abs(measured['shift_samples']), 10)
            with self.assertRaises(ValueError): module.replace(video, source, output)
            with self.assertRaises(ValueError): module.replace(video, source, video)
            aac, gap, refused = root / 'audio.m4a', root / 'gap.m4a', root / 'must-not-exist.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-c:a', 'aac', str(aac)], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(aac), '-c:a', 'copy', '-bsf:a',
                            r'setts=pts=PTS+gte(PTS*TB\,1)/TB:dts=DTS+gte(DTS*TB\,1)/TB', str(gap)], check=True)
            with self.assertRaisesRegex(ValueError, 'timestamps are discontinuous'):
                module.replace(video, gap, refused)
            self.assertFalse(refused.exists())


if __name__ == '__main__':
    unittest.main()
