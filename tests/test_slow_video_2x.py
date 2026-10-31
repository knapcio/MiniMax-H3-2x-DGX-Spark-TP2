import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/slow_video_2x.py'
spec = importlib.util.spec_from_file_location('slow_video_2x', SCRIPT)
slow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slow)


class SlowVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-slow-video-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sample(self, name='source.mp4', *, audio=True, fps=24, audio_duration=1.25, duration=1.25):
        path = self.root / name
        cmd = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
               f'testsrc2=size=128x96:rate={fps}:duration={duration}']
        if audio:
            cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=32000:duration={audio_duration}']
        cmd += ['-map', '0:v:0']
        if audio:
            cmd += ['-map', '1:a:0']
        cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        if audio:
            cmd += ['-c:a', 'aac']
        subprocess.run(cmd + [str(path)], check=True)
        return path

    def test_slow_copy_preserves_source_frames_and_retimes_audio(self):
        source, output = self.sample(), self.root / 'slow.mp4'
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        report = slow.create(source, output)
        self.assertEqual(report['source_frames'], 30)
        self.assertEqual(report['output_frames'], 60)
        self.assertEqual(report['retained_even_frames_identical'], 30)
        self.assertTrue(report['last_frame_repeats_source'])
        self.assertEqual(report['output_seconds'], 2.5)
        self.assertEqual(report['video_timestamps'], 'continuous 24 fps on source and output')
        self.assertIn('reencoded', report['audio'])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
        self.assertEqual(report['output_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        info = slow.exact.probe(output)
        video = [s for s in info['streams'] if s['codec_type'] == 'video'][0]
        audio = [s for s in info['streams'] if s['codec_type'] == 'audio'][0]
        self.assertEqual(int(video['nb_frames']), 60)
        self.assertAlmostEqual(float(audio['duration']), 2.5, places=2)

    def test_silent_input_keeps_no_audio_and_preserves_source_frames(self):
        source, output = self.sample(audio=False), self.root / 'silent.mp4'
        report = slow.create(source, output)
        self.assertEqual(report['audio'], 'none')
        self.assertEqual(report['retained_even_frames_identical'], 30)
        self.assertFalse(any(s['codec_type'] == 'audio' for s in slow.exact.probe(output)['streams']))

    def test_twenty_percent_slower_preserves_aligned_anchors_and_audio(self):
        source = self.sample(duration=1, audio_duration=1)
        output = self.root / 'slow125.mp4'
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        report = slow.create(source, output, factor='1.25')
        self.assertEqual((report['source_frames'], report['output_frames']), (24, 30))
        self.assertEqual((report['anchor_frames_identical'], report['anchor_source_stride'],
                          report['anchor_output_stride']), (6, 4, 5))
        self.assertEqual(report['output_seconds'], 1.25)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
        audio = [s for s in slow.exact.probe(output)['streams'] if s['codec_type'] == 'audio'][0]
        self.assertAlmostEqual(float(audio['duration']), 1.25, places=2)

    def test_fractional_frame_count_or_unknown_factor_is_rejected(self):
        source = self.sample(audio=False)
        for factor in ('1.25', '1.3'):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    slow.create(source, self.root / ('bad' + factor + '.mp4'), factor=factor)

    def test_blended_interpolation_retains_anchors_and_retimes_audio(self):
        source = self.sample(duration=1, audio_duration=1)
        output = self.root / 'slow-blended.mp4'
        report = slow.create(source, output, factor='1.25', interpolation='blend')
        self.assertEqual(report['interpolation'], 'blend')
        self.assertEqual((report['anchor_frames_identical'], report['output_frames']), (6, 30))
        audio = [s for s in slow.exact.probe(output)['streams'] if s['codec_type'] == 'audio'][0]
        self.assertAlmostEqual(float(audio['duration']), 1.25, places=2)

    def test_rejects_existing_or_dangling_output_without_touching_it(self):
        source = self.sample(audio=False)
        output = self.root / 'exists.mp4'; output.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'new file'):
            slow.create(source, output)
        self.assertEqual(output.read_bytes(), b'keep')
        link = self.root / 'dangling.mp4'
        os.symlink(self.root / 'missing.mp4', link)
        with self.assertRaisesRegex(ValueError, 'new file'):
            slow.create(source, link)
        self.assertTrue(link.is_symlink())

    def test_rejects_wrong_frame_rate_or_truncated_audio(self):
        for args in ({'fps': 30}, {'audio_duration': 0.2}):
            with self.subTest(args=args):
                source = self.sample(name='src' + str(len(list(self.root.iterdir()))) + '.mp4', **args)
                with self.assertRaises(ValueError):
                    slow.create(source, self.root / ('out' + str(len(list(self.root.iterdir()))) + '.mp4'))

    def test_does_not_publish_if_source_changes_during_processing(self):
        source, output = self.sample(audio=False), self.root / 'changed.mp4'
        real = slow.digest
        calls = []
        def changed(path):
            calls.append(path)
            result = real(path)
            return result if len(calls) == 1 else 'changed'
        with mock.patch.object(slow, 'digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                slow.create(source, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob('.h3-slow2x-*')))

    def test_timestamp_verifier_rejects_a_shifted_frame(self):
        good = {'frames': [{'best_effort_timestamp_time': x} for x in ('0.000000', '0.041667', '0.083333')]}
        bad = {'frames': [{'best_effort_timestamp_time': x} for x in ('0.000000', '0.041667', '0.090000')]}
        with mock.patch.object(slow.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(good))):
            slow.verify_timestamps(self.root / 'video', 3)
        with mock.patch.object(slow.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(bad))):
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                slow.verify_timestamps(self.root / 'video', 3)


if __name__ == '__main__':
    unittest.main()
