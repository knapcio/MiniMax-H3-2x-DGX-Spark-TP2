import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('check_decoded_audio_extent', SCRIPTS / 'check_decoded_audio_extent.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frames(second_pts=16000):
    return {'streams': [{'sample_rate': '32000', 'time_base': '1/32000', 'duration': '9.0'}],
            'frames': [{'pts': 0, 'nb_samples': 16000, 'duration': 90000},
                       {'pts': second_pts, 'nb_samples': 16000, 'duration': 90000}]}


class DecodedAudioExtent(unittest.TestCase):
    def test_complete_coverage_and_explicit_threshold(self):
        measured = module.assess(frames(), 0, 1)
        self.assertEqual(measured['status'], 'MEASURED')
        self.assertEqual(measured['decoded_samples'], 32000)
        self.assertEqual(measured['decoded_end_s'], 1)
        self.assertEqual(module.assess(frames(), 0, 1, .04)['status'], 'PASS')

    def test_stretched_declared_duration_does_not_hide_missing_audio(self):
        result = module.assess(frames(), 0, 1.5, .04)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['missing_end_s'], .5)
        self.assertEqual(result['decoded_end_s'], 1)

    def test_short_start_and_discontinuous_pts(self):
        data = frames(); data['frames'][0]['pts'] = 1000; data['frames'][1]['pts'] = 17000
        self.assertAlmostEqual(module.assess(data, 0, 1, .02)['missing_start_s'], .03125)
        self.assertEqual(module.assess(data, 0, 1, .02)['status'], 'FAIL')
        self.assertEqual(module.assess(data, 0, 1, .04)['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            module.assess(frames(16100), 0, 1, .04)

    def test_invalid_intervals_and_thresholds_rejected(self):
        for start, end, limit in ((True, 1, None), (0, 0, None), (0, float('nan'), None),
                                  (0, 1, -1), (0, 1, float('inf'))):
            with self.subTest(values=(start, end, limit)), self.assertRaises(ValueError):
                module.assess(frames(), start, end, limit)

    def test_invalid_decoded_frame_shapes_and_stream_types(self):
        invalid = [None, {}, {'streams': [None], 'frames': [{}]},
                   {'streams': frames()['streams'], 'frames': [None]},
                   {'streams': frames()['streams'], 'frames': [42]}]
        for rate in (True, 32000.1, '32000.1', '-1', None):
            data = frames(); data['streams'][0]['sample_rate'] = rate; invalid.append(data)
        for base in (None, 0.5, '1/0', '-1/32000'):
            data = frames(); data['streams'][0]['time_base'] = base; invalid.append(data)
        for field, value in (('pts', True), ('pts', '0'), ('nb_samples', True), ('nb_samples', 0)):
            data = frames(); data['frames'][0][field] = value; invalid.append(data)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                module.assess(data, 0, 1, .04)

    def test_fifo_substitution_after_snapshot_does_not_block(self):
        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / 'tone.wav'
            with wave.open(str(media), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(32000); wav.writeframes(b'\0\0' * 32000)
            code = '''
import os, pathlib, sys
sys.path.insert(0, sys.argv[1])
import check_decoded_audio_extent as target
source = pathlib.Path(sys.argv[2])
def swap(_):
    source.unlink()
    os.mkfifo(source)
    return {'streams': [{'sample_rate': '32000', 'time_base': '1/32000'}],
            'frames': [{'pts': 0, 'nb_samples': 32000}]}
target.probe = swap
try:
    target.inspect(source, 0, 1, .04)
except ValueError as exc:
    assert 'nonregular' in str(exc), exc
else:
    raise AssertionError('FIFO substitution was accepted')
'''
            run = subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPTS), str(media)],
                                 capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_real_decoded_wav_and_cli_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); media = root / 'tone.wav'; report = root / 'report.json'
            with wave.open(str(media), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(32000); wav.writeframes(b'\0\0' * 32000)
            result = module.inspect(media, 0, 1, .04)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['decoded_samples'], 32000)
            cmd = [sys.executable, str(SCRIPTS / 'check_decoded_audio_extent.py'), str(media), '--expected-end', '1',
                   '--max-missing', '.04', '--report', str(report)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)['status'], 'PASS')
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
            self.assertEqual(report.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
