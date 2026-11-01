import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('screen_speech_levels', SCRIPTS / 'screen_speech_levels.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RATE = 32000


def write_wav(path, data):
    pcm = np.clip(np.rint(data * 32767), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(RATE); f.writeframes(pcm.tobytes())


class SpeechLevels(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.wav'; self.output = self.root / 'output.wav'
        t = np.arange(4 * RATE) / RATE
        data = np.zeros(4 * RATE)
        data[RATE // 2:3 * RATE // 2] = .2 * np.sin(2 * np.pi * 440 * t[RATE // 2:3 * RATE // 2])
        data[2 * RATE:3 * RATE] = .3 * np.sin(2 * np.pi * 660 * t[2 * RATE:3 * RATE])
        altered = data.copy(); altered[2 * RATE:3 * RATE] *= .1
        write_wav(self.source, data); write_wav(self.output, altered)
        self.lines = [{'start': .5, 'end': 1.5, 'speaker': 'A'}, {'start': 2, 'end': 3, 'speaker': 'B'}]

    def test_detects_attenuation_but_no_default_quality_threshold(self):
        report = MODULE.assess(self.source, self.output, self.lines)
        self.assertEqual(report['status'], 'MEASURED')
        self.assertAlmostEqual(report['intervals'][0]['output_over_source_db'], 0, places=3)
        self.assertAlmostEqual(report['intervals'][1]['output_over_source_db'], -20, places=2)
        selected = MODULE.assess(self.source, self.output, self.lines, min_relative_db=-12)
        self.assertEqual(selected['status'], 'FAIL')
        self.assertEqual([x['outside_selected_threshold'] for x in selected['intervals']], [False, True])
        self.assertTrue(selected['frame_timing']['source']['continuous'])
        self.assertIn('not perceptual', selected['limitations'])

    def test_silent_output_and_silent_source_are_not_false_passes(self):
        write_wav(self.output, np.zeros(4 * RATE))
        selected = MODULE.assess(self.source, self.output, self.lines, min_relative_db=-12)
        self.assertEqual(selected['status'], 'FAIL')
        self.assertEqual(selected['intervals'][1]['status'], 'silent_output')
        self.assertIsNone(selected['intervals'][1]['output_over_source_db'])
        self.assertEqual(MODULE.assess(self.source, self.output, self.lines, max_relative_db=3)['status'], 'INCOMPLETE')
        missing_source = MODULE.assess(self.source, self.output, [{'start': 3.1, 'end': 3.8}], min_relative_db=-12)
        self.assertEqual(missing_source['status'], 'INCOMPLETE')
        self.assertEqual(missing_source['intervals'][0]['status'], 'low_source_energy')

    def test_discontinuous_pts_and_unbounded_inputs_are_rejected(self):
        with patch.object(MODULE, 'inspect_frame_timing', return_value={'continuous': False, 'status': 'measured'}):
            with self.assertRaisesRegex(ValueError, 'PTS'):
                MODULE.assess(self.source, self.output, self.lines)
        for lines in ([], [{'start': True, 'end': 2}], [{'start': 0, 'end': 3601}],
                      [{'start': 2, 'end': 2}], [{'start': 0, 'end': float('nan')}]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                MODULE.assess(self.source, self.output, lines)
        with self.assertRaisesRegex(ValueError, '20000000 samples'):
            MODULE.assess(self.source, self.output, [{'start': 0, 'end': 1000}])

    def test_original_aba_during_decode_cannot_change_snapshot_result(self):
        original = self.source.read_bytes(); weak = self.output.read_bytes()
        actual_decode = MODULE.decode_first_channel
        def mutate_original(path, rate, cap):
            if path.name.startswith('source'):
                self.source.write_bytes(weak)
                try:
                    return actual_decode(path, rate, cap)
                finally:
                    self.source.write_bytes(original)
            return actual_decode(path, rate, cap)
        with patch.object(MODULE, 'decode_first_channel', side_effect=mutate_original):
            result = MODULE.assess(self.source, self.output, self.lines, min_relative_db=-12)
        self.assertEqual(result['status'], 'FAIL')
        self.assertAlmostEqual(result['intervals'][1]['output_over_source_db'], -20, places=2)
        self.assertEqual(self.source.read_bytes(), original)

    def test_frame_report_is_bounded_before_json_parsing(self):
        def huge(_command, *, stdout, **_kwargs):
            stdout.write(b'x' * (16 * 1024**2 + 1))
        with patch.object(MODULE.subprocess, 'run', side_effect=huge):
            with self.assertRaisesRegex(ValueError, '16 MiB'):
                MODULE.inspect_frame_timing(self.source)

    def test_fifo_without_writer_is_rejected_without_blocking(self):
        fifo = self.root / 'fifo.wav'; os.mkfifo(fifo)
        code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "import screen_speech_levels as s; "
                "s.snapshot(Path(sys.argv[2]), Path(sys.argv[3]))")
        result = subprocess.run([sys.executable, '-c', code, str(SCRIPTS), str(fifo),
                                 str(self.root / 'copy.wav')], capture_output=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'nonregular', result.stderr)

    def test_cli_writes_fail_report_and_never_overwrites(self):
        intervals = self.root / 'intervals.json'; intervals.write_text(json.dumps(self.lines))
        report = self.root / 'report.json'
        command = [sys.executable, str(SCRIPTS / 'screen_speech_levels.py'), str(self.source), str(self.output),
                   '--intervals', str(intervals), '--report', str(report), '--min-relative-db=-12']
        first = subprocess.run(command, capture_output=True)
        self.assertEqual(first.returncode, 1, first.stderr.decode())
        saved = report.read_bytes()
        self.assertEqual(json.loads(saved)['status'], 'FAIL')
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
        self.assertEqual(report.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
