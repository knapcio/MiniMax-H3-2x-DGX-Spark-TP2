import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('slice_dialogue_intervals', SCRIPTS / 'slice_dialogue_intervals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import screen_speech_levels as screen


class DialogueSlices(unittest.TestCase):
    def test_boundaries_cover_each_line_and_merge_a_tiny_tail(self):
        result = module.plan([{'start': 1, 'end': 2.004, 'speaker': 'A'},
                              {'start': 4.1, 'end': 4.35, 'speaker': 'B'}])
        rows = result['dialogue']
        self.assertEqual([(x['start'], x['end']) for x in rows], [(1, 1.5), (1.5, 2.004), (4.1, 4.35)])
        self.assertEqual([x['source_interval_index'] for x in rows], [1, 1, 2])
        self.assertEqual([x['slice_index'] for x in rows], [1, 2, 1])

    def test_rejects_invalid_or_unbounded_input(self):
        for item in ([{'start': True, 'end': 2}], [{'start': 1, 'end': 1.001}],
                     [{'start': 3, 'end': 2}], [{'start': 0, 'end': float('nan')}],
                     [{'start': 0, 'end': 1, 'speaker': ['not', 'a', 'label']}], []):
            with self.subTest(item=item), self.assertRaises(ValueError):
                module.plan(item)
        for step in (0, .01, 61, float('inf'), True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                module.plan([{'start': 0, 'end': 1}], step)
        with self.assertRaisesRegex(ValueError, '1000'):
            module.plan([{'start': 0, 'end': 101}], .1)

    def test_real_audio_screen_exposes_a_fading_second_half(self):
        rate = 32000
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = np.arange(rate) / rate
            source = .2 * np.sin(2 * np.pi * 440 * t)
            output = source.copy(); output[rate // 2:] *= .1
            for name, samples in (('source.wav', source), ('output.wav', output)):
                with wave.open(str(root / name), 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
                    wav.writeframes(np.rint(samples * 32767).astype('<i2').tobytes())
            slices = module.plan([{'start': 0, 'end': 1, 'speaker': 'A'}])
            result = screen.assess(root / 'source.wav', root / 'output.wav', slices['dialogue'])
            self.assertEqual(result['status'], 'MEASURED')
            self.assertAlmostEqual(result['intervals'][0]['output_over_source_db'], 0, places=2)
            self.assertAlmostEqual(result['intervals'][1]['output_over_source_db'], -20, places=2)

    def test_cli_output_is_screen_input_and_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / 'dialogue.json'; out = root / 'slices.json'
            data.write_text(json.dumps({'dialogue': [{'start': .5, 'end': 2, 'speaker': 'A'}]}))
            cmd = [sys.executable, str(SCRIPTS / 'slice_dialogue_intervals.py'), '--intervals', str(data),
                   '--slice-seconds', '0.5', '--output', str(out)]
            first = subprocess.run(cmd, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            saved = out.read_bytes(); self.assertEqual(len(json.loads(saved)['dialogue']), 3)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
