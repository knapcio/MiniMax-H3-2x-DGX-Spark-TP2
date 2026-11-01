import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('plan_dialogue_windows', SCRIPTS / 'plan_dialogue_windows.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DialogueWindowTests(unittest.TestCase):
    def test_two_lines_belong_to_different_windows_without_repetition(self):
        lines = [{'start': .3, 'end': 3.342, 'speaker': 'Wokulski', 'text': 'Warszawa się zmienia.'},
                 {'start': 12, 'end': 14.786, 'speaker': 'Wokulski', 'text': 'Potrzeba cierpliwości.'}]
        result = module.plan(16, 'The same steady shot.', lines, lock_source=True)
        prompts = result['extra_params']['continuation_prompts']
        self.assertEqual(len(prompts), 2)
        self.assertIn('Warszawa się zmienia.', prompts[0])
        self.assertNotIn('Warszawa się zmienia.', prompts[1])
        self.assertNotIn('Potrzeba cierpliwości.', prompts[0])
        self.assertIn('Potrzeba cierpliwości.', prompts[1])
        self.assertEqual([x['window'] for x in result['dialogue']], [1, 2])
        self.assertEqual(result['extra_params']['audio_mode'], 'lock_source')

    def test_lengths_from_10_through_300_seconds_have_the_expected_plan(self):
        for seconds, count in ((10, 1), (15, 2), (30, 3), (60, 6), (300, 29)):
            with self.subTest(seconds=seconds):
                result = module.plan(seconds, 'One scene.', [{'start': 1, 'end': 2, 'text': 'Jedno zdanie.'}])
                prompts = result['extra_params']['continuation_prompts']
                self.assertEqual(len(prompts), count)
                self.assertEqual(sum('Jedno zdanie.' in p for p in prompts), 1)
                self.assertEqual(result['windows'][-1]['new_frames'][-1], round(seconds * 24))

    def test_crossing_a_new_frame_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'crosses'):
            module.plan(30, 'A scene.', [{'start': 11.4, 'end': 11.7, 'text': 'Linia.'}])
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            module.plan(10, 'A scene.', [{'start': 1, 'end': 3, 'text': 'A'},
                                        {'start': 2, 'end': 4, 'text': 'B'}])

    def test_invalid_text_and_times_are_refused(self):
        for line in ([{'start': -1, 'end': 2, 'text': 'A'}],
                     [{'start': 1, 'end': 11, 'text': 'A'}],
                     [{'start': 1, 'end': 2, 'text': '</d> extra'}],
                     [{'start': 1, 'end': 2, 'text': 'A\u0000B'}],
                     [{'start': 1, 'end': 2, 'text': 'A', 'unknown': 'x'}]):
            with self.subTest(line=line), self.assertRaises(ValueError):
                module.plan(10, 'A scene.', line)

    def test_cli_writes_the_explicit_extra_params(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'lines.json'; report = root / 'plan.json'; extra = root / 'extra.json'
            source.write_text(json.dumps([{'start': 1, 'end': 2, 'text': 'Dzień dobry.'}]))
            subprocess.run([sys.executable, str(SCRIPTS / 'plan_dialogue_windows.py'), '--seconds', '10',
                            '--scene', 'A steady scene.', '--dialogue', str(source), '--report', str(report),
                            '--extra-output', str(extra)], check=True)
            self.assertEqual(json.loads(extra.read_text()), json.loads(report.read_text())['extra_params'])
            result = subprocess.run([sys.executable, str(SCRIPTS / 'plan_dialogue_windows.py'), '--seconds', '10',
                                     '--scene', 'A steady scene.', '--dialogue', str(source), '--report', str(report),
                                     '--extra-output', str(extra)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
