import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('plan_story_windows', SCRIPTS / 'plan_story_windows.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StoryWindows(unittest.TestCase):
    def test_connected_300_second_beats_are_assigned_to_new_frames(self):
        beats = [{'start': 0, 'end': 60, 'action': 'Wokulski reads the letter in the shop.'},
                 {'start': 60, 'end': 180, 'action': 'He and Izabela discuss an invitation.'},
                 {'start': 180, 'end': 300, 'action': 'They decide to meet the writer.'}]
        result = MODULE.plan(300, 'The same people, places and story.\r\n', beats)
        self.assertEqual(len(result['windows']), 29)
        self.assertEqual(result['windows'][0]['new_frames'][0], 0)
        self.assertEqual(result['windows'][-1]['new_frames'][1], 7200)
        for i, report in enumerate(result['windows']):
            lo, hi = report['new_frames']
            expected = [j + 1 for j, b in enumerate(result['beats'])
                        if b['first_frame'] < hi and b['end_frame'] > lo]
            self.assertEqual(report['beat_indices'], expected)
            prompt = result['extra_params']['continuation_prompts'][i]
            self.assertTrue(prompt.startswith('The same people, places and story.\r\n'))
            for j, beat in enumerate(beats, 1):
                self.assertEqual(beat['action'] in prompt, j in expected)
        self.assertEqual((result['extra_params']['continuation_window_frames'],
                          result['extra_params']['continuation_overlap_frames']), (277, 22))
        self.assertIn('not frame-accurate', result['limitations'])

    def test_shared_window_is_explicit_and_no_cut_policy_is_invented(self):
        beats = [{'start': 0, 'end': 5, 'action': 'A letter arrives.'},
                 {'start': 5, 'end': 16, 'action': 'They discuss it elsewhere.'}]
        result = MODULE.plan(16, 'They remain the same characters.', beats, window=175, overlap=22)
        self.assertTrue(result['windows'][0]['multiple_beats_in_window'])
        self.assertIn('Continue the action already underway', result['extra_params']['continuation_prompts'][1])
        self.assertNotIn('no cuts', result['extra_params']['continuation_prompts'][0].lower())

    def test_invalid_beats_and_geometry_are_rejected(self):
        for beats in ([], [{'start': 0, 'end': 3, 'action': 'A'}, {'start': 2, 'end': 4, 'action': 'B'}],
                      [{'start': True, 'end': 3, 'action': 'A'}],
                      [{'start': 0, 'end': 3, 'action': ''}],
                      [{'start': 0, 'end': 3, 'action': 'A', 'unknown': 1}]):
            with self.subTest(beats=beats), self.assertRaises(ValueError):
                MODULE.plan(10, 'Continuity.', beats)
        with self.assertRaises(ValueError):
            MODULE.plan(10, 'Continuity.', [{'start': 0, 'end': 2, 'action': 'A'}], window=176)

    def test_fractional_requested_end_cannot_silently_lose_a_beat(self):
        for beat in ([{'start': 10.005, 'end': 10.01, 'action': 'A late event.'}],
                     [{'start': 9.9, 'end': 10.01, 'action': 'A partly truncated event.'}],
                     [{'start': 10.00000000001, 'end': 10.00000000002, 'action': 'A tiny late event.'}]):
            with self.subTest(beat=beat), self.assertRaisesRegex(ValueError, 'rounded to 24 fps'):
                MODULE.plan(10.01, 'Continuity.', beat)
        result = MODULE.plan(10.01, 'Continuity.', [{'start': 9.9, 'end': 10.0, 'action': 'Inside.'}])
        self.assertIn('Inside.', result['extra_params']['continuation_prompts'][-1])
        near = MODULE.plan(10.01, 'Continuity.', [{'start': 9.9, 'end': 10.00000000002, 'action': 'Near.'}])
        self.assertEqual(near['beats'][0]['end_frame'], near['requested_frames'])
        self.assertIn('Near.', near['extra_params']['continuation_prompts'][-1])

    def test_cli_preserves_literal_context_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            context, beats, report, extra = (root / x for x in ('context.txt', 'beats.json', 'report.json', 'extra.json'))
            context.write_bytes('Izabela i Wokulski;\r\nWarszawa.\n'.encode())
            beats.write_text(json.dumps([{'start': 0, 'end': 10, 'action': 'Czytają list.'}]))
            command = [sys.executable, str(SCRIPTS / 'plan_story_windows.py'), '--seconds', '10',
                       '--continuity-file', str(context), '--beats', str(beats), '--report', str(report),
                       '--extra-output', str(extra)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            saved = json.loads(extra.read_text())
            self.assertEqual(saved, json.loads(report.read_text())['extra_params'])
            self.assertTrue(saved['continuation_prompts'][0].startswith('Izabela i Wokulski;\r\nWarszawa.\n'))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
