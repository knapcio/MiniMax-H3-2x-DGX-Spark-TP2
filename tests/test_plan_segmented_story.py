import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import plan_segmented_story as module


class SegmentedStory(unittest.TestCase):
    def test_300_seconds_assigns_global_beats_across_all_six_requests(self):
        beats = [{'start': 0, 'end': 60, 'action': 'They read the letter.'},
                 {'start': 60, 'end': 180, 'action': 'They discuss an invitation.'},
                 {'start': 180, 'end': 300, 'action': 'They make a decision.'}]
        result = module.plan(300, 'Same characters and story.\r\n', beats,
                             [720, 1440, 1440, 1440, 1440, 720])
        segments = result['segments']
        self.assertEqual(len(segments), 6)
        self.assertEqual([s['global_frames'] for s in segments],
                         [[0, 720], [720, 2160], [2160, 3600], [3600, 5040],
                          [5040, 6480], [6480, 7200]])
        seen = set()
        for i, segment in enumerate(segments):
            lo, hi = segment['global_frames']
            self.assertEqual(segment['expected_output_frames'], hi - lo)
            self.assertEqual(segment['windows'][0]['new_global_frames'][0], lo)
            self.assertEqual(segment['windows'][-1]['new_global_frames'][1], hi)
            self.assertEqual(len(segment['windows']), len(segment['extra_params']['continuation_prompts']))
            for win, prompt in zip(segment['windows'], segment['extra_params']['continuation_prompts']):
                a, b = win['new_global_frames']
                expected = [j for j, beat in enumerate(result['beats'], 1)
                            if beat['first_frame'] < b and beat['end_frame'] > a]
                self.assertEqual(win['beat_indices'], expected)
                self.assertTrue(prompt.startswith('Same characters and story.\r\n'))
                for j, beat in enumerate(beats, 1):
                    self.assertEqual(beat['action'] in prompt, j in expected)
                seen.update(expected)
            if i:
                self.assertEqual(segment['first_frame_reference_plan'],
                                 {'kind': 'previous-segment-last-frame', 'segment': i,
                                  'source': 'verified-exact-duration-output',
                                  'expected_frames': [720, 1440, 1440, 1440, 1440][i - 1],
                                  'frame_index_zero_based': [719, 1439, 1439, 1439, 1439][i - 1]})
        self.assertEqual(seen, {1, 2, 3})
        self.assertIn('Continue the action already underway: They read',
                      segments[1]['extra_params']['continuation_prompts'][0])
        self.assertEqual(segments[0]['first_frame_reference_plan'], {'kind': 'initial-request-input'})

    def test_boundaries_do_not_lose_beats_and_reference_can_be_explicit(self):
        beats = [{'start': 0, 'end': 4, 'action': 'A'},
                 {'start': 4, 'end': 8, 'action': 'B'}]
        result = module.plan(8, 'Context.', beats, [96, 96], window=175, reference='explicit')
        self.assertEqual(result['segments'][0]['windows'][0]['beat_indices'], [1])
        self.assertEqual(result['segments'][1]['windows'][0]['beat_indices'], [2])
        self.assertEqual(result['segments'][1]['first_frame_reference_plan'], {'kind': 'user-supplied'})
        self.assertIn('Next: B', result['segments'][1]['extra_params']['continuation_prompts'][0])

    def test_invalid_lengths_input_and_geometry_fail_closed(self):
        beat = [{'start': 0, 'end': 8, 'action': 'A'}]
        for segments in ([], [96], [95, 97], [True, 191], [96.0, 96], [96] * 1001):
            with self.subTest(segments=segments[:3]), self.assertRaises(ValueError):
                module.plan(8, 'Context.', beat, segments)
        with self.assertRaises(ValueError):
            module.plan(8, 'Context.', beat, [96, 96], window=176)
        with self.assertRaises(ValueError):
            module.plan(8, 'Context.', beat, [96, 96], reference='bad')
        with self.assertRaises(ValueError):
            module.plan(8, 'Context.', [{'start': 0, 'end': 9, 'action': 'A'}], [96, 96])

    def test_cli_preserves_text_creates_private_plan_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            context, beats, output = (root / x for x in ('context.txt', 'beats.json', 'plan.json'))
            context.write_bytes('Izabela i Wokulski;\r\nWarszawa.\n'.encode())
            beats.write_text(json.dumps([{'start': 0, 'end': 8, 'action': 'Czytają list.'}]))
            command = [sys.executable, str(SCRIPTS / 'plan_segmented_story.py'), '--seconds', '8',
                       '--continuity-file', str(context), '--beats', str(beats), '--segment-frames', '96', '96',
                       '--output', str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            saved = json.loads(output.read_text())
            self.assertTrue(saved['segments'][0]['extra_params']['continuation_prompts'][0]
                            .startswith('Izabela i Wokulski;\r\nWarszawa.\n'))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(json.loads(output.read_text()), saved)


if __name__ == '__main__':
    unittest.main()
