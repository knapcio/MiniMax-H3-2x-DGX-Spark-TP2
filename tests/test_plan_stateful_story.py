import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('plan_stateful_story', SCRIPTS / 'plan_stateful_story.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

INITIAL = {'letter': 'in Wokulski’s hand', 'setting': 'the shop'}
EVENT = {'start': 9, 'end': 13, 'action': 'He places the letter on the counter.',
         'before': {'letter': 'in Wokulski’s hand'}, 'after': {'letter': 'on the counter'}}


class StatefulStory(unittest.TestCase):
    def test_change_is_not_assumed_before_a_spanning_action_finishes(self):
        result = module.plan(30, 'A continuous scene in the shop.', INITIAL, [EVENT], [720])
        segment = result['segments'][0]; windows = segment['windows']; prompts = segment['extra_params']['continuation_prompts']
        self.assertEqual([w['new_global_frames'][0] for w in windows], [0, 277, 532])
        self.assertEqual(windows[0]['assumed_stable_state_at_new_start']['letter'], 'in Wokulski’s hand')
        self.assertNotIn('letter', windows[1]['assumed_stable_state_at_new_start'])
        self.assertEqual(windows[1]['last_completed_planned_state']['letter'], 'in Wokulski’s hand')
        self.assertEqual(windows[1]['in_progress_state_keys'], ['letter'])
        self.assertEqual(windows[1]['ongoing_beat_indices'], [1])
        self.assertEqual(windows[2]['assumed_stable_state_at_new_start']['letter'], 'on the counter')
        self.assertTrue(all(w['state_verified_from_video'] is False for w in windows))
        self.assertIn('Next: He places', prompts[0])
        self.assertIn('Continue the action already underway: He places', prompts[1])
        self.assertNotIn('letter: in Wokulski’s hand', prompts[1])
        self.assertIn('without resetting these changing elements: letter', prompts[1])
        self.assertIn('When this action is complete, the intended state is:\nletter: on the counter', prompts[0])
        self.assertIn('When this action is complete, the intended state is:\nletter: on the counter', prompts[1])
        self.assertNotIn('He places', prompts[2])
        self.assertIn('letter: on the counter', prompts[2])
        self.assertEqual((segment['extra_params']['continuation_window_frames'],
                          segment['extra_params']['continuation_overlap_frames']), (277, 22))

    def test_target_is_communicated_without_assuming_it_already_happened(self):
        event = dict(EVENT, action='He moves the letter.', after={'letter': 'inside his coat pocket'})
        result = module.plan(30, 'Same shop.', INITIAL, [event], [720])
        segment = result['segments'][0]; prompt = segment['extra_params']['continuation_prompts'][1]
        self.assertNotIn('letter', segment['windows'][1]['assumed_stable_state_at_new_start'])
        self.assertIn('When this action is complete, the intended state is:\nletter: inside his coat pocket', prompt)
        self.assertEqual(segment['windows'][2]['assumed_stable_state_at_new_start']['letter'], 'inside his coat pocket')

    def test_state_and_ongoing_beat_cross_a_request_boundary(self):
        event = dict(EVENT, start=14, end=16)
        result = module.plan(30, 'Same shop.', INITIAL, [event], [360, 360])
        first, second = result['segments']
        self.assertEqual(second['first_frame_reference_plan']['frame_index_zero_based'], 359)
        self.assertEqual(second['windows'][0]['new_global_frames'][0], 360)
        self.assertNotIn('letter', second['windows'][0]['assumed_stable_state_at_new_start'])
        self.assertEqual(second['windows'][0]['in_progress_state_keys'], ['letter'])
        self.assertIn('Continue the action already underway', second['extra_params']['continuation_prompts'][0])
        self.assertEqual(second['windows'][1]['assumed_stable_state_at_new_start']['letter'], 'on the counter')
        self.assertEqual(second['global_frames'], [360, 720])

    def test_conflicting_or_unknown_state_transitions_are_rejected(self):
        conflicting = {'start': 14, 'end': 15, 'action': 'She reads it.',
                       'before': {'letter': 'in Wokulski’s hand'}}
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            module.plan(30, 'Same shop.', INITIAL, [EVENT, conflicting], [720])
        unknown = dict(EVENT, after={'hat': 'on the counter'})
        with self.assertRaisesRegex(ValueError, 'initial_state'):
            module.plan(30, 'Same shop.', INITIAL, [unknown], [720])
        with self.assertRaises(ValueError):
            module.plan(30, 'Same shop.', INITIAL, [dict(EVENT, start=20, end=10)], [720])
        with self.assertRaises(ValueError):
            module.plan(30, 'Same shop.', INITIAL, [EVENT], [719])

    def test_300_second_plan_preserves_request_and_window_geometry(self):
        result = module.plan(300, 'Same shop.', INITIAL, [EVENT], [720, 1440, 1440, 1440, 1440, 720])
        self.assertEqual(sum(s['expected_output_frames'] for s in result['segments']), 7200)
        self.assertEqual(len(result['segments']), 6)
        for segment in result['segments']:
            self.assertEqual(len(segment['windows']), len(segment['extra_params']['continuation_prompts']))
            self.assertTrue(all(not w['state_verified_from_video'] for w in segment['windows']))
        self.assertEqual(result['segments'][-1]['windows'][0]['assumed_stable_state_at_new_start']['letter'], 'on the counter')
        self.assertIn('not verified', result['state_status'])

    def test_cli_requires_new_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scene = root / 'scene.txt'; story = root / 'story.json'; output = root / 'out.json'
            scene.write_text('Same shop.'); story.write_text(json.dumps({'initial_state': INITIAL, 'beats': [EVENT]}))
            cmd = [sys.executable, str(SCRIPTS / 'plan_stateful_story.py'), '--seconds', '30',
                   '--continuity-file', str(scene), '--story', str(story), '--segment-frames', '720', '--output', str(output)]
            first = subprocess.run(cmd, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            prior = output.read_bytes()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), prior)


if __name__ == '__main__':
    unittest.main()
