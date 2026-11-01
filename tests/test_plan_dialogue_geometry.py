"""Check that a dialogue plan's geometry survives the actual render client.

Curl is captured by the existing CPU harness; no GPU or server is contacted.
The same-count cases matter: the server's prompt-count validation alone would
accept these requests with incorrect dialogue timing before the fix.
"""
import json
import unittest

import test_plan_dialogue_windows as planner_tests
import test_render_duration as render_tests
from h3_tp2 import cont_prompts


class DialogueGeometryTests(unittest.TestCase):
    def setUp(self):
        self.render = render_tests.RenderDurationTests()
        self.render.setUp()
        self.addCleanup(self.render.doCleanups)

    def transmitted(self, seconds, start, end, *, window=277, overlap=22):
        plan = planner_tests.module.plan(
            seconds, 'The same steady shot.',
            [{'start': start, 'end': end, 'text': 'Dzień dobry; proszę wejść.'}],
            window=window, overlap=overlap, lock_source=True)
        # render.sh merges these fields into its task/duration request and
        # CONT_PROMPTS=auto must leave explicit per-window prompts untouched.
        result, sent, _ = self.render.run_render(
            {**plan['extra_params'], 'duration': seconds})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent['continuation_prompts'], plan['extra_params']['continuation_prompts'])
        self.assertEqual(sent['continuation_window_frames'], window)
        self.assertEqual(sent['continuation_overlap_frames'], overlap)
        self.assertEqual(sent['audio_mode'], 'lock_source')
        self.assertEqual(cont_prompts.request_window_count(sent), len(plan['windows']))
        actual = planner_tests.module.geometry.plan(
            planner_tests.module.geometry.aligned_frames(seconds),
            sent['continuation_window_frames'], sent['continuation_overlap_frames'])
        self.assertEqual(actual, [tuple(row['frames']) for row in plan['windows']])
        owner = plan['dialogue'][0]['window'] - 1
        first_new = 0 if owner == 0 else actual[owner - 1][1]
        self.assertLessEqual(first_new / 24, start)
        self.assertGreaterEqual(actual[owner][1] / 24, end)
        return plan, sent

    def test_default_geometry_preserves_prompt_content(self):
        plan, sent = self.transmitted(30, 12, 13)
        self.assertEqual(plan['dialogue'][0]['window'], 2)
        self.assertEqual(sent['continuation_prompts'], [
            'The same steady shot. No new dialogue is spoken in this part.',
            'The same steady shot. <d>[Polish] Dzień dobry; proszę wejść.</d>',
            'The same steady shot. No new dialogue is spoken in this part.'])

    def test_custom_window_with_same_count_preserves_line_owner(self):
        plan, sent = self.transmitted(30, 12, 13, window=345)
        self.assertEqual(len(plan['windows']), 3)
        self.assertEqual(plan['dialogue'][0]['window'], 1)
        # Without the transmitted window, count validation would still pass,
        # but window 1 would finish at 277/24, before this line begins.
        default = dict(sent)
        default.pop('continuation_window_frames')
        self.assertEqual(cont_prompts.request_window_count(default), 3)
        self.assertLess(277 / 24, 12)

    def test_custom_overlap_with_same_count_preserves_line_owner(self):
        plan, _ = self.transmitted(30, 21.7, 22.0, overlap=39)
        self.assertEqual(len(plan['windows']), 3)
        self.assertEqual(plan['dialogue'][0]['window'], 3)
        # Default overlap ends window 2 at 532/24; overlap 39 at 515/24.
        self.assertLess(515 / 24, 21.7)
        self.assertGreater(532 / 24, 22.0)

    def test_custom_window_avoids_prompt_count_rejection(self):
        plan, sent = self.transmitted(60, 12, 13, window=345)
        self.assertEqual(len(plan['windows']), 5)
        default = dict(sent)
        default.pop('continuation_window_frames')
        self.assertEqual(cont_prompts.request_window_count(default), 6)

    def test_json_round_trip_keeps_nondefault_integer_geometry(self):
        plan, sent = self.transmitted(300, 1, 2, window=175, overlap=39)
        decoded = json.loads(json.dumps(plan['extra_params']))
        self.assertIs(type(decoded['continuation_window_frames']), int)
        self.assertIs(type(decoded['continuation_overlap_frames']), int)
        self.assertEqual(decoded['continuation_prompts'], sent['continuation_prompts'])


if __name__ == '__main__':
    unittest.main()
