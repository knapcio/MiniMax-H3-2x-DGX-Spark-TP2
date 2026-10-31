"""The direct-bench gait test is independent of Mac client smoke but keeps render guards."""
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_window175_20261031 as previous

SCRIPT = Path(__file__).with_name('queue-human-pace-direct-bench-20261031.sh')
NAME = 'quality30-human-pace-960x544-20261031'
TAG = 'd30_960x544_s20_seed1101_fl2va'
OLD = 'the tiny figures on the road move very slowly, taking small natural steps without sudden movements'
NEW = ('the tiny figures on the road walk naturally at a relaxed, unhurried pace, taking about one small step '
       'each second, without running, speeding up, time-lapse or jerky movements')


class DirectPaceQueueTests(unittest.TestCase):
    def setUp(self):
        self.base = previous.Window175QueueTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        result = self.base.run_queue()
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        self.results, self.f = self.base.results, self.base.f
        self.env = dict(self.base.env)
        self.env.update({key.replace('H3_W175_', 'H3_PACE_'): value for key, value in self.base.env.items()
                         if key.startswith('H3_W175_')})

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.env, **changes), text=True,
                              capture_output=True, timeout=35)

    def test_compares_only_gait_without_client_marker(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = json.loads((self.results / previous.previous.LOCKED / (TAG + '.mp4.timing.json')).read_text())
        candidate = json.loads((self.results / NAME / (TAG + '.mp4.timing.json')).read_text())
        self.assertEqual(candidate['prompt'], baseline['prompt'].replace(OLD, NEW))
        old_extra, new_extra = dict(baseline['extra_params']), dict(candidate['extra_params'])
        old_prompts, new_prompts = old_extra.pop('continuation_prompts'), new_extra.pop('continuation_prompts')
        self.assertEqual(old_extra, new_extra)
        self.assertEqual(new_prompts, [x.replace(OLD, NEW) for x in old_prompts])
        self.assertIn('H3-ASPECT-HUMANPACE-DONE', (self.results / 'aspect-events-20261031.log').read_text())
        self.assertFalse((self.results / 'client-smoke-proof-20261031.json').exists())

    def test_missing_baseline_or_window_hold_never_starts(self):
        for kind in ('baseline', 'window_hold', 'window_marker'):
            with self.subTest(kind=kind):
                other = DirectPaceQueueTests(); other.setUp()
                try:
                    if kind == 'baseline': (other.results / previous.previous.LOCKED / 'STATUS').write_text('failed\n')
                    if kind == 'window_hold':
                        with (other.results / 'aspect-events-20261031.log').open('a') as out:
                            out.write('H3-ASPECT-HOLD window175-20261031 phase=rendering rc=1\n')
                    if kind == 'window_marker':
                        p = other.results / 'aspect-events-20261031.log'
                        p.write_text(p.read_text().replace('H3-ASPECT-WINDOW175-DONE', 'REMOVED'))
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue().returncode, 0)
                    self.assertEqual(other.f.cases(), before)
                finally:
                    other.doCleanups()

    def test_busy_changed_service_or_failed_render_holds(self):
        for changes, attempted in (({'H3_TEST_GPU': '40'}, False), ({'H3_TEST_JOBS': '1'}, False),
                                   ({'H3_TEST_AGENT_CHANGES': '1'}, False), ({'H3_TEST_GUARD': 'bad'}, False),
                                   ({'H3_TEST_FAIL_CASE': NAME}, True)):
            with self.subTest(changes=changes):
                other = DirectPaceQueueTests(); other.setUp()
                try:
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue(**changes).returncode, 0)
                    self.assertEqual(other.f.cases(), before + ([NAME] if attempted else []))
                    self.assertIn('H3-ASPECT-HOLD human-pace-20261031',
                                  (other.results / 'aspect-events-20261031.log').read_text())
                finally:
                    other.doCleanups()


if __name__ == '__main__':
    unittest.main()
