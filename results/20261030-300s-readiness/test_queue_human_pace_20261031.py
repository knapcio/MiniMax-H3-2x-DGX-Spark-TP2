import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_window175_20261031 as previous


SCRIPT = Path(__file__).with_name('queue-human-pace-20261031.sh')
NAME = 'quality30-human-pace-960x544-20261031'
TAG = 'd30_960x544_s20_seed1101_fl2va'
CLIENT_SHA = 'b2c8500d83fb8cc22d9afe5f5b625fa50f4c2fcd9ce85b01d748df8d01d021aa'
OLD = 'the tiny figures on the road move very slowly, taking small natural steps without sudden movements'
NEW = ('the tiny figures on the road walk naturally at a relaxed, unhurried pace, taking about one small step '
       'each second, without running, speeding up, time-lapse or jerky movements')


class HumanPaceQueueTests(unittest.TestCase):
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
        image_sha = hashlib.sha256((self.results / 'koninck.png').read_bytes()).hexdigest()
        self.proof = self.results / 'client-smoke-proof-20261031.json'
        self.proof.write_text(json.dumps({'client_sha256': CLIENT_SHA, 'image_sha256': image_sha,
                                          'duration_requested': 5, 'exact_duration_requested': True,
                                          'canvas': [960, 544], 'output_sha256': 'a' * 64}))

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.env, **changes), text=True,
                              capture_output=True, timeout=35)

    def test_only_human_clause_changes_with_same_window_geometry(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = json.loads((self.results / previous.previous.LOCKED / (TAG + '.mp4.timing.json')).read_text())
        candidate = json.loads((self.results / NAME / (TAG + '.mp4.timing.json')).read_text())
        self.assertEqual(candidate['prompt'], baseline['prompt'].replace(OLD, NEW))
        old_extra, new_extra = dict(baseline['extra_params']), dict(candidate['extra_params'])
        old_prompts, new_prompts = old_extra.pop('continuation_prompts'), new_extra.pop('continuation_prompts')
        self.assertEqual(old_extra, new_extra)
        self.assertEqual(len(new_prompts), 3)
        self.assertEqual(new_prompts, [x.replace(OLD, NEW) for x in old_prompts])
        self.assertIn('H3-ASPECT-HUMANPACE-DONE', (self.results / 'aspect-events-20261031.log').read_text())

    def test_missing_or_changed_proof_never_starts_render(self):
        for kind in ('missing', 'changed'):
            with self.subTest(kind=kind):
                other = HumanPaceQueueTests()
                other.setUp()
                try:
                    if kind == 'missing':
                        other.proof.unlink()
                    else:
                        x = json.loads(other.proof.read_text()); x['image_sha256'] = '0' * 64
                        other.proof.write_text(json.dumps(x))
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue().returncode, 0)
                    self.assertEqual(other.f.cases(), before)
                finally:
                    other.doCleanups()

    def test_busy_or_changed_service_never_starts_render(self):
        for changes in ({'H3_TEST_GPU': '40'}, {'H3_TEST_JOBS': '1'}, {'H3_TEST_AGENT_CHANGES': '1'}):
            with self.subTest(changes=changes):
                other = HumanPaceQueueTests()
                other.setUp()
                try:
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue(**changes).returncode, 0)
                    self.assertEqual(other.f.cases(), before)
                finally:
                    other.doCleanups()

    def test_failed_render_holds_without_success_marker(self):
        before = self.f.cases()
        self.assertNotEqual(self.run_queue(H3_TEST_FAIL_CASE=NAME).returncode, 0)
        self.assertEqual(self.f.cases(), before + [NAME])
        events = (self.results / 'aspect-events-20261031.log').read_text()
        self.assertIn('H3-ASPECT-HOLD human-pace-20261031', events)
        self.assertNotIn('H3-ASPECT-HUMANPACE-DONE', events)

    def test_failed_client_smoke_never_starts_render(self):
        (self.results / 'client-smoke-failed-20261031').write_text('failed')
        before = self.f.cases()
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.f.cases(), before)


if __name__ == '__main__':
    unittest.main()
