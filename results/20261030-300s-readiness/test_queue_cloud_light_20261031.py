"""Verify the isolated cloud-light control and its fail-closed preconditions."""
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_human_pace_direct_bench_20261031 as previous


SCRIPT = Path(__file__).with_name('queue-cloud-light-20261031.sh')
NAME = 'quality30-cloud-light-960x544-20261031'
TAG = 'd30_960x544_s20_seed1101_fl2va'
OLD = 'the light on the landscape stays soft and nearly constant, without strong moving cloud shadows'
NEW = 'the ground remains evenly lit by soft daylight, with only faint diffuse shadows moving slowly in step with the clouds'


class CloudLightQueueTests(unittest.TestCase):
    def setUp(self):
        self.base = previous.DirectPaceQueueTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        result = self.base.run_queue()
        if result.returncode: raise AssertionError(result.stdout + result.stderr)
        self.results, self.f = self.base.results, self.base.f
        with (self.results / 'aspect-events-20261031.log').open('a') as out:
            out.write('H3-RETRY-FOLLOWUPS-DONE quality-review-required test\n')
        (self.results / 'followups.lock').touch()
        self.f.command('journalctl', '''
if [ "$1" = -k ]; then
  [ "${H3_TEST_LOCAL_JOURNAL_FAIL:-0}" != 1 ] || exit 1
  [ "${H3_TEST_LOCAL_XID:-0}" != 1 ] || echo 'kernel NVRM: Xid 13'
else echo 'phase diffuse 10 s'; fi
exit 0''')
        self.f.command('ssh', '''
case "$*" in
  *"journalctl -k"*)
    [ "${H3_TEST_REMOTE_JOURNAL_FAIL:-0}" != 1 ] || exit 1
    [ "${H3_TEST_REMOTE_XID:-0}" != 1 ] || echo 'kernel NVRM: Xid 13'
    exit 0;;
  *"show h3-tp2-agent.service"*)
    if [ "${H3_TEST_AGENT_CHANGES:-0}" = 1 ] && [ -e "$H3_TEST_CASES" ]; then echo 124; else echo 123; fi; exit 0;;
  *h3-tp2-agent.service*) [ "${H3_TEST_WORKER:-ok}" = ok ]; exit $?;;
  *sha256sum*) if [ "${H3_TEST_REMOTE_HASH_BAD:-0}" = 1 ]; then echo wrong; else sha256sum "$H3_TRACE_TXN"; fi; exit 0;;
  *verify-new*) [ "${H3_TEST_REMOTE_BAD:-0}" != 1 ]; exit $?;;
  *nvidia-smi*) printf '0\\n'; exit 0;;
  *MemAvailable*) printf '30000000\\n'; exit 0;;
esac
exit 1''')
        self.env = dict(self.base.env)
        self.env.update({key.replace('H3_PACE_', 'H3_CLOUD_'): value for key, value in self.base.env.items()
                         if key.startswith('H3_PACE_')})

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.env, **changes), text=True,
                              capture_output=True, timeout=35)

    def test_only_cloud_light_clause_changes(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = json.loads((self.results / 'quality30-camera-locked-960x544-20261031' /
                               (TAG + '.mp4.timing.json')).read_text())
        candidate = json.loads((self.results / NAME / (TAG + '.mp4.timing.json')).read_text())
        self.assertEqual(candidate['prompt'], baseline['prompt'].replace(OLD, NEW))
        old_extra, new_extra = dict(baseline['extra_params']), dict(candidate['extra_params'])
        old_prompts, new_prompts = old_extra.pop('continuation_prompts'), new_extra.pop('continuation_prompts')
        self.assertEqual(old_extra, new_extra)
        self.assertEqual(new_prompts, [p.replace(OLD, NEW) for p in old_prompts])
        self.assertIn('H3-ASPECT-CLOUDLIGHT-DONE', (self.results / 'aspect-events-20261031.log').read_text())

    def test_missing_or_failed_predecessor_does_not_start(self):
        for kind in ('done', 'pace', 'hold', 'retry_hold'):
            with self.subTest(kind=kind):
                other = CloudLightQueueTests(); other.setUp()
                try:
                    events = other.results / 'aspect-events-20261031.log'
                    if kind == 'done': events.write_text(events.read_text().replace('H3-RETRY-FOLLOWUPS-DONE', 'REMOVED'))
                    if kind == 'pace': (other.results / previous.NAME / 'STATUS').write_text('failed\n')
                    if kind in ('hold', 'retry_hold'):
                        prefix = 'H3-ASPECT-HOLD window175' if kind == 'hold' else 'H3-RETRY-FOLLOWUPS-HOLD phase=x'
                        with events.open('a') as out: out.write(prefix + ' test\n')
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue().returncode, 0)
                    self.assertEqual(other.f.cases(), before)
                finally: other.doCleanups()

    def test_xid_or_failed_journal_query_does_not_start(self):
        for variable in ('H3_TEST_LOCAL_XID', 'H3_TEST_REMOTE_XID',
                         'H3_TEST_LOCAL_JOURNAL_FAIL', 'H3_TEST_REMOTE_JOURNAL_FAIL'):
            with self.subTest(variable=variable):
                other = CloudLightQueueTests(); other.setUp()
                try:
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue(**{variable: '1'}).returncode, 0)
                    self.assertEqual(other.f.cases(), before)
                finally: other.doCleanups()

    def test_busy_or_failed_render_holds(self):
        for changes, attempted in (({'H3_TEST_GPU': '40'}, False), ({'H3_TEST_JOBS': '1'}, False),
                                   ({'H3_TEST_AGENT_CHANGES': '1'}, False), ({'H3_TEST_FAIL_CASE': NAME}, True)):
            with self.subTest(changes=changes):
                other = CloudLightQueueTests(); other.setUp()
                try:
                    before = other.f.cases()
                    self.assertNotEqual(other.run_queue(**changes).returncode, 0)
                    self.assertEqual(other.f.cases(), before + ([NAME] if attempted else []))
                    self.assertIn('H3-ASPECT-HOLD cloud-light-20261031',
                                  (other.results / 'aspect-events-20261031.log').read_text())
                finally: other.doCleanups()


if __name__ == '__main__': unittest.main()
