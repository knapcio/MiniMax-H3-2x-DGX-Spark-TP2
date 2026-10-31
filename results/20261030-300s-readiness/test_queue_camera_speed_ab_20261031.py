import json
from pathlib import Path
import subprocess
import unittest

import test_queue_aspect_experimental300_20261031 as previous
import test_queue_vae_trace as fixture


SCRIPT = Path(__file__).with_name('queue-camera-speed-ab-20261031.sh')
DEFAULT = 'quality30-camera-default-960x544-20261031'
LOCKED = 'quality30-camera-locked-960x544-20261031'
FAST = 'quality30-camera-locked-s10-960x544-20261031'


class CameraSpeedQueueTests(unittest.TestCase):
    def setUp(self):
        self.base=previous.Experimental300Tests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.f=self.base.f; self.root=self.base.root; self.results=self.base.results
        with (self.results/'aspect-events-20261031.log').open('a') as out:
            out.write('H3-ASPECT-EXPERIMENT300-SKIPPED test\n')
        self.f.command('ssh', '''
case "$*" in
  *"show h3-tp2-agent.service"*)
    if [ "${H3_TEST_AGENT_CHANGES:-0}" = 1 ] && [ -e "$H3_TEST_CASES" ]; then echo 124; else echo 123; fi; exit 0;;
  *h3-tp2-agent.service*) [ "${H3_TEST_WORKER:-ok}" = ok ]; exit $?;;
  *sha256sum*) if [ "${H3_TEST_REMOTE_HASH_BAD:-0}" = 1 ]; then echo wrong; else sha256sum "$H3_TRACE_TXN"; fi; exit 0;;
  *verify-new*) [ "${H3_TEST_REMOTE_BAD:-0}" != 1 ]; exit $?;;
  *nvidia-smi*) printf '0\\n'; exit 0;;
  *MemAvailable*) printf '30000000\\n'; exit 0;;
esac
exit 1''')
        self.env=dict(self.base.env,H3_AB_RESULTS=str(self.results),H3_AB_LIVE=str(self.root/'live'),
                      H3_AB_STAGE=str(self.root/'stage'),H3_AB_BACKUP=str(self.root/'backup'),
                      H3_AB_TXN=str(fixture.TXN),H3_AB_BENCH=str(self.f.bench),
                      H3_AB_METRICS=str(self.f.metrics),H3_AB_PYTHON='python3',
                      H3_AB_FLICKER=str(fixture.FLICKER),H3_AB_PROC_ROOT=str(self.root/'proc'),
                      H3_AB_MEMINFO=str(self.root/'meminfo'),H3_AB_MIN_DISK_KB='0',
                      H3_AB_WAIT_ATTEMPTS='1',H3_AB_IDLE_ATTEMPTS='3',H3_AB_WAIT_S='0',H3_AB_IDLE_S='0')

    def run_queue(self,**changes):
        return subprocess.run(['bash',str(SCRIPT)],env=dict(self.env,**changes),text=True,
                              capture_output=True,timeout=35)

    def events(self): return (self.results/'aspect-events-20261031.log').read_text()

    def timing(self,name,steps):
        return json.loads((self.results/name/f'd30_960x544_s{steps}_seed1101_fl2va.mp4.timing.json').read_text())

    def test_matched_prompt_pair_and_step_pair(self):
        result=self.run_queue()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(self.f.cases(),[DEFAULT,LOCKED,FAST])
        baseline=self.timing(DEFAULT,20); fixed=self.timing(LOCKED,20); fast=self.timing(FAST,10)
        self.assertNotIn('continuation_prompts',baseline['extra_params'])
        prompts=fixed['extra_params']['continuation_prompts']
        self.assertEqual(len(prompts),3)
        self.assertEqual(prompts[1],prompts[2])
        self.assertIn('camera remains locked on a tripod',prompts[1])
        self.assertNotIn('camera keeps moving',prompts[1])
        self.assertEqual(prompts[0],baseline['prompt'].rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.')
        self.assertEqual(fixed['extra_params'],fast['extra_params'])
        self.assertEqual(fixed['prompt'],fast['prompt'])
        self.assertIn('H3-ASPECT-CAMERAAB-DONE',self.events())

    def test_missing_predecessor_or_hold_never_starts(self):
        for kind in ('short','hold','unterminated'):
            with self.subTest(kind=kind):
                other=CameraSpeedQueueTests(); other.setUp()
                try:
                    if kind=='short': (other.results/'quality15-direct-960x544-20261031/STATUS').write_text('failed\n')
                    if kind=='hold':
                        with (other.results/'aspect-events-20261031.log').open('a') as out:
                            out.write('H3-ASPECT-HOLD experiment300-20261031 rc=1 test\n')
                    if kind=='unterminated':
                        p=other.results/'aspect-events-20261031.log'; p.write_text(p.read_text().replace('H3-ASPECT-EXPERIMENT300-SKIPPED test\n',''))
                    self.assertNotEqual(other.run_queue().returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_input_output_and_prompt_state_guards(self):
        for kind in ('source','output','trace','custom','disk'):
            with self.subTest(kind=kind):
                other=CameraSpeedQueueTests(); other.setUp()
                try:
                    changes={}
                    if kind=='source': (other.results/'koninck.png').write_bytes(b'changed')
                    if kind=='output': (other.results/DEFAULT).mkdir()
                    if kind=='trace':
                        p=other.root/'proc/123/environ'; p.write_bytes(p.read_bytes().replace(b'TRACE=0',b'TRACE=1'))
                    if kind=='custom':
                        p=other.root/'proc/123/environ'; p.write_bytes(p.read_bytes()+b'H3_TP2_CONT_PROMPTS_REST=custom\0')
                    if kind=='disk': changes['H3_AB_MIN_DISK_KB']='999999999999'
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_busy_or_changed_deployment_never_starts(self):
        for changes in ({'H3_TEST_GPU':'40'},{'H3_TEST_JOBS':'1'},{'H3_TEST_WORKER':'bad'},
                        {'H3_TEST_REMOTE_BAD':'1'},{'H3_TEST_GUARD':'bad'}):
            with self.subTest(changes=changes):
                other=CameraSpeedQueueTests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_failure_or_agent_restart_stops_following_cases(self):
        for changes,expected in (({'H3_TEST_FAIL_CASE':DEFAULT},[DEFAULT]),
                                 ({'H3_TEST_FAIL_CASE':LOCKED},[DEFAULT,LOCKED]),
                                 ({'H3_TEST_BAD_TIMING':'1'},[DEFAULT]),
                                 ({'H3_TEST_AGENT_CHANGES':'1'},[DEFAULT])):
            with self.subTest(changes=changes):
                other=CameraSpeedQueueTests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),expected)
                    self.assertIn('H3-ASPECT-HOLD camera-speed-ab-20261031',other.events())
                    self.assertNotIn('H3-ASPECT-CAMERAAB-DONE',other.events())
                finally: other.doCleanups()


if __name__ == '__main__': unittest.main()
