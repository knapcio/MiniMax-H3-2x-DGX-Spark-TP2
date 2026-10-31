import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_long300 as fixture


SCRIPT = Path(__file__).with_name('queue-aspect-long300-20261031.sh')
SIXTEEN = 'quality16-960x544-20261031'
SIXTY = 'quality60-960x544-20261031'
LONG = 'long300-960x544-20261031'
LOG = 'aspect-events-20261031.log'


class ResumeLong300Tests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.Long300QueueTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root, self.results = self.f.root, self.f.results
        (self.results / 'quality16-960-stream').rename(self.results / SIXTEEN)
        (self.results / 'quality60-960-stream').rename(self.results / SIXTY)
        (self.results / SIXTY / 'luma-flicker.txt').write_text(
            'periodic luma drops: none >= 2 YAVG for 2 successive 17-frame periods (1450 frames checked)\n')
        for folder,duration in ((self.results/SIXTEEN,16),(self.results/SIXTY,60)):
            old=f'd{duration}_960x576_s20_seed1101_fl2va'
            new=f'd{duration}_960x544_s20_seed1101_fl2va'
            for path in list(folder.glob(old+'*')):
                path.rename(folder/path.name.replace(old,new))
            (folder/'STATUS').write_text(new+' ok\n')
            path=folder/(new+'.mp4.timing.json')
            data=json.loads(path.read_text()); data['size']='960x544'; path.write_text(json.dumps(data))
            (folder/'source.sha256').write_text(hashlib.sha256((self.results/'koninck.png').read_bytes()).hexdigest()+'\n')
        self.f.t.output_case('quality16-base', 'd16_1344x768_s20_seed1101_fl2va')
        (self.results / LOG).write_text(
            'H3-ASPECT-OK ' + SIXTEEN + ' test\n' +
            'H3-ASPECT-OK ' + SIXTY + ' test\n' +
            'H3-ASPECT-SKIP-300 timeout or speed margin insufficient test\n')
        bench = self.f.t.bench
        bench.write_text(bench.read_text() + '''
python3 - "$1" "$2" <<'PY'
import json, os, pathlib, sys
d,w,h,steps,seed = map(int,sys.argv[1].split(':'))
tag=f'd{d}_{w}x{h}_s{steps}_seed{seed}_fl2va'
x={'http':'200','curl_rc':0,'duration':d,'size':f'{w}x{h}','steps':steps,'seed':seed,
   'task':'fl2va','prompt':os.environ['PROMPT'],'image':sys.argv[2],
   'extra_params':json.loads(os.environ['EXTRA_JSON'])}
if os.environ.get('H3_TEST_BAD_300_TIMING') == '1': x['seed']=99
(pathlib.Path(os.environ['OUT'])/(tag+'.mp4.timing.json')).write_text(json.dumps(x))
PY
''')

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.f.t.env, **changes),
                              text=True, capture_output=True, timeout=30)

    def events(self):
        return (self.results / LOG).read_text()

    def test_verified_time_skip_runs_300_and_retains_16h_limits(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.f.t.cases(), [LONG])
        self.assertIn('H3_TP2_REQUEST_TIMEOUT_S=57600', self.f.envfile.read_text())
        self.assertEqual(self.f.t.state.read_text(), 'active')
        self.assertIn('H3-ASPECT-LONG300-DONE', self.events())
        self.assertIn('H3-ASPECT-OK ' + LONG, self.events())

    def test_quality_skip_or_prior_success_never_restarts(self):
        for prior_success in (False, True):
            with self.subTest(prior_success=prior_success):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    if prior_success:
                        other.f.t.output_case('quality300-960x544-20261031', 'd300_960x544_s20_seed1101_fl2va')
                        text = 'H3-ASPECT-QUALITY-DONE test\n'
                    else:
                        text = 'H3-ASPECT-SKIP-300 960 control quality or memory samples did not pass test\n'
                    (other.results / LOG).write_text(text)
                    result = other.run_queue()
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(other.f.t.cases(), [])
                    self.assertNotIn('stop h3-tp2-head.service', other.f.t.systemctl())
                    self.assertIn('H3-ASPECT-LONG300-SKIPPED', other.events())
                finally:
                    other.doCleanups()

    def test_bad_controls_never_restart(self):
        for kind in ('flicker', 'flicker60', 'memory', 'prompt', 'marker', 'baseline', 'projection'):
            with self.subTest(kind=kind):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    p = other.results / SIXTEEN
                    if kind == 'flicker': (p / 'luma-flicker.txt').write_text('detected\n')
                    if kind == 'flicker60': (other.results / SIXTY / 'luma-flicker.txt').write_text('detected\n')
                    if kind == 'memory': (p / 'd16_960x544_s20_seed1101_fl2va.mem-head.csv').write_text('1,100\n2,100\n')
                    if kind == 'prompt':
                        path = p / 'd16_960x544_s20_seed1101_fl2va.mp4.timing.json'
                        x=json.loads(path.read_text()); x['prompt']='changed'; path.write_text(json.dumps(x))
                    if kind == 'marker':
                        path = other.results / LOG; path.write_text(path.read_text().replace('H3-ASPECT-OK ' + SIXTY, 'MISSING ' + SIXTY))
                    if kind == 'baseline': (other.results / 'quality16-base/d16_1344x768_s20_seed1101_fl2va.mp4').write_bytes(b'different')
                    if kind == 'projection': (other.results / SIXTY / 'phases.txt').write_text('phase diffuse 2000.0 s\n')
                    result = other.run_queue()
                    if kind == 'projection': self.assertEqual(result.returncode, 0)
                    else: self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(other.f.t.cases(), [])
                    self.assertNotIn('stop h3-tp2-head.service', other.f.t.systemctl())
                finally:
                    other.doCleanups()

    def test_busy_gpu_or_async_job_never_restarts(self):
        for change in ({'H3_TEST_GPU':'40'}, {'H3_TEST_JOBS':'1'}, {'H3_TEST_WORKER':'bad'}):
            with self.subTest(change=change):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**change).returncode, 0)
                    self.assertNotIn('stop h3-tp2-head.service', other.f.t.systemctl())
                finally:
                    other.doCleanups()

    def test_failed_render_or_bad_result_restores_limits_when_idle(self):
        for change in ({'H3_TEST_FAIL_CASE':LONG}, {'H3_TEST_BAD_300_TIMING':'1'}):
            with self.subTest(change=change):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    result = other.run_queue(**change)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(other.f.envfile.read_text(), other.f.original_env)
                    self.assertEqual(other.f.t.state.read_text(), 'active')
                    self.assertNotIn('H3-ASPECT-OK ' + LONG, other.events())
                    self.assertIn('H3-ASPECT-HOLD long300-20261031', other.events())
                finally:
                    other.doCleanups()

    def test_failure_does_not_stop_an_uncertain_or_busy_head(self):
        for flag in ('H3_TEST_BUSY_ON_FAIL', 'H3_TEST_JOBS_ON_FAIL'):
            with self.subTest(flag=flag):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(H3_TEST_FAIL_CASE=LONG, **{flag:'1'}).returncode, 0)
                    self.assertEqual(other.f.t.state.read_text(), 'active')
                    self.assertEqual(other.f.t.systemctl().count('stop h3-tp2-head.service'), 1)
                    self.assertIn('H3_TP2_REQUEST_TIMEOUT_S=57600', other.f.envfile.read_text())
                finally:
                    other.doCleanups()

    def test_failed_or_activating_start_restores_only_when_safe(self):
        for flag in ('H3_TEST_FAIL_START_ONCE', 'H3_TEST_START_ACTIVATING'):
            with self.subTest(flag=flag):
                other = ResumeLong300Tests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**{flag:'1'}).returncode, 0)
                    if flag == 'H3_TEST_FAIL_START_ONCE':
                        self.assertEqual(other.f.envfile.read_text(), other.f.original_env)
                        self.assertEqual(other.f.t.state.read_text(), 'active')
                    else:
                        self.assertEqual(other.f.t.state.read_text(), 'activating')
                        self.assertIn('H3_TP2_REQUEST_TIMEOUT_S=57600', other.f.envfile.read_text())
                finally:
                    other.doCleanups()


if __name__ == '__main__':
    unittest.main()
