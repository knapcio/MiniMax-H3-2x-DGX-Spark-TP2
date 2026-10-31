import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_vae_trace as fixture


SCRIPT = Path(__file__).with_name('queue-aspect-experimental300-20261031.sh')
NAME = 'experiment300-960x544-20261031'
PROMPT = ('The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, '
          'keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without '
          'strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees '
          'tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, '
          'taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at '
          'all; one continuous take; soft natural ambience.')


class Experimental300Tests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.VaeTraceQueueTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root, self.results = self.f.root, self.f.results
        for duration in (10, 15, 16, 60):
            name = (f'quality{duration}-direct-960x544-20261031' if duration in (10, 15)
                    else f'quality{duration}-960x544-20261031')
            tag = f'd{duration}_960x544_s20_seed1101_fl2va'
            self.f.output_case(name, tag)
            if duration in (16, 60):
                folder = self.results / name
                (folder / 'source.sha256').write_text(hashlib.sha256(b'painting').hexdigest() + '\n')
                (folder / 'metrics.txt').write_text('cuts (scene score > 0.1): none\n')
                (folder / 'luma-flicker.txt').write_text('known repeated drops\n')
                for rank in ('head', 'worker'):
                    (folder / (tag + '.mem-' + rank + '.csv')).write_text('1,15000,0\n2,15000,0\n')
                timing = {'http':'200','curl_rc':0,'duration':duration,'size':'960x544','steps':20,'seed':1101,
                          'task':'fl2va','prompt':PROMPT,'image':str(self.results/'koninck.png'),
                          'extra_params':{'long_video':True,'long_video_mode':'continuation','preencode_mp4':True}}
                (folder / (tag+'.mp4.timing.json')).write_text(json.dumps(timing))
                if duration == 60:
                    (folder / 'phases.txt').write_text('phase diffuse 500.0 s\n' * 6)
        (self.results/'aspect-events-20261031.log').write_text(
            'H3-ASPECT-OK quality16-960x544-20261031 test\n'
            'H3-ASPECT-OK quality60-960x544-20261031 test\n'
            'H3-ASPECT-SHORT-DONE direct-10-15 test\n')
        (self.root/'proc/123/environ').write_bytes(
            b'H3_TP2_VAE_BLEND_TRACE=0\0H3_TP2_REQUEST_TIMEOUT_S=28800\0'
            b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=28800\0VLLM_OMNI_VIDEO_SYNC_TIMEOUT=29400\0')
        self.f.bench.write_text(self.f.bench.read_text()+'''
python3 - "$1" "$2" <<'PY'
import json,os,pathlib,sys
d,w,h,steps,seed=map(int,sys.argv[1].split(':'))
tag=f'd{d}_{w}x{h}_s{steps}_seed{seed}_fl2va'; root=pathlib.Path(os.environ['OUT'])
x={'http':'200','curl_rc':0,'duration':d,'size':f'{w}x{h}','steps':steps,'seed':seed,
   'task':'fl2va','prompt':os.environ['PROMPT'],'image':sys.argv[2], 'extra_params':json.loads(os.environ['EXTRA_JSON'])}
if os.environ.get('H3_TEST_BAD_TIMING') == '1': x['seed']=2
(root/(tag+'.mp4.timing.json')).write_text(json.dumps(x))
for rank in ('head','worker'):
    (root/(tag+'.mem-'+rank+'.csv')).write_text('1,15000,0\\n2,15000,0\\n')
PY
''')
        self.env = dict(self.f.env, H3_EXP_RESULTS=str(self.results), H3_EXP_LIVE=str(self.root/'live'),
                        H3_EXP_STAGE=str(self.root/'stage'), H3_EXP_BACKUP=str(self.root/'backup'),
                        H3_EXP_TXN=str(fixture.TXN), H3_EXP_BENCH=str(self.f.bench),
                        H3_EXP_METRICS=str(self.f.metrics), H3_EXP_PYTHON='python3',
                        H3_EXP_FLICKER=str(fixture.FLICKER), H3_EXP_PROC_ROOT=str(self.root/'proc'),
                        H3_EXP_MEMINFO=str(self.root/'meminfo'), H3_EXP_CLIENT_TIMEOUT_S='30000',
                        H3_EXP_MIN_DISK_KB='0', H3_EXP_WAIT_ATTEMPTS='1', H3_EXP_IDLE_ATTEMPTS='3',
                        H3_EXP_WAIT_S='0', H3_EXP_IDLE_S='0')

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.env, **changes), text=True,
                              capture_output=True, timeout=25)

    def events(self):
        return (self.results/'aspect-events-20261031.log').read_text()

    def test_known_flicker_allows_only_explicit_experiment(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.f.cases(), [NAME])
        self.assertIn('H3-ASPECT-EXPERIMENT300-START known-control-flicker; structural-test-only', self.events())
        self.assertIn('H3-ASPECT-EXPERIMENT300-DONE structural-media-verification-only', self.events())
        self.assertNotIn('H3-ASPECT-QUALITY-DONE', self.events())

    def test_missing_short_result_or_predecessor_hold_never_starts(self):
        (self.results/'quality15-direct-960x544-20261031/STATUS').write_text('failed\n')
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.f.cases(), [])
        other=Experimental300Tests(); other.setUp()
        try:
            with (other.results/'aspect-events-20261031.log').open('a') as out:
                out.write('H3-ASPECT-HOLD vae-trace phase=restore rc=1\n')
            self.assertNotEqual(other.run_queue().returncode, 0)
            self.assertEqual(other.f.cases(), [])
        finally: other.doCleanups()

    def test_cut_memory_missing_timing_or_changed_source_never_starts(self):
        for kind in ('cut','memory','timing','source','output','disk'):
            with self.subTest(kind=kind):
                other=Experimental300Tests(); other.setUp()
                try:
                    folder=other.results/'quality60-960x544-20261031'; changes={}
                    if kind=='cut': (folder/'metrics.txt').write_text('cuts detected\n')
                    if kind=='memory': (folder/'d60_960x544_s20_seed1101_fl2va.mem-head.csv').write_text('1,2000,0\n2,2000,0\n')
                    if kind=='timing': (folder/'phases.txt').write_text('phase diffuse 500.0 s\n' * 5)
                    if kind=='source': (other.results/'koninck.png').write_bytes(b'different')
                    if kind=='output': (other.results/NAME).mkdir()
                    if kind=='disk': changes['H3_EXP_MIN_DISK_KB']='999999999999'
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_timeout_or_trace_skip_without_running(self):
        for kind in ('slow','trace','client'):
            with self.subTest(kind=kind):
                other=Experimental300Tests(); other.setUp()
                try:
                    changes={}
                    if kind=='slow': (other.results/'quality60-960x544-20261031/phases.txt').write_text('phase diffuse 1000.0 s\n'*6)
                    if kind=='trace':
                        p=other.root/'proc/123/environ'; p.write_bytes(p.read_bytes().replace(b'TRACE=0',b'TRACE=1'))
                    if kind=='client': changes['H3_EXP_CLIENT_TIMEOUT_S']='14400'
                    result=other.run_queue(**changes)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(other.f.cases(),[])
                    self.assertIn('H3-ASPECT-EXPERIMENT300-SKIPPED',other.events())
                finally: other.doCleanups()

    def test_busy_jobs_guard_or_changed_deployment_block_run(self):
        for changes in ({'H3_TEST_GPU':'40'},{'H3_TEST_JOBS':'1'},{'H3_TEST_WORKER':'bad'},
                        {'H3_TEST_REMOTE_BAD':'1'},{'H3_TEST_GUARD':'bad'}):
            with self.subTest(changes=changes):
                other=Experimental300Tests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_failed_render_or_wrong_result_never_claims_done(self):
        for changes in ({'H3_TEST_FAIL_CASE':NAME},{'H3_TEST_BAD_TIMING':'1'}):
            with self.subTest(changes=changes):
                other=Experimental300Tests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),[NAME])
                    self.assertNotIn('H3-ASPECT-EXPERIMENT300-DONE',other.events())
                    self.assertIn('H3-ASPECT-HOLD experiment300-20261031',other.events())
                finally: other.doCleanups()


if __name__ == '__main__': unittest.main()
