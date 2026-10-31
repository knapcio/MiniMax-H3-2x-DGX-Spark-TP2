import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_vae_trace as fixture


SCRIPT=Path(__file__).with_name('queue-aspect-short-20261031.sh')
TEN='quality10-direct-960x544-20261031'
FIFTEEN='quality15-direct-960x544-20261031'


class ShortQueueTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.VaeTraceQueueTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root,self.results=self.f.root,self.f.results
        self.f.output_case('aspect-trace-smoke5','d5_960x576_s10_seed0')
        self.f.output_case('aspect-trace-duration16','d16_960x544_s20_seed1101_fl2va')
        source=self.results/'quality16-960x544-20261031'; source.mkdir()
        (source/'source.sha256').write_text(hashlib.sha256(b'painting').hexdigest()+'\n')
        (self.results/'aspect-events-20261031.log').write_text('H3-ASPECT-TRACE-DONE vae-trace test\n')
        bench=self.f.bench
        bench.write_text(bench.read_text()+'''
python3 - "$1" "$2" <<'PY'
import json,os,pathlib,sys
d,w,h,steps,seed=map(int,sys.argv[1].split(':'))
tag=f'd{d}_{w}x{h}_s{steps}_seed{seed}_fl2va'
x={'http':'200','curl_rc':0,'duration':d,'size':f'{w}x{h}','steps':steps,'seed':seed,
   'task':'fl2va','prompt':os.environ['PROMPT'],'image':sys.argv[2],'extra_params':{'task':'fl2va','duration':d}}
if os.environ.get('H3_TEST_BAD_DIRECT') == str(d): x['extra_params']['long_video']=True
(pathlib.Path(os.environ['OUT'])/(tag+'.mp4.timing.json')).write_text(json.dumps(x))
PY
''')
        self.env=dict(self.f.env,H3_SHORT_RESULTS=str(self.results),H3_SHORT_LIVE=str(self.root/'live'),
                      H3_SHORT_STAGE=str(self.root/'stage'),H3_SHORT_BACKUP=str(self.root/'backup'),
                      H3_SHORT_TXN=str(fixture.TXN),H3_SHORT_BENCH=str(self.f.bench),
                      H3_SHORT_METRICS=str(self.f.metrics),H3_SHORT_PYTHON='python3',
                      H3_SHORT_FLICKER=str(fixture.FLICKER),H3_SHORT_MEMINFO=str(self.root/'meminfo'),
                      H3_SHORT_PROC_ROOT=str(self.root/'proc'),H3_SHORT_WAIT_ATTEMPTS='1',
                      H3_SHORT_IDLE_ATTEMPTS='3',H3_SHORT_WAIT_S='0',H3_SHORT_IDLE_S='0')

    def run_queue(self,**changes):
        return subprocess.run(['bash',str(SCRIPT)],env=dict(self.env,**changes),text=True,
                              capture_output=True,timeout=20)

    def events(self): return (self.results/'aspect-events-20261031.log').read_text()

    def test_runs_direct_ten_then_fifteen_after_trace(self):
        result=self.run_queue()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(self.f.cases(),[TEN,FIFTEEN])
        self.assertIn('H3-ASPECT-SHORT-DONE direct-10-15',self.events())
        for name,duration in ((TEN,10),(FIFTEEN,15)):
            tag=f'd{duration}_960x544_s20_seed1101_fl2va'
            item=json.loads((self.results/name/(tag+'.mp4.timing.json')).read_text())
            self.assertFalse(item['extra_params'].get('long_video',False))
            self.assertEqual(item['duration'],duration)

    def test_missing_trace_or_source_never_starts(self):
        (self.results/'aspect-trace-duration16/STATUS').write_text('failed\n')
        self.assertNotEqual(self.run_queue().returncode,0)
        self.assertEqual(self.f.cases(),[])

    def test_changed_source_or_active_trace_never_starts(self):
        for kind in ('source','trace'):
            with self.subTest(kind=kind):
                other=ShortQueueTests(); other.setUp()
                try:
                    if kind=='source': (other.results/'quality16-960x544-20261031/source.sha256').write_text('0'*64+'\n')
                    else: (other.root/'proc/123/environ').write_bytes(b'H3_TP2_VAE_BLEND_TRACE=1\0')
                    self.assertNotEqual(other.run_queue().returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_busy_or_async_jobs_never_start(self):
        for change in ({'H3_TEST_GPU':'40'},{'H3_TEST_JOBS':'1'},{'H3_TEST_WORKER':'bad'}):
            with self.subTest(change=change):
                other=ShortQueueTests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**change).returncode,0)
                    self.assertEqual(other.f.cases(),[])
                finally: other.doCleanups()

    def test_failed_or_mismatched_first_request_blocks_second(self):
        for change in ({'H3_TEST_FAIL_CASE':TEN},{'H3_TEST_BAD_DIRECT':'10'}):
            with self.subTest(change=change):
                other=ShortQueueTests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_queue(**change).returncode,0)
                    self.assertEqual(other.f.cases(),[TEN])
                    self.assertNotIn('H3-ASPECT-OK '+TEN,other.events())
                finally: other.doCleanups()

    def test_existing_output_or_changed_deployed_files_never_start(self):
        (self.results/TEN).mkdir()
        self.assertNotEqual(self.run_queue().returncode,0)
        self.assertEqual(self.f.cases(),[])
        other=ShortQueueTests(); other.setUp()
        try:
            self.assertNotEqual(other.run_queue(H3_TEST_REMOTE_BAD='1').returncode,0)
            self.assertEqual(other.f.cases(),[])
        finally: other.doCleanups()

    def test_failed_fifteen_keeps_ten_but_never_claims_done(self):
        result=self.run_queue(H3_TEST_FAIL_CASE=FIFTEEN)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.f.cases(),[TEN,FIFTEEN])
        self.assertIn('H3-ASPECT-OK '+TEN,self.events())
        self.assertNotIn('H3-ASPECT-SHORT-DONE',self.events())


if __name__ == '__main__': unittest.main()
