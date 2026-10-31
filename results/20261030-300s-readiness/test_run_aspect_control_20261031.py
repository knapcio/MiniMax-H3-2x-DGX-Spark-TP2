import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import test_queue_quality as fixture


SCRIPT=Path(__file__).with_name('run-aspect-control-20261031.sh')


class AspectControlsTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.QueueQualityTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.command('curl','''
case "$*" in
 */health*) printf 200;;
 */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
 */v1/videos*) if [ "${H3_TEST_BUSY_JOB:-0}" = 1 ]; then printf '%s' '{"data":[1],"has_more":false}'; else printf '%s' '{"data":[],"has_more":false}'; fi;;
 *) exit 1;;
esac''')
        original=self.f.bench
        wrapper=self.f.root/'bench-aspect-wrapper.sh'
        wrapper.write_text('''#!/bin/bash
set -e
bash "$H3_TEST_REAL_BENCH" "$@"
python3 - "$1" "$2" <<'PY'
import json,os,pathlib,sys
d,w,h,steps,seed=map(int,sys.argv[1].split(':'))
tag=f'd{d}_{w}x{h}_s{steps}_seed{seed}_fl2va'
x={'http':'200','curl_rc':0,'duration':d,'size':f'{w}x{h}','steps':steps,'seed':seed,
   'task':'fl2va','prompt':os.environ['PROMPT'],'image':sys.argv[2],
   'extra_params':json.loads(os.environ['EXTRA_JSON'])}
if os.environ.get('H3_TEST_BAD_TIMING') == '1': x['seed']=99
(pathlib.Path(os.environ['OUT'])/(tag+'.mp4.timing.json')).write_text(json.dumps(x))
PY
''')
        self.env=dict(self.f.env,H3_SINGLE_RESULTS=str(self.f.results),H3_SINGLE_BENCH=str(wrapper),
                      H3_TEST_REAL_BENCH=str(original),H3_SINGLE_METRICS=str(self.f.metrics),
                      H3_SINGLE_PYTHON='python3',H3_SINGLE_FLICKER=str(fixture.DETECTOR),
                      H3_SINGLE_MEMINFO=str(self.f.root/'meminfo'),H3_SINGLE_WAIT_ATTEMPTS='1',
                      H3_SINGLE_IDLE_ATTEMPTS='3',H3_SINGLE_GATE_ATTEMPTS='1',
                      H3_SINGLE_WAIT_S='0',H3_SINGLE_IDLE_S='0')

    def run_control(self,duration,**changes):
        return subprocess.run(['bash',str(SCRIPT),str(duration)],env=dict(self.env,**changes),
                              text=True,capture_output=True,timeout=20)

    def events(self):
        p=self.f.results/'aspect-events-20261031.log'
        return p.read_text() if p.exists() else ''

    def prepare5(self, sha=None):
        folder=self.f.results/'quality5-960x544-20261031'; folder.mkdir()
        tag='d5_960x544_s20_seed1101_fl2va'
        (folder/(tag+'.mp4')).write_bytes(b'video')
        (folder/'STATUS').write_text(tag+' ok\n')
        sha=sha or hashlib.sha256((self.f.results/'koninck.png').read_bytes()).hexdigest()
        (folder/'source.sha256').write_text(sha+'\n')
        (self.f.results/'aspect-events-20261031.log').write_text('H3-ASPECT-OK quality5-960x544-20261031 test\n')

    def test_five_second_smoke_records_shape_and_source(self):
        result=self.run_control(5)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('H3-ASPECT-OK quality5-960x544-20261031',self.events())
        out=self.f.results/'quality5-960x544-20261031'
        timing=json.loads((out/'d5_960x544_s20_seed1101_fl2va.mp4.timing.json').read_text())
        self.assertEqual((timing['duration'],timing['size']),(5,'960x544'))
        self.assertEqual((out/'source.sha256').read_text().strip(),hashlib.sha256(b'painting').hexdigest())

    def test_sixteen_waits_for_five_and_matching_source(self):
        self.assertNotEqual(self.run_control(16).returncode,0)
        self.assertEqual(self.f.calls(),[])
        self.prepare5(sha='0'*64)
        self.assertNotEqual(self.run_control(16).returncode,0)
        self.assertEqual(self.f.calls(),[])
        (self.f.results/'quality5-960x544-20261031/source.sha256').write_text(hashlib.sha256(b'painting').hexdigest()+'\n')
        result=self.run_control(16)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('H3-ASPECT-OK quality16-960x544-20261031',self.events())

    def test_busy_pair_bad_result_and_existing_output_fail_closed(self):
        for change in ({'H3_TEST_WORKER_GUARD':'off'}, {'H3_TEST_GPU':'40'}, {'H3_TEST_BUSY_JOB':'1'}, {'H3_TEST_BAD_TIMING':'1'}):
            with self.subTest(change=change):
                other=AspectControlsTests(); other.setUp()
                try:
                    self.assertNotEqual(other.run_control(5,**change).returncode,0)
                    self.assertNotIn('H3-ASPECT-OK',other.events())
                finally: other.doCleanups()
        (self.f.results/'quality5-960x544-20261031').mkdir()
        self.assertNotEqual(self.run_control(5).returncode,0)
        self.assertEqual(self.f.calls(),[])

    def test_failed_predecessor_blocks_sixteen(self):
        (self.f.results/'aspect-events-20261031.log').write_text('H3-ASPECT-HOLD quality5-960x544-20261031 phase=rendering rc=1 test\n')
        self.assertNotEqual(self.run_control(16).returncode,0)
        self.assertEqual(self.f.calls(),[])


if __name__ == '__main__': unittest.main()
