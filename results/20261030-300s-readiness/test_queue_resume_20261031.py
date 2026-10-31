import json
import re
from pathlib import Path
import subprocess
import unittest

import test_queue_quality as fixture


SCRIPT = Path(__file__).with_name("queue-resume-20261031.sh")
CONTROL = "quality16-960-resume-20261031"
SIXTY = "quality60-960-resume-20261031"
THREE_HUNDRED = "quality300-960-resume-20261031"


def combined_prompt():
    text = SCRIPT.read_text()
    def value(name):
        return re.search(r"^" + name + r"='([^']*)'", text, re.M).group(1)
    result = value("original")
    for old, new in (("old_figures", "new_figures"), ("old_shadows", "new_shadows"), ("old_clouds", "new_clouds")):
        result = result.replace(value(old), value(new))
    return result


class ResumeQueueTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.QueueQualityTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.command("curl", '''
case "$*" in
  */health*) printf 200;;
  */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
  */v1/videos*) if [ "${H3_TEST_BUSY_JOB:-0}" = 1 ]; then printf '%s' '{"data":[1],"has_more":false}'; else printf '%s' '{"data":[],"has_more":false}'; fi;;
  *) exit 1;;
esac''')
        original_bench = self.f.bench
        wrapper = self.f.root / "bench-wrapper.sh"
        wrapper.write_text('''#!/bin/bash
set -e
bash "$H3_TEST_REAL_BENCH" "$@"
python3 - "$1" "$2" <<'PY'
import json, os, pathlib, sys
dur,w,h,steps,seed=map(int,sys.argv[1].split(':'))
tag=f'd{dur}_{w}x{h}_s{steps}_seed{seed}_fl2va'
x={'http':'200','curl_rc':0,'duration':dur,'size':f'{w}x{h}','steps':steps,'seed':seed,
   'task':'fl2va','prompt':os.environ['PROMPT'],'image':sys.argv[2],
   'extra_params':json.loads(os.environ['EXTRA_JSON'])}
if os.environ.get('H3_TEST_BAD_TIMING') == str(dur): x['seed'] = 999
(pathlib.Path(os.environ['OUT'])/(tag+'.mp4.timing.json')).write_text(json.dumps(x))
PY
''')
        self.env = dict(self.f.env, H3_QUALITY_BENCH=str(wrapper), H3_TEST_REAL_BENCH=str(original_bench),
                        H3_QUALITY_CLIENT_TIMEOUT_S="30000")

    def prepare_control(self, marker=True):
        root = self.f.results
        out = root / CONTROL
        out.mkdir()
        tag = 'd16_960x576_s20_seed1101_fl2va'
        (out / 'STATUS').write_text(tag + ' ok\n')
        (out / (tag + '.mp4')).write_bytes(b'fake mp4')
        (out / 'metrics.txt').write_text('cuts (scene score > 0.1): none\n')
        timing = {'http':'200', 'curl_rc':0, 'duration':16, 'size':'960x576', 'steps':20, 'seed':1101,
                  'task':'fl2va', 'prompt':combined_prompt(), 'image':str(root / 'koninck.png'),
                  'extra_params':{'long_video':True, 'long_video_mode':'continuation', 'preencode_mp4':True}}
        (out / (tag + '.mp4.timing.json')).write_text(json.dumps(timing))
        for rank in ('head', 'worker'):
            (out / (tag + '.mem-' + rank + '.csv')).write_text('1,15000,0\n2,15000,0\n')
        if marker:
            (root / 'resume-events-20261031.log').write_text('H3-RESUME-OK ' + CONTROL + ' test\n')

    def run_queue(self, **changes):
        return subprocess.run(['bash', str(SCRIPT)], env=dict(self.env, **changes), text=True,
                              capture_output=True, timeout=30)

    def events(self):
        path = self.f.results / 'resume-events-20261031.log'
        return path.read_text() if path.exists() else ''

    def test_waits_for_verified_predecessor(self):
        self.prepare_control(marker=False)
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.f.calls(), [])
        self.assertIn('H3-RESUME-HOLD quality-20261031', self.events())

    def test_runs_60_then_guarded_300(self):
        self.prepare_control()
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.f.calls(), [SIXTY, THREE_HUNDRED])
        self.assertIn('H3-RESUME-QUALITY-DONE', self.events())
        args = json.loads((self.f.results / SIXTY / 'args.json').read_text())
        self.assertEqual(args['prompt'], combined_prompt())
        self.assertTrue(args['extra']['preencode_mp4'])

    def test_guards_reject_busy_service_bad_metadata_and_existing_output(self):
        for kind in ('busy', 'guard', 'input', 'existing'):
            with self.subTest(kind=kind):
                other = ResumeQueueTests()
                other.setUp()
                try:
                    other.prepare_control()
                    changes = {}
                    if kind == 'busy': changes['H3_TEST_BUSY_JOB'] = '1'
                    if kind == 'guard': changes['H3_TEST_WORKER_GUARD'] = 'off'
                    if kind == 'input':
                        p = other.f.results / CONTROL / 'd16_960x576_s20_seed1101_fl2va.mp4.timing.json'
                        x = json.loads(p.read_text()); x['seed'] = 99; p.write_text(json.dumps(x))
                    if kind == 'existing': (other.f.results / SIXTY).mkdir()
                    self.assertNotEqual(other.run_queue(**changes).returncode, 0)
                    self.assertEqual(other.f.calls(), [])
                finally:
                    other.doCleanups()

    def test_failing_60_or_bad_timing_never_starts_300(self):
        self.prepare_control()
        self.assertNotEqual(self.run_queue(H3_TEST_BAD_TIMING='60').returncode, 0)
        self.assertEqual(self.f.calls(), [SIXTY])
        self.assertNotIn('H3-RESUME-OK ' + SIXTY, self.events())

    def test_quality_memory_and_time_guards_skip_300(self):
        for changes in ({'H3_TEST_CUT':'1'}, {'H3_TEST_MIN_MEM':'5000'}, {'H3_TEST_PHASE':'1000'},
                        {'H3_TEST_FLICKER':'1'}, {'H3_TEST_FLICKER_FAIL':'1'},
                        {'H3_QUALITY_CLIENT_TIMEOUT_S':'14400'}):
            with self.subTest(changes=changes):
                other = ResumeQueueTests()
                other.setUp()
                try:
                    other.prepare_control()
                    result = other.run_queue(**changes)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(other.f.calls(), [SIXTY])
                    self.assertIn('H3-RESUME-SKIP-300 ', other.events())
                finally:
                    other.doCleanups()


if __name__ == '__main__':
    unittest.main()
