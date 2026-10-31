import json
from pathlib import Path
import subprocess
import unittest

import test_queue_camera_speed_ab_20261031 as previous
import test_queue_vae_trace as fixture


SCRIPT=Path(__file__).with_name('queue-window175-20261031.sh')
NAME='quality30-camera-locked-win175-960x544-20261031'
TAG='d30_960x544_s20_seed1101_fl2va'


class Window175QueueTests(unittest.TestCase):
    def setUp(self):
        self.base=previous.CameraSpeedQueueTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        result=self.base.run_queue()
        if result.returncode: raise AssertionError(result.stdout+result.stderr)
        self.f=self.base.f; self.root=self.base.root; self.results=self.base.results
        self.env=dict(self.base.env,H3_W175_RESULTS=str(self.results),H3_W175_LIVE=str(self.root/'live'),
                      H3_W175_STAGE=str(self.root/'stage'),H3_W175_BACKUP=str(self.root/'backup'),
                      H3_W175_TXN=str(fixture.TXN),H3_W175_BENCH=str(self.f.bench),
                      H3_W175_METRICS=str(self.f.metrics),H3_W175_PYTHON='python3',
                      H3_W175_FLICKER=str(fixture.FLICKER),H3_W175_PROC_ROOT=str(self.root/'proc'),
                      H3_W175_MEMINFO=str(self.root/'meminfo'),H3_W175_MIN_DISK_KB='0',
                      H3_W175_WAIT_ATTEMPTS='1',H3_W175_IDLE_ATTEMPTS='3',H3_W175_WAIT_S='0',H3_W175_IDLE_S='0')

    def run_queue(self,**changes):
        return subprocess.run(['bash',str(SCRIPT)],env=dict(self.env,**changes),text=True,capture_output=True,timeout=35)

    def events(self): return (self.results/'aspect-events-20261031.log').read_text()

    def test_uses_matching_camera_prompt_with_five_175_frame_windows(self):
        result=self.run_queue()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(self.f.cases()[-1],NAME)
        old=json.loads((self.results/previous.LOCKED/(TAG+'.mp4.timing.json')).read_text())
        new=json.loads((self.results/NAME/(TAG+'.mp4.timing.json')).read_text())
        self.assertEqual(new['extra_params']['continuation_window_frames'],175)
        self.assertEqual(new['extra_params']['continuation_prompts'],
                         [old['extra_params']['continuation_prompts'][0]]+[old['extra_params']['continuation_prompts'][1]]*4)
        self.assertEqual(new['prompt'],old['prompt'])
        self.assertIn('H3-ASPECT-WINDOW175-DONE',self.events())

    def test_missing_predecessor_or_bad_baseline_never_starts(self):
        for kind in ('marker','baseline','source','output'):
            with self.subTest(kind=kind):
                other=Window175QueueTests(); other.setUp()
                try:
                    if kind=='marker':
                        p=other.results/'aspect-events-20261031.log'; p.write_text(p.read_text().replace('H3-ASPECT-CAMERAAB-DONE three-30s-controls','REMOVED'))
                    if kind=='baseline':
                        p=other.results/previous.LOCKED/(TAG+'.mp4.timing.json'); x=json.loads(p.read_text()); x['extra_params']['continuation_prompts'][1]='different'; p.write_text(json.dumps(x))
                    if kind=='source': (other.results/'koninck.png').write_bytes(b'changed')
                    if kind=='output': (other.results/NAME).mkdir()
                    before=other.f.cases()
                    self.assertNotEqual(other.run_queue().returncode,0)
                    self.assertEqual(other.f.cases(),before)
                finally: other.doCleanups()

    def test_busy_changed_service_or_failed_render_holds(self):
        for changes,attempted in (({'H3_TEST_GPU':'40'},False),({'H3_TEST_JOBS':'1'},False),
                                  ({'H3_TEST_AGENT_CHANGES':'1'},False),({'H3_TEST_FAIL_CASE':NAME},True)):
            with self.subTest(changes=changes):
                other=Window175QueueTests(); other.setUp()
                try:
                    before=other.f.cases()
                    self.assertNotEqual(other.run_queue(**changes).returncode,0)
                    self.assertEqual(other.f.cases(),before+([NAME] if attempted else []))
                    self.assertIn('H3-ASPECT-HOLD window175-20261031',other.events())
                    self.assertNotIn('H3-ASPECT-WINDOW175-DONE',other.events())
                finally: other.doCleanups()


if __name__=='__main__': unittest.main()
