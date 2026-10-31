import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT=Path(__file__).with_name('compare_camera_speed.py')
spec=importlib.util.spec_from_file_location('compare_camera_speed',SCRIPT)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class CameraSpeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        prompt='A painting. The camera is locked on a tripod.'
        first=prompt.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
        rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+prompt
        extra={'task':'fl2va','duration':30,'frame_indices':[0],'long_video':True,
               'long_video_mode':'continuation','preencode_mp4':True}
        self.runs={}
        for name,steps,seconds in zip(module.NAMES,(20,20,10),(1800,1750,900),strict=True):
            folder=self.root/name; folder.mkdir()
            (folder/'source.sha256').write_text('a'*64+'\n')
            (folder/'service-pids.txt').write_text('123 456\n')
            params=copy.deepcopy(extra)
            if name!=module.NAMES[0]: params['continuation_prompts']=[first,rest,rest]
            timing={'duration':30,'effective_duration':30,'size':'960x544','seed':1101,'steps':steps,
                    'task':'fl2va','image':'painting.png','frame_indices':'first','prompt':prompt,
                    'audio_reference':None,'extra_params':params,'elapsed_s':seconds}
            self.runs[name]={'timing':timing,'diffuse_seconds':[500,500,400],
                             'memory':{'head':{'min_available_mib':20000}},'quality_signals':['diagnostic']}
        self.read=patch.object(module.speed,'read_run',side_effect=lambda _root,name,_tag:self.runs[name])
        self.read.start(); self.addCleanup(self.read.stop)
        self.frames=patch.object(module.exact,'frame_hashes',return_value=['same']*277)
        self.frames.start(); self.addCleanup(self.frames.stop)

    def test_reports_separate_camera_and_step_comparisons(self):
        report=module.compare(self.root)
        self.assertEqual(report['pre_overlap_frames_equal'],255)
        self.assertEqual(report['overlap_frames_equal'],22)
        self.assertEqual(report['wall_seconds']['locked20'],1750)
        self.assertEqual(report['locked_10_vs_20_wall_change_percent'],-48.57)
        self.assertIn('Review playback',report['caveat'])

    def test_rejects_changed_camera_prompt_or_step_extra(self):
        self.runs[module.NAMES[1]]['timing']['extra_params']['continuation_prompts'][1]='different'
        self.runs[module.NAMES[2]]['timing']['extra_params']['continuation_prompts'][1]='different'
        with self.assertRaisesRegex(ValueError,'camera controls'):
            module.compare(self.root)
        self.runs[module.NAMES[2]]['timing']['extra_params']['other']=True
        with self.assertRaisesRegex(ValueError,'step-control'):
            module.compare(self.root)

    def test_rejects_different_processes_source_or_shape(self):
        path=self.root/module.NAMES[2]/'service-pids.txt'; path.write_text('123 789\n')
        with self.assertRaisesRegex(ValueError,'process identities'): module.compare(self.root)
        path.write_text('123 456\n')
        image=self.root/module.NAMES[2]/'source.sha256'; image.write_text('b'*64+'\n')
        with self.assertRaisesRegex(ValueError,'source-image'): module.compare(self.root)
        image.write_text('a'*64+'\n')
        self.runs[module.NAMES[2]]['timing']['size']='960x576'
        with self.assertRaisesRegex(ValueError,'size'): module.compare(self.root)

    def test_requires_all_window_timings(self):
        self.runs[module.NAMES[2]]['diffuse_seconds']=[500,400]
        with self.assertRaisesRegex(ValueError,'per-window'):
            module.compare(self.root)


if __name__=='__main__': unittest.main()
