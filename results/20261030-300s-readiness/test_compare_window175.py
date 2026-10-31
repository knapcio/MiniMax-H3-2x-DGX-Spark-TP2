import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT=Path(__file__).with_name('compare_window175.py')
spec=importlib.util.spec_from_file_location('compare_window175',SCRIPT)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class WindowComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        prompt='A painting. The camera is locked on a tripod.'
        first=prompt.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
        rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+prompt
        extra={'task':'fl2va','duration':30,'frame_indices':[0],'long_video':True,
               'long_video_mode':'continuation','preencode_mp4':True,'continuation_prompts':[first,rest,rest]}
        self.runs={}
        for name,wall in ((module.BASE,1800),(module.SMALL,1400)):
            folder=self.root/name; folder.mkdir()
            (folder/'source.sha256').write_text('a'*64+'\n'); (folder/'service-pids.txt').write_text('123 456\n')
            params=copy.deepcopy(extra)
            if name==module.SMALL:
                params['continuation_window_frames']=175; params['continuation_prompts']=[first]+[rest]*4
            timing={'duration':30,'effective_duration':30,'size':'960x544','steps':20,'seed':1101,
                    'task':'fl2va','image':'painting.png','frame_indices':'first','prompt':prompt,
                    'audio_reference':None,'extra_params':params,'elapsed_s':wall}
            self.runs[name]={'timing':timing,'diffuse_seconds':[300]* (5 if name==module.SMALL else 3),
                             'memory':{'head':{'min_available_mib':20000}},'quality_signals':['diagnostic']}
        patched=patch.object(module.speed,'read_run',side_effect=lambda _root,name,_tag:self.runs[name])
        patched.start(); self.addCleanup(patched.stop)

    def test_reports_speed_and_seams_without_quality_promotion(self):
        report=module.compare(self.root)
        self.assertEqual(report['wall_change_percent'],-22.22)
        self.assertEqual(len(report['diffuse_seconds_per_window']['175']),5)
        self.assertIn('increase the number of seams',report['caveat'])

    def test_rejects_changed_prompt_extra_or_missing_window(self):
        self.runs[module.SMALL]['timing']['extra_params']['continuation_prompts'][3]='different'
        with self.assertRaisesRegex(ValueError,'beyond window size'): module.compare(self.root)
        self.runs[module.SMALL]['timing']['extra_params']=copy.deepcopy(self.runs[module.BASE]['timing']['extra_params'])
        with self.assertRaisesRegex(ValueError,'window sizes'): module.compare(self.root)

    def test_rejects_changed_source_or_process_and_incomplete_phases(self):
        path=self.root/module.SMALL/'source.sha256'; path.write_text('b'*64+'\n')
        with self.assertRaisesRegex(ValueError,'source-image'): module.compare(self.root)
        path.write_text('a'*64+'\n'); path=self.root/module.SMALL/'service-pids.txt'; path.write_text('123 789\n')
        with self.assertRaisesRegex(ValueError,'service PIDs'): module.compare(self.root)
        path.write_text('123 456\n'); self.runs[module.SMALL]['diffuse_seconds']=[300]*4
        with self.assertRaisesRegex(ValueError,'per-window'): module.compare(self.root)


if __name__=='__main__': unittest.main()
