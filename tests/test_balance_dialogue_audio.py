import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts';sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('balance_audio', SCRIPTS / 'balance_dialogue_audio.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DialogueBalance(unittest.TestCase):
    def test_bounded_line_gains_preserve_video_and_reduce_level_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);wav=root/'source.wav';video=root/'input.mp4';out=root/'balanced.mp4'
            rate=48000;rng=np.random.default_rng(42);samples=np.zeros(6*rate,dtype=np.float32)
            for start,end,amp in ((.3,2.4,.03),(3,5.5,.12)):
                a,b=round(start*rate),round(end*rate);samples[a:b]=np.clip(rng.normal(0,amp,b-a),-.9,.9)
            with wave.open(str(wav),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes((samples*32767).astype('<i2').tobytes())
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=160x96:r=24:d=6',
                            '-i',str(wav),'-c:v','libx264','-preset','ultrafast','-threads','1','-pix_fmt','yuv420p',
                            '-c:a','aac','-t','6',str(video)],check=True)
            lines=[{'start':.3,'end':2.4,'speaker':'A'},{'start':3,'end':5.5,'speaker':'B'}]
            result=module.balance(video,lines,out)
            self.assertEqual(result['frames'],144);self.assertTrue(result['decoded_video_frames_identical'])
            self.assertLess(result['output_mono_sample_peak'],.98)
            before=[x['input_rms'] for x in result['lines']];after=[x['output_mono_rms'] for x in result['lines']]
            self.assertLess(max(after)/min(after),max(before)/min(before))
            with self.assertRaises(ValueError):module.balance(video,lines,out)
            with self.assertRaises(ValueError):module.balance(video,[{'start':1,'end':2},{'start':1.5,'end':3}],root/'bad.mp4')


if __name__=='__main__':unittest.main()
