import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('check_timed_audio', SCRIPTS / 'check_timed_audio.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimedAudioChecks(unittest.TestCase):
    def test_source_passes_and_shifted_or_missing_audio_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, shifted, silent = (root / name for name in ('source.wav', 'shifted.wav', 'silent.wav'))
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'anoisesrc=seed=42:r=32000:d=8',
                            '-c:a', 'pcm_s16le', str(source)], check=True)
            for path, filt in ((shifted, 'adelay=500:all=1'), (silent, 'volume=0')):
                subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-af', filt,
                                '-c:a', 'pcm_s16le', str(path)], check=True)
            intervals = [{'start': .5, 'end': 3, 'speaker': 'A'}, {'start': 4, 'end': 6.5, 'speaker': 'B'}]
            normal = module.assess(source, source, intervals)
            self.assertTrue(normal['pass'])
            self.assertTrue(all(x['correlation'] > .99 and x['presentation_shift_seconds'] == 0 for x in normal['intervals']))
            delayed = module.assess(source, shifted, intervals)
            self.assertFalse(delayed['pass'])
            self.assertTrue(all(abs(x['presentation_shift_seconds'] - .5) < .005 for x in delayed['intervals']))
            missing = module.assess(source, silent, intervals)
            self.assertFalse(missing['pass'])
            self.assertTrue(all(x['status'] == 'low_output_energy' for x in missing['intervals']))
            with self.assertRaises(ValueError):
                module.assess(source, source, [{'start': 5, 'end': 4}])
            timeline = root / 'timeline.json'
            timeline.write_text(json.dumps({'dialogue': intervals}))
            report = root / 'report.json'
            result = subprocess.run([sys.executable, str(SCRIPTS / 'check_timed_audio.py'), str(source), str(source),
                                     '--intervals', str(timeline), '--report', str(report)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(report.read_text())['pass'])


if __name__ == '__main__':
    unittest.main()
