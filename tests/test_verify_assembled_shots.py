import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('verify_assembled_shots', SCRIPTS / 'verify_assembled_shots.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AssemblyChecks(unittest.TestCase):
    def make_shot(self, path, color, frequency):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={color}:s=160x96:r=24:d=2',
                        '-f', 'lavfi', '-i', f'sine=frequency={frequency}:sample_rate=32000:duration=2',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-threads', '1',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k', '-t', '2', str(path)], check=True)

    def make_film(self, path, shots):
        cmd = ['ffmpeg', '-v', 'error', '-filter_complex_threads', '1']
        for shot in shots:
            cmd.extend(['-i', str(shot)])
        filters = []
        for i in range(len(shots)):
            filters.extend([f'[{i}:v]trim=end_frame=48,setpts=PTS-STARTPTS[v{i}]',
                            f'[{i}:a]apad=whole_dur=2,atrim=duration=2,asetpts=PTS-STARTPTS[a{i}]'])
        filters.append(''.join(f'[v{i}][a{i}]' for i in range(len(shots))) +
                       f'concat=n={len(shots)}:v=1:a=1[v][a]')
        cmd.extend(['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '[a]', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '19', '-threads', '1', '-c:a', 'aac', '-ar', '32000', str(path)])
        subprocess.run(cmd, check=True)

    def test_ordered_sources_match_and_swapped_shots_are_detectable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shots = [root / f'shot-{i}.mp4' for i in range(3)]
            for path, color, frequency in zip(shots, ('red', 'green', 'blue'), (337, 619, 997), strict=True):
                self.make_shot(path, color, frequency)
            movie = root / 'film.mp4'
            self.make_film(movie, shots)
            correct = module.assess(movie, shots, shot_seconds=2, fps=24)
            for row in correct['shots']:
                self.assertLess(max(x['rgb_mean_absolute_difference'] for x in row['video_samples']), 3)
                self.assertGreater(row['median_audio_correlation'], .9)
            swapped = module.assess(movie, [shots[1], shots[0], shots[2]], shot_seconds=2, fps=24)
            self.assertGreater(swapped['shots'][0]['video_samples'][0]['rgb_mean_absolute_difference'], 10)
            self.assertLess(swapped['shots'][0]['median_audio_correlation'], .2)


if __name__ == '__main__':
    unittest.main()
