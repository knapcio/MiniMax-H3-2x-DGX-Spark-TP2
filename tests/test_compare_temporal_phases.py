import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/compare_temporal_phases.py'
spec = importlib.util.spec_from_file_location('compare_temporal_phases', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TemporalPhaseComparisonTests(unittest.TestCase):
    def test_matching_clips_report_all_frames_and_identical_phase_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, candidate = Path(tmp) / 'source.mp4', Path(tmp) / 'candidate.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x64:rate=24:duration=2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            shutil.copyfile(source, candidate)
            report = module.compare(source, candidate, start=2, end=44)
            self.assertEqual(report['decoded_frames'], 48)
            self.assertEqual(report['frame_interval'], [2, 44])
            self.assertEqual(report['source_sha256'], report['candidate_sha256'])
            self.assertEqual(report['source_phases'], report['candidate_phases'])
            self.assertIn('not a visual-quality verdict', report['caveat'])

    def test_rejects_mismatched_frame_count_or_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, candidate = Path(tmp) / 'a', Path(tmp) / 'b'
            source.write_bytes(b'a'); candidate.write_bytes(b'b')
            geometry = {'width': 4, 'height': 4, 'fps': '24'}
            with mock.patch.object(module, 'metadata', return_value=geometry):
                with mock.patch.object(module.analyzer, 'encoded_means', side_effect=[[1, 2], [1, 2, 3]]):
                    with self.assertRaisesRegex(ValueError, 'frame counts'):
                        module.compare(source, candidate)
                with mock.patch.object(module.analyzer, 'encoded_means', side_effect=[[1, 2], [1, 2]]):
                    with mock.patch.object(module, 'digest', side_effect=['a', 'b', 'changed', 'b']):
                        with self.assertRaisesRegex(ValueError, 'changed'):
                            module.compare(source, candidate)


if __name__ == '__main__': unittest.main()
