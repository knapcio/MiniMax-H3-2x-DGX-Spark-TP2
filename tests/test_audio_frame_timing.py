from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audio_frame_timing as timing
import check_timed_audio as screen


class FrameArithmetic(unittest.TestCase):
    def fixture(self):
        return {'streams': [{'sample_rate': '32000', 'time_base': '1/32000'}],
                'frames': [{'pts': 100 + 1024 * i, 'nb_samples': 1024} for i in range(10)]}

    def test_continuous_nonzero_origin_and_trimmed_first_frame(self):
        data = self.fixture()
        data['frames'][0] = {'pts': 100 + 1000, 'nb_samples': 24}
        result = timing.assess_frames(data)
        self.assertTrue(result['continuous'])
        self.assertEqual(result['first_decoded_pts_s'], 1100 / 32000)

    def test_missing_pts_and_samples_fail_closed(self):
        for field in ('pts', 'nb_samples'):
            data = self.fixture(); del data['frames'][2][field]
            self.assertFalse(timing.assess_frames(data)['continuous'])
            self.assertEqual(timing.assess_frames(data)['status'], 'unavailable')

    def test_gap_overlap_and_nonmonotonic_time_fail(self):
        for shift in (32000, -100, -2048):
            data = self.fixture()
            for frame in data['frames'][5:]: frame['pts'] += shift
            self.assertFalse(timing.assess_frames(data)['continuous'])

    def test_small_adjacent_drift_cannot_accumulate(self):
        data = self.fixture()
        for i, frame in enumerate(data['frames']): frame['pts'] += i
        result = timing.assess_frames(data)
        self.assertLess(result['max_adjacent_delta_s'], result['tolerance_s'])
        self.assertFalse(result['continuous'])

    def test_unknown_or_coarse_clock_cannot_pass(self):
        for tb in ('0/1', '1/1', 'bad'):
            data = self.fixture(); data['streams'][0]['time_base'] = tb
            self.assertFalse(timing.assess_frames(data)['continuous'])


class RealMediaTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='h3-frame-timing-')
        cls.root = Path(cls.temp.name)
        cls.wav, cls.mp3, cls.aac, cls.gap, cls.offset = (cls.root / name for name in
            ('source.wav', 'source.mp3', 'source.mp4', 'gap.mp4', 'offset.mp4'))
        def ff(*args):
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', *map(str, args)], check=True, timeout=60)
        ff('-f', 'lavfi', '-i', 'anoisesrc=sample_rate=32000:duration=16:seed=10', '-c:a', 'pcm_s16le', cls.wav)
        ff('-i', cls.wav, '-c:a', 'libmp3lame', cls.mp3)
        ff('-i', cls.wav, '-c:a', 'aac', cls.aac)
        ff('-i', cls.aac, '-c', 'copy', '-bsf:a',
           r'setts=pts=PTS+gte(PTS*TB\,10)/TB:dts=DTS+gte(DTS*TB\,10)/TB', cls.gap)
        ff('-itsoffset', '1', '-i', cls.aac, '-c', 'copy', cls.offset)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_wav_mp3_and_aac_priming_are_continuous(self):
        for path in (self.wav, self.mp3, self.aac):
            with self.subTest(path=path.name):
                self.assertTrue(timing.inspect(path)['continuous'])
                self.assertTrue(screen.assess(path, path, [{'start': 1, 'end': 4}])['pass'])

    def test_gap_holds_despite_identical_decoded_waveforms(self):
        result = screen.assess(self.aac, self.gap, [{'start': 12, 'end': 15}])
        self.assertFalse(result['pass'])
        self.assertGreater(result['intervals'][0]['correlation'], .999)
        self.assertEqual(result['intervals'][0]['shift_seconds'], 0)
        self.assertIsNone(result['intervals'][0]['presentation_shift_seconds'])
        self.assertEqual(result['frame_timing']['output']['max_cumulative_delta_s'], 1)

    def test_absolute_offset_is_not_an_internal_discontinuity(self):
        result = timing.inspect(self.offset)
        self.assertTrue(result['continuous'])
        self.assertGreater(result['first_decoded_pts_s'], .9)

    def test_unknown_decoded_frame_timing_cannot_pass(self):
        unknown = timing.assess_frames({})
        with patch.object(timing, 'inspect', return_value=unknown):
            result = screen.assess(self.aac, self.aac, [{'start': 1, 'end': 4}])
        self.assertFalse(result['pass'])
        self.assertIsNone(result['intervals'][0]['presentation_shift_seconds'])


if __name__ == '__main__': unittest.main()
