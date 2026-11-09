"""CPU check of what the speech guard is given to listen to: the part of a window the finished clip keeps."""

import os
import unittest
from unittest import mock

import torch

from h3_tp2.cont_match import heard_audio, overlap_audio_t, wrap_diffuse
from tests.test_cont_cut_check import smooth_window


def ramp(steps=8):
    """An audio latent [1, 2, steps] whose value is its time step, so a slice shows which steps it holds."""
    return torch.arange(float(steps)).repeat(1, 2, 1)


class OverlapTests(unittest.TestCase):
    def test_first_window_has_none(self):
        self.assertIsNone(overlap_audio_t({}))
        self.assertIsNone(overlap_audio_t({"audio_condition_lengths": []}))

    def test_length_of_the_last_audio_condition(self):
        self.assertEqual(overlap_audio_t({"audio_condition_lengths": [5, 3]}), 3)

    def test_edit_mask_counts_pinned_rows_of_two_channels(self):
        mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])   # 3 pinned steps x 2 channels, then free rows
        self.assertEqual(overlap_audio_t({"audio_edit_mask_rows": mask}), 3)


class HeardAudioTests(unittest.TestCase):
    def test_first_window_is_heard_whole(self):
        audio = ramp()
        self.assertIs(heard_audio(audio, {}), audio)

    def test_a_continuation_window_is_heard_without_its_copy_of_the_tail(self):
        out = heard_audio(ramp(), {"audio_condition_lengths": [3]})
        self.assertEqual(out.shape[-1], 5)
        self.assertEqual(out[0, 0].tolist(), [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_the_kept_part_is_what_the_loop_appends(self):
        audio = ramp()
        self.assertTrue(torch.equal(heard_audio(audio, {"audio_condition_lengths": [3]}), audio[..., 3:]))

    def test_switch_gives_the_whole_window(self):
        audio = ramp()
        with mock.patch.dict(os.environ, {"H3_TP2_SPEECH_WHOLE_WINDOW": "1"}):
            self.assertIs(heard_audio(audio, {"audio_condition_lengths": [3]}), audio)

    def test_an_overlap_as_long_as_the_window_is_not_cut_to_nothing(self):
        audio = ramp(3)
        self.assertIs(heard_audio(audio, {"audio_condition_lengths": [3]}), audio)

    def test_non_tensors_pass_through(self):
        self.assertIsNone(heard_audio(None, {"audio_condition_lengths": [3]}))


class LoopTests(unittest.TestCase):
    def run_two_windows(self, env=None):
        first = smooth_window(seed=1)
        second = smooth_window(seed=2); second[:, :, :6] = first[:, :, -6:]
        outputs, judged = iter((first, second)), []

        def diffuse(**kwargs):
            return next(outputs), ramp()

        def speech_fn(audio, index):
            judged.append((index, audio[0, 0].tolist()))
            return (0.0, 0.0, 100)

        with mock.patch.dict(os.environ, env or {}):
            run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, max_cut=0.0, speech_fn=speech_fn, max_cer=0.02)
            run(seed=1)
            _video, audio = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        return judged, audio

    def test_guard_hears_window_one_whole_and_window_two_without_the_overlap(self):
        judged, audio = self.run_two_windows()
        self.assertEqual(judged[0], (1, [float(i) for i in range(8)]))
        self.assertEqual(judged[1], (2, [float(i) for i in range(2, 8)]))
        self.assertEqual(audio.shape[-1], 8)   # the window itself is returned whole; the loop cuts it when appending

    def test_switch_restores_the_whole_window(self):
        judged, _audio = self.run_two_windows({"H3_TP2_SPEECH_WHOLE_WINDOW": "1"})
        self.assertEqual(judged[1], (2, [float(i) for i in range(8)]))


if __name__ == "__main__":
    unittest.main()
