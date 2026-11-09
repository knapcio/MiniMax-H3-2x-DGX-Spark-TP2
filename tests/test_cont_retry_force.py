"""CPU check of the two measurement switches of the retry loop in wrap_diffuse (forced re-sample, seed shift)."""

import os
import unittest
from unittest import mock

import torch

from h3_tp2.cont_match import window_map, wrap_diffuse
from tests.test_cont_checkpoint_loop import fake_diffuse, options, production_loop, request
from tests.test_cont_cut_check import smooth_window


class WindowMapTests(unittest.TestCase):
    def test_windows_with_and_without_a_value(self):
        self.assertEqual(window_map(""), {})
        self.assertEqual(window_map("3, 5"), {3: 1, 5: 1})
        self.assertEqual(window_map("3:7919,5:11"), {3: 7919, 5: 11})

    def test_a_bad_item_is_refused(self):
        for text in ("x", "3:y", "0", "-1:5"):
            with self.assertRaises(ValueError):
                window_map(text)


class ForcedRetryTests(unittest.TestCase):
    def run_two_windows(self, env):
        first = smooth_window(seed=1)
        second = smooth_window(seed=2); second[:, :, :6] = first[:, :, -6:]
        third = smooth_window(seed=3); third[:, :, :6] = first[:, :, -6:]
        seeds, outputs = [], iter((first, second, third))

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return next(outputs), torch.full((1, 2, 8), float(len(seeds)))

        with mock.patch.dict(os.environ, env):
            run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, max_cut=0.0)
        run(seed=1)
        video, audio = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        return seeds, video, audio, second, third

    def test_a_forced_window_is_sampled_twice_and_the_retry_kept(self):
        seeds, video, audio, _second, third = self.run_two_windows({"H3_TP2_CONT_RETRY_FORCE": "2"})
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertTrue(torch.equal(video, third))
        self.assertEqual(float(audio[0, 0, 0]), 3.0)

    def test_a_seed_shift_moves_the_first_attempt_to_the_retry_seed(self):
        seeds, video, _audio, second, _third = self.run_two_windows({"H3_TP2_CONT_RETRY_SEED_SHIFT": "2:7919"})
        self.assertEqual(seeds, [1, 2 + 7919])
        self.assertTrue(torch.equal(video, second))

    def test_the_first_window_can_be_forced_without_any_check(self):
        seeds = []

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return smooth_window(seed=len(seeds)), torch.zeros(1, 2, 8)

        with mock.patch.dict(os.environ, {"H3_TP2_CONT_RETRY_FORCE": "1"}):
            run = wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100)
        video, _ = run(seed=5)
        self.assertEqual(seeds, [5, 5 + 7919])
        self.assertTrue(torch.equal(video, smooth_window(seed=2)))

    def test_nothing_changes_without_the_switches(self):
        seeds, video, _audio, second, _third = self.run_two_windows({})
        self.assertEqual(seeds, [1, 2])
        self.assertTrue(torch.equal(video, second))


class PluginLoopTests(unittest.TestCase):
    """The plugin's own continuation loop (its real body, synthetic sampler)."""

    def render(self, env, **overrides):
        calls = []
        kwargs, texts = request()
        with mock.patch.dict(os.environ, env):
            video, audio = production_loop(options(retries=2, **overrides))(
                fake_diffuse(calls), kwargs, window_frames=277, overlap_frames=22, text_conditioning=texts)
        return [c[0] for c in calls], video, audio

    def test_a_forced_retry_equals_a_first_attempt_with_the_retry_seed(self):
        seeds_f, video_f, audio_f = self.render({"H3_TP2_CONT_RETRY_FORCE": "2"})
        seeds_s, video_s, audio_s = self.render({"H3_TP2_CONT_RETRY_SEED_SHIFT": "2:7919"})
        self.assertEqual(seeds_f, [1101, 1102, 1102 + 7919, 1103])
        self.assertEqual(seeds_s, [1101, 1102 + 7919, 1103])
        self.assertTrue(torch.equal(video_f, video_s))
        self.assertTrue(torch.equal(audio_f, audio_s))
        _seeds, plain, _audio = self.render({})
        self.assertFalse(torch.equal(video_f, plain))

    def test_the_loop_resamples_a_window_the_speech_guard_refuses(self):
        heard = []

        def make_speech_fn(_diffuse):
            def fn(audio, index):
                heard.append(index)
                return (3.0, 0.06, 130) if heard.count(2) == 1 and index == 2 else (0.0, 0.0, 130)
            return fn

        cfg = mock.Mock(max_cer=0.02)
        seeds, _video, _audio = self.render({}, speech_cfg=cfg, make_speech_fn=make_speech_fn)
        self.assertEqual(seeds, [1101, 1102, 1102 + 7919, 1103])
        self.assertEqual(heard, [1, 2, 2, 3])

    def test_the_checks_also_run_on_a_dropped_attempt(self):
        # as before a real retry: the guard hears the first attempt of the forced window too, its verdict is ignored
        heard = []

        def make_speech_fn(_diffuse):
            return lambda audio, index: heard.append(index) or (0.0, 0.0, 130)

        seeds, video_f, _audio = self.render({"H3_TP2_CONT_RETRY_FORCE": "2"}, speech_cfg=mock.Mock(max_cer=0.5),
                                             make_speech_fn=make_speech_fn)
        self.assertEqual(seeds, [1101, 1102, 1102 + 7919, 1103])
        self.assertEqual(heard, [1, 2, 2, 3])
        _seeds, video_s, _audio = self.render({"H3_TP2_CONT_RETRY_SEED_SHIFT": "2:7919"}, speech_cfg=mock.Mock(max_cer=0.5),
                                              make_speech_fn=make_speech_fn)
        self.assertTrue(torch.equal(video_f, video_s))


if __name__ == "__main__":
    unittest.main()
