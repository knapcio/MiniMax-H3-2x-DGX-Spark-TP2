"""CPU check of the speech guard's retry in wrap_diffuse (a fake speech_fn stands in for decode + transcription)."""

import unittest

import torch

from h3_tp2.cont_match import wrap_diffuse
from tests.test_cont_cut_check import smooth_window


class SpeechRetryTests(unittest.TestCase):
    def run_two_windows(self, verdicts, max_cer=0.02):
        first = smooth_window(seed=1)
        second = smooth_window(seed=2); second[:, :, :6] = first[:, :, -6:]
        third = smooth_window(seed=3); third[:, :, :6] = first[:, :, -6:]
        seeds, outputs, judged = [], iter((first, second, third)), []
        results = iter(verdicts)

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return next(outputs), torch.full((1, 2, 8), float(len(seeds)))

        def speech_fn(audio, index):
            judged.append((float(audio[0, 0, 0]), index))
            return next(results)

        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, max_cut=0.0, speech_fn=speech_fn, max_cer=max_cer)
        run(seed=1)
        video, audio = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        return seeds, judged, video, audio, second, third

    def test_a_window_with_a_slip_is_resampled_and_the_better_attempt_kept(self):
        # window 1 clean; window 2 attempt 1: CER 0.06 (score 3.0), attempt 2: CER 0.0 -> kept, no third attempt
        seeds, judged, video, audio, _second, third = self.run_two_windows([(0.0, 0.0, 120), (3.0, 0.06, 130), (0.0, 0.0, 130)])
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertEqual(judged, [(1.0, 1), (2.0, 2), (3.0, 2)])  # the window's own audio and index reach the guard
        self.assertTrue(torch.equal(video, third))
        self.assertEqual(float(audio[0, 0, 0]), 3.0)

    def test_a_clean_window_is_not_resampled(self):
        seeds, judged, video, _audio, second, _third = self.run_two_windows([(0.0, 0.0, 120), (0.5, 0.01, 130)])
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(len(judged), 2)
        self.assertTrue(torch.equal(video, second))

    def test_a_window_without_lines_passes(self):
        seeds, judged, *_ = self.run_two_windows([(0.0, 0.0, 0), (0.0, 0.0, 0)])
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(len(judged), 2)

    def test_the_guard_is_off_without_a_limit(self):
        calls = []
        run = wrap_diffuse(lambda **kw: (smooth_window(seed=kw.get("seed", 0)), torch.zeros(1, 2, 8)), apply=False, retries=2,
                           max_ratio=100, speech_fn=lambda a, i: calls.append(i) or (9.0, 0.5, 100), max_cer=0.0)
        run(seed=1)
        run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        self.assertEqual(calls, [])

    def test_a_failing_scorer_accepts_the_window(self):
        def boom(audio, index):
            raise RuntimeError("whisper missing")

        seeds = []

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return smooth_window(seed=kwargs.get("seed", 0)), torch.zeros(1, 2, 8)

        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, speech_fn=boom, max_cer=0.02)
        run(seed=1)
        self.assertEqual(seeds, [1])


if __name__ == "__main__":
    unittest.main()
