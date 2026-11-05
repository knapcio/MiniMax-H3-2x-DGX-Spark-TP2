"""CPU checks of the interior cut check of a continuation window (cut_ratio + the retry in wrap_diffuse)."""

import unittest

import torch

from h3_tp2.cont_match import BLOCK_TOKENS, cut_ratio, wrap_diffuse


def smooth_window(t=41, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 3, 1, 4, 4, generator=g)
    steps = torch.randn(1, 3, t, 4, 4, generator=g) * 0.05
    video = base + steps.cumsum(dim=2)
    # the VAE block-start tokens (5k) differ more from their neighbours, as in the real latents
    video[:, :, ::BLOCK_TOKENS] += 0.3
    return video


class CutRatioTests(unittest.TestCase):
    def test_a_smooth_window_scores_near_one_and_block_starts_are_ignored(self):
        peak, median, ratio, token = cut_ratio(smooth_window(), 6)
        self.assertLess(ratio, 1.8)
        self.assertGreaterEqual(token, 6)
        # a step on a transition into a block-start token is still found: each phase has its own median
        video = smooth_window(); video[:, :, 25:] += 2.0
        self.assertEqual(cut_ratio(video, 6)[3], 24)
        self.assertGreater(cut_ratio(video, 6)[2], 2.0)

    def test_a_step_inside_the_free_part_is_found_and_one_in_the_overlap_is_not(self):
        video = smooth_window()
        video[:, :, 23:] += 2.0                      # transition 22 -> 23: a cut away (22 % 5 = 2, 23 % 5 = 3: a plain one)
        peak, median, ratio, token = cut_ratio(video, 6)
        self.assertEqual(token, 22)
        self.assertGreater(ratio, 2.0)
        video = smooth_window()
        video[:, :, 3:] += 2.0                       # inside the overlap: the seam check's business
        self.assertLess(cut_ratio(video, 6)[2], 1.8)

    def test_a_short_window_returns_no_verdict(self):
        self.assertEqual(cut_ratio(smooth_window(t=18), 6)[2], 0.0)


class RetryTests(unittest.TestCase):
    @staticmethod
    def run_two_windows(max_cut, cut_second=True):
        first = smooth_window(seed=1)
        bad = smooth_window(seed=2); bad[:, :, :6] = first[:, :, -6:]
        if cut_second:
            bad[:, :, 23:] += 2.0
        good = smooth_window(seed=3); good[:, :, :6] = first[:, :, -6:]
        seeds, outputs = [], iter((first, bad, good))

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return next(outputs), torch.zeros(1, 2, 8)

        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, max_cut=max_cut)
        run(seed=1)
        video, _audio = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        return seeds, video, bad, good

    def test_a_cut_window_is_resampled_and_the_clean_attempt_kept(self):
        seeds, video, bad, good = self.run_two_windows(2.0)
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertTrue(torch.equal(video, good))

    def test_disabled_check_keeps_the_first_attempt(self):
        seeds, video, bad, good = self.run_two_windows(0.0)
        self.assertEqual(seeds, [1, 2])
        self.assertTrue(torch.equal(video, bad))

    def test_a_clean_window_is_not_resampled(self):
        seeds, video, bad, good = self.run_two_windows(2.0, cut_second=False)
        self.assertEqual(seeds, [1, 2])
        self.assertTrue(torch.equal(video, bad))


if __name__ == "__main__":
    unittest.main()
