"""CPU check of the identity guard's retry in wrap_diffuse (a fake identity_fn stands in for decode + face scoring)."""

import unittest

import torch

from h3_tp2.cont_match import wrap_diffuse
from tests.test_cont_cut_check import smooth_window


class IdentityRetryTests(unittest.TestCase):
    def run_two_windows(self, verdicts, max_unknown=0.2):
        first = smooth_window(seed=1)
        second = smooth_window(seed=2); second[:, :, :6] = first[:, :, -6:]
        third = smooth_window(seed=3); third[:, :, :6] = first[:, :, -6:]
        seeds, outputs, judged = [], iter((first, second, third)), []
        results = iter(verdicts)

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return next(outputs), torch.zeros(1, 2, 8)

        def identity_fn(video, skip_frames):
            judged.append((video, skip_frames))
            return next(results)

        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, max_cut=0.0, identity_fn=identity_fn, max_unknown=max_unknown)
        run(seed=1)
        video, _audio = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        return seeds, judged, video, second, third

    def test_a_window_with_unknown_faces_is_resampled_and_the_better_attempt_kept(self):
        # window 1 passes; window 2 attempt 1: 60 % unknown (score 3.0), attempt 2: 10 % (score 0.5) -> kept, no third
        seeds, judged, video, second, third = self.run_two_windows([(0.0, 2, 0.0), (3.0, 4, 0.6), (0.5, 4, 0.1)])
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertEqual([skip for _v, skip in judged], [0, 24, 24])  # window 1 whole, window 2 after the 6-token overlap
        self.assertTrue(torch.equal(video, third))

    def test_a_window_whose_faces_are_all_known_is_not_resampled(self):
        seeds, judged, video, second, _third = self.run_two_windows([(0.0, 2, 0.0), (0.0, 3, 0.0)])
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(len(judged), 2)
        self.assertTrue(torch.equal(video, second))

    def test_the_first_window_is_resampled_on_identity_alone(self):
        # no overlap, no seam or cut check: the guard still re-samples window 1 and keeps the better attempt
        first, again = smooth_window(seed=1), smooth_window(seed=4)
        outputs, results, seeds = iter((first, again)), iter([(2.0, 3, 0.67), (0.0, 3, 0.0)]), []

        def diffuse(**kwargs):
            seeds.append(kwargs.get("seed"))
            return next(outputs), torch.zeros(1, 2, 8)

        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, identity_fn=lambda v, s: next(results), max_unknown=0.2)
        video, _audio = run(seed=1)
        self.assertEqual(seeds, [1, 1 + 7919])
        self.assertTrue(torch.equal(video, again))

    def test_no_faces_passes(self):
        seeds, judged, video, second, _third = self.run_two_windows([(0.0, 0, 0.0), (0.0, 0, 0.0)])
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(len(judged), 2)

    def test_the_guard_is_off_without_a_limit(self):
        calls = []
        run = wrap_diffuse(lambda **kw: (smooth_window(seed=kw.get("seed", 0)), torch.zeros(1, 2, 8)), apply=False, retries=2,
                           max_ratio=100, identity_fn=lambda v, s: calls.append(v) or (9.0, 2, 1.0), max_unknown=0.0)
        run(seed=1)
        run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
