"""CPU check of the early speech tap in wrap_diffuse: which sampling, window and step reach the measurement, and that
the windows come out as without it."""

import types
import unittest

import torch

from h3_tp2 import speech_early as se
from h3_tp2.cont_match import wrap_diffuse
from tests.test_cont_cut_check import smooth_window


def make_diffuse(dl, steps=4):
    """A diffuse that walks a loop calling the (tapped) update function twice per step, like upstream."""
    seeds = []

    def diffuse(**kwargs):
        seeds.append(kwargs.get("seed"))
        for step in range(1, steps + 1):
            dl.minimax_h3_euler_eta0_step(None, torch.full((2,), float(step)))            # video
            dl.minimax_h3_euler_eta0_step(None, torch.full((2,), 100.0 * len(seeds) + step))   # audio
        video = smooth_window(seed=len(seeds))
        return video, torch.full((1, 2, 8), float(len(seeds)))

    return diffuse, seeds


def fake_loop_module():
    dl = types.ModuleType("fake_denoise_loop")
    dl.minimax_h3_euler_eta0_step = lambda target, x0, **kw: x0
    se.install(dl)
    return dl


class EarlyTapLoopTests(unittest.TestCase):
    def test_every_sampling_is_heard_at_the_chosen_steps_with_its_window_and_attempt(self):
        dl = fake_loop_module()
        diffuse, seeds = make_diffuse(dl)
        heard = []

        def early_fn(x0, index, attempt, step, args):
            heard.append((float(x0[0]), index, attempt, step, args.get("seed")))

        verdicts = iter([(0.0, 0.0, 100), (3.0, 0.06, 100), (0.0, 0.0, 100)])   # window 2 slips once
        run = wrap_diffuse(diffuse, apply=False, retries=2, max_ratio=100, speech_fn=lambda a, i: next(verdicts), max_cer=0.02,
                           early_fn=early_fn, early_steps=(2, 4))
        run(seed=1)
        run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}], audio_condition_lengths=[2])
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertEqual(heard, [
            (102.0, 1, 1, 2, 1), (104.0, 1, 1, 4, 1),
            (202.0, 2, 1, 2, 2), (204.0, 2, 1, 4, 2),
            (302.0, 2, 2, 2, 2 + 7919), (304.0, 2, 2, 4, 2 + 7919),
        ])

    def test_the_windows_are_the_ones_without_the_measurement(self):
        outs = []
        for early in (None, lambda *a: None):
            dl = fake_loop_module()
            diffuse, _seeds = make_diffuse(dl)
            run = wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100, early_fn=early, early_steps=(1, 2, 3, 4))
            outs.append((run(seed=1), run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 6}])))
        for (v0, a0), (v1, a1) in zip(outs[0], outs[1]):
            self.assertTrue(torch.equal(v0, v1))
            self.assertTrue(torch.equal(a0, a1))

    def test_without_steps_nothing_is_heard(self):
        dl = fake_loop_module()
        diffuse, _seeds = make_diffuse(dl)
        heard = []
        run = wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100, early_fn=lambda *a: heard.append(a), early_steps=())
        run(seed=1)
        self.assertEqual(heard, [])
        self.assertIsNone(se._ACTIVE)


if __name__ == "__main__":
    unittest.main()
