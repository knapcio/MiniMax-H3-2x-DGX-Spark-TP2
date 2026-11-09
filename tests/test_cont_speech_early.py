"""CPU check of the early speech tap in wrap_diffuse: which sampling, window and step reach the check, that the
windows come out as without it, and that an attempt the check stops is given up and sampled again."""

import types
import unittest
from unittest import mock

import torch

from h3_tp2 import pace
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

        def early_fn(x0, index, attempt, step, args, may_stop):
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


class EarlyStopLoopTests(unittest.TestCase):
    CONT = {"ref_blocks": [{"kind": "latent_guide", "latent_t": 6}], "audio_condition_lengths": [2]}

    def run_chain(self, stop_at, retries=2, verdicts=None, **kw):
        """Two windows; ``stop_at`` = {(window, attempt): step} gives those attempts up.  Returns what happened."""
        dl = fake_loop_module()
        diffuse, seeds = make_diffuse(dl)
        asked, judged = [], []

        def early_fn(x0, index, attempt, step, args, may_stop):
            asked.append((index, attempt, step, may_stop))
            if may_stop and stop_at.get((index, attempt)) == step:
                raise se.Stop(step, 0.05, 0.01)

        results = iter(verdicts or [(0.0, 0.0, 100)] * 9)

        def speech_fn(audio, index):
            judged.append((float(audio[0, 0, 0]), index))
            return next(results)

        run = wrap_diffuse(diffuse, apply=False, retries=retries, max_ratio=100, speech_fn=speech_fn, max_cer=0.01,
                           early_fn=early_fn, early_steps=(2,), **kw)
        run(seed=1)
        video, audio = run(seed=2, **self.CONT)
        return seeds, asked, judged, video, audio

    def test_a_stopped_attempt_is_given_up_and_the_next_seed_is_sampled(self):
        with self.assertLogs("vllm_omni.h3_tp2.cont_match", level="WARNING") as logs:
            seeds, asked, judged, video, audio = self.run_chain({(2, 1): 2})
        self.assertEqual(seeds, [1, 2, 2 + 7919])
        self.assertEqual(judged, [(1.0, 1), (3.0, 2)])            # the stopped attempt never reaches the final check
        self.assertEqual(float(audio[0, 0, 0]), 3.0)              # the window is the second attempt's
        self.assertTrue(torch.equal(video, smooth_window(seed=3)))
        self.assertTrue(any("window 2 attempt 1 (seed 2): given up at step 2" in line for line in logs.output))

    def test_the_last_attempt_is_never_stopped(self):
        seeds, asked, judged, _video, audio = self.run_chain({(2, 1): 2, (2, 2): 2, (2, 3): 2})
        self.assertEqual(seeds, [1, 2, 2 + 7919, 2 + 2 * 7919])
        self.assertEqual([a for a in asked if a[0] == 2], [(2, 1, 2, True), (2, 2, 2, True), (2, 3, 2, False)])
        self.assertEqual(judged, [(1.0, 1), (4.0, 2)])
        self.assertEqual(float(audio[0, 0, 0]), 4.0)

    def test_without_retries_nothing_may_stop(self):
        seeds, asked, _judged, _video, _audio = self.run_chain({(1, 1): 2, (2, 1): 2}, retries=0)
        self.assertEqual(seeds, [1, 2])
        self.assertEqual(asked, [(1, 1, 2, False), (2, 1, 2, False)])

    def test_the_best_complete_attempt_is_kept_when_one_in_between_was_stopped(self):
        # window 2: attempt 1 completes with a slip (score 2), attempt 2 is stopped, attempt 3 completes worse (score 5)
        verdicts = [(0.0, 0.0, 100), (2.0, 0.02, 100), (5.0, 0.05, 100)]
        seeds, _asked, judged, _video, audio = self.run_chain({(2, 2): 2}, verdicts=verdicts)
        self.assertEqual(seeds, [1, 2, 2 + 7919, 2 + 2 * 7919])
        self.assertEqual(judged, [(1.0, 1), (2.0, 2), (4.0, 2)])
        self.assertEqual(float(audio[0, 0, 0]), 2.0)

    def test_a_stopped_attempt_does_not_enter_the_pace_record(self):
        # the clock is read at the start of every sampling and at the end of every complete one: window 1 takes 500 s,
        # attempt 1 of window 2 starts at 1000 and is stopped (no reading), attempt 2 runs from 1100 to 1600
        clock = iter([0.0, 500.0, 1000.0, 1100.0, 1600.0])
        with mock.patch.object(pace, "clock", side_effect=lambda: next(clock)), mock.patch.dict(pace._BEST, {}, clear=True):
            with self.assertNoLogs("vllm_omni.h3_tp2.pace", level="WARNING"):
                self.run_chain({(2, 1): 2})
            self.assertEqual(list(pace._BEST.values()), [500.0, 500.0])   # the first window's size and the continued one's
            self.assertEqual(next(clock, None), None)

    def test_a_stop_with_no_attempt_left_is_an_error_not_a_missing_window(self):
        dl = fake_loop_module()
        diffuse, _seeds = make_diffuse(dl)

        def early_fn(x0, index, attempt, step, args, may_stop):
            raise se.Stop(step, 0.05, 0.01)

        run = wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100, early_fn=early_fn, early_steps=(2,))
        with self.assertRaisesRegex(RuntimeError, "no attempt left"):
            run(seed=1)
        self.assertIsNone(se._ACTIVE)


if __name__ == "__main__":
    unittest.main()
