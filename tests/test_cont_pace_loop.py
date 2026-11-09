"""CPU check that wrap_diffuse times every sampling and reports it through h3_tp2.pace (wall time only)."""

import unittest
from unittest import mock

import torch

from h3_tp2 import pace
from h3_tp2.cont_match import wrap_diffuse
from tests.test_cont_cut_check import smooth_window

SIZE = {"task": "ref2va", "latent_t": 12, "latent_h": 4, "latent_w": 4, "audio_t": 8, "num_steps": 20}
GUIDE = {"ref_blocks": [{"kind": "latent_guide", "latent_t": 6}], "audio_condition_lengths": [2],
         "visual_condition_shapes": [(6, 4, 4)]}


class PaceInTheLoopTests(unittest.TestCase):
    def run_windows(self, ticks, windows=4):
        first = smooth_window(seed=1)
        calls = []

        def diffuse(**kwargs):
            calls.append(kwargs)
            video = smooth_window(seed=len(calls))
            if len(calls) > 1:
                video[:, :, :6] = first[:, :, -6:]
            return (first if len(calls) == 1 else video), torch.zeros(1, 2, 8)

        run = wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100)
        with mock.patch.object(pace, "clock", side_effect=ticks), mock.patch.dict(pace._BEST, {}, clear=True), \
                self.assertLogs("vllm_omni.h3_tp2.pace", level="INFO") as logs:
            run(seed=1, **SIZE)
            for index in range(2, windows + 1):
                run(seed=index, **SIZE, **GUIDE)
            best = dict(pace._BEST)
        return calls, logs, best

    def test_every_sampling_is_timed_and_a_slow_one_warned(self):
        # window 1: 581 s; windows 2-3: 555 s; window 4: 694 s
        ticks = [0.0, 581.0, 1000.0, 1555.0, 2000.0, 2555.0, 3000.0, 3694.0]
        calls, logs, best = self.run_windows(ticks)
        self.assertEqual(len(calls), 4)
        self.assertEqual([record.levelname for record in logs.records], ["INFO", "INFO", "INFO", "WARNING"])
        self.assertIn("window 1 attempt 1: 581 s (29.1 s/step)", logs.output[0])
        self.assertIn("window 4 attempt 1: 694 s (34.7 s/step), 25 % slower", logs.output[3])
        self.assertEqual(sorted(best.values()), [555.0, 581.0])   # the first window and the continued ones are two sizes

    def test_the_arguments_reach_the_sampler_unchanged(self):
        calls, _logs, _best = self.run_windows([float(10 * i) for i in range(4)], windows=2)
        self.assertEqual(calls[0], {"seed": 1, **SIZE})
        self.assertEqual(calls[1], {"seed": 2, **SIZE, **GUIDE})

    def test_the_reference_survives_the_request(self):
        first = smooth_window(seed=1)

        def diffuse(**_kwargs):
            return first, torch.zeros(1, 2, 8)

        with mock.patch.dict(pace._BEST, {}, clear=True):
            with mock.patch.object(pace, "clock", side_effect=[0.0, 555.0]):
                wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100)(seed=1, **SIZE)
            with mock.patch.object(pace, "clock", side_effect=[0.0, 700.0]), \
                    self.assertLogs("vllm_omni.h3_tp2.pace", level="WARNING") as logs:
                wrap_diffuse(diffuse, apply=False, retries=0, max_ratio=100)(seed=2, **SIZE)   # the next request
        self.assertIn("26 % slower than the fastest sampling of this size in this process (555 s)", logs.output[0])


if __name__ == "__main__":
    unittest.main()
