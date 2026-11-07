"""The step-gated audio pin inside upstream's own denoise loop (the verbatim copy in tests/fake_h3_upstream.py).

Three runs of the same request over 20 steps: audio edit pinned for the whole run (stock), no audio edit, and the
gate at 0.3.  The gated run must differ from both, agree with the stock run while the pin holds (the audio rows after
the release step are what the stock loop also produced up to there), and log the release line once.
"""
import logging
import unittest

import torch

from h3_tp2 import audio_pin_steps as aps
from tests import fake_h3_upstream as fake
from tests.test_step_cache import make_case, make_branch

SIGMAS = [round(1.0 - i / 20, 4) for i in range(21)]


def run(mod, case, *, audio_edit_rows, fraction=None):
    model = mod.MiniMaxH3DiTModel(backend="CUDNN_ATTN", seed=0)
    pipe = mod.MiniMaxH3Pipeline(model)
    if fraction is not None:
        aps.install(mod, mod, fraction)
    audio_edit = None if audio_edit_rows is None else mod.MiniMaxH3LatentEdit.from_rows(*(t.clone() for t in audio_edit_rows))
    seen = []
    v, a = pipe.diffuse(
        branch=make_branch(mod, case),
        initial_video_rows=case["video"].clone(), initial_audio_rows=case["audio"].clone(),
        keyframe_cond_rows=None, audio_ref_rows=None, video_edit=None, audio_edit=audio_edit,
        sigmas_video=SIGMAS, sigmas_audio=SIGMAS,
        on_step=lambda step, video, audio: seen.append(audio.clone()),
    )
    return a, seen


class GateInsideUpstreamLoop(unittest.TestCase):
    def setUp(self):
        self.case = make_case(sigmas_v=SIGMAS, sigmas_a=SIGMAS, seed=3)
        g = torch.Generator().manual_seed(7)
        n = self.case["audio"].shape[0]
        mask = torch.ones(n)
        mask[:4] = 0.0  # four pinned audio rows, like the continuation prefix
        self.edit_rows = (torch.randn(n, 32, generator=g), torch.randn(n, 32, generator=g), mask, mask.clone())

    def test_release_after_30_percent_differs_from_both_ends_and_logs_once(self):
        stock, stock_steps = run(fake.load("stock"), self.case, audio_edit_rows=self.edit_rows)
        free, _ = run(fake.load("free"), self.case, audio_edit_rows=None)
        with self.assertLogs("vllm_omni.h3_tp2.audio_pin_steps", level="INFO") as logs:
            gated, gated_steps = run(fake.load("gated"), self.case, audio_edit_rows=self.edit_rows, fraction=0.3)
        self.assertEqual([m for m in logs.output if "released at step 6 of 20" in m].__len__(), 1, logs.output)
        self.assertFalse(torch.equal(gated, stock))
        self.assertFalse(torch.equal(gated, free))
        # the pin holds for steps 0-5: the rows after step 5 are the stock loop's rows after step 5
        self.assertTrue(torch.equal(gated_steps[5], stock_steps[5]))
        self.assertFalse(torch.equal(gated_steps[6], stock_steps[6]))
        # released: the pinned rows move again
        self.assertFalse(torch.equal(gated[:4], stock[:4]))

    def test_fraction_one_is_the_stock_loop(self):
        stock, _ = run(fake.load("stock2"), self.case, audio_edit_rows=self.edit_rows)
        mod = fake.load("noop")
        self.assertIsNone(aps.parse_fraction("1"))
        same, _ = run(mod, self.case, audio_edit_rows=self.edit_rows)
        self.assertTrue(torch.equal(same, stock))


if __name__ == "__main__":
    unittest.main()
