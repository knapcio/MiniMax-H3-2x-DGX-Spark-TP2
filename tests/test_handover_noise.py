"""CPU tests of h3_tp2/handover_noise.py: the noised pin as an edit, and inside upstream's own denoise loop (the
verbatim copy in tests/fake_h3_upstream.py)."""
import logging
import unittest

import torch

from h3_tp2 import handover_noise as hn
from tests import fake_h3_upstream as fake
from tests.test_step_cache import make_branch, make_case

SIGMAS = [round(1.0 - i / 20, 4) for i in range(21)]
STOCK_T = 0.999


def stock_edit(mod, n=12, width=96, pinned=4, seed=5, ramp=False):
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(n, width, generator=g)
    noise = torch.randn(n, width, generator=g)
    mask = torch.ones(n)
    mask[:pinned] = 0.0
    if ramp:
        mask[pinned] = 0.5
    edit = mod.MiniMaxH3LatentEdit.from_rows(clean, STOCK_T * clean + (1.0 - STOCK_T) * noise, mask, mask.clone())
    return edit, clean, noise, mask


class ParseTests(unittest.TestCase):
    def test_off(self):
        for text in (None, "", " ", "0", "0.0", "off"):
            self.assertIsNone(hn.parse_level(text))

    def test_levels(self):
        self.assertEqual(hn.parse_level("0.05"), 0.05)
        self.assertEqual(hn.parse_level(" 0.5 "), 0.5)

    def test_out_of_range_fails_loudly(self):
        for text in ("-0.1", "0.51", "1", "nan", "inf"):
            with self.assertRaises(ValueError):
                hn.parse_level(text)
        with self.assertRaises(ValueError):
            hn.parse_level("a lot")


class NoisedEditTests(unittest.TestCase):
    def setUp(self):
        self.mod = fake.load("hn_unit")

    def test_nothing_to_do(self):
        self.assertIsNone(hn.noised(None, 0.1, STOCK_T))
        edit, *_ = stock_edit(self.mod, pinned=0)
        self.assertIs(hn.noised(edit, 0.1, STOCK_T), edit)

    def test_unusable_stock_timestep_leaves_the_edit(self):
        edit, *_ = stock_edit(self.mod)
        with self.assertLogs(hn.logger, level="WARNING"):
            self.assertIs(hn.noised(edit, 0.1, 1.0), edit)

    def test_pinned_rows_are_the_clean_rows_at_the_level_with_the_windows_own_noise(self):
        edit, clean, noise, _mask = stock_edit(self.mod)
        with self.assertLogs(hn.logger, level="INFO") as logs:
            out = hn.noised(edit, 0.1, STOCK_T)
        self.assertIn("4 of 12 pinned video rows shown at noise level 0.100", logs.output[0])
        state = torch.full((12, 96), 7.0)
        rows, timesteps = out.prepare(state, 0.2, STOCK_T, sigma=0.8)
        self.assertTrue(torch.allclose(rows[:4], 0.9 * clean[:4] + 0.1 * noise[:4], atol=2e-3))
        self.assertTrue(torch.equal(rows[4:], state[4:]))                    # free rows: the sampler's state
        self.assertTrue(torch.allclose(timesteps[:4], torch.full((4,), 0.9)))
        stock_rows, stock_t = edit.prepare(state, 0.2, STOCK_T, sigma=0.8)
        self.assertTrue(torch.equal(timesteps[4:], stock_t[4:]))             # free rows: the stock timesteps
        self.assertTrue(torch.allclose(stock_t[:4], torch.full((4,), STOCK_T)))
        self.assertFalse(torch.allclose(rows[:4], stock_rows[:4], atol=1e-2))

    def test_ramped_rows_keep_the_stock_anchor_and_timestep(self):
        edit, _clean, _noise, _mask = stock_edit(self.mod, ramp=True)
        out = hn.noised(edit, 0.2, STOCK_T)
        state = torch.zeros(12, 96)
        rows, timesteps = out.prepare(state, 0.2, STOCK_T, sigma=0.8)
        stock_rows, stock_t = edit.prepare(state, 0.2, STOCK_T, sigma=0.8)
        self.assertTrue(torch.equal(rows[4], stock_rows[4]))
        self.assertTrue(torch.equal(timesteps[4], stock_t[4]))

    def test_x0_restores_the_clean_rows(self):
        edit, clean, _noise, _mask = stock_edit(self.mod)
        out = hn.noised(edit, 0.2, STOCK_T)
        state = torch.randn(12, 96, generator=torch.Generator().manual_seed(1))
        rows, _ = out.prepare(state, 0.2, STOCK_T, sigma=0.8)
        velocity = torch.randn(12, 96, generator=torch.Generator().manual_seed(2))
        x0 = out.x0(rows, velocity, 0.2)
        self.assertTrue(torch.equal(x0[:4], clean[:4]))
        self.assertTrue(torch.equal(x0[4:], edit.x0(rows, velocity, 0.2)[4:]))

    def test_to_keeps_the_level_and_the_anchor(self):
        edit, *_ = stock_edit(self.mod)
        out = hn.noised(edit, 0.15, STOCK_T)
        moved = out.to(device=torch.device("cpu"), dtype=torch.float32)
        self.assertIsInstance(moved, hn.NoisedPin)
        self.assertEqual(moved.level, 0.15)
        self.assertTrue(torch.equal(moved.anchor_rows, out.anchor_rows))
        self.assertTrue(torch.equal(moved.mask_rows, edit.mask_rows))
        self.assertTrue(torch.equal(moved.clean_rows, edit.clean_rows))
        self.assertTrue(torch.equal(moved.restore_mask_rows, edit.restore_mask_rows))

    def test_level_of_the_stock_anchor_reproduces_it(self):
        edit, *_ = stock_edit(self.mod)
        out = hn.noised(edit, 1.0 - STOCK_T, STOCK_T)
        state = torch.zeros(12, 96)
        rows, timesteps = out.prepare(state, 0.2, STOCK_T, sigma=0.8)
        stock_rows, stock_t = edit.prepare(state, 0.2, STOCK_T, sigma=0.8)
        self.assertTrue(torch.allclose(rows, stock_rows, atol=1e-5))
        self.assertTrue(torch.allclose(timesteps, stock_t, atol=1e-6))


def run(mod, case, edit_rows, level=None):
    """One request through upstream's loop; returns the video rows and what the model was shown at every step."""
    model = mod.MiniMaxH3DiTModel(backend="CUDNN_ATTN", seed=0)
    pipe = mod.MiniMaxH3Pipeline(model)
    if level is not None:
        hn.install(mod, mod, level)
    video_edit = mod.MiniMaxH3LatentEdit.from_rows(*(t.clone() for t in edit_rows))
    branch = make_branch(mod, case)
    shown, stock_kwargs = [], branch.forward_kwargs

    def spy(**kw):
        shown.append((kw["video_rows"].clone(), kw["video_target_timesteps"].clone()))
        return stock_kwargs(**kw)

    branch.forward_kwargs = spy
    video, _audio = pipe.diffuse(
        branch=branch,
        initial_video_rows=case["video"].clone(), initial_audio_rows=case["audio"].clone(),
        keyframe_cond_rows=None, audio_ref_rows=None, video_edit=video_edit, audio_edit=None,
        sigmas_video=SIGMAS, sigmas_audio=SIGMAS,
    )
    return video, shown


class NoisedPinInsideUpstreamLoop(unittest.TestCase):
    def setUp(self):
        self.case = make_case(sigmas_v=SIGMAS, sigmas_a=SIGMAS, seed=3)
        g = torch.Generator().manual_seed(11)
        n, width = self.case["video"].shape
        clean = torch.randn(n, width, generator=g)
        mask = torch.ones(n)
        mask[:6] = 0.0   # six pinned video rows, like the continuation prefix
        anchor = STOCK_T * clean + (1.0 - STOCK_T) * self.case["video"]
        self.edit_rows = (clean, anchor, mask, mask.clone())

    def test_the_model_is_shown_the_noised_tail_at_its_timestep_and_the_pinned_rows_come_back_clean(self):
        stock, stock_shown = run(fake.load("hn_stock"), self.case, self.edit_rows)
        with self.assertLogs("vllm_omni.h3_tp2.handover_noise", level="INFO") as logs:
            noisy, shown = run(fake.load("hn_noisy"), self.case, self.edit_rows, level=0.2)
        self.assertEqual(len([m for m in logs.output if "6 of" in m and "noise level 0.200" in m]), 1, logs.output)
        clean, noise = self.edit_rows[0], self.case["video"]
        self.assertEqual(len(shown), 20)
        for (rows, timesteps), (stock_rows, stock_t) in zip(shown, stock_shown):
            self.assertTrue(torch.allclose(rows[:6], 0.8 * clean[:6] + 0.2 * noise[:6], atol=2e-3))
            self.assertTrue(torch.allclose(timesteps[:6], torch.full((6,), 0.8)))
            self.assertTrue(torch.allclose(stock_rows[:6], STOCK_T * clean[:6] + (1 - STOCK_T) * noise[:6], atol=1e-6))
            self.assertTrue(torch.allclose(stock_t[:6], torch.full((6,), STOCK_T)))
            self.assertTrue(torch.equal(timesteps[6:], stock_t[6:]))          # the free rows keep the schedule
        self.assertTrue(torch.equal(stock[:6], clean[:6]))
        self.assertTrue(torch.equal(noisy[:6], clean[:6]))       # what the clip keeps of the hand-over is untouched
        self.assertTrue(torch.isfinite(noisy).all())
        # the stand-in model has no attention between video rows, so its free rows cannot react to the pinned ones:
        # they must be the stock run's exactly, which shows the wrapper changes nothing but the pinned rows' view
        self.assertTrue(torch.equal(noisy[6:], stock[6:]))

    def test_the_stock_level_gives_the_stock_window(self):
        stock, _ = run(fake.load("hn_stock2"), self.case, self.edit_rows)
        same, _ = run(fake.load("hn_same"), self.case, self.edit_rows, level=1.0 - STOCK_T)
        self.assertTrue(torch.allclose(same, stock, atol=1e-3))

    def test_no_edit_passes_through(self):
        mod = fake.load("hn_none")
        hn.install(mod, mod, 0.2)
        model = mod.MiniMaxH3DiTModel(backend="CUDNN_ATTN", seed=0)
        ref = fake.load("hn_none_ref")
        a = mod.MiniMaxH3Pipeline(model).diffuse(
            branch=make_branch(mod, self.case), initial_video_rows=self.case["video"].clone(),
            initial_audio_rows=self.case["audio"].clone(), keyframe_cond_rows=None, audio_ref_rows=None,
            video_edit=None, audio_edit=None, sigmas_video=SIGMAS, sigmas_audio=SIGMAS)
        b = ref.MiniMaxH3Pipeline(ref.MiniMaxH3DiTModel(backend="CUDNN_ATTN", seed=0)).diffuse(
            branch=make_branch(ref, self.case), initial_video_rows=self.case["video"].clone(),
            initial_audio_rows=self.case["audio"].clone(), keyframe_cond_rows=None, audio_ref_rows=None,
            video_edit=None, audio_edit=None, sigmas_video=SIGMAS, sigmas_audio=SIGMAS)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
