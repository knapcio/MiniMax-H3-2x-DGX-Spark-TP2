"""CPU checks for the continuation look-lock reference and pinned prefixes."""

import unittest

import torch

from h3_tp2.cont_match import ref_slice, wrap_diffuse


class LookLockReferenceTests(unittest.TestCase):
    def test_reference_slice_uses_the_time_axis_for_video_and_audio(self):
        video = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(1, 2, 9, 2, 2)
        audio = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12)
        self.assertTrue(torch.equal(ref_slice(video, "window", dim=2), video))
        self.assertTrue(torch.equal(ref_slice(video, "tail", dim=2), video[:, :, -2:]))
        self.assertTrue(torch.equal(ref_slice(audio, "tail", dim=2), audio[..., -3:]))

    @staticmethod
    def render_two_windows(mode=None, locked_source=False):
        first_video = torch.arange(64, dtype=torch.float32).reshape(1, 2, 8, 2, 2) / 20
        next_video = first_video + 3
        next_video[:, :, :2] = first_video[:, :, -2:]
        first_audio = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8) / 20
        next_audio = first_audio + 3
        next_audio[..., :2] = first_audio[..., -2:]
        frames = iter(((first_video.clone(), first_audio.clone()), (next_video.clone(), next_audio.clone())))

        def diffuse(**_kwargs):
            return next(frames)

        options = {} if mode is None else {"lock_ref": mode}
        run = wrap_diffuse(diffuse, apply=False, lock="first", lock_audio=True, lock_ramp=0.25, max_ratio=100, **options)
        out_first = run(seed=1)
        source = {"locked_audio_rows": torch.zeros((16, 32))} if locked_source else {}
        out_second = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 2}], audio_condition_lengths=[2], **source)
        return (first_video, first_audio), (next_video, next_audio), out_first, out_second

    def test_default_equals_explicit_window_reference(self):
        default = self.render_two_windows()
        explicit = self.render_two_windows("window")
        for default_tensor, explicit_tensor in zip(default[3], explicit[3]):
            self.assertTrue(torch.equal(default_tensor, explicit_tensor))

    def test_tail_changes_free_frames_but_preserves_both_pinned_prefixes(self):
        initial, following, output_first, output_tail = self.render_two_windows("tail")
        output_window = self.render_two_windows("window")[3]
        for expected, actual in zip(initial, output_first):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(output_tail[0][:, :, :2], following[0][:, :, :2]))
        self.assertTrue(torch.equal(output_tail[1][..., :2], following[1][..., :2]))
        self.assertFalse(torch.equal(output_tail[0][:, :, 2:], output_window[0][:, :, 2:]))
        self.assertFalse(torch.equal(output_tail[1][..., 2:], output_window[1][..., 2:]))
        self.assertTrue(torch.isfinite(output_tail[0]).all())
        self.assertTrue(torch.isfinite(output_tail[1]).all())

    def test_locked_source_audio_is_not_recolored_by_the_ambient_look_lock(self):
        with self.assertLogs('vllm_omni.h3_tp2.cont_match', level='INFO') as messages:
            _, following, _, locked_output = self.render_two_windows(locked_source=True)
        _, _, _, native_output = self.render_two_windows(locked_source=False)
        self.assertTrue(torch.equal(locked_output[1], following[1]))
        self.assertFalse(torch.equal(native_output[1], following[1]))
        self.assertTrue(torch.equal(locked_output[0], native_output[0]))
        self.assertTrue(any('audio look-lock bypassed for locked source window 2' in line for line in messages.output))


if __name__ == "__main__":
    unittest.main()


class RobustLookLockTests(unittest.TestCase):
    """The robust look-lock (lock_robust=True) takes its statistics from the positions that did not change."""

    C, T, H, W = 2, 8, 6, 8
    OV = 2

    @classmethod
    def scene(cls, drift=0.0, figure=True, seed=0):
        """Window 1: a static background (fixed noise), nobody in its tail.  Window 2: the same background, plus
        ``drift`` on every position and, with ``figure``, a bright 2x3 patch in channel 0 (a character back in frame)."""
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(1, cls.C, 1, cls.H, cls.W, generator=g) * 0.3
        noise1 = torch.randn(1, cls.C, cls.T, cls.H, cls.W, generator=g) * 0.05
        noise2 = torch.randn(1, cls.C, cls.T, cls.H, cls.W, generator=g) * 0.05
        first = base + noise1
        second = base + noise2 + drift
        second[:, :, :cls.OV] = first[:, :, -cls.OV:]
        if figure:
            second[:, 0, cls.OV:, 1:3, 2:5] += 5.0
        return first, second

    def render(self, first, second, *, robust, classes=False):
        frames = iter(((first.clone(), None), (second.clone(), None)))
        run = wrap_diffuse(lambda **_kw: next(frames), apply=False, lock="first", lock_ramp=0.05, max_ratio=1e9,
                           lock_ref="tail", lock_classes=classes, lock_robust=robust)
        run(seed=1)
        out, _ = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": self.OV}])
        return run, out

    def test_a_returning_character_does_not_move_the_robust_lock(self):
        first, second = self.scene()
        _, plain = self.render(first, second, robust=False)
        _, robust = self.render(first, second, robust=True)
        patch = (slice(None), 0, slice(self.OV, None), slice(1, 3), slice(2, 5))
        bg = second[:, 0, self.OV:, 4:, :]
        # the plain lock pulls the whole channel down by the patch's share and squeezes its contrast
        self.assertLess(plain[patch].mean().item(), second[patch].mean().item() - 0.5)
        self.assertLess(plain[:, 0, self.OV:, 4:, :].mean().item(), bg.mean().item() - 0.1)
        # the robust lock sees no drift on the background and leaves the window as sampled (within the noise)
        self.assertAlmostEqual(robust[patch].mean().item(), second[patch].mean().item(), delta=0.15)
        self.assertAlmostEqual(robust[:, 0, self.OV:, 4:, :].mean().item(), bg.mean().item(), delta=0.03)
        self.assertTrue(torch.equal(robust[:, :, :self.OV], second[:, :, :self.OV]))

    def test_the_robust_lock_still_removes_whole_frame_drift(self):
        first, second = self.scene(drift=1.0)
        _, robust = self.render(first, second, robust=True)
        bg = (slice(None), slice(None), slice(self.OV, None), slice(4, None), slice(None))
        # the background is brought back to the reference tail's level, the figure keeps its contrast to it
        self.assertAlmostEqual(robust[bg].mean().item(), first[:, :, -self.OV:, 4:, :].mean().item(), delta=0.05)
        self.assertAlmostEqual((robust[:, 0, self.OV:, 1:3, 2:5] - robust[:, 0, self.OV:, 4:, :].mean()).mean().item(), 5.0, delta=0.3)

    def test_the_robust_maps_survive_a_snapshot_and_restore(self):
        first, second = self.scene(drift=0.4)
        for classes in (False, True):
            frames = iter(((first.clone(), None), (second.clone(), None)))
            run = wrap_diffuse(lambda **_kw: next(frames), apply=False, lock="first", lock_ramp=0.05, max_ratio=1e9,
                               lock_ref="tail", lock_classes=classes, lock_robust=True)
            run(seed=1)
            state = run.snapshot_state()
            self.assertIn("match.maps_mean", state)
            self.assertEqual("match.maps_A_sq" in state, classes)
            resumed = wrap_diffuse(lambda **_kw: (second.clone(), None), apply=False, lock="first", lock_ramp=0.05, max_ratio=1e9,
                                   lock_ref="tail", lock_classes=classes, lock_robust=True)
            resumed.restore_state(state, device="cpu")
            direct, _ = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": self.OV}])
            restored, _ = resumed(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": self.OV}])
            self.assertTrue(torch.equal(direct, restored))
            bad = dict(state); bad["match.maps_sq"] = torch.zeros_like(state["match.maps_sq"])
            fresh = wrap_diffuse(lambda **_kw: (second.clone(), None), apply=False, lock="first", lock_robust=True)
            with self.assertRaisesRegex(ValueError, "maps"):
                fresh.restore_state(bad, device="cpu")
            partial = dict(state); partial.pop("match.maps_sq")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                fresh.restore_state(partial, device="cpu")
