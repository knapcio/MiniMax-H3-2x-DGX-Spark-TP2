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
    def render_two_windows(mode=None):
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
        out_second = run(seed=2, ref_blocks=[{"kind": "latent_guide", "latent_t": 2}], audio_condition_lengths=[2])
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


if __name__ == "__main__":
    unittest.main()
