"""CPU tests of h3_tp2/voice_anchor.py: window 1's opening audio as an audio reference block for later windows."""
import unittest

import torch

from h3_tp2 import voice_anchor as va
from h3_tp2.cont_match import overlap_audio_t


def pack(latent):
    c, d, t = latent.shape
    return latent.permute(0, 2, 1).reshape(c * t, d).contiguous()


class ParseTests(unittest.TestCase):
    def test_off_and_values(self):
        for text in (None, "", "0", "off"):
            self.assertIsNone(va.parse_seconds(text))
        self.assertEqual(va.parse_seconds("5"), 5.0)
        self.assertEqual(va.steps(5), 200)
        for text in ("0.2", "16", "x"):
            with self.assertRaises(ValueError):
                va.parse_seconds(text)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.audio = torch.arange(2 * 32 * 600, dtype=torch.float32).reshape(2, 32, 600)

    def test_first_window_or_other_task_untouched(self):
        args = {"task": "ref2va", "seed": 1}
        self.assertIs(va.apply(args, None, 5.0, pack), args)
        args = {"task": "t2va"}
        self.assertIs(va.apply(args, self.audio, 5.0, pack), args)

    def test_anchor_block_first_and_lengths(self):
        args = {"task": "ref2va", "ref_blocks": [{"kind": "image", "latent_h": 64, "latent_w": 112},
                                                  {"kind": "audio", "ref_audio_t": 0}], "audio_condition": None}
        out = va.apply(args, self.audio, 5.0, pack)
        self.assertEqual(out["ref_audio_t"], 200)
        self.assertEqual(out["audio_condition_lengths"], [200])
        self.assertEqual(out["audio_condition"].shape, (400, 32))
        self.assertTrue(torch.equal(out["audio_condition"], pack(self.audio[..., :200])))
        self.assertEqual([b["kind"] for b in out["ref_blocks"]], ["image", "audio"])
        self.assertEqual(out["ref_blocks"][1]["ref_audio_t"], 200)
        self.assertIsNone(args.get("ref_audio_t"))   # the input dict is not modified

    def test_guide_block_stays_last_and_overlap_reads_the_guide(self):
        guide_rows = torch.zeros(2 * 7, 32)
        args = {"task": "ref2va", "audio_condition": guide_rows, "audio_condition_lengths": [7],
                "ref_blocks": [{"kind": "image", "latent_h": 64, "latent_w": 112},
                               {"kind": "latent_guide", "latent_t": 7, "ref_audio_t": 7}]}
        out = va.apply(args, self.audio, 2.0, pack)
        self.assertEqual(out["audio_condition_lengths"], [80, 7])
        self.assertEqual([b["kind"] for b in out["ref_blocks"]], ["image", "audio", "latent_guide"])
        self.assertEqual(out["audio_condition"].shape, (160 + 14, 32))
        self.assertEqual(overlap_audio_t(out), 7)

    def test_pin_mask_wins_over_lengths_for_the_overlap(self):
        mask = torch.ones(20)
        mask[:3] = 0.0
        mask[10:13] = 0.0
        self.assertEqual(overlap_audio_t({"audio_condition_lengths": [200], "audio_edit_mask_rows": mask}), 3)

    def test_short_chain_audio_is_clipped(self):
        out = va.apply({"task": "ref2va"}, self.audio[..., :50], 5.0, pack)
        self.assertEqual(out["ref_audio_t"], 50)
        self.assertEqual(out["audio_condition_lengths"], [50])
        self.assertNotIn("ref_blocks", out)   # upstream builds [image, audio] from ref_audio_t when the list is absent


if __name__ == "__main__":
    unittest.main()
