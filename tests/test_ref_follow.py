"""CPU tests of h3_tp2/ref_follow.py: image reference rows move with the window's temporal offset."""
import types
import unittest

import torch

from h3_tp2 import ref_follow as rf


def packed(origin=10.0, n_image=2, n_guide=0, n_target=6, offset=0.0):
    """A packed layout like upstream's: image blocks at origin + i, an optional latent guide (already offset), targets."""
    rows = n_image * 4 + n_guide + n_target
    img_pos = torch.arange(rows)
    update = torch.zeros(rows, dtype=torch.bool)
    update[n_image * 4 + n_guide:] = True
    g = torch.zeros(rows, 3, dtype=torch.float64)
    for i in range(n_image):
        g[i * 4:(i + 1) * 4, 0] = origin + i
    g[n_image * 4:n_image * 4 + n_guide, 0] = 100.0 + offset
    g[n_image * 4 + n_guide:, 0] = torch.arange(n_target, dtype=torch.float64) + 100.0 + offset
    return {"img_pos": img_pos, "update_mask": update, "img_position_ids": g}


class ShiftTests(unittest.TestCase):
    def test_window_one_is_untouched(self):
        p = packed()
        before = p["img_position_ids"].clone()
        self.assertEqual(rf.shift_image_rows(p, 0.0, 10.0), 0)
        self.assertTrue(torch.equal(p["img_position_ids"], before))

    def test_image_rows_move_targets_and_guides_do_not(self):
        p = packed(n_guide=3, offset=425.0)
        before = p["img_position_ids"].clone()
        self.assertEqual(rf.shift_image_rows(p, 425.0, 10.0), 8)
        g = p["img_position_ids"]
        self.assertTrue(torch.equal(g[:8, 0], before[:8, 0] + 425.0))
        self.assertTrue(torch.equal(g[8:], before[8:]))
        self.assertTrue(torch.equal(g[:, 1:], before[:, 1:]))

    def test_no_references(self):
        p = packed(n_image=0)
        self.assertEqual(rf.shift_image_rows(p, 425.0, 10.0), 0)

    def test_origin_from_text_len_when_unset(self):
        mod = types.SimpleNamespace()
        calls = []

        def packer(*a, **kw):
            calls.append(kw)
            return packed(origin=float(kw["text_len"]), offset=kw.get("temporal_offset", 0.0))

        mod.minimax_h3_packed_sequence_ref2va_blocks = packer
        rf.install(mod)
        rf.install(mod)   # once
        self.assertTrue(getattr(mod.minimax_h3_packed_sequence_ref2va_blocks, "_h3_tp2_ref_follow", False))
        with self.assertLogs(rf.logger, level="INFO") as logs:
            out = mod.minimax_h3_packed_sequence_ref2va_blocks(text_len=10, temporal_offset=850.0)
        self.assertIn("8 image reference rows moved by 850.0", logs.output[0])
        self.assertTrue(torch.equal(out["img_position_ids"][:8, 0], torch.full((8,), 10.0, dtype=torch.float64) + torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64) + 850.0))
        self.assertEqual(len(calls), 1)

    def test_a_packer_failure_is_not_hidden_but_a_shift_failure_is(self):
        mod = types.SimpleNamespace()
        mod.minimax_h3_packed_sequence_ref2va_blocks = lambda *a, **kw: {"img_pos": None}
        rf.install(mod)
        with self.assertLogs(rf.logger, level="WARNING"):
            out = mod.minimax_h3_packed_sequence_ref2va_blocks(text_len=3, temporal_offset=425.0)
        self.assertEqual(out, {"img_pos": None})


if __name__ == "__main__":
    unittest.main()
