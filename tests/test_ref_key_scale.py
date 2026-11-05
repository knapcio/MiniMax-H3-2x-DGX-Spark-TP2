"""CPU checks of h3_tp2.ref_key_scale: factor parsing, the image block rows, the per-rank slice and the in-place scale."""
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from h3_tp2 import ref_key_scale as m


@unittest.skipIf(torch is None, "torch not installed")
class RefKeyScaleTests(unittest.TestCase):
    def setUp(self):
        m.configure("")

    def test_factors(self):
        self.assertIsNone(m.parse_factors(""))
        self.assertIsNone(m.parse_factors("1"))
        self.assertIsNone(m.parse_factors("1,1.0"))
        self.assertEqual(m.parse_factors("1.3"), [1.3])
        self.assertEqual(m.parse_factors("1,1.5,1.5"), [1.0, 1.5, 1.5])
        for bad in ("0", "16", "-1", "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.parse_factors(bad)

    def test_image_blocks_are_the_leading_image_rows_after_the_text(self):
        blocks = [{"kind": "image", "latent_h": 64, "latent_w": 112}, {"kind": "image", "latent_h": 102, "latent_w": 64},
                  {"kind": "audio", "ref_audio_t": 5}, {"kind": "image", "latent_h": 8, "latent_w": 8}]
        self.assertEqual(m.image_blocks(300, blocks), [(300, 300 + 32 * 56), (300 + 32 * 56, 300 + 32 * 56 + 51 * 32)])
        self.assertEqual(m.image_blocks(10, []), [])

    def test_scale_hits_only_this_ranks_reference_rows(self):
        m.configure("1,2,2")
        m.record_layout(4, [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "image", "latent_h": 2, "latent_w": 4},
                            {"kind": "image", "latent_h": 2, "latent_w": 2}], 40)
        # rows: scene 4..8 (factor 1), crop A 8..10 (2), crop B 10..11 (2); rank 0 holds rows 0..20, rank 1 rows 20..40
        k0 = torch.ones(2, 20, 3, 4); k1 = torch.ones(2, 20, 3, 4)
        self.assertEqual(m.scale_send_k(k0, (0, 20)), 3)
        self.assertEqual(m.scale_send_k(k1, (20, 20)), 0)
        self.assertTrue(torch.equal(k1, torch.ones(2, 20, 3, 4)))
        expect = torch.ones(2, 20, 3, 4); expect[:, 8:11] = 2.0
        self.assertTrue(torch.equal(k0, expect))
        # a span of another length is not this sequence (self-check sub-calls): untouched
        other = torch.ones(2, 7, 3, 4)
        self.assertEqual(m.scale_send_k(other, (0, 7)), 0)
        self.assertTrue(torch.equal(other, torch.ones(2, 7, 3, 4)))

    def test_single_factor_covers_every_block_and_a_split_block_is_partial(self):
        m.configure("1.5")
        m.record_layout(0, [{"kind": "image", "latent_h": 2, "latent_w": 6}], 12)   # rows 0..3
        k0 = torch.ones(2, 6, 1, 1); k1 = torch.ones(2, 6, 1, 1)
        m.record_layout(0, [{"kind": "image", "latent_h": 2, "latent_w": 16}], 12)  # rows 0..8 -> split 0..6 | 6..8
        self.assertEqual(m.scale_send_k(k0, (0, 6)), 6)
        self.assertEqual(m.scale_send_k(k1, (6, 6)), 2)
        self.assertEqual(k1[0, :, 0, 0].tolist(), [1.5, 1.5, 1.0, 1.0, 1.0, 1.0])

    def test_off_is_a_no_op(self):
        m.configure("1")
        m.record_layout(0, [{"kind": "image", "latent_h": 2, "latent_w": 2}], 8)
        k = torch.ones(2, 8, 1, 1)
        self.assertEqual(m.scale_send_k(k, (0, 8)), 0)
        self.assertTrue(torch.equal(k, torch.ones(2, 8, 1, 1)))


if __name__ == "__main__":
    unittest.main()
