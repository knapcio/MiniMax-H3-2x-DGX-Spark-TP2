"""The uint8 conversion must produce the same bytes as the stock FP32 path (post_process clamp -> numpy
clip/scale/rint in the encoder) for every finite input.  NaN is out of scope on both paths (float-to-integer
conversion of NaN is undefined in numpy and torch alike).  CPU here; the same-seed frame digest in the README is the
GPU evidence.  Needs torch; skipped where it is not installed."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
    from h3_tp2.h3_output import to_uint8_frames
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class U8IdentityTests(unittest.TestCase):
    def _stock(self, v):
        """The stock path: post_process (float, permute, clamp) then the encoder's per-frame clip/scale/rint."""
        f = v.detach().float().cpu().permute(0, 2, 3, 4, 1).clamp(0, 1).numpy()
        out = np.empty(f.shape, dtype=np.uint8)
        for b in range(f.shape[0]):
            for i, fr in enumerate(f[b]):
                s = np.clip(fr.astype(np.float32, copy=False), 0.0, 1.0)
                s *= 255.0
                np.rint(s, out=s)
                out[b, i] = s
        return out

    def test_identical_to_stock_path(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            v = (torch.randn(2, 3, 37, 64, 48, generator=g) * 0.6 + 0.5).to(torch.float16).float()
            flat = v.view(-1)
            flat[:2000] = torch.tensor([k / 255.0 + 0.5 / 255.0 for k in range(2000)])  # near-half cases (saturating above 1)
            # exact ties and their fp32 neighbours: x*255 == k + 0.5 exactly for x = (2k+1)/510 rounded to fp32
            ties = torch.tensor([(2 * k + 1) / 510.0 for k in range(255)], dtype=torch.float32)
            flat[2000:2255] = ties
            flat[2255:2510] = torch.nextafter(ties, torch.tensor(1.0))
            flat[2510:2765] = torch.nextafter(ties, torch.tensor(0.0))
            flat[2765:2769] = torch.tensor([float("inf"), float("-inf"), 1.0, 0.0])
            got = to_uint8_frames(v)
            self.assertEqual(got.shape, (2, 37, 64, 48, 3))
            self.assertTrue(np.array_equal(got.numpy(), self._stock(v)))

    def test_input_not_mutated(self):
        v = torch.rand(1, 3, 4, 8, 6) * 1.5 - 0.2
        before = v.clone()
        to_uint8_frames(v)
        self.assertTrue(torch.equal(v, before))

    def test_shape_and_dtype(self):
        v = torch.rand(2, 3, 5, 8, 6)
        u = to_uint8_frames(v)
        self.assertEqual(tuple(u.shape), (2, 5, 8, 6, 3))
        self.assertEqual(u.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
