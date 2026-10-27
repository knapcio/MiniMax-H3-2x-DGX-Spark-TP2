"""The GPU-side uint8 conversion must produce the same bytes as the stock FP32 path (post_process clamp -> numpy
clip/scale/rint in the encoder).  Needs torch; skipped where it is not installed."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
    from h3_tp2.h3_output import to_uint8_frames
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class U8IdentityTests(unittest.TestCase):
    def _stock(self, v):
        f = v.detach().float().cpu().permute(0, 2, 3, 4, 1).clamp(0, 1).numpy()[0]
        out = np.empty(f.shape, dtype=np.uint8)
        for i, fr in enumerate(f):
            s = np.clip(fr.astype(np.float32, copy=False), 0.0, 1.0)
            s *= 255.0
            np.rint(s, out=s)
            out[i] = s
        return out

    def test_identical_to_stock_path(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(3):
            v = (torch.randn(1, 3, 37, 64, 48, generator=g) * 0.6 + 0.5).to(torch.float16).float()
            v.view(-1)[:2000] = torch.tensor([k / 255.0 + 0.5 / 255.0 for k in range(2000)])  # near-half cases
            self.assertTrue(np.array_equal(to_uint8_frames(v)[0].numpy(), self._stock(v)))

    def test_shape_and_dtype(self):
        v = torch.rand(2, 3, 5, 8, 6)
        u = to_uint8_frames(v)
        self.assertEqual(tuple(u.shape), (2, 5, 8, 6, 3))
        self.assertEqual(u.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
