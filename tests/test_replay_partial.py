"""H3_TP2_REPLAY_PARTIAL: saved latents stand in for the first windows, the rest is denoised (CPU, no server)."""
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:  # pragma: no cover - the suite runs on the Sparks' venv
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class ReplayPartialTests(unittest.TestCase):
    def setUp(self):
        from h3_tp2 import h3_output
        self.m = h3_output
        self.m._REPLAY.update({"files": None, "next": 0})
        self.tmp = tempfile.TemporaryDirectory(); root = pathlib.Path(self.tmp.name)
        self.kw = {"latent_t": 4, "latent_h": 3, "latent_w": 5, "audio_t": 7, "seed": 11}
        for n in (1, 2):
            torch.save((torch.full((1, 2, 4, 3, 5), float(n)), torch.full((1, 2, 7), float(n))), root / f"w{n}.pt")
        os.environ["H3_TP2_REPLAY_LATENTS"] = str(root / "w*.pt")
        os.environ.pop("H3_TP2_REPLAY_PARTIAL", None)

    def tearDown(self):
        os.environ.pop("H3_TP2_REPLAY_LATENTS", None); os.environ.pop("H3_TP2_REPLAY_PARTIAL", None)
        self.m._REPLAY.update({"files": None, "next": 0}); self.tmp.cleanup()

    def test_without_the_flag_a_third_window_is_refused(self):
        self.assertEqual(float(self.m._replay_take(self.kw)[0].flatten()[0]), 1.0)
        self.assertEqual(float(self.m._replay_take(self.kw)[0].flatten()[0]), 2.0)
        with self.assertRaisesRegex(RuntimeError, "only 2 saved"):
            self.m._replay_take(self.kw)

    def test_with_the_flag_the_later_windows_are_denoised(self):
        os.environ["H3_TP2_REPLAY_PARTIAL"] = "1"
        self.assertIsNotNone(self.m._replay_take(self.kw)); self.assertIsNotNone(self.m._replay_take(self.kw))
        self.assertIsNone(self.m._replay_take(self.kw))   # window 3: real denoising
        self.assertIsNone(self.m._replay_take(self.kw))   # window 4 too, no error
        self.assertEqual(self.m._REPLAY["next"], 4)

    def test_the_flag_does_not_skip_the_geometry_check(self):
        os.environ["H3_TP2_REPLAY_PARTIAL"] = "1"
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            self.m._replay_take({**self.kw, "latent_w": 6})


if __name__ == "__main__":
    unittest.main()
