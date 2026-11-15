"""scripts/audio_seams.py: the floor jump sees a hand-over level step and ignores a line against a pause."""
import importlib.util
import unittest
from pathlib import Path

np = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    pass

_SPEC = importlib.util.spec_from_file_location("audio_seams", Path(__file__).resolve().parents[1] / "scripts" / "audio_seams.py")


@unittest.skipUnless(np is not None, "needs numpy")
class FloorJumpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.util.module_from_spec(_SPEC)
        _SPEC.loader.exec_module(cls.m)

    def test_speech_against_a_pause_is_not_a_jump_but_a_floor_step_is(self):
        sr = 16000
        rng = np.random.default_rng(0)
        noise = rng.normal(size=6 * sr).astype(np.float32) * 0.001          # -60 dB room tone
        y = noise.copy()
        t = np.arange(sr) / sr
        y[int(2.0 * sr):int(3.0 * sr)] += (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)   # a line ending at the seam at 3.0 s
        self.assertLess(abs(self.m.floor_jump(y, 3.0)), 3.0)
        plain = 20 * np.log10(np.sqrt(np.mean(y[int(3.0 * sr):int(3.25 * sr)] ** 2))) - 20 * np.log10(np.sqrt(np.mean(y[int(2.75 * sr):int(3.0 * sr)] ** 2)))
        self.assertLess(plain, -20)                                          # the old measure would call this a -20 dB jump
        z = noise.copy()
        z[int(3.0 * sr):] *= 10.0                                            # the room tone steps up 20 dB at the seam
        self.assertAlmostEqual(self.m.floor_jump(z, 3.0), 20.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
