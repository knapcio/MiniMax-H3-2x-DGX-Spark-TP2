"""CPU tests of scripts/speaker_drift.py: the pitch split and the per-window summary (no model, no audio files)."""
import importlib.util
import unittest
from pathlib import Path

np = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    pass

_SPEC = importlib.util.spec_from_file_location("speaker_drift", Path(__file__).resolve().parents[1] / "scripts" / "speaker_drift.py")


@unittest.skipUnless(np is not None, "needs numpy")
class SpeakerDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sd = importlib.util.module_from_spec(_SPEC)
        _SPEC.loader.exec_module(cls.sd)

    def test_import_loads_no_model(self):
        self.assertIsNone(self.sd.model)

    def test_pitch_splits_a_low_and_a_high_voice(self):
        t = np.arange(16000 * 1.5) / 16000
        low = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        high = (0.3 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)
        self.assertAlmostEqual(self.sd.pitch(low), 110, delta=4)
        self.assertAlmostEqual(self.sd.pitch(high), 230, delta=8)
        self.assertIsNone(self.sd.pitch(np.zeros(16000, dtype=np.float32)))

    def test_a_strong_second_harmonic_does_not_double_the_pitch(self):
        t = np.arange(16000 * 1.5) / 16000
        voice = (0.1 * np.sin(2 * np.pi * 104 * t) + 0.3 * np.sin(2 * np.pi * 208 * t) + 0.1 * np.sin(2 * np.pi * 312 * t)).astype(np.float32)
        self.assertAlmostEqual(self.sd.pitch(voice), 104, delta=4)
        even_only = (0.3 * np.sin(2 * np.pi * 208 * t) + 0.1 * np.sin(2 * np.pi * 416 * t)).astype(np.float32)
        self.assertAlmostEqual(self.sd.pitch(even_only), 208, delta=8)   # no odd harmonics: that is a 208 Hz voice

    def test_summary_means_and_detail(self):
        e = np.eye(4)
        rows = [{"w": 1, "spk": "low", "e": e[0]}, {"w": 1, "spk": "low", "e": e[0] * 0.8 + e[1] * 0.6},
                {"w": 2, "spk": "low", "e": e[0]}, {"w": 2, "spk": "low", "e": e[2]},
                {"w": 3, "spk": "high", "e": e[3]}]        # no window-1 segment for the high voice: left out
        out, detail = self.sd.summarise(rows)
        self.assertEqual(sorted(out), ["low"])
        ref = (e[0] + (e[0] * 0.8 + e[1] * 0.6)) / 2
        ref = ref / np.linalg.norm(ref)
        self.assertAlmostEqual(out["low"][1], round(float((e[0] @ ref + (e[0] * 0.8 + e[1] * 0.6) @ ref) / 2), 3), places=3)
        self.assertAlmostEqual(out["low"][2], round(float(e[0] @ ref / 2), 3), places=3)   # e[2] is orthogonal: cosine 0
        self.assertEqual(detail["low"][2], {"n": 2, "min": 0.0, "relabelled": 0})
        self.assertEqual(detail["low"][1]["n"], 2)
        self.assertEqual(detail["low"]["w1_loo"], [0.8, 0.8])   # each window-1 segment against the other one

    def test_a_line_goes_to_the_voice_not_the_pitch(self):
        man, woman = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        rows = [{"w": 1, "spk": "low", "e": man}, {"w": 1, "spk": "high", "e": woman},
                {"w": 3, "spk": "high", "e": man * 0.95 + woman * np.sqrt(1 - 0.95 ** 2)},    # the pitch tracker doubled his F0
                {"w": 3, "spk": "high", "e": woman}]
        out, detail = self.sd.summarise(rows)
        self.assertEqual(out["low"][3], 0.95)
        self.assertEqual(out["high"][3], 1.0)
        self.assertEqual(detail["low"][3], {"n": 1, "min": 0.95, "relabelled": 1})
        self.assertEqual(detail["high"][3], {"n": 1, "min": 1.0, "relabelled": 0})
        # with one reference only, the pitch label stands
        out, detail = self.sd.summarise([{"w": 1, "spk": "low", "e": man}, {"w": 2, "spk": "high", "e": man}])
        self.assertEqual(out, {"low": {1: 1.0}})

    def test_one_reference_segment_has_no_leave_one_out(self):
        out, detail = self.sd.summarise([{"w": 1, "spk": "high", "e": [1.0, 0.0]}, {"w": 2, "spk": "high", "e": [0.0, 1.0]}])
        self.assertEqual(out, {"high": {1: 1.0, 2: 0.0}})
        self.assertEqual(detail["high"]["w1_loo"], [])


if __name__ == "__main__":
    unittest.main()
