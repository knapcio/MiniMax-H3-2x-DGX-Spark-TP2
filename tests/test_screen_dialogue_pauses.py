import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave


PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'screen_dialogue_pauses.py'
SPEC = importlib.util.spec_from_file_location('screen_dialogue_pauses', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PauseScreenTest(unittest.TestCase):
    def test_actual_quiet_and_voiced_intervals(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'sample.wav'
            rate = 8000
            samples = [int(12000 * math.sin(2 * math.pi * 440 * i / rate)) if rate <= i < 2 * rate else 0
                       for i in range(4 * rate)]
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
            result = MODULE.screen(path)
            self.assertEqual(len(result['quiet_intervals']), 2)
            self.assertAlmostEqual(result['quiet_total_s'], 3, delta=.01)
            self.assertAlmostEqual(result['longest_quiet_s'], 2, delta=.01)
            self.assertAlmostEqual(result['quiet_fraction'], .75, delta=.005)

    def test_threshold_must_be_valid(self):
        with self.assertRaises(ValueError):
            MODULE.screen(Path('unused'), threshold_db=float('nan'))
        with self.assertRaises(ValueError):
            MODULE.screen(Path('unused'), minimum_s=0)


if __name__ == '__main__':
    unittest.main()
