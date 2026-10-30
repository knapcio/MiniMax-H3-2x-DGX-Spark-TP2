import importlib.util
from pathlib import Path
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "results/20261030-painting-fl2va-continuation/experimental-regional-deflicker.py"
spec = importlib.util.spec_from_file_location("regional_deflicker", SCRIPT)
regional = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regional)


class RegionalLumaTests(unittest.TestCase):
    def image(self, value):
        return Image.new("L", (32, 32), value)

    def test_constant_drop_restored(self):
        out = regional.correct_luma(self.image(100), self.image(94), self.image(100), 0.5)
        self.assertEqual(out.getpixel((10, 10)), 100)

    def test_limit_bounds_correction(self):
        out = regional.correct_luma(self.image(100), self.image(80), self.image(100), 0.5, limit=8)
        self.assertEqual(out.getpixel((10, 10)), 88)

    def test_unaffected_frame_stays_unchanged(self):
        original = self.image(120)
        out = regional.correct_luma(original, original, original, 0.5)
        self.assertEqual(out.tobytes(), original.tobytes())


if __name__ == "__main__":
    unittest.main()
