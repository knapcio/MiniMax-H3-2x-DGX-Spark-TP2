"""Detect real edge bands without treating colored interior as removable pixels."""
import importlib.util
from pathlib import Path
import unittest
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location("border", Path(__file__).resolve().parents[1] / "scripts/input_border_check.py")
border = importlib.util.module_from_spec(spec)
spec.loader.exec_module(border)


class BorderTests(unittest.TestCase):
    def test_white_bottom_band_and_colored_picture(self):
        image = Image.new("RGB", (64, 48), (40, 90, 150))
        draw = ImageDraw.Draw(image)
        for x in range(64):
            draw.line((x, 0, x, 42), fill=(x * 3, 80, 140))
        draw.rectangle((0, 43, 63, 47), fill="white")
        result = border.inspect(image, tolerance=0)
        self.assertEqual(result["bands"]["bottom"]["pixels"], 5)
        self.assertFalse(result["bands"]["bottom"]["scan_limit_reached"])
        self.assertEqual(result["bands"]["bottom"]["rgb"], [255, 255, 255])

    def test_fully_uniform_image_is_ambiguous(self):
        result = border.inspect(Image.new("RGB", (80, 80), "white"))
        self.assertTrue(all(row["scan_limit_reached"] for row in result["bands"].values()))
        self.assertIn("intentional", result["note"])

    def test_transparent_bottom_band(self):
        image = Image.new("RGBA", (64, 48), (10, 60, 100, 255))
        ImageDraw.Draw(image).rectangle((0, 44, 63, 47), fill=(90, 0, 180, 0))
        result = border.inspect(image)
        self.assertEqual(result["bands"]["bottom"]["pixels"], 4)
        self.assertEqual(result["bands"]["bottom"]["kind"], "transparent")


if __name__ == "__main__":
    unittest.main()
