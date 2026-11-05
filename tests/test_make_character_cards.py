"""CPU checks of scripts/make_character_cards.py with explicit boxes (no detector)."""
import importlib.util
import pathlib
import tempfile
import unittest

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_character_cards", ROOT / "scripts/make_character_cards.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class CardTests(unittest.TestCase):
    def test_crop_box_is_clamped_and_centred_on_the_face(self):
        self.assertEqual(m.crop_box((100, 100, 200, 220), 1344, 768), (30, 0, 270, 358))
        self.assertEqual(m.crop_box((0, 0, 10, 10), 100, 100), (0, 0, 17, 21))
        with self.assertRaises(ValueError):
            m.crop_box((5, 5, 5, 9), 100, 100)

    def test_small_crops_are_upscaled_to_the_minimum_edge(self):
        self.assertEqual(m.upscale(Image.new("RGB", (236, 376)), 512).size, (512, 816))
        self.assertEqual(m.upscale(Image.new("RGB", (600, 900)), 512).size, (600, 900))

    def test_cli_writes_one_crop_per_box_left_to_right(self):
        with tempfile.TemporaryDirectory() as d:
            scene = pathlib.Path(d) / "scene.png"; Image.new("RGB", (640, 360), "gray").save(scene)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = m.main([str(scene), "--out-dir", d, "--box", "400,80,460,150", "--box", "100,90,150,150"])
            self.assertEqual(rc, 0)
            line = buf.getvalue().strip()
            self.assertTrue(line.startswith("H3_REF_IMAGES=" + str(scene.resolve()) + ","))
            parts = line.split("=", 1)[1].split(",")
            self.assertEqual([pathlib.Path(p).name for p in parts], ["scene.png", "scene_face0.png", "scene_face1.png"])
            self.assertGreaterEqual(min(Image.open(parts[1]).size), 512)


if __name__ == "__main__":
    unittest.main()
