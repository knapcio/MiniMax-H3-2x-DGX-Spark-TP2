"""Tests of scripts/first_frame_pin.py: the latent count, the mask grid, and the source clip when ffmpeg is installed."""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("first_frame_pin", Path(__file__).resolve().parents[1] / "scripts" / "first_frame_pin.py")
fp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fp)


class LatentTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(fp.latent_frames(192), 57)    # 8.0 s shot
        self.assertEqual(fp.latent_frames(277), 82)    # the serving window
        self.assertEqual(fp.latent_frames(5), 2)
        for bad in (4, 240, 100):
            with self.assertRaises(ValueError):
                fp.latent_frames(bad)


class MaskTests(unittest.TestCase):
    def test_token_grid_with_the_first_latent_pinned(self):
        m = fp.mask(57, 768, 1344)
        self.assertEqual((len(m), len(m[0]), len(m[0][0])), (57, 24, 42))
        self.assertTrue(all(v == 0.0 for row in m[0] for v in row))
        self.assertTrue(all(v == 1.0 for t in m[1:] for row in t for v in row))
        self.assertEqual(sum(v == 0.0 for t in fp.mask(57, 768, 1344, pinned=3) for row in t for v in row), 3 * 24 * 42)

    def test_bad_sizes(self):
        with self.assertRaises(ValueError):
            fp.mask(57, 770, 1344)
        with self.assertRaises(ValueError):
            fp.mask(57, 768, 1344, pinned=0)
        with self.assertRaises(ValueError):
            fp.mask(57, 768, 1344, pinned=58)

    def test_tail_pins_the_last_latents(self):
        m = fp.mask(57, 768, 1344, pinned=1, pinned_tail=1)
        zero = [t for t in range(57) if all(v == 0.0 for row in m[t] for v in row)]
        self.assertEqual(zero, [0, 56])
        self.assertEqual([t for t in range(5) if all(v == 0.0 for row in fp.mask(5, 64, 96, 2, 2)[t] for v in row)], [0, 1, 3, 4])
        with self.assertRaises(ValueError):
            fp.mask(5, 64, 96, 3, 3)
        with self.assertRaises(ValueError):
            fp.mask(5, 64, 96, 1, -1)

    def test_json_round_trip_is_plain_floats(self):
        text = json.dumps(fp.mask(2, 64, 96))
        self.assertEqual(json.loads(text), [[[0.0, 0.0, 0.0]] * 2, [[1.0, 1.0, 1.0]] * 2])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class SourceTests(unittest.TestCase):
    def test_main_writes_a_sized_clip_and_the_mask(self):
        d = tempfile.mkdtemp()
        try:
            still = os.path.join(d, "still.png")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=teal:s=320x180",
                            "-frames:v", "1", still], check=True)
            fp.main([still, "--frames", "22", "--width", "192", "--height", "96", "--out-dir", d])
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                    "stream=width,height,nb_read_packets", "-of", "csv=p=0", os.path.join(d, "source.mp4")],
                                   capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(probe, "192,96,24")
            m = json.load(open(os.path.join(d, "mask.json")))
            self.assertEqual((len(m), len(m[0]), len(m[0][0])), (7, 3, 6))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_last_still_fills_the_final_four_frames_of_a_full_length_source(self):
        d = tempfile.mkdtemp()
        try:
            first, last = os.path.join(d, "first.png"), os.path.join(d, "last.png")
            for path, colour in ((first, "0x0000ff"), (last, "0xff0000")):
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=320x180",
                                "-frames:v", "1", path], check=True)
            fp.main([first, "--last", last, "--frames", "22", "--width", "192", "--height", "96", "--out-dir", d])
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                    "stream=nb_read_packets", "-of", "csv=p=0", os.path.join(d, "source.mp4")],
                                   capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(probe, "22")
            raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", os.path.join(d, "source.mp4"), "-vf", "scale=1:1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            pixels = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
            self.assertEqual(len(pixels), 22)
            self.assertTrue(all(p[2] > 200 and p[0] < 60 for p in pixels[:18]), pixels[:18])   # blue up to frame 17
            self.assertTrue(all(p[0] > 200 and p[2] < 60 for p in pixels[18:]), pixels[18:])   # red on the last four
            m = json.load(open(os.path.join(d, "mask.json")))
            self.assertEqual([t for t in range(7) if m[t][0][0] == 0.0], [0, 6])
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
