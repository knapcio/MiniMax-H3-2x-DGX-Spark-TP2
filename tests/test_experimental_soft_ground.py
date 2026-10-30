import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "results/20261030-painting-fl2va-continuation/experimental-soft-ground.py"
spec = importlib.util.spec_from_file_location("soft_ground", SCRIPT)
soft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soft)


class SoftGroundTests(unittest.TestCase):
    def test_mask_leaves_sky_and_limits_ground_change(self):
        reference = Image.new("L", (20, 20), 100)
        current = Image.new("L", (20, 20), 140)
        mask = soft.lower_mask(20, 20, 0.55, 0.72)
        result = soft.correct(reference, current, mask, radius=2, strength=0.5, limit=10)
        self.assertEqual(result.getpixel((10, 5)), 140)
        self.assertEqual(result.getpixel((10, 19)), 130)
        self.assertEqual(soft.correct(reference, reference, mask, 2, 0.5, 10).tobytes(), reference.tobytes())

    def test_video_preserves_sky_chroma_and_audio_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "source.mp4", root / "target.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24",
                            "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=48000", "-t", "1",
                            "-vf", "eq=brightness=0.2*t:eval=frame", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", str(source)], check=True)
            def decoded(path):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                                "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            def audio(path):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                                "-c:a", "copy", "-f", "data", "-"])
            subprocess.run([sys.executable, str(SCRIPT), str(source), str(target), "--radius", "3"],
                           check=True, capture_output=True, text=True)
            before, after = decoded(source), decoded(target)
            ysize, frame_size = 64 * 48, 64 * 48 * 3 // 2
            self.assertEqual(len(before), len(after))
            self.assertEqual(len(before) // frame_size, 24)
            self.assertEqual(audio(source), audio(target))
            for offset in range(0, len(before), frame_size):
                self.assertEqual(before[offset + ysize:offset + frame_size], after[offset + ysize:offset + frame_size])
                self.assertEqual(before[offset:offset + 64 * 20], after[offset:offset + 64 * 20])
            self.assertTrue(before != after, "no luminance frame was altered")
            prior = target.read_bytes()
            result = subprocess.run([sys.executable, str(SCRIPT), str(source), str(target)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_bytes(), prior)
            self.assertFalse(list(root.glob(".h3-soft-ground-*")))


if __name__ == "__main__":
    unittest.main()

