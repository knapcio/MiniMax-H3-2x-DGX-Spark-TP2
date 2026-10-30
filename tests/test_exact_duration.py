import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/exact_duration.py"
spec = importlib.util.spec_from_file_location("exact_duration", SCRIPT)
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


class ExactDurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24:duration=2",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=2",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", "-y", str(cls.source)], check=True)
        cls.silent = cls.root / "silent.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(cls.source), "-map", "0:v:0", "-c:v", "copy",
                        "-an", "-y", str(cls.silent)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frame_rounding_and_invalid_durations(self):
        self.assertEqual(exact.target_frames("30"), 720)
        self.assertEqual(exact.target_frames("0.1"), 2)
        for bad in ("0", "-1", "NaN", "Infinity", "not-a-number"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                exact.target_frames(bad)

    def test_video_and_aac_are_verified_and_input_remains_intact(self):
        output = self.root / "exact-audio.mp4"
        before = hashlib.sha256(self.source.read_bytes()).digest()
        frames, seconds, gap = exact.trim(self.source, output, "1.5")
        self.assertEqual((frames, seconds), (36, 1.5))
        self.assertGreaterEqual(gap, 0)
        self.assertLess(gap, 0.04)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), before)
        self.assertEqual(exact.frame_hashes(self.source, frames), exact.frame_hashes(output, frames))

    def test_silent_video_and_fractional_frame_rounding(self):
        output = self.root / "exact-silent.mp4"
        frames, seconds, gap = exact.trim(self.silent, output, "1.1")
        self.assertEqual(frames, 26)
        self.assertAlmostEqual(seconds, 26 / 24)
        self.assertIsNone(gap)

    def test_never_overwrites_or_extends_the_source(self):
        output = self.root / "exists.mp4"
        output.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(ValueError, "new file"):
            exact.trim(self.source, output, "1")
        self.assertEqual(output.read_bytes(), b"do not overwrite")
        absent = self.root / "too-long.mp4"
        with self.assertRaisesRegex(ValueError, "too few frames"):
            exact.trim(self.source, absent, "3")
        self.assertFalse(absent.exists())

    def test_failed_frame_verification_does_not_publish_partial_output(self):
        output = self.root / "rejected.mp4"
        with patch.object(exact, "frame_hashes", side_effect=[["source"], ["different"]]):
            with self.assertRaisesRegex(ValueError, "frames changed"):
                exact.trim(self.source, output, "1")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".h3-exact-*")))


if __name__ == "__main__":
    unittest.main()
