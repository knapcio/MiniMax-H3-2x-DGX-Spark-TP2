"""Window geometry of the speech guard dry run."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import speech_guard_dryrun as dr  # noqa: E402


class WindowStarts(unittest.TestCase):
    def test_windows_of_the_default_geometry(self):
        self.assertEqual(dr.window_starts(277, 277, 22), [(0, 277)])
        self.assertEqual(dr.window_starts(124, 277, 22), [(0, 124)])
        # 64.6 s = 1552 frames = six full windows; 60 s = 1440 frames = five full windows and one of 165 frames
        self.assertEqual(dr.window_starts(1552, 277, 22), [(k * 255, 277) for k in range(6)])
        self.assertEqual(dr.window_starts(1440, 277, 22), [(k * 255, 277) for k in range(5)] + [(1275, 165)])


if __name__ == "__main__":
    unittest.main()
