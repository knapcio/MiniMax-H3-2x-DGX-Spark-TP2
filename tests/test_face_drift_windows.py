"""`face_drift.py` reports per window: the clip's tail belongs to the last window, not to a window of its own."""
import importlib.util
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("face_drift", SCRIPTS / "face_drift.py")
face_drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face_drift)


class Windows(unittest.TestCase):
    def test_window_count_matches_the_renderer(self):
        self.assertEqual(face_drift.window_count(277, 277, 22), 1)
        self.assertEqual(face_drift.window_count(1440, 277, 22), 6)   # 60 s: five full steps and a short sixth window
        self.assertEqual(face_drift.window_count(1552, 277, 22), 6)   # 64.6 s: six full windows
        self.assertEqual(face_drift.window_count(1553, 277, 22), 7)
        self.assertEqual(face_drift.window_count(100, 277, 22), 1)
        with self.assertRaises(ValueError):
            face_drift.window_count(1000, 22, 22)

    def test_the_tail_stays_in_the_last_window(self):
        count = face_drift.window_count(1552, 277, 22)
        self.assertEqual(face_drift.window_of(0, 277, 22, count), 1)
        self.assertEqual(face_drift.window_of(254, 277, 22, count), 1)
        self.assertEqual(face_drift.window_of(255, 277, 22, count), 2)
        self.assertEqual(face_drift.window_of(1530, 277, 22, count), 6)  # frame 1530 = 63.75 s used to print "window 7"
        self.assertEqual(face_drift.window_of(1551, 277, 22, count), 6)


if __name__ == "__main__":
    unittest.main()
