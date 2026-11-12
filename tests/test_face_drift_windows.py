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


class Turn(unittest.TestCase):
    def test_frontal_three_quarter_profile(self):
        eyes = [(100.0, 100.0), (140.0, 100.0)]
        self.assertAlmostEqual(face_drift.head_turn(eyes + [(120.0, 120.0), (105, 140), (135, 140)]), 0.0)
        self.assertAlmostEqual(face_drift.head_turn(eyes + [(140.0, 120.0), (105, 140), (135, 140)]), 0.5)
        self.assertAlmostEqual(face_drift.head_turn(eyes + [(100.0, 120.0), (105, 140), (135, 140)]), -0.5)
        # a profile: the eyes almost coincide and the nose sits far out; clamped
        self.assertEqual(face_drift.head_turn([(100.0, 100.0), (101.0, 100.0), (130.0, 118.0), (0, 0), (0, 0)]), 3.0)
        self.assertEqual(face_drift.head_turn([(100.0, 100.0), (101.0, 100.0), (70.0, 118.0), (0, 0), (0, 0)]), -3.0)

    def test_pose_matched_is_about_the_reference_pose(self):
        self.assertTrue(face_drift.pose_matched(0.1, 0.0))
        self.assertTrue(face_drift.pose_matched(-0.7, 0.56))     # the mirror side of the same turn
        self.assertFalse(face_drift.pose_matched(0.9, 0.56))     # a profile against a three-quarter reference
        self.assertFalse(face_drift.pose_matched(0.0, 0.56))     # frontal against a three-quarter reference

    def test_tilt_does_not_count_as_turn(self):
        # the head rolled 30 degrees, nose on the eye midline
        import math
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        le, re = (-20.0, 0.0), (20.0, 0.0)
        rot = lambda x, y: (x * c - y * s + 100, x * s + y * c + 100)   # noqa: E731
        self.assertAlmostEqual(face_drift.head_turn([rot(*le), rot(*re), rot(0.0, 20.0), (0, 0), (0, 0)]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
