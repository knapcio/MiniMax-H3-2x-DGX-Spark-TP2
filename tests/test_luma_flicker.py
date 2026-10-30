import unittest

from scripts.luma_flicker import periodic_drops


class PeriodicDropsTests(unittest.TestCase):
    def test_detects_only_consecutive_equal_phase_dips(self):
        luma = [100.0] * 140
        for frame in (17, 34, 51, 102, 120):
            luma[frame] = 94.0
        runs = periodic_drops(luma)
        self.assertEqual([[frame for frame, _ in run] for run in runs], [[17, 34, 51]])
        self.assertEqual([drop for _, drop in runs[0]], [-6.0, -6.0, -6.0])

    def test_small_or_nonperiodic_changes_are_not_runs(self):
        luma = [100.0] * 100
        for frame, value in ((17, 98.1), (34, 98.1), (51, 98.1), (70, 90.0), (87, 90.0)):
            luma[frame] = value
        self.assertEqual(periodic_drops(luma), [])


if __name__ == "__main__":
    unittest.main()
