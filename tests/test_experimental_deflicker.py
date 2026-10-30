import importlib.util
from pathlib import Path
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "results" / "20261030-painting-fl2va-continuation" /
          "experimental-deflicker.py")
spec = importlib.util.spec_from_file_location("experimental_deflicker", SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class DeflickerExperimentTests(unittest.TestCase):
    def test_only_detected_short_dips_receive_bounded_luma_corrections(self):
        means = [100.0] * 100
        for frame in (17, 34, 51):
            means[frame:frame + 3] = [94.0, 96.0, 99.0]
        groups, corrections = repair.make_adjustments(means)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(corrections), {17, 18, 19, 34, 35, 36, 51, 52, 53})
        self.assertEqual([corrections[i] for i in (17, 18, 19)], [6.0, 4.0, 1.0])
        self.assertLessEqual(max(corrections.values()), 8.0)

    def test_no_periodic_run_means_no_adjustment(self):
        means = [100.0] * 80
        means[17] = means[34] = 94.0
        self.assertEqual(repair.make_adjustments(means), ([], {}))


if __name__ == "__main__":
    unittest.main()
