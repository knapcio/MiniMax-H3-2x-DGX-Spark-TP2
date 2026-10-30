import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/regional_flicker.py"
spec = importlib.util.spec_from_file_location("regional_flicker", SCRIPT)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


class RegionalFlickerTests(unittest.TestCase):
    def test_finds_repeated_local_spikes_with_stable_baseline(self):
        changes = [0.2] * 120
        for i in (20, 37, 54, 71, 88):
            changes[i] = 1.1
        groups = detector.periodic_spikes(changes)
        self.assertEqual([[item[0] for item in run] for run in groups], [[20, 37, 54, 71, 88]])

    def test_requires_consecutive_periods_and_local_contrast(self):
        changes = [0.4] * 100
        for i in (20, 37, 71):
            changes[i] = 1.0  # under 3x nearby; also not three consecutive periods
        self.assertEqual(detector.periodic_spikes(changes), [])

    def test_rejects_invalid_parameters(self):
        for params in ({"period": 0}, {"threshold": 0}, {"ratio": 1}, {"minimum": 1}, {"radius": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                detector.periodic_spikes([0.1] * 30, **params)


if __name__ == "__main__":
    unittest.main()
