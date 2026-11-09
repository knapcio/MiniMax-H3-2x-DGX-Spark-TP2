"""CPU tests of h3_tp2/ladder.py: the extended sigma ladder."""
import unittest

from h3_tp2 import ladder


class ParseTests(unittest.TestCase):
    def test_off(self):
        for text in (None, "", "0", "off"):
            self.assertIsNone(ladder.parse_tail(text))

    def test_values(self):
        self.assertEqual(ladder.parse_tail("0.025,0.0125"), (0.025, 0.0125))
        self.assertEqual(ladder.parse_tail(" 0.03 "), (0.03,))

    def test_bad(self):
        for text in ("0.0125,0.025", "0.05,0.05", "1", "0.1,nan", "x"):
            with self.assertRaises(ValueError):
                ladder.parse_tail(text)


class PositionsTests(unittest.TestCase):
    def test_stock_grid_plus_tail(self):
        grid = ladder.positions(20, (0.025, 0.0125))
        self.assertEqual(len(grid), 23)
        self.assertEqual(grid[0], 1.0)
        self.assertAlmostEqual(grid[19], 0.05)
        self.assertEqual(grid[20:], [0.025, 0.0125, 0.0])
        self.assertTrue(all(a > b for a, b in zip(grid, grid[1:])))

    def test_video_sigmas_of_the_tail(self):
        # the flow shift of 12: sigma = 12u / (1 + 11u); the stock ladder ends 0.39 -> 0, the tail 0.39 -> 0.24 -> 0.13 -> 0
        def sigma(u):
            return 12 * u / (1 + 11 * u)
        self.assertAlmostEqual(sigma(0.05), 0.387, places=3)
        self.assertAlmostEqual(sigma(0.025), 0.235, places=3)
        self.assertAlmostEqual(sigma(0.0125), 0.132, places=3)

    def test_tail_must_sit_below_the_last_stock_position(self):
        with self.assertRaises(ValueError):
            ladder.positions(20, (0.05,))
        with self.assertRaises(ValueError):
            ladder.positions(0, (0.01,))


class ApplyTests(unittest.TestCase):
    def test_off_returns_the_same_dict(self):
        args = {"num_steps": 20, "base_schedule": None}
        self.assertIs(ladder.apply(args, None), args)
        self.assertIs(ladder.apply(args, ()), args)

    def test_extends(self):
        out = ladder.apply({"num_steps": 20, "base_schedule": None, "seed": 3}, (0.025, 0.0125))
        self.assertEqual(out["num_steps"], 22)
        self.assertEqual(len(out["base_schedule"]), 23)
        self.assertEqual(out["seed"], 3)

    def test_refuses_a_pinned_schedule(self):
        with self.assertRaises(ValueError):
            ladder.apply({"num_steps": 4, "base_schedule": [1.0, 0.5, 0.0]}, (0.1,))


if __name__ == "__main__":
    unittest.main()
