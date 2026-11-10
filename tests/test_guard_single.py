"""CPU tests of h3_tp2/guard_single.py: a one-window request gets the one-window continuation geometry."""
import unittest

from h3_tp2 import guard_single as gs


class OneWindowTests(unittest.TestCase):
    def test_grid_and_range(self):
        self.assertEqual(gs.one_window(243), (243, 22))     # 10 s
        self.assertEqual(gs.one_window(124), (124, 22))     # 4.5 s
        self.assertEqual(gs.one_window(345), (345, 22))
        self.assertIsNone(gs.one_window(362))              # 15 s: longer than one window
        self.assertIsNone(gs.one_window(240))              # off the 17n+5 grid
        self.assertIsNone(gs.one_window(90))
        self.assertIsNone(gs.one_window(None))
        self.assertIsNone(gs.one_window("x"))


class JudgeTests(unittest.TestCase):
    def test_sets_geometry_once(self):
        ctx = {"continuation": None, "num_frames": 243}
        self.assertTrue(gs.judge(ctx))
        self.assertEqual(ctx["continuation"], (243, 22))

    def test_leaves_long_video_and_long_requests_alone(self):
        ctx = {"continuation": (277, 22), "num_frames": 1552}
        self.assertFalse(gs.judge(ctx))
        self.assertEqual(ctx["continuation"], (277, 22))
        ctx = {"continuation": None, "num_frames": 362}
        self.assertFalse(gs.judge(ctx))
        self.assertIsNone(ctx["continuation"])


class InstallTests(unittest.TestCase):
    def test_wraps_the_method_once_and_never_breaks_it(self):
        class Pipe:
            def _prepare_request_inputs(self, raw_prompt, sampling):
                return {"continuation": None, "num_frames": 124, "seed": 7}

        gs.install(Pipe)
        gs.install(Pipe)
        self.assertTrue(getattr(Pipe._prepare_request_inputs, "_h3_tp2_guard_single", False))
        with self.assertLogs(gs.logger, level="INFO"):
            ctx = Pipe()._prepare_request_inputs("p", None)
        self.assertEqual(ctx["continuation"], (124, 22))
        self.assertEqual(ctx["seed"], 7)

        class Odd:
            def _prepare_request_inputs(self, raw_prompt, sampling):
                return "not a dict"

        gs.install(Odd)
        self.assertEqual(Odd()._prepare_request_inputs("p", None), "not a dict")


if __name__ == "__main__":
    unittest.main()
