"""The tap on the denoising loop's update call (h3_tp2/speech_early.py): which calls it hears, and that it only reads."""

import types
import unittest

from h3_tp2 import speech_early as se


def loop_module():
    """A stand-in for upstream's denoise_loop module: the update function and a loop that calls it like upstream does."""
    dl = types.ModuleType("fake_denoise_loop")
    dl.calls = []

    def minimax_h3_euler_eta0_step(target, x0, *, sigma_curr, sigma_next):
        dl.calls.append((target, x0, sigma_curr, sigma_next))
        return ("next", target, x0)

    def loop(steps):
        out = []
        for step in range(1, steps + 1):   # video first, audio second, as in denoise_loop.py
            out.append(dl.minimax_h3_euler_eta0_step(f"v{step}", f"x0v{step}", sigma_curr=step, sigma_next=step + 1))
            out.append(dl.minimax_h3_euler_eta0_step(f"a{step}", f"x0a{step}", sigma_curr=step, sigma_next=step + 1))
        return out

    dl.minimax_h3_euler_eta0_step = minimax_h3_euler_eta0_step
    dl.loop = loop
    return dl


class ParseTests(unittest.TestCase):
    def test_steps_are_sorted_and_unique(self):
        self.assertEqual(se.parse_steps("10, 6,8,6"), (6, 8, 10))
        self.assertEqual(se.parse_steps(""), ())

    def test_stops_carry_their_own_limit_or_the_guards(self):
        self.assertEqual(se.parse_stops("14, 10:0.03"), {10: 0.03, 14: None})
        self.assertEqual(list(se.parse_stops("14,10:0.03")), [10, 14])
        self.assertEqual(se.parse_stops(""), {})
        for bad in ("ten", "0:0.1", "10:2", "10:0", "10:x"):
            with self.assertRaises(ValueError):
                se.parse_stops(bad)

    def test_bad_steps_are_named(self):
        with self.assertRaisesRegex(ValueError, "'x'"):
            se.parse_steps("6,x")
        with self.assertRaisesRegex(ValueError, "start at 1"):
            se.parse_steps("0")


class TapTests(unittest.TestCase):
    def test_the_audio_estimate_of_the_chosen_steps_reaches_the_sink(self):
        dl = loop_module()
        self.assertTrue(se.install(dl))
        heard = []
        with se.tapping((2, 4), lambda step, x0: heard.append((step, x0))):
            dl.loop(5)
        self.assertEqual(heard, [(2, "x0a2"), (4, "x0a4")])

    def test_the_update_is_called_and_answers_as_without_the_tap(self):
        plain, tapped = loop_module(), loop_module()
        se.install(tapped)
        with se.tapping((1, 2, 3), lambda step, x0: None):
            got = tapped.loop(3)
        self.assertEqual(got, plain.loop(3))
        self.assertEqual(tapped.calls, plain.calls)

    def test_nothing_is_heard_outside_the_block_or_without_steps(self):
        dl = loop_module()
        se.install(dl)
        heard = []
        dl.loop(2)
        with se.tapping((), lambda step, x0: heard.append(step)):
            dl.loop(2)
        with se.tapping((1,), None):
            dl.loop(2)
        with se.tapping((1,), lambda step, x0: heard.append(step)):
            pass
        dl.loop(2)
        self.assertEqual(heard, [])
        self.assertIsNone(se._ACTIVE)

    def test_every_sampling_counts_its_own_steps(self):
        dl = loop_module()
        se.install(dl)
        heard = []
        for _ in range(2):
            with se.tapping((1,), lambda step, x0: heard.append((step, x0))):
                dl.loop(2)
        self.assertEqual(heard, [(1, "x0a1"), (1, "x0a1")])

    def test_a_failing_sink_does_not_break_the_loop(self):
        dl = loop_module()
        se.install(dl)

        def boom(step, x0):
            raise RuntimeError("no decoder")

        with self.assertLogs("vllm_omni.h3_tp2.speech", level="WARNING") as logs:
            with se.tapping((1,), boom):
                out = dl.loop(2)
        self.assertEqual(len(out), 4)
        self.assertIn("skipped at step 1", logs.output[0])

    def test_a_stop_from_the_sink_ends_the_loop(self):
        dl = loop_module()
        se.install(dl)

        def sink(step, x0):
            raise se.Stop(step, 0.052, 0.01)

        with self.assertRaises(se.Stop) as caught:
            with se.tapping((2,), sink):
                dl.loop(5)
        self.assertEqual((caught.exception.step, caught.exception.cer, caught.exception.limit), (2, 0.052, 0.01))
        self.assertEqual(len(dl.calls), 3)   # video 1, audio 1, video 2: the audio update of step 2 was not made
        self.assertIsNone(se._ACTIVE)

    def test_the_block_ends_the_tap_when_the_sampling_raises(self):
        dl = loop_module()
        se.install(dl)
        with self.assertRaises(KeyError):
            with se.tapping((1,), lambda step, x0: None):
                raise KeyError("cancelled")
        self.assertIsNone(se._ACTIVE)

    def test_installing_twice_wraps_once(self):
        dl = loop_module()
        se.install(dl)
        first = dl.minimax_h3_euler_eta0_step
        se.install(dl)
        self.assertIs(dl.minimax_h3_euler_eta0_step, first)
        heard = []
        with se.tapping((1,), lambda step, x0: heard.append(step)):
            dl.loop(1)
        self.assertEqual(heard, [1])

    def test_an_unreviewed_loop_is_left_alone(self):
        dl = loop_module()
        before = dl.minimax_h3_euler_eta0_step
        with self.assertLogs("vllm_omni.h3_tp2.speech", level="WARNING"):
            self.assertFalse(se.install(dl, source_ok=False))
        self.assertIs(dl.minimax_h3_euler_eta0_step, before)
        with self.assertLogs("vllm_omni.h3_tp2.speech", level="WARNING"):
            self.assertFalse(se.install(types.ModuleType("empty")))


if __name__ == "__main__":
    unittest.main()
